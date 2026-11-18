"""ROS front end for the unchanged walking KCF-iCEM controller.
"""

import argparse
import json
import os
from time import monotonic
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String
from nav_msgs.msg import Odometry
from kcf_interface import COMMAND_TOPIC, FEEDBACK_TOPIC, SHADOW_TOPIC, MOCAP_TOPIC, StateProvider
from kcf_runtime import load_controller, load_profile
from kcf_keyboard import KEY_HELP, KeyboardOperator, TerminalKeyboard
from utils.experiment_utils import EXPERIMENT_INFO_TOPIC, experiment_info_qos


class KCFNode(Node):
    def __init__(self, args, *, keyboard=None):
        profile = load_profile(getattr(args, 'profile', 'walking'))
        decoder = profile.symbol('feedback_decoder')
        state_source = getattr(args, 'state_source', None)
        if state_source is None:
            state_source = profile.values['state_source']
        self.command_limits = np.asarray(profile.symbol('robot_config_factory')().cmd_scale, dtype=float)
        super().__init__("kcf_mpc_node")
        self.enabled = not args.shadow
        self.keyboard = keyboard
        self.keyboard_operator = KeyboardOperator() if keyboard is not None else None
        self.quit_requested = False
        self._keyboard_tick = None
        self._keyboard_publish = -float('inf')
        self.valid_feedback_wall, self.valid_feedback_time = -float('inf'), None
        self.timeout = 0.1
        self.fsm, self.fsm_wall = "init", -float("inf")
        self.operator_wall = -float("inf")
        self.feedback_wall, self.feedback_time = -float("inf"), None
        self.last_command_wall = -float("inf")
        self.command = np.zeros(3)
        self.command_scale = None
        self.state_provider = StateProvider(state_source)
        self.active, self.ready = False, False
        self.status_text = None

        # Define the ROS publisher and subscriber
        self.command_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 1)
        self.shadow_pub = self.create_publisher(Float64MultiArray, SHADOW_TOPIC, 1)
        self.status_pub = self.create_publisher(String, "/deploy_kcf/status", 1)
        self.create_subscription(Float64MultiArray, FEEDBACK_TOPIC, self.feedback_callback, 1)
        if self.state_provider.source == 'mocap':
            self.create_subscription(Odometry, MOCAP_TOPIC, self.mocap_callback, 1)
        if keyboard is None:
            # Embedders can supply their own operator; the CLI owns its keyboard.
            self.create_subscription(String, "/deploy_robot/fsm", self.fsm_callback, 1)
            self.create_subscription(Float32MultiArray, "/deploy_robot/joystick", self.operator_callback, 1)
        else:
            self.fsm_pub = self.create_publisher(String, '/deploy_robot/fsm', 1)
            self.operator_pub = self.create_publisher(Float32MultiArray, '/deploy_robot/joystick', 1)
            self.create_subscription(String, EXPERIMENT_INFO_TOPIC, self.consumer_callback,
                                     experiment_info_qos())
        
        self.source = None
        self.profile, self.decode_feedback = profile, decoder
        self.status("warming_up; shadow" if not self.enabled else "warming_up; kcf_control")
        try:
            self.source, self.state_from_feedback = load_controller(self.profile.name, args.reference_mode)
            if keyboard is not None:
                keyboard.discard()  # Never execute keys queued during GPU warmup.
        except BaseException:
            try:
                if self.source is not None:
                    self.source.close()
            finally:
                super().destroy_node()
            raise
        self.timeout = min(self.timeout, self.source.live_feedback_timeout)
        self.ready = True
        self.timer = self.create_timer(0.005, self.poll)
        self.status("ready; waiting_for_complete_feedback")
        if keyboard is not None:
            print(KEY_HELP, flush=True)

    def status(self, reason):
        if reason != self.status_text:
            self.get_logger().info(reason)
            self.status_pub.publish(String(data=reason))
            self.status_text = reason

    def invalidate(self, reason):
        if (self.keyboard_operator is not None and self.keyboard_operator.state == 'track'
                and reason not in ('fsm_changed', 'waiting_for_operator_track')):
            self.keyboard_operator.stop()
            self.fsm, self.operator_wall = 'damp', -float('inf')
        was_active = self.active
        self.active = False
        self.command = np.zeros(3)
        self.command_pub.publish(Float64MultiArray(data=[0.0] * 6))
        self.shadow_pub.publish(Float64MultiArray(data=[0.0] * 6))
        if self.keyboard is not None and self.enabled:
            self.operator_pub.publish(Float32MultiArray(data=[0.] * 9))
        self.status(reason)
        if was_active:
            try:
                self.source.reset()
            except Exception as error:
                self.ready = False
                self.get_logger().error(f'KCF reset failed; restart required: {error}')

    def fsm_callback(self, msg):
        if msg.data != self.fsm:
            self.invalidate("fsm_changed")
        self.fsm, self.fsm_wall = msg.data, monotonic()

    def operator_callback(self, msg):
        a = np.asarray(msg.data, dtype=float)
        if a.ndim == 1 and len(a) >= 4 and np.isfinite(a[:4]).all() and a[0] == 1:
            self.operator_wall = monotonic()
        else:
            self.operator_wall = -float("inf")
            self.invalidate("operator_disconnected")

    def permitted(self, now):
        return (not self.enabled or (self.fsm == "track"
                and now - self.fsm_wall < self.timeout
                and now - self.operator_wall < self.timeout))

    def consumer_callback(self, msg):
        # Legacy policy consumers multiply joystick axes by cmd_scale. Undo
        # that wire scaling, so KCF's physical body velocities are unchanged.
        try:
            config = yaml.safe_load(json.loads(msg.data)['config_yaml'])
            scale = np.asarray(config['cmd_scale'], dtype=float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError('invalid command scale')
        except (ValueError, TypeError, KeyError, yaml.YAMLError):
            self.command_scale = None
            self.invalidate('invalid_policy_command_scale')
            return
        if self.command_scale is not None and not np.array_equal(scale, self.command_scale):
            self.invalidate('policy_command_scale_changed; re-arm required')
        self.command_scale = scale

    def mocap_callback(self, msg):
        try:
            self.state_provider.set_mocap(msg, monotonic())
        except ValueError as error:
            self.valid_feedback_wall = -float('inf')
            self.invalidate(str(error))

    def feedback_callback(self, msg):
        now = monotonic()
        try:
            feedback = self.decode_feedback(msg.data, self.timeout)
            feedback = self.state_provider.select(feedback, now, self.timeout)
        except ValueError as error:
            self.valid_feedback_wall = -float('inf')
            self.invalidate(str(error))
            return
        if feedback.time != self.valid_feedback_time:
            self.valid_feedback_wall, self.valid_feedback_time = now, feedback.time
        if not self.ready or not self.permitted(now):
            self.invalidate("runtime_not_ready; restart required" if not self.ready else "waiting_for_operator_track")
            return
        if feedback.time == self.feedback_time:
            return
        if self.feedback_time is not None and feedback.time < self.feedback_time:
            self.invalidate("policy_clock_reset")
            if self.keyboard is not None:
                self.feedback_time = feedback.time
                return  # A restarted simulator must be explicitly re-armed.
        try:
            if not self.active:
                self.source.t_start = feedback.time
                self.source.reset()
            self.feedback_wall, self.feedback_time = now, feedback.time
            self.active = True
            self.source.observe_applied_command(feedback.applied_time, feedback.applied_command)
            state = self.state_from_feedback(feedback)
            # get() retains the existing asynchronous GPU solve worker. It does
            # not wait for an optimization in this ROS callback.
            cmd = np.asarray(self.source.get(state), dtype=float)
            reason = getattr(self.source, 'command_reason', 'initializing')
            stamp = self.source._last_application_time
            stamp = feedback.time + self.source.dt if stamp is None else float(stamp)
            valid = bool(cmd.shape == (3,) and np.isfinite(cmd).all()
                         and np.all(np.abs(cmd) <= self.command_limits + 1e-6)
                         and np.isfinite(stamp) and reason in ('buffered', 'committed'))
            finished = monotonic()
            if finished - self.feedback_wall >= self.timeout or not self.permitted(finished):
                self.invalidate('feedback_or_operator_timeout')
                return
            if not valid:
                cmd = np.zeros(3)
                stamp = feedback.time + self.source.dt
            self.command = cmd.copy()
            packet = [float(valid), *cmd.tolist(), stamp, feedback.time]
            self.shadow_pub.publish(Float64MultiArray(data=packet))
            self.command_pub.publish(Float64MultiArray(data=packet if self.enabled else [0.0]*6))
            self.last_command_wall = finished
            if self.keyboard is not None and self.enabled:
                self.publish_operator(True, finished)
            self.status(('kcf_control; ' if self.enabled else 'shadow; ') + reason)
        except Exception as error:
            self.ready = False
            self.get_logger().error(f'KCF runtime failed: {error}')
            self.invalidate('runtime_failed; restart required')

    def publish_operator(self, connected, now):
        connected = bool(connected and self.ready and self.command_scale is not None)
        live = (connected and self.active and self.fsm == 'track'
                and now - self.last_command_wall < self.timeout)
        velocity = self.command / self.command_scale if live else np.zeros(3)
        # Exact joystick packet: connected, vx, vy, wz, then five unused axes/buttons.
        operator = Float32MultiArray(data=[float(connected), *velocity.tolist(), *([0.] * 5)])
        self.operator_callback(operator)
        self.operator_pub.publish(operator)

    def keyboard_tick(self, now):
        try:
            keys = self.keyboard.read_keys()
            connected = keys is not None
        except (EOFError, OSError) as error:
            self.get_logger().error(f'Keyboard disconnected: {error}')
            connected, keys, self.quit_requested = False, '', True
        delayed = self._keyboard_tick is not None and now - self._keyboard_tick >= self.timeout
        self._keyboard_tick = now
        if not connected or delayed:
            self.keyboard_operator.stop()
            self.invalidate('keyboard_unavailable; re-arm required')
            keys = ''  # Ignore queued arming input after a pause/background job.
            if connected:
                try:
                    self.keyboard.discard()
                except EOFError:
                    connected, self.quit_requested = False, True
        fresh = (self.ready and self.command_scale is not None
                 and now - self.valid_feedback_wall < self.timeout)
        if self.keyboard_operator.state == 'track' and not fresh:
            self.keyboard_operator.stop()
            self.invalidate('feedback_timeout; re-arm required')
        if self.enabled:
            self.keyboard_operator.update(keys or '', feedback_ready=fresh)
        elif keys and any(key in keys.lower() for key in ('q', '\x03', '\x04')):
            self.keyboard_operator.update('q', feedback_ready=False)
        self.quit_requested |= self.keyboard_operator.quit_requested
        # Shadow mode never publishes hardware-visible enable/FSM messages.
        if self.enabled and (now - self._keyboard_publish >= 0.02
                             or self.fsm != self.keyboard_operator.state or not connected):
            state = String(data=self.keyboard_operator.state)
            self.fsm_callback(state)
            self.fsm_pub.publish(state)
            self.publish_operator(connected, now)
            self._keyboard_publish = now

    def poll(self):
        now = monotonic()
        if self.keyboard is not None:
            self.keyboard_tick(now)
        if self.active and (now - self.feedback_wall >= self.timeout or not self.permitted(now)):
            self.invalidate("feedback_or_operator_timeout")
        if self.active and now - self.last_command_wall >= self.timeout:
            self.invalidate('command_timeout; re-arm required')

    # terminate joy_node on shutdown
    def destroy_node(self):
        try:
            if self.context.ok():
                if self.keyboard is not None and self.enabled:
                    self.keyboard_operator.stop()
                    self.fsm_pub.publish(String(data='damp'))
                    self.operator_pub.publish(Float32MultiArray(data=[0.] * 9))
                self.invalidate("shutdown")
        finally:
            try:
                if self.source is not None:
                    self.source.close()
            finally:
                super().destroy_node()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.environ.get('KCF_MODEL_PROFILE', 'walking'), help="Model profile declared in deploy/configs/g1 (default: walking)")
    parser.add_argument("--reference-mode", help="Optional reference selection; MPC gains/settings remain unchanged")
    parser.add_argument('--shadow', action='store_true', help='Compute only; never publish actuation or FSM enable messages')
    parser.add_argument('--state-source', choices=('simulation', 'mocap'),
                        help='Override kcf.state_source in the deployment YAML for this run')
    args, ros_args = parser.parse_known_args(argv)
    if ros_args and ros_args[0] != '--ros-args':
        parser.error('unrecognized arguments: ' + ' '.join(ros_args))
    return args, ros_args


def main():
    args, ros_args = parse_arguments()
    # Validate the terminal before initializing ROS or warming the GPU.
    with TerminalKeyboard() as keyboard:
        run(args, ros_args, keyboard)


def run(args, ros_args, keyboard):
    rclpy.init(args=ros_args)
    node = None
    try:
        node = KCFNode(args, keyboard=keyboard)
        while rclpy.ok() and not node.quit_requested:
            rclpy.spin_once(node, timeout_sec=0.005)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

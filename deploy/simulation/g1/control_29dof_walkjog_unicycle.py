##
#
# Control node for 29DoF MjLab UNICYCLE WALK + JOG in one policy (G1-WalkJog-Unicycle).
#
##


# standard imports
import argparse

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.unitree_utils import get_gravity_orientation
from utils.policy import Policy
from utils.experiment_utils import publish_experiment_info
from utils.locomotion.jog_modes import JogTwistCommander
from utils.locomotion.gait_library import GaitLibraryClock


TRIGGER_OFFSET = 1e-6  # [sec] only absorbs float rounding (sim time has no jitter)


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """Asynchronous control node for the unicycle walk + jog policy (G1-WalkJog-Unicycle).

    Observation: pelvis gyro, joint pos / vel, last action, phase clock, twist, projected gravity.
    ONE library holds two strides, walk (70 frames) and jog (43): each tick the twist snaps to
    its nearest clip (utils/locomotion/gait_library.py), the phase is t / n of THAT clip, and
    crossing the cut remaps the frame to the same phase. The idle clip zeroes twist and phase.
    Joystick (utils/locomotion/jog_modes.py), RT = TURBO: released, the walk bands (forward
    0.50..0.90, backward -0.50..-0.70, pivots 0.50..1.45); pressed, the jog bands.
    No joystick -> default_twist (autonomous, sim only; a hardware node must stand).
    """

    def __init__(self, config_path: str):

        super().__init__('control_node')

        # load config file
        self.config = self.load_config(config_path)

        # load params
        self.init_policy()

        # broadcast which experiment is running so the logger can record it
        self.experiment_info_pub = publish_experiment_info(self, config_path, self.config, self.policy)

        # ROS publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/command', 10)

        # policy step trigger state (see time_callback)
        self.last_step = None     # slot index of the last policy step
        self.step_advance = 1     # slots since the last policy step
        self.slot_time = 0.0      # [sec] ideal sim time of the current policy step

        # ROS subscribers (simulation_time last, so this step's sensors are taken first)
        self.cmd_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick', self.cmd_callback, 10)
        self.pelvis_imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_sensor_callback, 1)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 1)
        self.sim_time_sub = self.create_subscription(Float64, 'deploy_robot/simulation_time', self.time_callback, 1)

        # sensor state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # pelvis (w, x, y, z)
        self.omega = np.zeros(3)                      # pelvis gyro (base_ang_vel)
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # joystick command [vx, vy, wz] sticks in [-1, 1], RT turbo in [0, 1], connection flag
        self.cmd = np.zeros(3)
        self.boost = 0.0
        self._last_boost_print = None
        self.joystick_connected = False

        # active mode (idle / forward / backward / pivot) and gait (walk / jog) for logging
        self.mode = "idle"
        self._last_mode = None
        self._last_period = None

        # initialize the action
        self.action = np.zeros(self.act_size)

        # the gait phase clock lives in self.clock (GaitLibraryClock): integer frame index that
        # wraps at the snapped clip's own length; its observation is blanked on the idle clip

        print("Control node initialized.")


    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it (accept the name with or without the .yaml extension)
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        config_path_full = ROOT_DIR + "/deploy/configs/g1/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from [{config_path_full}].")

        return config

    # initialize the policy
    def init_policy(self):

        # default_joint_pos and action_scale are loaded from the policy metadata
        self.qpos_joints_default = self.config.get('default_joint_pos')
        self.action_scale = self.config.get('action_scale')

        # PD gains
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # joystick shaping (utils/locomotion/jog_modes.py); the gait clock needs the policy
        self.default_twist = np.array(self.config["default_twist"], dtype=np.float32)
        self.commander = JogTwistCommander(self.config)
        self.gait_names = {int(k): str(v) for k, v in self.config.get("gait_names", {}).items()}

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/g1/" + policy_path

        # load the policy
        self.policy = Policy(policy_path_full)

        # alias for convenience
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        # deployment params embedded in the policy
        self.qpos_joints_default = self.policy.get_param('default_joint_pos', self.qpos_joints_default)
        self.action_scale = self.policy.get_param('action_scale', self.action_scale)
        assert len(self.qpos_joints_default) == self.act_size, \
            f"default_joint_pos has {len(self.qpos_joints_default)} values, expected {self.act_size}."
        assert len(self.action_scale) == self.act_size, \
            f"action_scale has {len(self.action_scale)} values, expected {self.act_size}."
        for _k in ('default_joint_pos', 'action_scale'):
            print(f"    {_k}: from {'policy metadata' if _k in self.policy.metadata else 'yaml config'}")

        # the observation this node builds must be the one the policy was trained on:
        # base_ang_vel(3) + joint_pos(n) + joint_vel(n) + actions(n) + twist(3) + phase(2) + gravity(3)
        expected_obs = 3 + 3 * self.act_size + 3 + 2 + 3
        assert self.obs_size == expected_obs, \
            f"policy expects a {self.obs_size}-wide observation, this node builds {expected_obs}."

        # the yaml's home pose must be the policy's default pose: the sim spawns / hardware
        # ramps to home_joint_pos, and a zero action holds default_joint_pos -- they must agree
        home = np.array(self.config["home_joint_pos"], dtype=np.float32)
        gap = float(np.abs(home - self.qpos_joints_default).max())
        assert gap < 2e-3, \
            f"home_joint_pos differs from the policy's default_joint_pos by {gap:.4f} rad; " \
            f"they must be the same standing idle pose."

        # the gait clock: nearest-clip snap + per-clip period, from the policy's library metadata
        self.clock = GaitLibraryClock(self.policy)
        assert self.clock.idle_idx is not None, "the library has no zero-twist idle clip to stand on"

        # the sticks must only ever ask for twists the library actually covers
        lib = self.clock.lib_twists
        for name, lo, hi, axis in (("fwd_vx_range", *self.commander.fwd_vx, 0),
                                   ("fwd_vx_range_boost", *self.commander.forward_band(1.0), 0),
                                   ("bwd_vx_range_boost", *self.commander.backward_band(1.0), 0),
                                   ("pivot_wz_range_boost", *self.commander.pivot_wz_band(1.0), 2),
                                   ("bwd_vx_range", *self.commander.bwd_vx, 0),
                                   ("curve_wz_range", *self.commander.curve_wz, 2),
                                   ("pivot_wz_range", *self.commander.pivot_wz, 2)):
            assert lib[:, axis].min() - 1e-3 <= min(lo, hi) and max(lo, hi) <= lib[:, axis].max() + 1e-3, \
                f"{name} [{lo}, {hi}] leaves the library's span [{lib[:, axis].min()}, {lib[:, axis].max()}]"

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")
        print(f"    Gait library: {self.clock.describe()}")
        print("    Gait periods: " + ", ".join(
            f"{self.gait_name(p)} {p} frames ({p * self.ctrl_dt:.2f} s)" for p in self.clock.periods))
        print(f"    Joystick: {self.commander.describe()}")
        print(f"    No joystick -> autonomous twist {self.default_twist.tolist()} (sim only)")


    #################################################################
    # HELPERS
    #################################################################

    # joystick command: [is_connected, vx, vy, omega, ..., RT at index 9] (sticks in [-1, 1])
    def cmd_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.joystick_connected = (data[0] > 0.5)
        self.cmd = np.array([data[1], data[2], data[3]], dtype=np.float32)
        # RT is turbo (walk bands -> jog bands); older 9-wide senders mean no turbo
        self.boost = float(np.clip(data[9], 0.0, 1.0)) if len(data) > 9 else 0.0

    # pelvis IMU data: [rpy(3), quat(4), gyro(3), acc(3)]
    def pelvis_imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.quat = data[3:7]
        self.omega = data[7:10]

    # joint data: [q(n), dq(n), ddq(n), tau_est(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # simulation time; steps the policy once per control_dt slot of sim time
    def time_callback(self, msg):
        prev_time = self.sim_time
        self.sim_time = msg.data
        step = int((self.sim_time + TRIGGER_OFFSET) // self.ctrl_dt)
        # sim restarted: a new episode of slots
        if self.last_step is None or self.sim_time < prev_time:
            self.last_step, self.step_advance = step, 1
        elif step != self.last_step:
            self.step_advance = step - self.last_step
            self.last_step = step
        else:
            return
        self.slot_time = self.last_step * self.ctrl_dt
        self.control_callback()

    # commanded planar twist [vx, vy, wz]: left stick Y = drive (forward/back), right stick X =
    # steer (left +), both at once. The lateral stick (vy) is ignored: the library has no vy.
    def commanded_twist(self):
        # autonomous jog when no joystick is connected (for viewing in sim)
        if not self.joystick_connected:
            self.mode = "forward" if self.default_twist[0] > 0 else "idle"
            return self.default_twist

        self.mode, twist = self.commander.update(fwd_stick=self.cmd[0], turn_stick=self.cmd[2],
                                                 boost=self.boost)
        # announce the forward band in 25% RT steps
        step = int(round(self.boost * 4))
        if step != self._last_boost_print:
            lo, hi = self.commander.fwd_band
            print(f"[RT] turbo {self.boost:.2f} -> forward vx [{lo:+.2f}, {hi:+.2f}] m/s")
            self._last_boost_print = step
        return twist

    # label for a gait period (walk / jog), falling back to the frame count
    def gait_name(self, period):
        return self.gait_names.get(int(period), f"T={int(period)}")

    # build the observation vector for the policy
    def build_observation(self):

        # proprioception (joint pos relative to default, joint vel)
        qj = self.qpos_joints - self.qpos_joints_default
        dqj = self.qvel_joints

        # commanded twist -> nearest library clip. The clock remaps the frame to the same phase
        # when the snap crosses the walk/jog cut, and on the idle clip zeroes the twist the policy
        # sees (training's idle stop: idle clip <=> zero twist <=> blanked phase)
        twist = self.clock.update(self.commanded_twist())

        # gait phase clock: (sin, cos) of 2*pi*t/n with n the SNAPPED clip's own length (70 walk,
        # 43 jog), BLANKED to (0, 0) on the idle clip so a stopped robot stands still instead of
        # shuffling. Matches mjlab crawling_fwd/mdp/observations.py: motion_phase.
        motion_phase = self.clock.phase_obs()

        # projected gravity from the pelvis IMU (roll/pitch; yaw-invariant)
        proj_grav = get_gravity_orientation(self.quat).astype(np.float32)

        # concatenate in the trained order (from the policy's observation_names metadata):
        # base_ang_vel(3), joint_pos(29), joint_vel(29), actions(29),
        # commanded_twist(3), motion_phase(2), projected_gravity(3) = 98
        obs = np.concatenate([
            self.omega,
            qj, dqj, self.action,
            twist, motion_phase, proj_grav,
        ]).astype(np.float32)

        return obs, twist

    # control published at the control frequency
    def control_callback(self):

        # get the current observation
        obs, twist = self.build_observation()

        # announce mode changes (idle / forward / backward / pivot) with the twist
        if self.mode != self._last_mode:
            print(f"[mode] {self._last_mode} -> {self.mode}  twist vx {twist[0]:+.2f} wz {twist[2]:+.2f}")
            self._last_mode = self.mode

        # announce stride changes (walk <-> jog); idle is its own state, not a stride
        period = None if self.clock.is_idle else self.clock.period
        if period != self._last_period:
            gait = "idle" if period is None else self.gait_name(period)
            clip = self.clock.clip_twist
            print(f"[gait] -> {gait}  clip #{self.clock.clip} (vx {clip[0]:+.2f} wz {clip[2]:+.2f}), "
                  f"frame {self.clock.t}/{self.clock.period}")
            self._last_period = period

        # target joint positions (PD control)
        self.action = self.policy.inference(obs)

        # build the command: [q_des, dq_des, Kp, Kd, tau_ff]
        qpos_des = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.act_size, dtype=np.float32)
        tau_ff = np.zeros(self.act_size, dtype=np.float32)

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]).tolist()
        self.command_pub.publish(cmd_msg)

        # advance the free-running gait clock (wraps at the current clip's length)
        for _ in range(self.step_advance):   # by the slots elapsed
            self.clock.advance()


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Asynchronous Control Node for the MjLab unicycle walk + jog policy.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file. Example: "g1_29dof_walkjog_unicycle.yaml".'
    )
    args = parser.parse_args()

    # create the control node
    ctrl_node = ControlNode(args.config)

    # execute the policy
    try:
        # spin the node
        rclpy.spin(ctrl_node)

    except KeyboardInterrupt:
        pass

    finally:
        # close everything
        ctrl_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

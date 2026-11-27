##
#
# Hardware control node for KCF: the 29dof running policy runs, kcf_ros.py sends the twist (FSM 'track').
# The native-104 KCF model receives all 29 joints and previous raw policy actions directly.
# Pelvis pose/velocity come from deploy/mocap/estimator.py. Policy clock: steps * control_dt, never reset;
# mocap stamps (ROS clock) are moved onto it by their age at each policy step.
# Guards: a lost IMU or mocap stream is held at its last value (mocap for HOLD_TIME), with warnings.
#
##


# standard imports
import argparse
from time import monotonic

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, Float64MultiArray, String
from nav_msgs.msg import Odometry

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.policy import Policy
from utils.experiment_utils import publish_experiment_info
from deploy.kcf.kcf_interface import CommandGate, Feedback, mocap_base, COMMAND_TOPIC, FEEDBACK_TOPIC, MOCAP_TOPIC
from kcf_mpc.profiles import load_profile
from kcf_mpc.robot_config import G1Config, COMMAND_BANDS
from kcf_mpc.state.math_utils import gravity_from_quat
from utils.locomotion.gait_library import GaitLibraryClock
import onnxruntime as ort


TRIGGER_OFFSET = 0.005  # [sec] half a hardware.py tick, so tick jitter can't move a policy step
IMU_TIMEOUT = 0.1       # [sec] no pelvis IMU message for this long counts as lost (the last one is held)
HOLD_TIME = 0.5         # [sec] a lost mocap Odometry is held this long, then the KCF feedback is invalid (-> control)


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Run the 29dof velocity policy with KCF twists and publish complete native-104 model feedback.
    """

    def __init__(self, config_path: str):

        super().__init__('control_node')

        # One deployment YAML owns both the policy and the matching KCF model.
        self.config = self.load_config(config_path)
        self.kcf_profile = load_profile(self.config['kcf_profile'])
        self.kcf_profile.validate_assets()
        if self.kcf_profile.values['state_source'] != 'simulation':
            raise ValueError("Set kcf.state_source: simulation in the KCF yaml: this node already merges the "
                             "mocap pose into the feedback on the policy clock (kcf_ros must use it as is).")
        # load params
        self.init_policy()

        # broadcast which experiment is running (the logger records it, kcf_ros reads cmd_scale from it)
        self.experiment_info_pub = publish_experiment_info(self, config_path, self.config, self.policy)

        # ROS publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/command', 10)
        self.feedback_pub = self.create_publisher(Float64MultiArray, FEEDBACK_TOPIC, 1)

        # KCF command gate: operator twist in control (zero from kcf_ros), timed MPC twist in track, else zero
        self.gate = CommandGate(dt=self.ctrl_dt, limits=self.command_limits, command_bands=self.command_bands)
        self.fsm = "init"

        # policy step trigger state and the never-reset policy clock (see time_callback)
        self.last_step = None     # fsm_time slot index of the last policy step
        self.fsm_time = 0.0
        self.policy_time = 0.0    # [sec] KCF/policy clock, on the control_dt grid

        # sensor state (full 29dof hardware order)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.omega = np.zeros(3)
        self.qpos_joints = self.qpos_joints_default.copy()
        self.qvel_joints = np.zeros(self.act_size)
        self.mocap = None          # (stamp [s, ROS clock], base [pos, quat], world linear velocity)

        # guards: arrival time [s, monotonic] of the last IMU message, and when each stream was lost
        self.imu_wall = None
        self.imu_lost_at = None
        self.mocap_lost_at = None
        self.mocap_expired = False

        # initialize command and action
        self.cmd = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(self.act_size)

        # ROS subscribers (fsm_time last, so this tick's sensors are taken first)
        self.cmd_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick',
                                                lambda msg: self.gate.set_manual(msg.data, monotonic()), 10)
        self.fsm_sub = self.create_subscription(String, 'deploy_robot/fsm', self.fsm_callback, 1)
        self.mpc_sub = self.create_subscription(Float64MultiArray, COMMAND_TOPIC,
                                                lambda msg: self.gate.set_mpc(msg.data, monotonic()), 1)
        self.mocap_sub = self.create_subscription(Odometry, MOCAP_TOPIC, self.mocap_callback, 1)
        self.pelvis_imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_sensor_callback, 1)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 1)
        self.fsm_time_sub = self.create_subscription(Float64, 'deploy_robot/fsm_time', self.time_callback, 1)

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

    # Initialize the running policy and validate the complete policy/model contract.
    def init_policy(self):
        cfg = G1Config()
        profile = self.kcf_profile
        if (profile.values['state_dim'], profile.values['joint_count'],
                profile.values['nH'], profile.values['recurrent_state_dim']) != (104, 29, 256, 0):
            raise ValueError('Running requires the native-104, 29-joint, lift-256 profile')
        self.ctrl_dt = float(self.config['control_dt'])
        if self.ctrl_dt != cfg.control_dt:
            raise ValueError('Running policy control_dt must be 0.02 seconds')
        self.Kp = np.asarray(self.config['Kp'], dtype=np.float32)
        self.Kd = np.asarray(self.config['Kd'], dtype=np.float32)
        self.cmd_scale = np.asarray(self.config['cmd_scale'], dtype=np.float32)
        self.command_limits = np.asarray(self.config['command_limits'], dtype=np.float32)
        if not np.array_equal(self.cmd_scale, [1., 1., 1.]) or not np.array_equal(self.command_limits, cfg.cmd_scale):
            raise ValueError('Running uses physical commands with limits [2, 0, 2] and unit wire scaling')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.policy = Policy(str(profile.path(profile.values['policy_path'])), session_options=options)
        self.obs_size, self.act_size = self.policy.input_size, self.policy.output_size
        if (self.obs_size, self.act_size) != (98, 29) or len(self.policy.inputs) != 1:
            raise ValueError('Running requires one 98-input, 29-output actor')
        self.qpos_joints_default = self.policy.get_param('default_joint_pos')
        self.action_scale = self.policy.get_param('action_scale')
        for name, actual, expected in (
            ('Kp', self.Kp, cfg.Kp), ('Kd', self.Kd, cfg.Kd),
            ('home_joint_pos', self.config['home_joint_pos'], cfg.default_joint_pos),
            ('default_joint_pos', self.qpos_joints_default, cfg.default_joint_pos),
            ('action_scale', self.action_scale, cfg.action_scale),
        ):
            if np.shape(actual) != (29,) or not np.allclose(actual, expected, atol=1e-6, rtol=0):
                raise ValueError(f'{name} does not match the running actor')
        settings = profile.settings()
        if (np.any(np.asarray(settings['input_lower_bounds']) < -self.command_limits)
                or np.any(np.asarray(settings['input_upper_bounds']) > self.command_limits)
                or settings['icem']['command_bands'] != COMMAND_BANDS):
            raise ValueError('Running MPC bounds/bands must match the actor contract')
        self.command_bands = COMMAND_BANDS
        self.clock = GaitLibraryClock(self.policy)
        if not {37, 43}.issubset(self.clock.periods) or self.clock.idle_idx is None or self.clock.period != 43:
            raise ValueError('Running actor requires idle and 37/43-frame gait clips')


    #################################################################
    # CALLBACKS
    #################################################################

    # FSM state: the gate follows it; the previous action restarts at zero when the policy engages
    def fsm_callback(self, msg):
        if msg.data in ("control", "track") and self.fsm not in ("control", "track"):
            self.action[:] = 0.0
            self.clock.reset()
        self.fsm = msg.data
        self.gate.set_fsm(msg.data, monotonic())

    # mocap pelvis Odometry from deploy/mocap/estimator.py (an invalid message keeps the last pose)
    def mocap_callback(self, msg):
        try:
            self.mocap = mocap_base(msg)
        except ValueError:
            print("WARNING: invalid mocap Odometry ignored (keeping the last pose).", flush=True)

    # pelvis IMU data: [rpy(3), quat(4), gyro(3), acc(3)]
    def pelvis_imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.quat = data[3:7]
        self.omega = data[7:10]
        self.imu_wall = monotonic()

    # joint data: [q(29), dq(29), ddq(29), tau_est(29)]; we only need q and dq
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = self.act_size
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # fsm time; steps the policy once per control_dt slot, advancing the policy clock by the slots elapsed
    def time_callback(self, msg):
        prev_time = self.fsm_time
        self.fsm_time = msg.data
        step = int((self.fsm_time + TRIGGER_OFFSET) // self.ctrl_dt)
        # fsm_time restarted (FSM transition): count it as one slot, the policy clock never goes back
        if self.last_step is None or self.fsm_time < prev_time:
            advance = 1 if self.last_step is not None else 0
        elif step != self.last_step:
            advance = step - self.last_step
        else:
            return
        self.last_step = step
        self.policy_time = round(self.policy_time / self.ctrl_dt + advance) * self.ctrl_dt
        if self.fsm in ('control', 'track'):
            for _ in range(max(0, advance - 1)):
                self.clock.advance()
        self.control_callback()


    #################################################################
    # CONTROL
    #################################################################

    # publish the KCF feedback for this policy step and select the twist (operator, KCF, or zero)
    def prepare_kcf_feedback(self):
        now = monotonic()
        t = self.policy_time

        # Full joint state in the trained policy/model order, without recentering.
        q = np.zeros(36)
        v = np.zeros(35)
        q[3:7] = self.quat
        q[7:] = self.qpos_joints
        v[3:6] = self.omega
        v[6:] = self.qvel_joints

        # IMU guard: the last pelvis IMU message stays in use; warn on loss and recovery
        if self.imu_wall is not None and now - self.imu_wall >= IMU_TIMEOUT:
            if self.imu_lost_at is None:
                self.imu_lost_at = self.imu_wall
                print("WARNING: pelvis IMU lost. Holding the last IMU message.", flush=True)
        elif self.imu_lost_at is not None:
            print(f"WARNING: pelvis IMU recovered after {now - self.imu_lost_at:.2f} s.", flush=True)
            self.imu_lost_at = None

        # pelvis pose and world velocity from mocap, if fresh: its age moves the stamp onto the policy clock
        pose_time, valid = t, False
        if self.mocap is not None:
            stamp, base, world_linear = self.mocap
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp
            if -1e-3 <= age <= self.gate.timeout:
                pose_time, valid = t - max(age, 0.0), True
                q[:7], v[:3] = base, world_linear
                if self.mocap_lost_at is not None:
                    print(f"WARNING: mocap recovered after {now - self.mocap_lost_at:.2f} s.", flush=True)
                    self.mocap_lost_at, self.mocap_expired = None, False

            # mocap guard: hold the last pose (as current) for up to HOLD_TIME, then the feedback is invalid
            else:
                if self.mocap_lost_at is None:
                    self.mocap_lost_at = now
                    print(f"WARNING: mocap lost (no fresh Odometry). Holding the last pose for up to {HOLD_TIME} s.", flush=True)
                if now - self.mocap_lost_at <= HOLD_TIME:
                    pose_time, valid = t, True
                    q[:7], v[:3] = base, world_linear
                elif not self.mocap_expired:
                    print(f"WARNING: mocap lost for over {HOLD_TIME} s: KCF feedback invalid (KCF falls back to control).", flush=True)
                    self.mocap_expired = True

        # twist; missing mocap in track gives zero velocity (never act on a stale pose)
        self.cmd, self.command_source = self.gate.select(t, now)
        if not valid and self.gate.fsm == 'track':
            self.cmd[:] = 0
            self.gate.candidates.clear()

        # Echo the current physical twist and the previous complete raw policy action.
        feedback = Feedback(t, pose_time, t, valid, q, v, self.action.copy(), self.cmd.copy(), self.clock.state.copy())
        self.feedback_pub.publish(Float64MultiArray(data=feedback.pack()))

    # build the observation vector for the policy
    def build_observation(self):
        self.clock.update(self.cmd)
        # Trained running order: gyro, relative joint positions, velocities,
        # previous raw action, physical twist, clip phase, projected gravity.
        return np.concatenate((
            self.omega, self.qpos_joints - self.qpos_joints_default,
            self.qvel_joints, self.action, self.cmd, self.clock.phase_obs(),
            gravity_from_quat(np.asarray(self.quat, dtype=np.float32)),
        )).astype(np.float32)

    # control published at the control frequency
    def control_callback(self):

        # KCF feedback and twist for this step
        self.prepare_kcf_feedback()

        # run the policy (29dof action)
        if self.fsm not in ('control', 'track'):
            self.action[:] = 0.0
            self.clock.reset()
        else:
            obs = self.build_observation()
            self.action = self.policy.inference(obs)
            self.clock.advance()

        # build the command: [q_des, dq_des, Kp, Kd, tau_ff]
        qpos_des = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.act_size, dtype=np.float32)
        tau_ff = np.zeros(self.act_size, dtype=np.float32)

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]).tolist()
        self.command_pub.publish(cmd_msg)


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Hardware control node: 29dof running policy with native-104 KCF feedback.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        default='g1_29dof_kcf',
        help='Path to the config yaml file. Example: "g1_29dof_kcf.yaml".'
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
        # close everything (Ctrl+C may already have shut the ROS context down)
        ctrl_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

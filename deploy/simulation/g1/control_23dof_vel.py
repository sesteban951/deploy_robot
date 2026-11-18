##
#
# Control node for 23DoF MjLab velocity tracking robot.
#
##


# standard imports
import argparse
from time import monotonic

# other imports
import math
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, Float64MultiArray, String

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.unitree_utils import get_gravity_orientation
from utils.policy import Policy
from utils.experiment_utils import publish_experiment_info
from deploy.kcf.kcf_interface import (
    CommandGate, Feedback, mocap_base, COMMAND_TOPIC, FEEDBACK_TOPIC,
    SIM_STATE_TOPIC, MOCAP_TOPIC,
)


TRIGGER_OFFSET = 1e-6  # [sec] only absorbs float rounding (sim time has no jitter)


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Asynchronous control node that runs the policy and sends actions to the simulation.
    """

    def __init__(self, config_path: str, *, pose_source=None):

        super().__init__('control_node')

        # load config file
        self.config = self.load_config(config_path)
        self.kcf_enabled = bool(self.config.get('kcf_enabled', False))
        if self.kcf_enabled:
            from kcf_mpc.profiles import load_profile
            self.kcf_profile = load_profile(self.config['kcf_profile'])
            self.kcf_profile.validate_assets()
            if self.kcf_profile.values['joint_count'] != 23:
                raise ValueError('Select the policy controller matching this model profile')
        if pose_source is not None and not self.kcf_enabled:
            raise ValueError('--pose-source requires the KCF simulation configuration')

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
        if self.kcf_enabled:
            if (self.obs_size, self.act_size) != (80, 23) or not np.array_equal(self.cmd_scale, [1, 1, 1]):
                raise ValueError("KCF requires walking 80/23 policy and unscaled physical commands")
            if self.policy.metadata.get('normalization_formula') != '(obs-mean)/sqrt(var+0.01)':
                raise ValueError('KCF requires the validated walking ONNX export, not the legacy policy')
            for name in ('control_dt', 'gait_period', 'stand_cmd_threshold'):
                value = np.asarray(self.policy.metadata.get(name, []), dtype=float).reshape(-1)
                if value.size != 1 or not np.isclose(value[0], self.config[name], atol=1e-7, rtol=0):
                    raise ValueError(f'KCF {name} must match the exported walking policy')
            limits = np.asarray(self.policy.metadata.get('command_clip_limits', []), dtype=float)
            if limits.shape != (3,) or not np.allclose(limits, self.config['command_limits'], atol=1e-7, rtol=0):
                raise ValueError('KCF physical command limits must match the exported policy')
            self.gate = CommandGate(dt=self.ctrl_dt, limits=self.config['command_limits'])
            self.feedback_pub = self.create_publisher(Float64MultiArray, FEEDBACK_TOPIC, 1)
            self.create_subscription(String, '/deploy_robot/fsm',
                                     lambda msg: self.gate.set_fsm(msg.data, monotonic()), 1)
            self.create_subscription(Float64MultiArray, COMMAND_TOPIC,
                                     lambda msg: self.gate.set_mpc(msg.data, monotonic()), 1)
            self.create_subscription(Float64MultiArray, SIM_STATE_TOPIC, self.kcf_state_callback, 1)
            self.kcf_pose_source = pose_source or self.config.get('kcf_pose_source', 'simulation')
            if self.kcf_pose_source not in ('simulation', 'mocap'):
                raise ValueError("KCF simulation pose source must be simulation or mocap")
            self.mocap, self.mocap_wall = None, -float('inf')
            if self.kcf_pose_source == 'mocap':
                from nav_msgs.msg import Odometry
                self.create_subscription(Odometry, self.config.get('kcf_mocap_topic', MOCAP_TOPIC),
                                         self.mocap_callback, 1)
        else:
            self.pelvis_imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_sensor_callback, 1)
            self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 1)
            self.sim_time_sub = self.create_subscription(Float64, 'deploy_robot/simulation_time', self.time_callback, 1)

        # sensor state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.omega = np.zeros(3)
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # initialize command
        self.cmd = np.zeros(3)

        # initialize the action
        self.action = np.zeros(self.act_size)

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

        # scaling params
        self.action_scale = self.config.get('action_scale')
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)

        # PD gains
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # gait phase params
        self.gait_period = self.config["gait_period"]
        self.stand_cmd_threshold = self.config["stand_cmd_threshold"]

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

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")


    #################################################################
    # HELPERS
    #################################################################

    # command from the command node
    def cmd_callback(self, msg):
        if self.kcf_enabled:
            self.gate.set_manual(msg.data, monotonic())
            return
        data = np.array(msg.data, dtype=np.float32)
        
        # joystick
        joystick_is_connected = (data[0] > 0.5)
        vx_cmd = data[1]
        vy_cmd = data[2]
        omega_cmd = data[3]

        # update the command with the scaling
        self.cmd = np.array([vx_cmd, vy_cmd, omega_cmd], dtype=np.float32)

    def mocap_callback(self, msg):
        try:
            self.mocap = mocap_base(msg)
            self.mocap_wall = monotonic()
        except ValueError:
            self.mocap = None

    def kcf_state_callback(self, msg):
        state = np.asarray(msg.data, dtype=float)
        if (state.shape != (60,) or not np.isfinite(state).all()
                or abs(np.linalg.norm(state[4:8]) - 1) > 0.01):
            self.gate.candidates.clear()
            return
        self.kcf_state = state.copy()
        self.quat = state[4:8].copy()
        self.omega = state[34:37].copy()
        self.qpos_joints = state[8:31].copy()
        self.qvel_joints = state[37:60].copy()
        if state[0] < self.sim_time:
            self.action[:] = 0
            self.gate.candidates.clear()
        self.time_callback(Float64(data=float(state[0])))

    def prepare_kcf_feedback(self):
        now = monotonic()
        t = float(self.kcf_state[0])
        q, v = self.kcf_state[1:31].copy(), self.kcf_state[31:60].copy()
        pose_time, valid = t, True
        if self.kcf_pose_source == 'mocap':
            valid = self.mocap is not None and now - self.mocap_wall < self.gate.timeout
            if valid:
                pose_time, base, world_linear = self.mocap
                valid = -1e-6 <= t - pose_time <= self.gate.timeout
                q[:7], v[:3] = base, world_linear
        self.cmd, self.command_source = self.gate.select(t, now)
        if not valid and self.gate.fsm == 'track':
            self.cmd[:] = 0
            self.gate.candidates.clear()
        # Echo the CURRENT physical command, but PREVIOUS raw RL action.
        # Missing mocap sets valid=False: never substitute truth as valid mocap.
        feedback = Feedback(t, pose_time, t, valid, q, v, self.action.copy(), self.cmd.copy())
        self.feedback_pub.publish(Float64MultiArray(data=feedback.pack()))

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

    # build the observation vector for the policy
    def build_observation(self):

        # base orientation state
        gravity_orientation = get_gravity_orientation(self.quat)

        # joint position and velocity errors
        qj = (self.qpos_joints - self.qpos_joints_default)
        dqj = self.qvel_joints
        
        # gait phase clock from the slot time (zeroed when standing)
        phase = (self.slot_time % self.gait_period) / self.gait_period
        two_pi = 2.0 * math.pi
        gait_phase = np.array([
            math.sin(two_pi * phase),
            math.cos(two_pi * phase),
        ], dtype=np.float32)
        if np.linalg.norm(self.cmd) < self.stand_cmd_threshold:
            gait_phase[:] = 0.0

        # build the observation vector
        # ['base_ang_vel', 'projected_gravity', 'command', 'phase', 'joint_pos', 'joint_vel', 'actions']
        n = len(qj)
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[0:3]              = self.omega
        obs[3:6]              = gravity_orientation
        obs[6:9]              = self.cmd * self.cmd_scale
        obs[9:11]             = gait_phase
        obs[11:11+n]          = qj
        obs[11+n:11+2*n]      = dqj
        obs[11+2*n:11+3*n]    = self.action

        return obs

    # control published at the control frequency
    def control_callback(self):

        if self.kcf_enabled:
            self.prepare_kcf_feedback()

        # get the current observation
        obs = self.build_observation()

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


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Asynchronous Simulation Node using Mujoco.'
    )
    # config path argument
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        '--config',
        type=str,
        help='Path to the Mujoco config yaml file. Example: "g1_23dof_vel.yaml".'
    )
    selection.add_argument('--profile', help='Select the complete model bundle in deploy/configs/kcf')
    parser.add_argument('--pose-source', choices=('simulation', 'mocap'),
                        help='Pelvis pose: direct simulation or the shared mocap interface')
    args = parser.parse_args()
    if args.profile is not None:
        from kcf_mpc.profiles import load_profile
        args.config = load_profile(args.profile).values['policy_config']

    # create the simulation node
    ctrl_node = ControlNode(args.config, pose_source=args.pose_source)

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

##
#
# Control node for running a 23dof policy on the 29dof G1 hardware.
# WARNING: This only works when the mechanical lock is on the waist roll/pitch.
#          Because we set the waist PD gains to 0.0.
#
##


# standard imports
import argparse

# other imports
import math
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


############################################################################
# JOINT MAPPING (29dof hardware <-> 23dof policy)
############################################################################

# Hardware indices KEPT by the 23dof policy (in policy order).
# Discarded (commanded to 0.0): 13 (waist_roll),    14 (waist_pitch),
#                               20 (L_wrist_pitch), 21 (L_wrist_yaw), 
#                               27 (R_wrist_pitch), 28 (R_wrist_yaw).
POLICY_JOINT_IDX_23DOF = [
    0, 1, 2, 3, 4, 5,         # left leg
    6, 7, 8, 9, 10, 11,       # right leg
    12,                       # waist yaw
    15, 16, 17, 18, 19,       # left arm (through wrist roll)
    22, 23, 24, 25, 26,       # right arm (through wrist roll)
]


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Asynchronous control node that runs a 23dof policy on the 29dof G1
    hardware and sends actions to the hardware node.
    """

    def __init__(self, config_path: str):

        super().__init__('control_node')

        # load config file
        self.config = self.load_config(config_path)

        # load params
        self.init_policy()

        # ROS publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/command', 10)

        # ROS subscribers
        self.cmd_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick', self.cmd_callback, 10)
        self.pelvis_imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_sensor_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 10)
        self.fsm_time_sub = self.create_subscription(Float64, 'deploy_robot/fsm_time', self.time_callback, 10)

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state (full 29dof)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.omega = np.zeros(3)
        self.qpos_joints_full = self.qpos_joints_default_full.copy()
        self.qvel_joints_full = np.zeros(self.n_full)
        self.fsm_time = 0.0

        # initialize command
        self.cmd = np.zeros(3)

        # initialize the action (23dof policy output)
        self.action = np.zeros(self.act_size)

        print("Control node initialized.")


    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from [{config_path_full}].")

        return config

    # initialize the policy
    def init_policy(self):

        # mapping from policy's 23dof order to the full 29dof hardware order
        self.policy_joint_indices = np.array(POLICY_JOINT_IDX_23DOF, dtype=np.int64)

        # default joint positions (full 29dof) and the 23dof subset used by the policy
        self.qpos_joints_default_full = np.array(self.config['default_joint_pos'], dtype=np.float32)
        self.qpos_joints_default = self.qpos_joints_default_full[self.policy_joint_indices]

        # dimensions
        self.n_full = len(self.qpos_joints_default_full)
        self.n_policy = len(self.policy_joint_indices)

        # scaling params (action_scale is 23dof, matches policy output)
        self.action_scale = np.array(self.config["action_scale"], dtype=np.float32)
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)

        # PD gains (full 29dof, published straight through to the hardware)
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # gait phase params
        self.gait_period = self.config["gait_period"]
        self.stand_cmd_threshold = self.config["stand_cmd_threshold"]

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/" + policy_path

        # load the policy
        self.policy = Policy(policy_path_full)

        # alias for convenience
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        # sanity checks on sizes
        assert len(self.Kp) == self.n_full, f"Kp must have {self.n_full} values, got {len(self.Kp)}."
        assert len(self.Kd) == self.n_full, f"Kd must have {self.n_full} values, got {len(self.Kd)}."
        assert len(self.action_scale) == self.n_policy, (f"action_scale must have {self.n_policy} values, "
                                                         f"got {len(self.action_scale)}.")
        assert self.act_size == self.n_policy, (f"Policy output size ({self.act_size}) must match "
                                                f"policy_joint_indices length ({self.n_policy}).")

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size} (policy dof)")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")


    #################################################################
    # HELPERS
    #################################################################

    # command from the command node
    def cmd_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)

        # joystick
        joystick_is_connected = (data[0] > 0.5)
        vx_cmd = data[1]
        vy_cmd = data[2]
        omega_cmd = data[3]

        # update the command with the scaling
        self.cmd = np.array([vx_cmd, vy_cmd, omega_cmd], dtype=np.float32)

    # pelvis IMU data: [rpy(3), quat(4), gyro(3), acc(3)]
    def pelvis_imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.quat = data[3:7]
        self.omega = data[7:10]

    # joint data: [q(n_full), dq(n_full), ddq(n_full), tau_est(n_full)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = self.n_full
        self.qpos_joints_full = data[:n]
        self.qvel_joints_full = data[n:2*n]

    # hardware fsm time
    def time_callback(self, msg):
        self.fsm_time = msg.data

    # build the observation vector for the policy
    def build_observation(self):

        # base orientation state
        gravity_orientation = get_gravity_orientation(self.quat)

        # slice the full sensor arrays down to the 23 policy joints
        qpos_policy = self.qpos_joints_full[self.policy_joint_indices]
        qvel_policy = self.qvel_joints_full[self.policy_joint_indices]

        # joint position and velocity errors (23dof)
        qj = (qpos_policy - self.qpos_joints_default)
        dqj = qvel_policy

        # gait phase clock (zeroed when standing)
        phase = (self.fsm_time % self.gait_period) / self.gait_period
        two_pi = 2.0 * math.pi
        gait_phase = np.array([
            math.sin(two_pi * phase),
            math.cos(two_pi * phase),
        ], dtype=np.float32)
        if np.linalg.norm(self.cmd) < self.stand_cmd_threshold:
            gait_phase[:] = 0.0

        # build the observation vector
        # ['base_ang_vel', 'projected_gravity', 'command', 'phase', 'joint_pos', 'joint_vel', 'actions']
        n = self.n_policy
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

        # get the current observation
        obs = self.build_observation()

        # run policy (23dof action)
        self.action = self.policy.inference(obs)

        # build the 29dof command: excluded joints (waist roll/pitch, L/R wrist
        # pitch/yaw at hw indices 13, 14, 20, 21, 27, 28) are commanded to 0.0
        qpos_des = np.zeros(self.n_full, dtype=np.float32)
        qpos_des[self.policy_joint_indices] = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.n_full, dtype=np.float32)
        tau_ff = np.zeros(self.n_full, dtype=np.float32)

        # publish the command: [q_des, dq_des, Kp, Kd, tau_ff]
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
        description='Asynchronous Hardware Control Node for 23dof-on-29dof G1.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file. Example: "g1_23to29dof_vel.yaml".'
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

##
#
# Control node for 29DoF MjLab velocity tracking robot.
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
from utils.unitree_rotation import get_gravity_orientation
from utils.policy import Policy


############################################################################
# JOINT MAPPING FOR REDUCED POLICIES
############################################################################

# Indices of joints removed in the 23-DOF policy (out of the full 29-DOF):
#   waist_roll(13), waist_pitch(14),
#   left_wrist_pitch(20), left_wrist_yaw(21),
#   right_wrist_pitch(27), right_wrist_yaw(28)
REMOVED_JOINTS_23DOF = [13, 14, 20, 21, 27, 28]

# High PD gains to lock removed joints at their default position
LOCK_KP = 200.0
LOCK_KD = 10.0


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Asynchronous control node that runs the policy and sends actions to the simulation.
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
        self.sim_time_sub = self.create_subscription(Float64, 'deploy_robot/simulation_time', self.time_callback, 10)

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.omega = np.zeros(3)
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # initialize command
        self.cmd = np.zeros(3)

        # gait phase params
        self.gait_period = 1.0
        self.gait_offset = 0.5

        # initialize the action
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

        # default joint positions
        self.qpos_joints_default = np.array(self.config['default_joint_pos'])

        # scaling params
        self.action_scale = np.array(self.config["action_scale"], dtype=np.float32)
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)

        # PD gains
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/" + policy_path
        
        # load the policy
        self.policy = Policy(policy_path_full)

        # alias for convenience
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        # joint mapping: determine which joints the policy controls
        self.num_full_joints = len(self.qpos_joints_default)
        self.num_policy_joints = self.act_size

        if self.num_policy_joints == self.num_full_joints:
            # full policy — all joints are controlled
            self._policy_joint_idx = np.arange(self.num_full_joints)
            self._removed_joint_idx = np.array([], dtype=int)
        elif self.num_policy_joints == 23:
            # 23-DOF policy — lock removed joints at default
            self._removed_joint_idx = np.array(REMOVED_JOINTS_23DOF)
            self._policy_joint_idx = np.array(
                [i for i in range(self.num_full_joints) if i not in REMOVED_JOINTS_23DOF]
            )
            self.Kp[self._removed_joint_idx] = LOCK_KP
            self.Kd[self._removed_joint_idx] = LOCK_KD
        else:
            raise ValueError(
                f"Unsupported policy joint count: {self.num_policy_joints} "
                f"(expected {self.num_full_joints} or 23)"
            )

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Full joints: {self.num_full_joints}, Policy joints: {self.num_policy_joints}")
        if len(self._removed_joint_idx) > 0:
            print(f"    Locked joints: {self._removed_joint_idx.tolist()} (Kp={LOCK_KP}, Kd={LOCK_KD})")
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

    # joint data: [q(n), dq(n), ddq(n), tau_est(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # hardware time
    def time_callback(self, msg):
        self.sim_time = msg.data

    # build the observation vector for the policy
    def build_observation(self):

        # base orientation state
        gravity_orientation = get_gravity_orientation(self.quat)

        # joint position and velocity (select only the policy's joint subset)
        idx = self._policy_joint_idx
        qj = (self.qpos_joints - self.qpos_joints_default)[idx]
        dqj = self.qvel_joints[idx]

        # gait phase clock
        phase = (self.sim_time % self.gait_period) / self.gait_period
        phase_right = (phase + self.gait_offset) % 1.0
        two_pi = 2.0 * math.pi
        gait_phase = np.array([
            math.sin(two_pi * phase),
            math.cos(two_pi * phase),
            math.sin(two_pi * phase_right),
            math.cos(two_pi * phase_right),
        ], dtype=np.float32)

        # build the observation vector
        # ['base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions', 'command', 'gait_phase']
        n = self.num_policy_joints
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[0:3]             = self.omega
        obs[3:6]             = gravity_orientation
        obs[6:6+n]           = qj
        obs[6+n:6+2*n]       = dqj
        obs[6+2*n:6+3*n]     = self.action
        obs[6+3*n:6+3*n+3]   = self.cmd * self.cmd_scale
        obs[6+3*n+3:6+3*n+7] = gait_phase

        return obs

    # control published at the control frequency
    def control_callback(self):

        # get the current observation
        obs = self.build_observation()

        # target joint positions (PD control)
        self.action = self.policy.inference(obs)
        # print(f"Policy action (raw): {self.action}")

        # expand policy actions to full joint space (removed joints stay at 0 → default pos)
        full_action = np.zeros(self.num_full_joints, dtype=np.float32)
        full_action[self._policy_joint_idx] = self.action

        # build the command: [q_des, dq_des, Kp, Kd, tau_ff]
        qpos_des = full_action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.num_full_joints, dtype=np.float32)
        tau_ff = np.zeros(self.num_full_joints, dtype=np.float32)

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
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the Mujoco config yaml file. Example: "g1_29dof_vel.yaml".'
    )
    args = parser.parse_args()

    # create the simulation node
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
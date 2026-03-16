##
#
# Control node for the MuJoCo simulation.
#
##

# standard imports
import argparse
import time

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64, Float32MultiArray

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.kinematics import get_gravity_orientation
from utils.policy_utils import *


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
        self.action_pub = self.create_publisher(Float32MultiArray, 'action', 10)

        # ROS subscribers
        self.imu_sensor_sub = self.create_subscription(Float32MultiArray, 'imu_data', self.imu_sensor_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'joint_data', self.joint_sensor_callback, 10)
        self.time_sub = self.create_subscription(Float64, 'sim_time', self.time_callback, 10)

        # control timer
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.omega = np.zeros(3)
        self.qpos_joints = np.zeros(len(self.qpos_joints_default))
        self.qvel_joints = np.zeros(len(self.qpos_joints_default))
        self.sim_time = 0.0

        # initalize the action
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
        self.ang_vel_scale = self.config["ang_vel_scale"]
        self.dof_pos_scale = self.config["dof_pos_scale"]
        self.dof_vel_scale = self.config["dof_vel_scale"]
        self.action_scale = self.config["action_scale"]
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/" + policy_path
        
        # load the policy
        self.policy, self.policy_type = load_policy(policy_path_full)

        # get input and output sizes
        self.obs_size, self.act_size = get_policy_io_size(self.policy, self.policy_type)

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy_type}.")
        print(f"    Input size: {self.obs_size}.")
        print(f"    Output size: {self.act_size}.")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz.")


    #################################################################
    # HELPERS
    #################################################################

    # IMU data: [quat(4), omega(3)]
    def imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.quat = data[:4]
        self.omega = data[4:7]

    # joint data: [qpos(n), qvel(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # sim time
    def time_callback(self, msg):
        self.sim_time = msg.data

    # build the observation vector for the policy
    def build_observation(self):

        # command vector
        cmd = np.zeros(3)

        # base orientation state
        omega = self.omega * self.ang_vel_scale
        gravity_orientation = get_gravity_orientation(self.quat)

        # joint position and velocity errors
        qj = (self.qpos_joints - self.qpos_joints_default) * self.dof_pos_scale
        dqj = self.qvel_joints * self.dof_vel_scale

        # compute the phase
        period = 0.8
        phase = self.sim_time % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        
        # build the observation vector
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[:3] = omega
        obs[3:6] = gravity_orientation
        obs[6:9] = cmd * self.cmd_scale
        obs[9 : 9 + self.act_size] = qj
        obs[9 + self.act_size : 9 + 2 * self.act_size] = dqj
        obs[9 + 2 * self.act_size : 9 + 3 * self.act_size] = self.action
        obs[9 + 3 * self.act_size : 9 + 3 * self.act_size + 2] = np.array([sin_phase, cos_phase])

        return obs

    # control published at the control frequency
    def control_callback(self):

        # get the current observation
        obs = self.build_observation()
        
        # target joint positions (PD control)
        action = policy_inference(self.policy, self.policy_type, obs)
        self.action = action

        # scale the action
        qpos_joints_des = action * self.action_scale + self.qpos_joints_default

        # publish the action
        action_msg = Float32MultiArray()
        action_msg.data = qpos_joints_des.tolist()

        self.action_pub.publish(action_msg)


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
        help='Path to the Mujoco config yaml file. Example: "g1_29dof.yaml".'
    )
    args = parser.parse_args()

    # create the simulation node
    ctrl_node = ControlNode(args.config)

    # execute the simulation
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
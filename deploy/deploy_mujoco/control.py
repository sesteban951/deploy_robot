##
#
# Control node for the MuJoCo simulation.
#
##

# standard imports
import time

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64, Float32MultiArray

# import policy
import onnx
import torch

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.kinematics import get_gravity_orientation


############################################################################
# CONTROLLER NODE
############################################################################

class AsyncControlNode(Node):
    """
    Asynchronous control node that runs the policy and sends actions to the simulation.
    """

    def __init__(self, config_path: str):

        super().__init__('async_sim_node')

        # load config file
        self.config = self.load_config(config_path)

        # load params
        self.init_policy()


    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        return config
    
    # initialize the policy
    def init_policy(self):
        # PD gains
        self.Kp = np.array(self.config['kps'])
        self.Kd = np.array(self.config['kds'])

        # default joint positions
        self.qpos_joints_default = np.array(self.config['default_angles'])

        # scaling params
        self.ang_vel_scale = self.config["ang_vel_scale"]
        self.dof_pos_scale = self.config["dof_pos_scale"]
        self.dof_vel_scale = self.config["dof_vel_scale"]
        self.action_scale = self.config["action_scale"]
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policies/" + policy_path
        if "pt" in policy_path_full.lower():
            self.policy = torch.jit.load(policy_path_full)
        elif "onnx" in policy_path_full.lower():
            self.policy = onnx.load(policy_path_full)
        else:
            raise ValueError("Unsupported policy format. Please use .pt or .onnx files.")

##
#
# Control node for BeyondMimic motion tracking policies.
#
##

# standard imports
import argparse
import xml.etree.ElementTree as ET    # for parsing joint names in the XML file

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
from utils.math_utils import quat_conjugate, quat_multiply, quat_to_rot6d
from utils.policy import Policy


############################################################################
# JOINT ORDER MAPPING
############################################################################

def parse_xml_joint_names(xml_path):
    """Parse actuated joint names from MuJoCo XML (skips floating base)."""
    joint_names = []
    tree = ET.parse(xml_path)
    for joint in tree.getroot().iter("joint"):
        name = joint.attrib.get("name")
        if name is not None:
            joint_names.append(name)
    # first joint is the floating base
    return joint_names[1:]


def build_joint_mappings(xml_joint_names, onnx_joint_names):
    """Build index mappings between MuJoCo (XML) and Isaac (ONNX) joint orders."""
    xml_to_onnx = []
    for name in xml_joint_names:
        xml_to_onnx.append(onnx_joint_names.index(name))

    onnx_to_xml = []
    for name in onnx_joint_names:
        onnx_to_xml.append(xml_joint_names.index(name))

    return np.array(xml_to_onnx), np.array(onnx_to_xml)


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
        self.command_pub = self.create_publisher(Float32MultiArray, 'command', 10)

        # ROS subscribers
        self.imu_sensor_sub = self.create_subscription(Float32MultiArray, 'imu_data', self.imu_sensor_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'joint_data', self.joint_sensor_callback, 10)
        self.time_sub = self.create_subscription(Float64, 'sim_time', self.time_callback, 10)

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state (MuJoCo order)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w, x, y, z)
        self.omega = np.zeros(3, dtype=np.float32)
        self.qpos_joints = np.zeros(self.num_joints, dtype=np.float32)
        self.qvel_joints = np.zeros(self.num_joints, dtype=np.float32)
        self.sim_time = 0.0

        # policy state (ONNX/Isaac order)
        self.action = np.zeros(self.num_joints, dtype=np.float32)

        # motion playback state
        self.motion_frame_idx = 0

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

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # load the policy
        policy_path = ROOT_DIR + "/policy/" + self.config['policy_path']
        self.policy = Policy(policy_path)
        meta = self.policy.metadata

        # joint info from ONNX metadata (all in ONNX/Isaac order)
        self.num_joints = meta["num_joints"]
        self.onnx_joint_names = meta["joint_names"]
        self.default_joint_pos = meta["default_joint_pos"]
        self.action_scale = meta["action_scale"]
        self.kps = meta["kps"]
        self.kds = meta["kds"]
        self.observation_names = meta["observation_names"]

        # joint order mapping (MuJoCo XML <-> ONNX)
        xml_path = ROOT_DIR + "/models/" + self.config['xml_path']
        self.xml_joint_names = parse_xml_joint_names(xml_path)
        self.xml_to_onnx, self.onnx_to_xml = build_joint_mappings(
            self.xml_joint_names, self.onnx_joint_names
        )

        # load motion trajectory
        motion_path = ROOT_DIR + "/motions/" + self.config['motion_npz_path']
        self.motion = np.load(motion_path)
        self.motion_num_frames = self.motion["joint_pos"].shape[0]

        # alias for convenience
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        print(f"Policy: {policy_path}")
        print(f"    Obs size: {self.obs_size}, Act size: {self.act_size}")
        print(f"    Control freq: {1.0 / self.ctrl_dt} Hz")
        print(f"    Observations: {self.observation_names}")
        print(f"    Motion: {motion_path} ({self.motion_num_frames} frames)")


    #################################################################
    # JOINT ORDER CONVERSION
    #################################################################

    def isaac_to_mujoco(self, arr):
        """Convert array from ONNX/Isaac order to MuJoCo/XML order."""
        return arr[self.onnx_to_xml]

    def mujoco_to_isaac(self, arr):
        """Convert array from MuJoCo/XML order to ONNX/Isaac order."""
        return arr[self.xml_to_onnx]

    #################################################################
    # CALLBACKS
    #################################################################

    # IMU data: [quat(4), omega(3)]
    def imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.quat = data[:4]
        self.omega = data[4:7]

    # joint data: [qpos(n), qvel(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = self.num_joints
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # sim time
    def time_callback(self, msg):
        self.sim_time = msg.data

    #################################################################
    # OBSERVATION BUILDERS
    #################################################################

    def obs_command_imitate(self):
        """Reference joint pos and vel from the motion trajectory (ONNX order)."""
        joint_pos = self.motion["joint_pos"][self.motion_frame_idx]
        joint_vel = self.motion["joint_vel"][self.motion_frame_idx]
        return np.concatenate([joint_pos, joint_vel]).astype(np.float32)

    def obs_motion_anchor_ori_b(self):
        """Orientation error between motion anchor body and robot, as 6D encoding."""
        motion_quat = self.motion["body_quat_w"][self.motion_frame_idx, 0]  # anchor body
        q_inv = quat_conjugate(self.quat)
        q_rel = quat_multiply(q_inv, motion_quat)
        return quat_to_rot6d(q_rel)

    def obs_base_ang_vel(self):
        """Angular velocity of the robot base."""
        return self.omega

    def obs_joint_pos(self):
        """Joint positions relative to default pose (ONNX order)."""
        return self.mujoco_to_isaac(self.qpos_joints) - self.default_joint_pos

    def obs_joint_vel(self):
        """Joint velocities (ONNX order)."""
        return self.mujoco_to_isaac(self.qvel_joints)

    def obs_actions(self):
        """Previous action output."""
        return self.action

    def build_observation(self):
        """Build observation vector by calling obs functions in metadata-specified order."""
        obs_parts = []
        for name in self.observation_names:
            func = getattr(self, f"obs_{name}")
            obs_parts.append(func())
        return np.concatenate(obs_parts).astype(np.float32)

    #################################################################
    # CONTROL LOOP
    #################################################################

    # control callback at the control frequency
    def control_callback(self):

        # build observation
        obs = self.build_observation()

        # run ONNX inference with obs and time_step
        obs_input = obs.reshape(1, -1).astype(np.float32)
        time_step_input = np.array([[self.motion_frame_idx]], dtype=np.float32)
        outputs = self.policy._onnx_session.run(None, {
            "obs": obs_input,
            "time_step": time_step_input,
        })
        self.action = outputs[0].squeeze().astype(np.float32)

        # compute desired joint positions (ONNX order -> MuJoCo order)
        qpos_des = self.isaac_to_mujoco(self.default_joint_pos + self.action * self.action_scale)
        qvel_des = np.zeros(self.num_joints, dtype=np.float32)
        tau_ff = np.zeros(self.num_joints, dtype=np.float32)
        kps = self.isaac_to_mujoco(self.kps)
        kds = self.isaac_to_mujoco(self.kds)

        # publish [qpos_des, qvel_des, tau_ff, kp, kd]
        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, tau_ff, kps, kds]).tolist()
        self.command_pub.publish(cmd_msg)

        # advance motion frame
        if self.motion_frame_idx < self.motion_num_frames - 1:
            self.motion_frame_idx += 1


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='BeyondMimic control node for MuJoCo simulation.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Config file name. Example: "g1_29dof_beyond.yaml".'
    )
    args = parser.parse_args()

    # create the control node
    ctrl_node = ControlNode(args.config)

    # spin
    try:
        rclpy.spin(ctrl_node)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

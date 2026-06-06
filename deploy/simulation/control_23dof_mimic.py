##
#
# Control node for running a 23dof mimic policy on the 23dof G1 in simulation.
#
##


# standard imports
import argparse
import time

# other imports
import mujoco
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, String

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.policy import Policy
from utils.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rot6d,
    yaw_quat,
)


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Asynchronous control node that runs a 23dof mimic policy on the 23dof G1
    in simulation, replaying a motion reference from a .npz file.
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
        self.pelvis_imu_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_callback, 10)
        if self.anchor != "pelvis":
            self.anchor_imu_sub = self.create_subscription(Float32MultiArray, f'deploy_robot/{self.anchor}_imu_state', self.anchor_imu_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 10)
        self.sim_time_sub = self.create_subscription(Float64, 'deploy_robot/simulation_time', self.time_callback, 10)
        self.joystick_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick', self.joystick_callback, 10)
        self.fsm_sub = self.create_subscription(String, 'deploy_robot/fsm', self.fsm_callback, 10)

        # sensor state
        self.anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w, x, y, z)
        self.pelvis_omega = np.zeros(3, dtype=np.float32)
        self.qpos_joints = self.qpos_joints_default.copy()
        self.qvel_joints = np.zeros(self.n_joints, dtype=np.float32)
        self.sim_time = 0.0

        # initialize the action
        self.action = np.zeros(self.act_size)

        # yaw alignment between robot-at-policy-start and motion frame 0
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.policy_start_time = None

        # joystick / FSM gating of the motion playback
        self.joystick_connected = False
        self.fsm_state = "control"
        self.track_start_time = None

        # upfront check: is a joystick connected?
        self.use_fsm = self.check_joystick_connected()
        if self.use_fsm:
            print("Joystick connected: holding the first motion frame until FSM enters 'track'.")
        else:
            print("No joystick connected: looping the motion trajectory continuously.")

        # control timer (created last, after joystick check)
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        print("Control node initialized.")


    #################################################################
    # INITIALIZATION
    #################################################################

    def load_config(self, config_path: str):
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from [{config_path_full}].")
        return config

    def init_policy(self):

        # joint defaults and dimensions
        self.qpos_joints_default = np.array(self.config['default_joint_pos'], dtype=np.float32)
        self.n_joints = len(self.qpos_joints_default)

        # scaling params
        self.action_scale = np.array(self.config["action_scale"], dtype=np.float32)

        # PD gains
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # load the policy
        policy_path = ROOT_DIR + "/policy/" + self.config['policy_path']
        self.policy = Policy(policy_path)
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        print(f"Loading policy from [{policy_path}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")

        # load motion reference data
        motion_path = ROOT_DIR + "/motions/" + self.config['motion_path']
        motion = np.load(motion_path)
        self.motion_fps = float(motion['fps'])
        self.motion_joint_pos = motion['joint_pos'].astype(np.float32)
        self.motion_joint_vel = motion['joint_vel'].astype(np.float32)
        self.motion_body_quat_w = motion['body_quat_w'].astype(np.float32)
        self.motion_num_frames = self.motion_joint_pos.shape[0]

        print(f"Loaded motion from [{motion_path}].")
        print(f"    FPS: {self.motion_fps}")
        print(f"    Frames: {self.motion_num_frames}")
        print(f"    Duration: {self.motion_num_frames / self.motion_fps:.1f}s")

        # find anchor body index
        anchor_name = self.policy.metadata['anchor_body_name']
        xml_path = ROOT_DIR + "/models/" + self.config['xml_path']
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        body_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, mj_model.nbody)  # skip world (id 0)
        ]
        self.anchor_body_idx = body_names.index(anchor_name)

        if "pelvis" in anchor_name.lower():
            self.anchor = "pelvis"
        elif "torso" in anchor_name.lower():
            self.anchor = "torso"
        else:
            raise ValueError(f"Unsupported anchor body name: {anchor_name}")

        print(f"    Anchor body: {anchor_name} (index {self.anchor_body_idx})")

        # sanity checks
        assert len(self.Kp) == self.n_joints
        assert len(self.Kd) == self.n_joints
        assert len(self.action_scale) == self.n_joints
        assert self.act_size == self.n_joints

    def check_joystick_connected(self, timeout: float = 2.0):
        print(f"Checking for a joystick connection ({timeout:.0f}s)...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joystick_connected:
                return True
        return False


    #################################################################
    # CALLBACKS
    #################################################################

    def anchor_imu_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.anchor_quat = data[3:7]

    def pelvis_imu_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.pelvis_omega = data[7:10]
        if self.anchor == "pelvis":
            self.anchor_quat = data[3:7]

    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.qpos_joints = data[:self.n_joints]
        self.qvel_joints = data[self.n_joints:2*self.n_joints]

    def time_callback(self, msg):
        self.sim_time = msg.data

    def joystick_callback(self, msg):
        self.joystick_connected = (msg.data[0] > 0.5)

    def fsm_callback(self, msg):
        state = msg.data
        if state == "track" and self.fsm_state != "track":
            self.track_start_time = self.sim_time
            print("FSM entered 'track': starting motion trajectory.")
        elif state != "track" and self.fsm_state == "track":
            self.track_start_time = None
            print(f"FSM left 'track' (now '{state}'): holding the first frame.")
        self.fsm_state = state


    #################################################################
    # OBSERVATION
    #################################################################

    # ['command', 'motion_anchor_ori_b', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']
    def build_observation(self):

        if not self.use_fsm:
            elapsed = self.sim_time - self.policy_start_time
            frame = int(elapsed / self.ctrl_dt) % self.motion_num_frames
        elif self.track_start_time is None:
            frame = 0
        else:
            elapsed = self.sim_time - self.track_start_time
            frame = min(int(elapsed / self.ctrl_dt), self.motion_num_frames - 1)

        # command (46): motion reference joint_pos + joint_vel
        command = np.concatenate([
            self.motion_joint_pos[frame],
            self.motion_joint_vel[frame],
        ])

        # motion_anchor_ori_b (6): desired anchor orientation in base frame
        motion_anchor_quat_w = self.motion_body_quat_w[frame, self.anchor_body_idx]
        ref_quat_corrected = quat_multiply(self.init_quat, motion_anchor_quat_w)
        rel_quat = quat_multiply(quat_conjugate(self.anchor_quat), ref_quat_corrected)
        anchor_ori_b = quat_to_rot6d(rel_quat)

        # base_ang_vel (3)
        base_ang_vel_b = self.pelvis_omega

        # joint_pos (23): relative to default
        qj = self.qpos_joints - self.qpos_joints_default

        # joint_vel (23)
        dqj = self.qvel_joints

        # 46 + 6 + 3 + 23 + 23 + 23 = 124
        obs = np.concatenate([
            command, anchor_ori_b,
            base_ang_vel_b,
            qj, dqj, self.action,
        ]).astype(np.float32)

        return obs, frame


    #################################################################
    # CONTROL
    #################################################################

    def control_callback(self):

        if self.policy_start_time is None:
            self.policy_start_time = self.sim_time
            motion_anchor_quat_0 = self.motion_body_quat_w[0, self.anchor_body_idx]
            self.init_quat = quat_multiply(
                yaw_quat(self.anchor_quat),
                quat_conjugate(yaw_quat(motion_anchor_quat_0)),
            )

        obs, frame = self.build_observation()
        self.action = self.policy.inference(obs, time_step=frame)

        qpos_des = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.n_joints, dtype=np.float32)
        tau_ff = np.zeros(self.n_joints, dtype=np.float32)

        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]).tolist()
        self.command_pub.publish(cmd_msg)


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    rclpy.init()

    parser = argparse.ArgumentParser(
        description='Simulation Control Node for 23dof Mimic Policy on 23dof G1.'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file. Example: "g1_23dof_mimic.yaml".'
    )
    args = parser.parse_args()

    ctrl_node = ControlNode(args.config)

    try:
        rclpy.spin(ctrl_node)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

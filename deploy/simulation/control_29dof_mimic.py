##
#
# Control node for 29DoF MjLab mimic tracking.
#
##


# standard imports
import argparse

# other imports
import mujoco
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
    Asynchronous control node that runs the mimic policy and sends actions to the simulation.
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

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state
        self.anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w, x, y, z) from anchor IMU
        self.pelvis_omega = np.zeros(3, dtype=np.float32)  # base_ang_vel, always from pelvis IMU
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # initialize the action
        self.action = np.zeros(self.act_size)

        # yaw alignment between robot-at-policy-start and motion frame 0 (identity until captured on first tick)
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.policy_start_time = None

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

        print(f"Loading policy from [{policy_path_full}].")
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

        # find anchor body index against robot's full body list
        anchor_name = self.policy.metadata['anchor_body_name']
        xml_path = ROOT_DIR + "/models/" + self.config['xml_path']
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        motion_body_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, mj_model.nbody)  # skip world (id 0)
        ]
        self.anchor_body_idx = motion_body_names.index(anchor_name)

        # select IMU based on anchor body
        if "pelvis" in anchor_name.lower():
            self.anchor = "pelvis"
        elif "torso" in anchor_name.lower():
            self.anchor = "torso"
        else:
            raise ValueError(f"Unsupported anchor body name: {anchor_name}")

        print(f"    Anchor body: {anchor_name} (index {self.anchor_body_idx})")


    #################################################################
    # CALLBACKS
    #################################################################

    # anchor IMU: [rpy(3), quat(4), gyro(3), acc(3)] — orientation when anchor != pelvis
    def anchor_imu_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.anchor_quat = data[3:7]

    # pelvis IMU: [rpy(3), quat(4), gyro(3), acc(3)] — base_ang_vel plus anchor_quat when anchor = pelvis
    def pelvis_imu_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.pelvis_omega = data[7:10]
        if self.anchor == "pelvis":
            self.anchor_quat = data[3:7]

    # joint data: [q(n), dq(n), ddq(n), tau_est(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # hardware time
    def time_callback(self, msg):
        self.sim_time = msg.data


    #################################################################
    # OBSERVATION
    #################################################################

    # build the observation vector for the policy
    # ['command', 'motion_anchor_ori_b', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']
    def build_observation(self):

        # motion frame: 1 frame per control_dt, matching training (time_steps += 1 per step_dt)
        elapsed = self.sim_time - self.policy_start_time
        frame = int(elapsed / self.ctrl_dt) % self.motion_num_frames

        # --- command (58) : motion reference joint_pos + joint_vel ---
        command = np.concatenate([
            self.motion_joint_pos[frame],
            self.motion_joint_vel[frame],
        ])

        # --- motion_anchor_ori_b (6) : desired anchor orientation in base frame (6D rotation) ---
        # apply the captured yaw offset so the motion is replayed in the robot's initial heading
        motion_anchor_quat_w = self.motion_body_quat_w[frame, self.anchor_body_idx]
        ref_quat_corrected = quat_multiply(self.init_quat, motion_anchor_quat_w)
        rel_quat = quat_multiply(quat_conjugate(self.anchor_quat), ref_quat_corrected)
        anchor_ori_b = quat_to_rot6d(rel_quat)

        # --- base_ang_vel (3) : pelvis angular velocity (training uses imu_in_pelvis site) ---
        base_ang_vel_b = self.pelvis_omega

        # --- joint_pos (29) : relative to default ---
        qj = self.qpos_joints - self.qpos_joints_default

        # --- joint_vel (29) ---
        dqj = self.qvel_joints

        # --- actions (29) : previous action ---
        # concatenate: 58 + 6 + 3 + 29 + 29 + 29 = 154
        obs = np.concatenate([
            command, anchor_ori_b,
            base_ang_vel_b,
            qj, dqj, self.action,
        ]).astype(np.float32)

        return obs, frame


    #################################################################
    # CONTROL
    #################################################################

    # control published at the control frequency
    def control_callback(self):

        # on the first tick, align motion frame 0 with the robot's current yaw
        if self.policy_start_time is None:
            self.policy_start_time = self.sim_time
            motion_anchor_quat_0 = self.motion_body_quat_w[0, self.anchor_body_idx]
            self.init_quat = quat_multiply(
                yaw_quat(self.anchor_quat),
                quat_conjugate(yaw_quat(motion_anchor_quat_0)),
            )

        # get the current observation and motion frame index
        obs, frame = self.build_observation()

        # target joint positions (PD control)
        self.action = self.policy.inference(obs, time_step=frame)

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
        description='Asynchronous Control Node for MjLab Mimic Policy.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file. Example: "g1_29dof_mimic.yaml".'
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

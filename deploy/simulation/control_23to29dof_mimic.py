##
#
# Control node for running a 23dof mimic policy on the 29dof G1 in simulation.
# Excluded joints (waist roll/pitch, L/R wrist pitch/yaw) are commanded to 0.0.
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
    Asynchronous control node that runs a 23dof mimic policy on the 29dof G1
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

        # sensor state (full 29dof)
        self.anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w, x, y, z)
        self.pelvis_omega = np.zeros(3, dtype=np.float32)
        self.qpos_joints_full = self.qpos_joints_default_full.copy()
        self.qvel_joints_full = np.zeros(self.n_full, dtype=np.float32)
        self.sim_time = 0.0

        # initialize the action (23dof policy output)
        self.action = np.zeros(self.act_size)

        # yaw alignment between robot-at-policy-start and motion frame 0
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.policy_start_time = None

        # joystick / FSM gating of the motion playback
        self.joystick_connected = False
        self.fsm_state = "control"       # default: hold frame 0 until joystick sends "track"
        self.track_start_time = None     # sim time the current "track" episode began

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

        # mapping from policy's 23dof order to the full 29dof hardware order
        self.policy_joint_indices = np.array(POLICY_JOINT_IDX_23DOF, dtype=np.int64)

        # full 29dof defaults, then slice to the 23dof subset used by the policy
        self.qpos_joints_default_full = np.array(self.config['default_joint_pos'], dtype=np.float32)
        self.qpos_joints_default = self.qpos_joints_default_full[self.policy_joint_indices]

        # dimensions
        self.n_full = len(self.qpos_joints_default_full)
        self.n_policy = len(self.policy_joint_indices)

        # scaling params (23dof, matches policy output)
        self.action_scale = np.array(self.config["action_scale"], dtype=np.float32)

        # PD gains (full 29dof, published straight through)
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/" + policy_path
        self.policy = Policy(policy_path_full)

        # alias for convenience
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size} (policy dof)")
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

        # find anchor body index in the motion model (23dof), not the deployment model (29dof) —
        # body ordering differs between the two and the npz was generated from the motion model
        anchor_name = self.policy.metadata['anchor_body_name']
        motion_xml_path = ROOT_DIR + "/models/" + self.config['motion_xml_path']
        mj_model = mujoco.MjModel.from_xml_path(motion_xml_path)
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

        # sanity checks on sizes
        assert len(self.Kp) == self.n_full, f"Kp must have {self.n_full} values, got {len(self.Kp)}."
        assert len(self.Kd) == self.n_full, f"Kd must have {self.n_full} values, got {len(self.Kd)}."
        assert len(self.action_scale) == self.n_policy, (f"action_scale must have {self.n_policy} values, "
                                                          f"got {len(self.action_scale)}.")
        assert self.act_size == self.n_policy, (f"Policy output size ({self.act_size}) must match "
                                                f"policy_joint_indices length ({self.n_policy}).")

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

    # joint data: [q(n_full), dq(n_full), ddq(n_full), tau_est(n_full)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = self.n_full
        self.qpos_joints_full = data[:n]
        self.qvel_joints_full = data[n:2*n]

    # simulation time
    def time_callback(self, msg):
        self.sim_time = msg.data

    # joystick command: [is_connected, vx, vy, omega]
    def joystick_callback(self, msg):
        self.joystick_connected = (msg.data[0] > 0.5)

    # FSM state from the joystick node
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

    # build the observation vector for the policy
    # ['command', 'motion_anchor_ori_b', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']
    def build_observation(self):

        # motion frame: 1 frame per control_dt, matching training
        if not self.use_fsm:
            # no joystick: loop the trajectory continuously
            elapsed = self.sim_time - self.policy_start_time
            frame = int(elapsed / self.ctrl_dt) % self.motion_num_frames
        elif self.track_start_time is None:
            # in "control" (not tracking): hold the first frame
            frame = 0
        else:
            # tracking: advance from frame 0, freeze at the last frame (no loop)
            elapsed = self.sim_time - self.track_start_time
            frame = min(int(elapsed / self.ctrl_dt), self.motion_num_frames - 1)

        # --- command (46) : 23dof motion reference joint_pos + joint_vel ---
        command = np.concatenate([
            self.motion_joint_pos[frame],
            self.motion_joint_vel[frame],
        ])

        # --- motion_anchor_ori_b (6) : desired anchor orientation in base frame (6D rotation) ---
        motion_anchor_quat_w = self.motion_body_quat_w[frame, self.anchor_body_idx]
        ref_quat_corrected = quat_multiply(self.init_quat, motion_anchor_quat_w)
        rel_quat = quat_multiply(quat_conjugate(self.anchor_quat), ref_quat_corrected)
        anchor_ori_b = quat_to_rot6d(rel_quat)

        # --- base_ang_vel (3) : pelvis angular velocity ---
        base_ang_vel_b = self.pelvis_omega

        # --- joint_pos (23) : relative to default, policy joints only ---
        qpos_policy = self.qpos_joints_full[self.policy_joint_indices]
        qj = qpos_policy - self.qpos_joints_default

        # --- joint_vel (23) : policy joints only ---
        dqj = self.qvel_joints_full[self.policy_joint_indices]

        # --- actions (23) : previous action ---
        # concatenate: 46 + 6 + 3 + 23 + 23 + 23 = 124
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

        # capture yaw alignment on the first policy tick
        if self.policy_start_time is None:
            self.policy_start_time = self.sim_time
            motion_anchor_quat_0 = self.motion_body_quat_w[0, self.anchor_body_idx]
            self.init_quat = quat_multiply(
                yaw_quat(self.anchor_quat),
                quat_conjugate(yaw_quat(motion_anchor_quat_0)),
            )

        obs, frame = self.build_observation()
        self.action = self.policy.inference(obs, time_step=frame)

        # expand 23dof action to 29dof command; excluded joints stay at default
        qpos_des = self.qpos_joints_default_full.copy()
        qpos_des[self.policy_joint_indices] = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.n_full, dtype=np.float32)
        tau_ff = np.zeros(self.n_full, dtype=np.float32)

        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]).tolist()
        self.command_pub.publish(cmd_msg)


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    rclpy.init()

    parser = argparse.ArgumentParser(
        description='Simulation Control Node for 23dof Mimic Policy on 29dof G1.'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file. Example: "g1_23to29dof_mimic.yaml".'
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

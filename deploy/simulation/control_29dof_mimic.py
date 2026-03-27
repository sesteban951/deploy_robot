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
    quat_to_rotation_matrix,
    quat_to_rot6d,
)


############################################################################
# JOINT MAPPING FOR REDUCED POLICIES
############################################################################

# Indices of joints removed in the 23-DOF policy (out of the full 29-DOF):
#   waist_roll(13), waist_pitch(14),
#   left_wrist_pitch(20), left_wrist_yaw(21),
#   right_wrist_pitch(27), right_wrist_yaw(28)
REMOVED_JOINTS_23DOF = [13, 14, 20, 21, 27, 28]


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
        self.pelvis_imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_sensor_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 10)
        self.sim_time_sub = self.create_subscription(Float64, 'deploy_robot/simulation_time', self.time_callback, 10)
        self.body_state_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/body_state', self.body_state_callback, 10)

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state
        self.pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w, x, y, z)
        self.pelvis_omega = np.zeros(3, dtype=np.float32)
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # body state (from simulation body_state topic)
        self.body_xpos_w = np.zeros((self._mj_nbody, 3), dtype=np.float32)
        self.body_xquat_w = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (self._mj_nbody, 1))
        self.base_lin_vel_w = np.zeros(3, dtype=np.float32)

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
            # 23-DOF policy — removed joints hold default pos with nominal config gains
            self._removed_joint_idx = np.array(REMOVED_JOINTS_23DOF)
            self._policy_joint_idx = np.array(
                [i for i in range(self.num_full_joints) if i not in REMOVED_JOINTS_23DOF]
            )
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
            print(f"    Locked joints: {self._removed_joint_idx.tolist()} (using nominal config Kp/Kd)")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")

        # load motion reference data
        motion_path = ROOT_DIR + "/motions/" + self.config['motion_path']
        motion = np.load(motion_path)
        self.motion_fps = float(motion['fps'])
        self.motion_joint_pos = motion['joint_pos'].astype(np.float32)
        self.motion_joint_vel = motion['joint_vel'].astype(np.float32)
        self.motion_body_pos_w = motion['body_pos_w'].astype(np.float32)
        self.motion_body_quat_w = motion['body_quat_w'].astype(np.float32)
        self.motion_num_frames = self.motion_joint_pos.shape[0]

        print(f"Loaded motion from [{motion_path}].")
        print(f"    FPS: {self.motion_fps}")
        print(f"    Frames: {self.motion_num_frames}")
        print(f"    Duration: {self.motion_num_frames / self.motion_fps:.1f}s")

        # load MuJoCo model for body name→index mapping
        xml_path = ROOT_DIR + "/models/" + self.config['xml_path']
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_nbody = mj_model.nbody

        # find anchor body index in the motion file using body_names from policy metadata
        anchor_name = self.policy.metadata.get('anchor_body_name', 'pelvis')
        body_names = self.policy.metadata.get('body_names')
        self.anchor_body_idx = body_names.index(anchor_name)

        # map policy body_names to MuJoCo body indices (for body_state topic)
        self._body_mj_indices = np.array([
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        ], dtype=int)
        self._pelvis_mj_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

        print(f"    Anchor body: {anchor_name} (index {self.anchor_body_idx})")
        print(f"    Tracked bodies: {len(body_names)} -> MuJoCo indices {self._body_mj_indices.tolist()}")


    #################################################################
    # CALLBACKS
    #################################################################

    # pelvis IMU data: [rpy(3), quat(4), gyro(3), acc(3)]
    def pelvis_imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.pelvis_quat = data[3:7]
        self.pelvis_omega = data[7:10]

    # joint data: [q(n), dq(n), ddq(n), tau_est(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # hardware time
    def time_callback(self, msg):
        self.sim_time = msg.data

    # body state: [xpos(nbody*3), xquat(nbody*4), base_lin_vel_w(3)]
    def body_state_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        nb = self._mj_nbody
        self.body_xpos_w = data[:nb*3].reshape(nb, 3)
        self.body_xquat_w = data[nb*3:nb*7].reshape(nb, 4)
        self.base_lin_vel_w = data[nb*7:nb*7+3]


    #################################################################
    # OBSERVATION
    #################################################################

    # build the observation vector for the policy
    def build_observation(self):

        # motion frame: 1 frame per control_dt, matching training (time_steps += 1 per step_dt)
        frame = int(self.sim_time / self.ctrl_dt) % self.motion_num_frames

        # joint index for the policy subset
        idx = self._policy_joint_idx

        # shared terms
        base_quat_conj = quat_conjugate(self.pelvis_quat)

        # --- motion_anchor_ori_b (6) ---
        motion_anchor_quat_w = self.motion_body_quat_w[frame, self.anchor_body_idx]
        anchor_ori_b = quat_to_rot6d(quat_multiply(base_quat_conj, motion_anchor_quat_w))

        # --- joint obs (policy joints only) ---
        qj = (self.qpos_joints - self.qpos_joints_default)[idx]
        dqj = self.qvel_joints[idx]
        base_ang_vel_b = self.pelvis_omega

        if self.num_policy_joints == self.num_full_joints:
            # 29-DOF obs: [command(58), anchor_ori(6), ang_vel(3), qj(29), dqj(29), act(29)] = 154
            command = np.concatenate([
                self.motion_joint_pos[frame],
                self.motion_joint_vel[frame],
            ])
            obs = np.concatenate([
                command, anchor_ori_b, base_ang_vel_b,
                qj, dqj, self.action,
            ]).astype(np.float32)
        else:
            # 23-DOF obs: [command(46), anchor_pos(3), anchor_ori(6),
            #              lin_vel(3), ang_vel(3), qj(23), dqj(23), act(23)] = 130
            R_base = quat_to_rotation_matrix(self.pelvis_quat)
            base_pos_w = self.body_xpos_w[self._pelvis_mj_idx]

            # --- command (46) : motion joint_pos + joint_vel (policy joints only) ---
            command = np.concatenate([
                self.motion_joint_pos[frame][idx],
                self.motion_joint_vel[frame][idx],
            ])

            # --- motion_anchor_pos_b (3) ---
            motion_anchor_pos_w = self.motion_body_pos_w[frame, self.anchor_body_idx]
            anchor_pos_b = R_base.T @ (motion_anchor_pos_w - base_pos_w)

            # --- base_lin_vel (3) : in base frame ---
            base_lin_vel_b = R_base.T @ self.base_lin_vel_w

            obs = np.concatenate([
                command, anchor_pos_b, anchor_ori_b,
                base_lin_vel_b, base_ang_vel_b,
                qj, dqj, self.action,
            ]).astype(np.float32)

        return obs, frame


    #################################################################
    # CONTROL
    #################################################################

    # control published at the control frequency
    def control_callback(self):

        # get the current observation and motion frame index
        obs, frame = self.build_observation()

        # target joint positions (PD control)
        self.action = self.policy.inference(obs, time_step=frame)

        # expand policy actions to full joint space (removed joints stay at 0 → default pos)
        full_action = np.zeros(self.num_full_joints, dtype=np.float32)
        full_action[self._policy_joint_idx] = self.action

        # build the command: [qpos_des, qvel_des, Kp, Kd, tau_ff]
        qpos_des = full_action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.num_full_joints, dtype=np.float32)
        tau_ff = np.zeros(self.num_full_joints, dtype=np.float32)

        # print action debug info
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print(f"[frame {frame}] action (policy): {self.action}")
        print(f"[frame {frame}] qpos_des (full): {qpos_des}")

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

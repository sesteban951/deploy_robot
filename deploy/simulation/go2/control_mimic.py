##
#
# Control node for Go2 MjLab mimic tracking (Go2-Tracking-Custom).
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
from utils.experiment_utils import publish_experiment_info
from utils.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rot6d,
    heading_about_z_world,
)

# PD gains while the policy is OFF and a frame is held ('control' before 'track', after the clip);
# the policy's gains sag the stand. Same as POSE_KP / POSE_KD in the foxy branch's go2 hardware.py.
HOLD_KP = 60.0  # [Nm / rad]
HOLD_KD = 5.0   # [Nm s / rad]


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

        # broadcast which experiment is running so the logger can record it
        self.experiment_info_pub = publish_experiment_info(self, config_path, self.config, self.policy)

        # ROS publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/command', 10)

        # ROS subscribers
        self.imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/imu_state', self.imu_sensor_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 10)
        self.sim_time_sub = self.create_subscription(Float64, 'deploy_robot/simulation_time', self.time_callback, 10)
        self.joystick_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick', self.joystick_callback, 10)
        self.fsm_sub = self.create_subscription(String, 'deploy_robot/fsm', self.fsm_callback, 10)

        # sensor state
        self.anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w, x, y, z) from the base IMU
        self.base_omega = np.zeros(3, dtype=np.float32)  # base_ang_vel from the base IMU
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # initialize the action
        self.action = np.zeros(self.act_size)

        # yaw alignment between the robot's current heading and motion frame 0
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.init_quat_captured = False
        self.policy_start_time = None
        self.prev_frame = 0  # to detect the loop wrap in no-joystick mode

        # joystick / FSM gating of the motion playback
        self.joystick_connected = False   # latest connection flag from the joystick node
        self.fsm_state = "control"        # connected default: "control" = frozen on the first frame
        self.track_start_time = None      # sim time the current "track" episode began (None = not tracking)

        # upfront check: is a joystick connected? this locks the playback mode for the run
        self.use_fsm = self.check_joystick_connected()
        if self.use_fsm:
            print("Joystick connected: holding the first motion frame until FSM enters 'track'.")
        else:
            print("No joystick connected: looping the motion trajectory continuously.")

        # control timer to run the policy at a fixed frequency (created last, after the joystick check)
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        print("Control node initialized.")


    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it (accept the name with or without the .yaml extension)
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        config_path_full = ROOT_DIR + "/deploy/configs/go2/" + config_path
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

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/go2/" + policy_path

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
        # mjlab exports a single action_scale when every joint shares it
        if len(self.action_scale) == 1:
            self.action_scale = np.full(self.act_size, self.action_scale[0], dtype=np.float32)
        assert len(self.action_scale) == self.act_size, \
            f"action_scale has {len(self.action_scale)} values, expected {self.act_size}."
        for _k in ('default_joint_pos', 'action_scale'):
            print(f"    {_k}: from {'policy metadata' if _k in self.policy.metadata else 'yaml config'}")

        # PD gains while the policy runs: from the yaml, so they can be fine-tuned
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)
        assert len(self.Kp) == self.act_size and len(self.Kd) == self.act_size, \
            f"yaml Kp / Kd have {len(self.Kp)} / {len(self.Kd)} values, expected {self.act_size}."

        # flag any gain that differs from what the policy was trained with (its metadata)
        for name, used, key in (("Kp", self.Kp, "joint_stiffness"), ("Kd", self.Kd, "joint_damping")):
            trained = self.policy.get_param(key)
            if trained is None:
                print(f"    {name}: yaml (policy has no '{key}' metadata to compare against)")
            elif np.allclose(used, trained, atol=1e-3):
                print(f"    {name}: yaml, matches the trained gains {np.unique(trained).tolist()}")
            else:
                diff = [f"{i}: {t:g} -> {u:g}" for i, (t, u) in enumerate(zip(trained, used)) if abs(t - u) > 1e-3]
                print(f"    NOTE {name}: yaml differs from the trained gains at joint " + ", ".join(diff))

        # stiffer gains for holding a frame with the policy off
        self.hold_Kp = np.full(self.act_size, HOLD_KP, dtype=np.float32)
        self.hold_Kd = np.full(self.act_size, HOLD_KD, dtype=np.float32)

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

        # the Go2 has one IMU, on the floating base: the anchor must be that body
        free_joint = int(np.flatnonzero(mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
        base_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, mj_model.jnt_bodyid[free_joint])
        if anchor_name != base_name:
            raise ValueError(f"Anchor body '{anchor_name}' is not the IMU body '{base_name}'.")

        print(f"    Anchor body: {anchor_name} (index {self.anchor_body_idx})")

    # spin briefly to detect whether a joystick is connected.
    def check_joystick_connected(self, timeout: float = 0.5):
        print("Checking for a joystick connection...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joystick_connected:
                return True
        return False


    #################################################################
    # CALLBACKS
    #################################################################

    # base IMU data: [rpy(3), quat(4), gyro(3), acc(3)]
    def imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.anchor_quat = data[3:7]
        self.base_omega = data[7:10]

    # joint data: [q(n), dq(n), ddq(n), tau_est(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]

    # hardware time
    def time_callback(self, msg):
        self.sim_time = msg.data

    # joystick command: [is_connected, vx, vy, omega]
    def joystick_callback(self, msg):
        self.joystick_connected = (msg.data[0] > 0.5)

    # FSM state from the joystick node; the trajectory only advances while in "track"
    def fsm_callback(self, msg):
        state = msg.data
        entering_control = (state == "control" and self.fsm_state != "control")
        if state == "track" and self.fsm_state != "track":
            # entering track: (re)start the trajectory from the first frame, from a fresh
            # episode (zero last action) aligned to the robot's heading right now
            self.track_start_time = self.sim_time
            self.action = np.zeros(self.act_size)
            self.init_quat_captured = False
            print("FSM entered 'track': starting motion trajectory.")
        elif state != "track" and self.fsm_state == "track":
            # leaving track: reset so the next 'track' replays from the start
            self.track_start_time = None
            print(f"FSM left 'track' (now '{state}'): holding the first frame.")

        # re-align the yaw to the robot's current heading on every (re)entry to control
        if entering_control:
            self.init_quat_captured = False
        self.fsm_state = state


    #################################################################
    # OBSERVATION
    #################################################################

    # current motion frame index for this tick
    # 1 frame per control_dt, matching training (time_steps += 1 per step_dt)
    def compute_frame(self):
        if not self.use_fsm:
            # no joystick: loop the trajectory continuously
            elapsed = self.sim_time - self.policy_start_time
            return int(elapsed / self.ctrl_dt) % self.motion_num_frames
        elif self.track_start_time is None:
            # in "control" (not tracking): hold the first frame
            return 0
        else:
            # tracking: advance from frame 0, then freeze at the last frame (no loop)
            elapsed = self.sim_time - self.track_start_time
            return min(int(elapsed / self.ctrl_dt), self.motion_num_frames - 1)

    # build the observation vector for the policy
    # ['command', 'motion_anchor_ori_b', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']
    def build_observation(self, frame):

        # --- command (24) : motion reference joint_pos + joint_vel ---
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

        # --- base_ang_vel (3) : base IMU gyro ---
        base_ang_vel_b = self.base_omega

        # --- joint_pos (12) : relative to default ---
        qj = self.qpos_joints - self.qpos_joints_default

        # --- joint_vel (12) ---
        dqj = self.qvel_joints

        # --- actions (12) : previous action ---
        # concatenate: 24 + 6 + 3 + 12 + 12 + 12 = 69
        obs = np.concatenate([
            command, anchor_ori_b,
            base_ang_vel_b,
            qj, dqj, self.action,
        ]).astype(np.float32)

        return obs


    #################################################################
    # CONTROL
    #################################################################

    # control published at the control frequency
    def control_callback(self):

        # one-time timing origin (used by the no-FSM continuous-loop mode)
        if self.policy_start_time is None:
            self.policy_start_time = self.sim_time

        # current motion frame for this tick
        frame = self.compute_frame()

        # no-FSM loop: re-align heading at each wrap so a turning motion chains
        # (the FSM path does this on control re-entry instead)
        if not self.use_fsm and frame < self.prev_frame:
            self.init_quat_captured = False
        self.prev_frame = frame

        # capture the heading offset once per control/loop (re)entry; lets a turning motion chain
        if not self.init_quat_captured:
            motion_anchor_quat_0 = self.motion_body_quat_w[0, self.anchor_body_idx]
            # heading offset (about world z) between robot and reference
            q_rel = quat_multiply(
                self.anchor_quat, quat_conjugate(motion_anchor_quat_0)
            )
            self.init_quat = heading_about_z_world(q_rel)
            self.init_quat_captured = True

        # outside the clip (before 'track', or once the last frame is reached) hold that frame's
        # joint angles with plain PD. The policy never saw a frozen reference in training (time
        # always advances), and frozen on a frame it crouches, turns, or lies down.
        holding_first = self.use_fsm and self.track_start_time is None
        holding_last = self.use_fsm and self.track_start_time is not None and \
            (self.sim_time - self.track_start_time) / self.ctrl_dt >= self.motion_num_frames
        holding = holding_first or holding_last
        if holding:
            qpos_des = self.motion_joint_pos[0 if holding_first else -1]
        else:
            # get the current observation
            obs = self.build_observation(frame)

            # target joint positions (PD control)
            self.action = self.policy.inference(obs, time_step=frame)
            qpos_des = self.action * self.action_scale + self.qpos_joints_default

        # build the command: [q_des, dq_des, Kp, Kd, tau_ff]
        qvel_des = np.zeros(self.act_size, dtype=np.float32)
        tau_ff = np.zeros(self.act_size, dtype=np.float32)

        # publish the command
        cmd_msg = Float32MultiArray()
        Kp, Kd = (self.hold_Kp, self.hold_Kd) if holding else (self.Kp, self.Kd)
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, Kp, Kd, tau_ff]).tolist()
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
        help='Path to the config yaml file. Example: "go2_mimic.yaml".'
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

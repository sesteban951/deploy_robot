##
#
# Control node for 29DoF MjLab omnidirectional crawling (G1-Crawling-Omni), HARDWARE.
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
from std_msgs.msg import Float64, Float32MultiArray, String

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.unitree_utils import get_gravity_orientation
from utils.policy import Policy
from utils.crawl_modes import resolve_crawl_twist
from utils.experiment_utils import publish_experiment_info


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Asynchronous control node for the omnidirectional crawling policy
    (G1-Crawling-Omni) on hardware.

    The actor sees proprioception, a free-running gait phase clock, a commanded
    planar twist [vx, vy, wz], and projected gravity. No reference motion is loaded
    at runtime. The commanded twist is assigned to one of three library motions --
    forward, backward, or in-place turn -- by its magnitude (see utils.crawl_modes),
    then clamped to that motion's trained range; anything else -> the zero-twist
    idle/stop pose. This keeps combined commands on the gait-library manifold instead
    of collapsing a big forward+turn into a pure in-place spin.

    Ranges match the G1-Crawling-Omni "alldir" gait library and the mjlab training
    TWIST_RANGE: vx in [-0.25, 0.30] (reverse crawl INCLUDED), vy in [-0.12, 0.12],
    wz in [-0.60, 0.60]. Keep the yaml twist_clip_* in sync with the trained range
    (mjlab tasks/crawling_omni/config/g1/env_cfgs.py: TWIST_RANGE).

    Hardware harness (hardware.py) FSM: init -> damp -> home -> control <-> track.
    The 'home' state ramps the robot from its current pose into the crawl idle/laydown
    pose (home_joint_pos == the policy's default qpos) with ramped gains, so this node
    only runs the policy in the 'control'/'track' states and stays silent otherwise
    (the low level holds the ramped pose). The gait clock resets to 0 while idle so
    'control' always starts the gait at phase 0 from the idle pose.
    """

    def __init__(self, config_path: str):

        super().__init__('control_node')

        # load config file
        self.config = self.load_config(config_path)

        # load params
        self.init_policy()

        # ROS publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/command', 10)

        # broadcast which experiment is running so the logger can record it
        self.experiment_info_pub = publish_experiment_info(self, config_path, self.config, self.policy)

        # ROS subscribers
        self.cmd_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick', self.cmd_callback, 10)
        self.pelvis_imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_sensor_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 10)
        self.fsm_sub = self.create_subscription(String, 'deploy_robot/fsm', self.fsm_callback, 10)
        self.fsm_time_sub = self.create_subscription(Float64, 'deploy_robot/fsm_time', self.time_callback, 10)

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # pelvis (w, x, y, z)
        self.omega = np.zeros(3)                      # pelvis gyro (base_ang_vel)
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.fsm_state = "init"
        self.fsm_time = 0.0

        # joystick command [vx, vy, wz] + connection flag
        self.cmd = np.zeros(3)
        self.joystick_connected = False

        # initialize the action
        self.action = np.zeros(self.act_size)

        # free-running gait phase clock (integer frame index, wraps at T)
        self.phase_step = 0

        # active crawl mode (forward / backward / turn / idle) for logging
        self.mode = "idle"
        self._last_mode = None

        print("Control node initialized.")


    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it (accept the name with or without the .yaml extension)
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from [{config_path_full}].")

        return config

    # initialize the policy
    def init_policy(self):

        # default_joint_pos and action_scale are loaded from the policy metadata
        self.qpos_joints_default = self.config.get('default_joint_pos')
        self.action_scale = self.config.get('action_scale')

        # PD gains
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # crawl gait + omnidirectional twist command params
        self.motion_period_frames = int(self.config["motion_period_frames"])  # T
        self.default_twist = np.array(self.config["default_twist"], dtype=np.float32)
        self.twist_scale = np.array(self.config["twist_scale"], dtype=np.float32)
        self.twist_clip_lo = np.array(self.config["twist_clip_lo"], dtype=np.float32)
        self.twist_clip_hi = np.array(self.config["twist_clip_hi"], dtype=np.float32)

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/" + policy_path

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
        assert len(self.action_scale) == self.act_size, \
            f"action_scale has {len(self.action_scale)} values, expected {self.act_size}."
        for _k in ('default_joint_pos', 'action_scale'):
            print(f"    {_k}: from {'policy metadata' if _k in self.policy.metadata else 'yaml config'}")

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")
        print(f"    Gait period: {self.motion_period_frames} frames "
              f"({self.motion_period_frames * self.ctrl_dt:.2f} s)")


    #################################################################
    # HELPERS
    #################################################################

    # joystick command: [is_connected, vx, vy, omega]
    def cmd_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.joystick_connected = (data[0] > 0.5)
        self.cmd = np.array([data[1], data[2], data[3]], dtype=np.float32)

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

    # FSM state (from the joystick node)
    def fsm_callback(self, msg):
        self.fsm_state = msg.data

    # fsm time -- time since entering the current FSM state (hardware.py resets it on each transition)
    def time_callback(self, msg):
        self.fsm_time = msg.data

    # commanded planar twist [vx, vy, wz]: raw joystick command -> mode-select +
    # clamp to the gait library (utils.crawl_modes), so a big combined command
    # can't leave the reachable set and collapse to a pure in-place turn.
    def commanded_twist(self):
        # hardware safety: no joystick -> idle/stop pose. (The sim node crawls forward
        # autonomously here for viewing; on hardware a dropped joystick must NOT keep driving.)
        if not self.joystick_connected:
            self.mode = "idle"
            return np.zeros(3, dtype=np.float32)

        twist = self.cmd * self.twist_scale
        self.mode, shaped = resolve_crawl_twist(twist, self.config)
        return shaped

    # build the observation vector for the policy
    def build_observation(self):

        # proprioception (joint pos relative to default, joint vel)
        qj = self.qpos_joints - self.qpos_joints_default
        dqj = self.qvel_joints

        # commanded twist
        twist = self.commanded_twist()

        # gait phase clock: (sin, cos) of 2*pi*t/T, matching training (time_steps / T)
        phase = self.phase_step / self.motion_period_frames
        ang = 2.0 * math.pi * phase
        motion_phase = np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)

        # projected gravity from the pelvis IMU (roll/pitch; yaw-invariant)
        proj_grav = get_gravity_orientation(self.quat).astype(np.float32)

        # concatenate in the trained order (from the policy's observation_names metadata):
        # base_ang_vel(3), joint_pos(29), joint_vel(29), actions(29),
        # commanded_twist(3), motion_phase(2), projected_gravity(3) = 98
        obs = np.concatenate([
            self.omega,
            qj, dqj, self.action,
            twist, motion_phase, proj_grav,
        ]).astype(np.float32)

        return obs

    # control published at the control frequency
    def control_callback(self):

        # idle states (init/damp/home): stay silent; the low level holds the ramped pose.
        # reset the gait clock so 'control' always starts the gait from phase 0 at the idle pose.
        if self.fsm_state not in ("control", "track"):
            self.action = np.zeros(self.act_size)
            self.phase_step = 0
            return

        # get the current observation
        obs = self.build_observation()

        # announce crawl-mode changes (forward / backward / turn / idle)
        if self.mode != self._last_mode:
            print(f"[mode] {self._last_mode} -> {self.mode}")
            self._last_mode = self.mode

        # target joint positions (PD control); time_step feeds the gait frame index
        self.action = self.policy.inference(obs, time_step=self.phase_step)

        # build the command: [q_des, dq_des, Kp, Kd, tau_ff]
        qpos_des = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.act_size, dtype=np.float32)
        tau_ff = np.zeros(self.act_size, dtype=np.float32)

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]).tolist()
        self.command_pub.publish(cmd_msg)

        # advance the free-running gait clock
        self.phase_step = (self.phase_step + 1) % self.motion_period_frames


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Asynchronous Control Node for the MjLab omnidirectional crawling policy (hardware).'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file. Example: "g1_29dof_crawl_omni.yaml".'
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

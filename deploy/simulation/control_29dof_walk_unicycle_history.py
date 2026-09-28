##
#
# Control node for 29DoF MjLab UNICYCLE upright WALKING with an OBSERVATION HISTORY
# (G1-Walk-Unicycle-History).
#
##


# standard imports
import argparse
from collections import deque

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
from utils.unitree_utils import get_gravity_orientation
from utils.policy import Policy
from utils.experiment_utils import publish_experiment_info
# the jog's joystick shaper, imported unchanged (as mjlab imports the jog's sampler): nothing in
# it is gait-specific, the walk only changes the bands in the yaml
from utils.locomotion.jog_modes import JogTwistCommander


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Asynchronous control node for the unicycle upright walking policy with an observation
    history (G1-Walk-Unicycle-History).

    Everything the plain walk node does (control_29dof_walk_unicycle.py): the same joystick
    shaping through utils/locomotion/jog_modes.py, the same free-running gait clock blanked at
    idle, the same zero-action = standing idle pose. The ONE difference is the observation: the
    actor's dynamics terms are stacked over the last H control steps (H = 5 @ 50 Hz = 100 ms,
    the current frame included) so the reference-free policy can read its speed off the stance
    leg, tell which foot is planted and see the actuator delay (mjlab
    tasks/walk_unicycle/config/g1_history/env_cfgs.py). Stacked: base_ang_vel, joint_pos,
    joint_vel, actions, projected_gravity. NOT stacked (one frame, as before): the commanded
    twist (piecewise constant) and the motion phase (a known ramp).

    The layout is the one mjlab's ObservationManager produces with flatten_history_dim=True:
    TERM-MAJOR, and inside each stacked term the H frames run OLDEST -> NEWEST (CircularBuffer
    .buffer is chronological, reshaped row-major), so e.g. joint_pos occupies 5 x 29 = 145 slots
    as [q(t-4), q(t-3), q(t-2), q(t-1), q(t)]. The window advances exactly once per control tick
    (mjlab appends once per env step) and, on the first tick, every slot is BACKFILLED with the
    first frame -- what mjlab does on the first append after an episode reset -- so the policy
    never sees a zero-padded or half-filled window.

    No joystick -> default_twist (an autonomous walk, sim only; a hardware node must stand).
    """

    # the actor terms mjlab stacks (HISTORY_TERMS in the env cfg); the others stay at one frame
    HISTORY_TERMS = ("base_ang_vel", "joint_pos", "joint_vel", "actions", "projected_gravity")

    # the trained term order (the policy's observation_names metadata)
    OBS_TERMS = ("base_ang_vel", "joint_pos", "joint_vel", "actions",
                 "commanded_twist", "motion_phase", "projected_gravity")

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
        self.cmd_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick', self.cmd_callback, 10)
        self.pelvis_imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_sensor_callback, 10)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 10)
        self.sim_time_sub = self.create_subscription(Float64, 'deploy_robot/simulation_time', self.time_callback, 10)

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)

        # sensor state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # pelvis (w, x, y, z)
        self.omega = np.zeros(3)                      # pelvis gyro (base_ang_vel)
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.sim_time = 0.0

        # joystick command [vx, vy, wz] sticks in [-1, 1] + connection flag
        self.cmd = np.zeros(3)
        self.joystick_connected = False

        # active mode (idle / forward / backward / pivot) for logging
        self.mode = "idle"
        self._last_mode = None

        # initialize the action
        self.action = np.zeros(self.act_size)

        # gait phase clock (integer frame index, wraps at T). Always free-running; the OBSERVATION
        # derived from it is blanked at idle (see build_observation).
        self.phase_step = 0

        # observation history: one window per stacked term holding its last H frames, oldest ->
        # newest (mjlab CircularBuffer). Empty until the first control tick, which backfills every
        # slot with the first frame exactly as mjlab does on an episode reset.
        self.history = {name: deque(maxlen=self.history_length) for name in self.HISTORY_TERMS}

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

        # walk gait clock + joystick shaping (utils/locomotion/jog_modes.py)
        self.motion_period_frames = int(self.config["motion_period_frames"])  # T
        self.default_twist = np.array(self.config["default_twist"], dtype=np.float32)
        self.commander = JogTwistCommander(self.config)

        # observation history depth H (frames, the current one included); cross-checked against
        # the policy metadata below
        self.history_length = int(self.config["history_length"])
        assert self.history_length >= 1, f"history_length must be >= 1, got {self.history_length}."

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

        # the observation this node builds must be the one the policy was trained on. Term-major,
        # every stacked term H frames deep (flattened oldest -> newest), twist and phase one frame:
        # base_ang_vel(3H) + joint_pos(nH) + joint_vel(nH) + actions(nH) + twist(3) + phase(2) + gravity(3H)
        H = self.history_length
        expected_obs = 3 * H + 3 * self.act_size * H + 3 + 2 + 3 * H
        assert self.obs_size == expected_obs, \
            f"policy expects a {self.obs_size}-wide observation, this node builds {expected_obs} " \
            f"(history_length = {H})."

        # the policy metadata records the term order and how deep each term was stacked: the
        # order must be the one hard-wired here and every depth must be H on the stacked terms and
        # 0 (one frame) on the rest. A wrong depth on one term can still add up to the right total
        # width (e.g. stacking the twist instead of the gyro), so the size check alone is not enough.
        obs_names = self.policy.metadata.get('observation_names')
        hist_lens = self.policy.metadata.get('observation_terms_history_length')
        if obs_names is not None:
            assert list(obs_names) == list(self.OBS_TERMS), \
                f"policy observation order {list(obs_names)} != this node's {list(self.OBS_TERMS)}."
        if hist_lens is not None:
            for name, h in zip(self.OBS_TERMS, np.atleast_1d(hist_lens)):
                want = H if name in self.HISTORY_TERMS else 0
                assert int(h) == want, \
                    f"policy stacks '{name}' over {int(h)} frames, this node uses {want} " \
                    f"(history_length = {H}; 0 = one frame)."

        # the yaml's home pose must be the policy's default pose: the sim spawns / hardware
        # ramps to home_joint_pos, and a zero action holds default_joint_pos -- they must agree
        home = np.array(self.config["home_joint_pos"], dtype=np.float32)
        gap = float(np.abs(home - self.qpos_joints_default).max())
        assert gap < 2e-3, \
            f"home_joint_pos differs from the policy's default_joint_pos by {gap:.4f} rad; " \
            f"they must be the same standing idle pose."

        # the policy's gait period must match the yaml's clock
        motion_len = self.policy.motion_period_frames
        if motion_len is not None:
            assert motion_len == self.motion_period_frames, \
                f"policy bundles a {motion_len}-frame motion but motion_period_frames = {self.motion_period_frames}."

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")
        print(f"    Gait period: {self.motion_period_frames} frames "
              f"({self.motion_period_frames * self.ctrl_dt:.2f} s)")
        print(f"    Observation history: {H} frames ({H * self.ctrl_dt * 1e3:.0f} ms) on "
              f"{list(self.HISTORY_TERMS)}; one frame on the rest")
        print(f"    Joystick: {self.commander.describe()}")
        print(f"    No joystick -> autonomous twist {self.default_twist.tolist()} (sim only)")


    #################################################################
    # HELPERS
    #################################################################

    # joystick command: [is_connected, vx, vy, omega] (sticks in [-1, 1])
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

    # simulation time
    def time_callback(self, msg):
        self.sim_time = msg.data

    # commanded planar twist [vx, vy, wz]: left stick Y = drive (forward/back), right stick X =
    # steer (left +), both at once. The lateral stick (vy) is ignored: the library has no vy.
    def commanded_twist(self):
        # autonomous walk when no joystick is connected (for viewing in sim)
        if not self.joystick_connected:
            self.mode = "forward" if self.default_twist[0] > 0 else "idle"
            return self.default_twist

        self.mode, twist = self.commander.update(fwd_stick=self.cmd[0], turn_stick=self.cmd[2])
        return twist

    # build the observation vector for the policy. Call it ONCE per control tick: it advances the
    # history window (mjlab appends once per env step).
    def build_observation(self):

        # proprioception (joint pos relative to default, joint vel)
        qj = self.qpos_joints - self.qpos_joints_default
        dqj = self.qvel_joints

        # commanded twist
        twist = self.commanded_twist()

        # gait phase clock: (sin, cos) of 2*pi*t/T, matching training (time_steps / T), but
        # BLANKED to (0, 0) whenever the commanded twist is zero (idle). The idle clip is one pose
        # held for the whole period, so the reference stands perfectly still -- but a live clock
        # still reads as "swing your limbs" to a policy trained on periodic gaits, and that is what
        # makes a stopped robot shuffle instead of standing statically. Training blanks it the same
        # way (mjlab crawling_fwd/mdp/observations.py: motion_phase, shared by the library tasks),
        # so leaving it live here would put the policy OFF-DISTRIBUTION at exactly the moment we
        # want it quiet. (0, 0) is off the unit circle -> unreachable during gait -> an unambiguous
        # "no gait" flag. The clock itself keeps free-running (see control_callback); only the
        # observation is blanked, so leaving idle resumes mid-cycle exactly as it did in training.
        if np.any(twist):
            ang = 2.0 * math.pi * (self.phase_step / self.motion_period_frames)
            motion_phase = np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)
        else:
            motion_phase = np.zeros(2, dtype=np.float32)

        # projected gravity from the pelvis IMU (roll/pitch; yaw-invariant)
        proj_grav = get_gravity_orientation(self.quat).astype(np.float32)

        # push this tick's frame of every stacked term into its window. self.action is still the
        # PREVIOUS tick's action here (inference runs after this), so the 'actions' window is
        # [a(t-H), ..., a(t-1)], as in training. The first tick backfills every slot with the first
        # frame -- mjlab CircularBuffer's first append after a reset -- so there is never a
        # zero-padded window. Snapshot each frame: the sensor callbacks replace the arrays.
        frames = {
            "base_ang_vel": self.omega,
            "joint_pos": qj,
            "joint_vel": dqj,
            "actions": self.action,
            "projected_gravity": proj_grav,
        }
        for name, frame in frames.items():
            frame = np.array(frame, dtype=np.float32)
            window = self.history[name]
            if len(window) == 0:
                window.extend(frame.copy() for _ in range(self.history_length))
            else:
                window.append(frame)

        # each stacked term flattened oldest -> newest ((H, dim) row-major), then the terms in the
        # trained order (from the policy's observation_names metadata), H = 5, n = 29:
        # base_ang_vel(15), joint_pos(145), joint_vel(145), actions(145),
        # commanded_twist(3), motion_phase(2), projected_gravity(15) = 470
        obs = np.concatenate([
            np.concatenate(self.history["base_ang_vel"]),
            np.concatenate(self.history["joint_pos"]),
            np.concatenate(self.history["joint_vel"]),
            np.concatenate(self.history["actions"]),
            twist, motion_phase,
            np.concatenate(self.history["projected_gravity"]),
        ]).astype(np.float32)

        return obs, twist

    # control published at the control frequency
    def control_callback(self):

        # get the current observation (advances the history window)
        obs, twist = self.build_observation()

        # announce mode changes (idle / forward / backward / pivot) with the twist
        if self.mode != self._last_mode:
            print(f"[mode] {self._last_mode} -> {self.mode}  twist vx {twist[0]:+.2f} wz {twist[2]:+.2f}")
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
        description='Asynchronous Control Node for the MjLab unicycle walking policy with an observation history.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file. Example: "g1_29dof_walk_unicycle_history.yaml".'
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

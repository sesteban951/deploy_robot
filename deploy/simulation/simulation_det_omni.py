##
#
# Deterministic (single-process) simulation for the 29DoF MjLab omnidirectional
# crawling policy (G1-Crawling-Omni).
#
# The normal deployment splits the controller (control_29dof_crawl_omni.py) and
# the physics (simulation.py) into two async ROS2 nodes that talk over DDS. That
# is convenient but NOT reproducible: the two nodes run on independent wall-clock
# timers, so the number of physics steps between policy updates -- and thus the
# resulting trajectory -- jitters from run to run.
#
# This script fuses both into ONE synchronous loop with no ROS2 and no DDS. Given
# the same config, seed, and command stream it produces the exact same trajectory
# every time. Sources of nondeterminism removed:
#   1. async node timing         -> single loop, fixed decimation (physics steps
#                                    per policy step is an exact integer)
#   2. wall-clock-driven stepping -> stepping is step-count driven; the wall clock
#                                    only paces the (state-invariant) viewer
#   3. random initial state       -> mj_resetData + fixed home pose + mj_forward
#   4. RNG (optional sensor noise) -> seeded; noise is OFF by default
#
# The command comes from a pygame joystick by default. When NO joystick is
# connected the command falls back to the config's default_twist (a forward
# crawl) and the whole run is fully reproducible. A live joystick is the one
# intentional source of variation: the physics/stepping stay deterministic, but
# human input obviously does not reproduce. Pass --twist to force a fixed command
# (ignores the joystick) or --no-joystick to always use the default forward crawl.
#
# Use this to tune the omni controller: change knobs in g1_29dof_crawl_omni.yaml
# (twist range, gait period, gains, ...) and compare runs on equal footing.
#
# Velocity-tracking experiments: --profile {fwd,bwd,crab,turn,all} runs a scripted
# staircase of body-frame twist setpoints, logs commanded vs achieved base velocity
# to an npz, and you visualize it with plot_velocity_tracking.py.
#
##


# standard imports
import argparse
import math
import time

# mujoco imports
import mujoco
import mujoco.viewer

# other imports
import numpy as np
import yaml

# joystick (optional; the command falls back to the default twist without it)
try:
    import pygame
except Exception:
    pygame = None

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.unitree_utils import get_gravity_orientation
from utils.policy import Policy
from utils.joystick_utils import JoystickState, pygame_to_joystick_state
from utils.crawl_modes import resolve_crawl_twist


############################################################################
# SIMULATION SETTINGS
############################################################################

# Physics integrates at SIM_HZ; the viewer renders at RENDER_HZ (decoupled).
# The policy runs at 1/control_dt Hz (50 Hz for this config), so the decimation
# (physics steps per policy step) is SIM_HZ * control_dt -- kept an exact integer.
SIM_HZ = 500.0     # [Hz] simulation rate
RENDER_HZ = 50.0   # [Hz] viewer render rate


############################################################################
# VELOCITY-TRACKING STAIRCASE PROFILES
############################################################################

# Each profile is a list of body-frame twist setpoints [vx, vy, wz]. The sim holds
# each for an integer number of GAIT CYCLES (after an initial idle settle), so the
# steady-state average spans whole cycles -- the crawl gait makes the base velocity
# oscillate at the gait period, so a whole-cycle window cancels that swing.
N_HOLD_CYCLES = 3     # setpoint hold length, in gait cycles (multiple of the ~2.6 s period)
N_SETTLE_CYCLES = 1   # initial idle settle, in gait cycles

_FWD  = [[v,   0.0,  0.0] for v  in (0.10, 0.15, 0.20, 0.25, 0.30)]
_BWD  = [[v,   0.0,  0.0] for v  in (-0.05, -0.10, -0.15, -0.20, -0.25)]
_CRAB = [[0.20, vy,  0.0] for vy in (0.0, 0.06, 0.12, -0.06, -0.12)]
_TURN = [[0.0, 0.0,  wz] for wz in (0.28, 0.40, 0.50, 0.60, -0.28, -0.40, -0.60)]
PROFILES = {
    "fwd":  _FWD,     # forward vx staircase
    "bwd":  _BWD,     # backward vx staircase
    "crab": _CRAB,    # lateral vy staircase at a fixed forward vx
    "turn": _TURN,    # in-place yaw-rate staircase
    "all":  _FWD + _BWD + _CRAB + _TURN,
}

# --- "combo" paper experiment: ONE run through the three mode regions, sampling
# real twist clips from the gait library --------------------------------------
# N_PER_REGION commands are drawn per motion (forward / backward / in-place turn)
# from the ACTUAL library twist table (bundled snapshot of the mjlab crawl_omni
# clips), seeded by --seed for reproducibility. Regions run back-to-back.
N_PER_REGION = 2

# bundled snapshot of the mjlab crawl_omni gait-library twist labels (285 clips)
_TWIST_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crawl_omni_twists.npy")

# fallback grid, used only if the twist-table file is missing
_LIB_FWD_VX   = (0.10, 0.15, 0.20, 0.25, 0.30)
_LIB_BWD_VX   = (-0.05, -0.10, -0.15, -0.20, -0.25)
_LIB_CRAB_VY  = (-0.12, -0.06, 0.0, 0.06, 0.12)
_LIB_CURVE_WZ = (-0.12, -0.06, 0.0, 0.06, 0.12)
_LIB_TURN_WZ  = tuple(sorted(s * round(0.28 + 0.02 * i, 2) for i in range(17) for s in (-1.0, 1.0)))


def sample_combo_setpoints(seed, n=N_PER_REGION):
    """Sample n real gait-library twist clips from each motion (seeded -> reproducible)."""
    rng = np.random.default_rng(int(seed))

    if os.path.exists(_TWIST_TABLE):
        T = np.load(_TWIST_TABLE)
        vx, vy, wz = T[:, 0], T[:, 1], T[:, 2]
        fwd_mask  = vx >= 0.10
        bwd_mask  = vx <= -0.05
        turn_mask = (np.abs(vx) < 1e-6) & (np.abs(vy) < 1e-6) & (np.abs(wz) > 0.25)

        # stratified draw: pick n DISTINCT values of the primary axis, then a real
        # library clip at each -> genuine library points that span the motion's range
        def draw(mask, key_col):
            idx = np.where(mask)[0]
            keys = np.unique(np.round(T[idx, key_col], 3))
            chosen = rng.choice(keys, size=min(n, len(keys)), replace=False)
            out = []
            for kv in chosen:
                sub = idx[np.isclose(T[idx, key_col], kv)]
                out.append(T[int(rng.choice(sub))].astype(float).tolist())
            return out

        fwd  = sorted(draw(fwd_mask, 0),  key=lambda c: c[0])    # ascending vx
        bwd  = sorted(draw(bwd_mask, 0),  key=lambda c: -c[0])   # increasingly negative vx
        turn = sorted(draw(turn_mask, 2), key=lambda c: c[2])    # ascending wz
        return fwd + bwd + turn

    # fallback: reconstruct from the known grid
    def region(vx_set):
        vxs = rng.choice(vx_set, size=n, replace=False)
        return [[float(v), float(rng.choice(_LIB_CRAB_VY)), float(rng.choice(_LIB_CURVE_WZ))] for v in vxs]
    fwd = sorted(region(_LIB_FWD_VX), key=lambda c: c[0])
    bwd = sorted(region(_LIB_BWD_VX), key=lambda c: -c[0])
    wzs = rng.choice(_LIB_TURN_WZ, size=n, replace=False)
    turn = sorted([[0.0, 0.0, float(w)] for w in wzs], key=lambda c: c[2])
    return fwd + bwd + turn


############################################################################
# DETERMINISTIC SIMULATION
############################################################################

class DeterministicOmniSim:
    """
    Single-process, deterministic sim for the G1-Crawling-Omni policy.

    The observation is built EXACTLY as control_29dof_crawl_omni.py builds it
    (same sensors, same concatenation order) and the PD torque is computed
    EXACTLY as simulation.py computes it, so this loop reproduces the deployed
    control pipeline -- just synchronously and reproducibly.
    """

    def __init__(self, config_path: str, seed: int = 0,
                 twist_override=None, use_joystick: bool = True,
                 apply_noise: bool = False,
                 headless: bool = False, realtime: bool = True,
                 profile=None, log_path=None):

        # determinism: seed every RNG we might touch (noise path only, but seed
        # unconditionally so nothing downstream can depend on process entropy)
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.apply_noise = apply_noise
        self.headless = headless
        self.realtime = realtime
        self.twist_override = twist_override
        self.use_joystick = use_joystick
        self._pygame_inited = False

        # velocity-tracking profile run: scripted + deterministic, so it never
        # grabs a live joystick or honors a --twist override
        self.profile_name = profile
        if profile is not None:
            self.use_joystick = False
            self.twist_override = None
        self.log_path = log_path
        self.log_enabled = log_path is not None

        # load config, control params, profile, joystick, policy, and the model
        self.config = self.load_config(config_path)
        self.init_control_params()
        self.init_profile()
        self.init_joystick()
        self.init_policy()
        self.init_simulation()


    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file (same loader as the ROS2 nodes)
    def load_config(self, config_path: str):
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from [{config_path_full}].")
        return config

    # load the crawl-gait + twist-command params (mirrors the control node)
    def init_control_params(self):
        # PD gains
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = float(self.config["control_dt"])

        # crawl gait clock (T frames per cycle). twist_scale maps input->physical
        # twist; twist_clip is the raw safety box. Mode selection/shaping
        # (utils.crawl_modes) then clamps to the trained gait library.
        self.motion_period_frames = int(self.config["motion_period_frames"])
        self.default_twist = np.array(self.config["default_twist"], dtype=np.float32)
        self.twist_scale = np.array(self.config["twist_scale"], dtype=np.float32)
        self.twist_clip_lo = np.array(self.config["twist_clip_lo"], dtype=np.float32)
        self.twist_clip_hi = np.array(self.config["twist_clip_hi"], dtype=np.float32)

        # optional fixed-command override: a DIRECT physical twist [vx, vy, wz]
        # (not a raw stick), clipped to the raw safety box. When set it takes
        # precedence over the joystick and makes the whole run reproducible.
        if self.twist_override is not None:
            self.fixed_twist = np.clip(
                np.array(self.twist_override, dtype=np.float32),
                self.twist_clip_lo, self.twist_clip_hi,
            )
        else:
            self.fixed_twist = None

        # current commanded twist + active crawl mode (recomputed each control step)
        self.twist = self.default_twist.copy()
        self.mode = "idle"
        self._last_mode = None

    # build the staircase schedule for a --profile run (list of timed segments)
    def init_profile(self):
        # gait period in seconds; hold/settle are integer multiples of it so
        # steady-state averaging spans whole gait cycles
        self.period_s = self.motion_period_frames * self.ctrl_dt
        self.hold_s = 0.0
        self.settle_s = 0.0
        self.profile_segments = None   # [(t_start, t_end, twist, seg_id), ...]
        self.profile_total_s = None
        self._seg = -1                 # current setpoint index (-1 = settle/idle)
        if self.profile_name is None:
            return
        if self.profile_name == "combo":
            setpoints = sample_combo_setpoints(self.seed)
        elif self.profile_name in PROFILES:
            setpoints = PROFILES[self.profile_name]
        else:
            raise ValueError(f"unknown --profile '{self.profile_name}'; "
                             f"choices: {list(PROFILES) + ['combo']}")

        self.hold_s = N_HOLD_CYCLES * self.period_s
        self.settle_s = N_SETTLE_CYCLES * self.period_s

        segs = []
        t = 0.0
        segs.append((t, t + self.settle_s, np.zeros(3, dtype=np.float32), -1))  # settle at idle
        t += self.settle_s
        for i, tw in enumerate(setpoints):
            segs.append((t, t + self.hold_s, np.array(tw, dtype=np.float32), i))
            t += self.hold_s
        self.profile_segments = segs
        self.profile_total_s = t
        print(f"Profile '{self.profile_name}': {len(setpoints)} setpoints x {self.hold_s:.1f}s "
              f"({N_HOLD_CYCLES} gait cycles) + {self.settle_s:.1f}s settle = {t:.0f}s total.")
        if self.profile_name == "combo":
            print(f"    sampled from library (seed {self.seed}), {N_PER_REGION} per motion:")
            for c in setpoints:
                print(f"      [vx={c[0]:+.2f}, vy={c[1]:+.2f}, wz={c[2]:+.2f}]")

    # the scripted setpoint (twist, seg_id) active at sim time t
    def profile_twist(self, t):
        for (t0, t1, tw, seg) in self.profile_segments:
            if t0 <= t < t1:
                return tw, seg
        last = self.profile_segments[-1]
        return last[2], last[3]

    # set up the pygame joystick (the default command source). Falls back to the
    # config default_twist when no joystick is present / pygame is unavailable.
    def init_joystick(self):
        self.joystick = None
        self.joystick_state = JoystickState()
        self.joystick_drove = False   # did a live joystick ever drive this run?

        # skip the joystick entirely for fixed-command or --no-joystick runs
        if self.fixed_twist is not None:
            print(f"Fixed --twist command {self.fixed_twist.tolist()}; joystick ignored.")
            return
        if not self.use_joystick:
            print(f"Joystick disabled (--no-joystick); using default forward "
                  f"command {self.default_twist.tolist()}.")
            return
        if pygame is None:
            print(f"pygame unavailable; using default forward command "
                  f"{self.default_twist.tolist()}.")
            return

        # initialize pygame + joystick (same path as deploy/joystick/joystick_pygame.py)
        try:
            pygame.init()
            pygame.joystick.init()
            self._pygame_inited = True
        except Exception as e:
            print(f"Could not initialize pygame ({e}); using default forward "
                  f"command {self.default_twist.tolist()}.")
            return

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick connected: [{self.joystick.get_name()}]. "
                  f"LS=vx (fore/aft), RS=vy (lateral), LT=turn+, RT=turn-.")
        else:
            print(f"No joystick found; using default forward command "
                  f"{self.default_twist.tolist()}. Plug one in to take over.")

    # load the policy and the metadata-backed deploy params (mirrors control node)
    def init_policy(self):
        # yaml fallbacks (may be absent; policy metadata is authoritative)
        self.qpos_joints_default = self.config.get('default_joint_pos')
        self.action_scale = self.config.get('action_scale')

        # load the policy from the policy/ folder
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/" + policy_path
        self.policy = Policy(policy_path_full)

        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        # default_joint_pos and action_scale come from the policy metadata,
        # falling back to the yaml if absent (same precedence as the control node)
        self.qpos_joints_default = self.policy.get_param('default_joint_pos', self.qpos_joints_default)
        self.action_scale = self.policy.get_param('action_scale', self.action_scale)
        assert len(self.qpos_joints_default) == self.act_size, \
            f"default_joint_pos has {len(self.qpos_joints_default)} values, expected {self.act_size}."
        assert len(self.action_scale) == self.act_size, \
            f"action_scale has {len(self.action_scale)} values, expected {self.act_size}."

        print(f"Loaded policy from [{policy_path_full}].")
        print(f"    Input size:  {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Gait period: {self.motion_period_frames} frames "
              f"({self.motion_period_frames * self.ctrl_dt:.2f} s)")

    # initialize the mujoco model in a deterministic initial state
    def init_simulation(self):
        # load the XML model
        models_path = ROOT_DIR + "/models/"
        xml_path = models_path + self.config['xml_path']
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # integrate at the global SIM_HZ (overrides the model's <option timestep>)
        self.mj_model.opt.timestep = 1.0 / SIM_HZ

        # model sizes
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu
        self.sim_dt = self.mj_model.opt.timestep

        assert self.nu == self.act_size, \
            f"model has {self.nu} actuators but policy outputs {self.act_size}."

        # decimation: physics steps per policy step. Must be an exact integer so
        # the policy always sees the same phase of the physics, run to run.
        decimation_f = self.ctrl_dt / self.sim_dt
        self.decimation = int(round(decimation_f))
        assert abs(decimation_f - self.decimation) < 1e-9, (
            f"control_dt/sim_dt = {decimation_f} is not an integer; "
            f"choose SIM_HZ and control_dt so their ratio is exact."
        )

        # render at RENDER_HZ: sync the viewer every Nth control step. The loop
        # runs at the control rate (1/ctrl_dt = 50 Hz), so N = 1 here and the
        # viewer renders at exactly 50 Hz, decoupled from the 500 Hz physics.
        control_hz = 1.0 / self.ctrl_dt
        self.render_every = max(1, int(round(control_hz / RENDER_HZ)))

        # deterministic initial state: reset, plant the home pose, zero velocity
        self.home_base = np.array(self.config['home_base_pos'], dtype=np.float64)
        self.home_joints = np.array(self.config['home_joint_pos'], dtype=np.float64)
        assert len(self.home_joints) == self.nu, \
            f"home_joint_pos must be size {self.nu}, got {len(self.home_joints)}."

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:7] = self.home_base
        self.mj_data.qpos[7:7 + self.nu] = self.home_joints
        self.mj_data.qvel[:] = 0.0
        # populate sensordata / derived quantities for the very first observation
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # per-joint sensor names, in actuator order (matches simulation.py)
        self.joint_pos_sensor_names = []
        self.joint_vel_sensor_names = []
        for i in range(self.nu):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.joint_pos_sensor_names.append(f"{joint_name}_pos_sensor")
            self.joint_vel_sensor_names.append(f"{joint_name}_vel_sensor")

        # controller state
        self.action = np.zeros(self.act_size, dtype=np.float32)   # last policy action
        self.phase_step = 0                                       # free-running gait clock
        self.qpos_des = self.qpos_joints_default.astype(np.float64).copy()  # PD target (held)

        # velocity-tracking measurement state: track the floating base (pelvis).
        # The achieved twist is read in the robot's heading frame, where "heading"
        # is the yaw of the base +z axis -- in the crawl pose the base is pitched
        # ~90 deg nose-down, so body +z points along ground-forward and is nearly
        # horizontal, making its yaw well-conditioned (no gimbal issue).
        self.track_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.track_body_id < 0:
            self.track_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.log_rows = []   # (t, cmd_vx, cmd_vy, cmd_wz, ach_vx, ach_vy, ach_wz, seg)

        print(f"Loaded Mujoco model from [{xml_path}].")
        print(f"    Sim dt: {self.sim_dt:.6f} s ({SIM_HZ:.0f} Hz), "
              f"decimation: {self.decimation} physics steps / policy step")
        print(f"    Control: {1.0 / self.ctrl_dt:.0f} Hz, render: {RENDER_HZ:.0f} Hz "
              f"(viewer sync every {self.render_every} control step)")
        print(f"    Seed: {self.seed}, noise: {self.apply_noise}")

        # viewer (skipped in headless mode so runs are watch-free and CI-friendly)
        self.viewer = None
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data,
                show_left_ui=False, show_right_ui=False,
            )
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 2.5
            self.viewer.cam.lookat[:] = list(self.home_base[0:3])


    #################################################################
    # COMMAND  (joystick, fixed override, or default forward fallback)
    #################################################################

    # poll pygame for hot-plug events and refresh the joystick state
    def update_joystick(self):
        if self.fixed_twist is not None or not self.use_joystick or pygame is None:
            return

        # handle connect / disconnect while running
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED and self.joystick is None:
                self.joystick = pygame.joystick.Joystick(event.device_index)
                self.joystick.init()
                print(f"Joystick connected: [{self.joystick.get_name()}].")
            elif event.type == pygame.JOYDEVICEREMOVED and self.joystick is not None:
                print("Joystick disconnected; using default forward command.")
                self.joystick = None
                self.joystick_state = JoystickState()

        if self.joystick is None:
            return

        # read the sticks (get_axis needs the event pump above)
        try:
            self.joystick_state = pygame_to_joystick_state(self.joystick)
        except pygame.error:
            print("Joystick error; using default forward command.")
            self.joystick = None
            self.joystick_state = JoystickState()

    # resolve the commanded twist [vx, vy, wz] for this control step: take the raw
    # physical command from the active source, then mode-select + clamp it to the
    # gait library (utils.crawl_modes) so it can never leave the reachable set.
    def commanded_twist(self):
        if self.profile_segments is not None:
            # scripted staircase setpoint for the current sim time
            twist, self._seg = self.profile_twist(self.mj_data.time)
        elif self.fixed_twist is not None:
            # explicit fixed command (deterministic; ignores the joystick)
            twist = self.fixed_twist
        elif self.joystick is not None:
            # live joystick. Input map matches deploy/joystick/joystick_pygame.py:
            #   vx = LS_Y (left stick fore/aft), vy = RS_X (right stick lateral),
            #   wz = LT - RT (left trigger turns +, right trigger turns -)
            raw = np.array([self.joystick_state.LS_Y,
                            self.joystick_state.RS_X,
                            self.joystick_state.LT - self.joystick_state.RT],
                           dtype=np.float32)
            twist = raw * self.twist_scale
        else:
            # no joystick connected -> autonomous forward crawl
            twist = self.default_twist

        self.mode, shaped = resolve_crawl_twist(twist, self.config)
        return shaped


    #################################################################
    # OBSERVATION / SENSING  (identical to control_29dof_crawl_omni.py)
    #################################################################

    # read the sensors the controller consumes, straight from mj_data
    def read_sensors(self):
        # pelvis IMU: quaternion (w, x, y, z) and gyro (base_ang_vel)
        quat = self.mj_data.sensor('pelvis_imu_quat_sensor').data.copy()
        omega = self.mj_data.sensor('pelvis_imu_gyro_sensor').data.copy()
        # per-joint position / velocity, in actuator order
        qpos_joints = np.array([self.mj_data.sensor(n).data[0] for n in self.joint_pos_sensor_names])
        qvel_joints = np.array([self.mj_data.sensor(n).data[0] for n in self.joint_vel_sensor_names])
        return quat, omega, qpos_joints, qvel_joints

    # build the 98-d observation in the trained order (from observation_names):
    # base_ang_vel(3), joint_pos(29), joint_vel(29), actions(29),
    # commanded_twist(3), motion_phase(2), projected_gravity(3)
    def build_observation(self, quat, omega, qpos_joints, qvel_joints):
        qj = qpos_joints - self.qpos_joints_default
        dqj = qvel_joints

        # gait phase clock: (sin, cos) of 2*pi*t/T
        phase = self.phase_step / self.motion_period_frames
        ang = 2.0 * math.pi * phase
        motion_phase = np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)

        # projected gravity from the pelvis quaternion (roll/pitch; yaw-invariant)
        proj_grav = get_gravity_orientation(quat).astype(np.float32)

        obs = np.concatenate([
            omega,
            qj, dqj, self.action,
            self.twist, motion_phase, proj_grav,
        ]).astype(np.float32)
        return obs


    #################################################################
    # PHYSICS / ACTUATION  (identical to simulation.py)
    #################################################################

    # add per-sensor Gaussian noise in place, using the std devs from the XML
    def apply_sensor_noise(self):
        for i in range(self.mj_model.nsensor):
            std = self.mj_model.sensor_noise[i]
            if std <= 0.0:
                continue
            adr = self.mj_model.sensor_adr[i]
            dim = self.mj_model.sensor_dim[i]
            self.mj_data.sensordata[adr:adr + dim] += np.random.normal(0.0, std, size=dim)

    # PD control + feedforward: tau = Kp*(q_des - q) + Kd*(0 - dq)
    def compute_torque(self):
        qpos_joints = self.mj_data.qpos[7:7 + self.nu]
        qvel_joints = self.mj_data.qvel[6:6 + self.nu]
        tau = (self.Kp * (self.qpos_des - qpos_joints)
               + self.Kd * (0.0 - qvel_joints))
        return tau


    #################################################################
    # VELOCITY-TRACKING MEASUREMENT
    #################################################################

    # achieved twist [vx, vy, wz] of the base, expressed in the robot's heading
    # frame (the frame the twist command lives in). World linear/angular velocity
    # come from mj_objectVelocity; the heading is the instantaneous yaw of the base
    # +z axis (ground-forward in the crawl pose) read straight from the orientation
    # -- no integration, so no gait-wag accumulation or drift.
    def achieved_twist(self):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data,
                                 mujoco.mjtObj.mjOBJ_BODY, self.track_body_id, vel6, 0)  # 0 = world
        omega_world = vel6[0:3]
        v_world = vel6[3:6]
        wz = float(omega_world[2])

        # forward-on-ground = base +z axis in world (3rd column of world_R_body)
        R = self.mj_data.xmat[self.track_body_id].reshape(3, 3)
        heading = math.atan2(R[1, 2], R[0, 2])
        c, s = math.cos(heading), math.sin(heading)
        vx = float(c * v_world[0] + s * v_world[1])
        vy = float(-s * v_world[0] + c * v_world[1])
        return vx, vy, wz


    #################################################################
    # MAIN LOOP
    #################################################################

    # one policy step: observe -> infer -> hold target over `decimation` physics steps
    def control_step(self):
        # refresh the command (live joystick, fixed --twist, or default forward)
        self.update_joystick()
        self.twist = self.commanded_twist()
        if self.joystick is not None:
            self.joystick_drove = True

        # announce crawl-mode changes (forward / backward / turn / idle)
        if self.mode != self._last_mode:
            print(f"[mode] {self._last_mode} -> {self.mode}  cmd={np.round(self.twist, 3).tolist()}")
            self._last_mode = self.mode

        # build the observation from the CURRENT sensor state, then infer
        quat, omega, qpos_joints, qvel_joints = self.read_sensors()
        obs = self.build_observation(quat, omega, qpos_joints, qvel_joints)
        self.action = self.policy.inference(obs, time_step=self.phase_step)

        # policy action -> PD position target (held across the physics substeps)
        self.qpos_des = (self.action * self.action_scale
                         + self.qpos_joints_default).astype(np.float64)

        # step physics `decimation` times with the target held fixed
        for _ in range(self.decimation):
            self.mj_data.ctrl[:] = self.compute_torque()
            mujoco.mj_step(self.mj_model, self.mj_data)
            if self.apply_noise:
                self.apply_sensor_noise()

        # measure achieved body-frame velocity and log it (velocity-tracking exps)
        if self.log_enabled:
            avx, avy, awz = self.achieved_twist()
            self.log_rows.append((float(self.mj_data.time),
                                  float(self.twist[0]), float(self.twist[1]), float(self.twist[2]),
                                  avx, avy, awz, float(self._seg)))

        # advance the free-running gait clock (wraps at T)
        self.phase_step = (self.phase_step + 1) % self.motion_period_frames

    # run for a fixed number of policy steps, or indefinitely if num_steps is None
    # (interactive joystick mode -- stop by closing the viewer or pressing Ctrl+C)
    def run(self, num_steps=None):
        real_start = time.perf_counter()
        indefinite = num_steps is None

        step = 0
        try:
            while indefinite or step < num_steps:
                self.control_step()
                step += 1

                # optional viewer render + real-time pacing (state-invariant: only sleeps)
                if self.viewer is not None:
                    if not self.viewer.is_running():
                        break
                    # render at exactly RENDER_HZ (every Nth control step)
                    if (step - 1) % self.render_every == 0:
                        self.viewer.sync()
                    if self.realtime:
                        # pace to the sim clock; sleeping never changes physics state
                        target_wall = real_start + step * self.ctrl_dt
                        remaining = target_wall - time.perf_counter()
                        if remaining > 0.0:
                            time.sleep(remaining)
        except KeyboardInterrupt:
            pass

        self.report(step)
        self.save_log()

    # deterministic run summary -- a fingerprint you can diff across runs
    def report(self, num_steps: int):
        base_xy = self.mj_data.qpos[0:2].copy()
        start_xy = self.home_base[0:2]
        dist = float(np.linalg.norm(base_xy - start_xy))
        # cheap reproducibility checksum over the final full state
        state = np.concatenate([self.mj_data.qpos, self.mj_data.qvel])
        checksum = float(np.sum(state * np.arange(1, state.size + 1)))
        print("\n=== deterministic run summary ===")
        print(f"    policy steps:       {num_steps}  ({num_steps * self.ctrl_dt:.2f} s sim time)")
        print(f"    base start xy:      [{start_xy[0]:+.4f}, {start_xy[1]:+.4f}]")
        print(f"    base final xy:      [{base_xy[0]:+.4f}, {base_xy[1]:+.4f}]")
        print(f"    planar displacement:{dist:.4f} m")
        if self.joystick_drove:
            print(f"    state checksum:     {checksum:.6f}")
            print("    NOTE: a live joystick drove this run -> not reproducible (expected).")
        else:
            print(f"    state checksum:     {checksum:.6f}   (identical across runs => deterministic)")

    # write the velocity-tracking log to an npz for plot_velocity_tracking.py
    def save_log(self):
        if not self.log_enabled or not self.log_rows:
            return
        arr = np.array(self.log_rows, dtype=np.float64)
        out_dir = os.path.dirname(self.log_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(self.log_path,
                 t=arr[:, 0], cmd=arr[:, 1:4], ach=arr[:, 4:7], seg=arr[:, 7],
                 profile=str(self.profile_name),
                 hold_s=self.hold_s, settle_s=self.settle_s, period_s=self.period_s)
        print(f"Saved velocity-tracking log: {len(self.log_rows)} rows -> {self.log_path}")

    def close(self):
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()
        if self._pygame_inited:
            pygame.quit()


############################################################################
# MAIN FUNCTION
############################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Deterministic single-process sim for the MjLab omnidirectional crawling policy.'
    )
    parser.add_argument('--config', type=str, default='g1_29dof_crawl_omni.yaml',
                        help='Config yaml (default: g1_29dof_crawl_omni.yaml).')
    parser.add_argument('--duration', type=float, default=None,
                        help='Sim time to run, in seconds. Default: interactive joystick runs '
                             'are indefinite; fixed-command/headless runs use 20 s.')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed (default: 0). Only affects the optional sensor noise.')
    parser.add_argument('--twist', type=float, nargs=3, metavar=('VX', 'VY', 'WZ'), default=None,
                        help='Force a FIXED physical twist [vx, vy, wz] (clipped to the trained '
                             'range), ignoring the joystick. Makes the run reproducible.')
    parser.add_argument('--no-joystick', action='store_true',
                        help='Do not open a joystick; always use the default forward command.')
    parser.add_argument('--profile', type=str, default=None, choices=list(PROFILES) + ["combo"],
                        help='Run a velocity-tracking staircase instead of live driving '
                             '(overrides joystick/--twist). "combo" samples library commands '
                             'across the 3 motions. Logs commanded vs achieved velocity.')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to write the velocity-tracking npz. Default for a --profile '
                             'run: logs/veltrack_<profile>.npz.')
    parser.add_argument('--no-plot', action='store_true',
                        help='For a --profile run, skip auto-opening the tracking plot afterward.')
    parser.add_argument('--noise', action='store_true',
                        help='Enable seeded per-sensor Gaussian noise (off by default).')
    parser.add_argument('--headless', action='store_true',
                        help='Run without the viewer (fastest; for reproducibility checks).')
    parser.add_argument('--fast', action='store_true',
                        help='Do not pace to real time (run as fast as possible).')
    args = parser.parse_args()

    # a --profile run always logs; default the path if none was given
    log_path = args.log
    if args.profile is not None and log_path is None:
        log_path = f"{ROOT_DIR}/logs/veltrack_{args.profile}.npz"

    sim = DeterministicOmniSim(
        config_path=args.config,
        seed=args.seed,
        twist_override=args.twist,
        use_joystick=not args.no_joystick,
        apply_noise=args.noise,
        headless=args.headless,
        realtime=not args.fast,
        profile=args.profile,
        log_path=log_path,
    )

    # a --profile run is bounded to the staircase length. Otherwise: an explicit
    # --duration wins; an interactive joystick session runs indefinitely; and
    # fixed-command / headless runs default to 20 s.
    joystick_mode = sim.use_joystick and sim.fixed_twist is None and not sim.headless
    if sim.profile_segments is not None:
        num_steps = int(round(sim.profile_total_s / sim.ctrl_dt))
    elif args.duration is not None:
        num_steps = int(round(args.duration / sim.ctrl_dt))
    elif joystick_mode:
        num_steps = None
    else:
        num_steps = int(round(20.0 / sim.ctrl_dt))

    if num_steps is None:
        print("Joystick mode: running indefinitely. Close the viewer or press Ctrl+C to stop.")
    else:
        print(f"Running {num_steps} policy steps ({num_steps * sim.ctrl_dt:.1f} s sim time).")

    try:
        sim.run(num_steps)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()

    # auto-open the tracking plot right after a profile run (fast iterate)
    if args.profile is not None and log_path and not args.no_plot:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from plot_velocity_tracking import render_log
            render_log(log_path, show=True)
        except Exception as e:
            print(f"(auto-plot failed: {e})")

    print("Deterministic simulation complete.")


if __name__ == "__main__":
    main()

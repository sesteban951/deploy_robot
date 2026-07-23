##
#
# Offline reconstruction + replay of the 29-DoF mimic observation from a log.
#
# This MIRRORS deploy/hardware/control_29dof_mimic.py:
#   - build_observation()      lines 233-265  (obs layout + math)
#   - init_quat capture         lines 314-321  (heading alignment at track start)
#   - init_policy() anchor idx  lines 170-186  (anchor body / IMU selection)
# The obs math is already duplicated between the hardware and simulation control
# nodes; this is a third, read-only copy for diagnostics. KEEP IN SYNC if the
# observation definition changes.
#
# Obs layout (154): [command(58), motion_anchor_ori_b(6), base_ang_vel(3),
#                    joint_pos(29), joint_vel(29), actions(29)]
#
##

import numpy as np
import mujoco

from utils.policy import Policy
from utils.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rot6d,
    quat_to_rpy,
    heading_about_z_world,
)

N_JOINTS = 29

# Obs vector block layout (154), matching the assembly in assemble_obs_steps /
# replay: [command(58)=motion_joint_pos+motion_joint_vel, motion_anchor_ori_b(6),
# base_ang_vel(3), joint_pos(29), joint_vel(29), prev_action(29)]. Single source of
# truth for the diagnostics that slice the obs by block (all-obs / compare / ablation
# / sensitivity). Each entry is (name, start, stop) with stop exclusive.
OBS_BLOCKS = [
    ("motion_joint_pos", 0, 29),
    ("motion_joint_vel", 29, 58),
    ("motion_anchor_ori_b", 58, 64),
    ("base_ang_vel", 64, 67),
    ("joint_pos", 67, 96),
    ("joint_vel", 96, 125),
    ("prev_action", 125, 154),
]
OBS_SIZE = OBS_BLOCKS[-1][2]  # 154


class MimicReplay:
    """Reconstructs the mimic observation from logged sensors and replays the ONNX policy."""

    def __init__(self, config: dict, root_dir: str):
        self.root_dir = root_dir
        self.ctrl_dt = float(config["control_dt"])

        # policy (+ embedded deployment metadata)
        self.policy = Policy(root_dir + "/policy/" + config["policy_path"])
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size
        self.default = np.asarray(self.policy.get_param("default_joint_pos"), dtype=np.float32)
        self.action_scale = np.asarray(self.policy.get_param("action_scale"), dtype=np.float32)

        # motion reference
        motion = np.load(root_dir + "/motions/" + config["motion_path"])
        self.motion_fps = float(motion["fps"])
        self.motion_joint_pos = motion["joint_pos"].astype(np.float32)
        self.motion_joint_vel = motion["joint_vel"].astype(np.float32)
        self.motion_body_quat_w = motion["body_quat_w"].astype(np.float32)
        self.num_frames = self.motion_joint_pos.shape[0]

        # anchor body index + IMU, resolved exactly like the control node
        self.anchor_name = self.policy.metadata["anchor_body_name"]
        mj_model = mujoco.MjModel.from_xml_path(root_dir + "/models/" + config["xml_path"])
        body_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, mj_model.nbody)  # skip world
        ]
        self.anchor_body_idx = body_names.index(self.anchor_name)
        self.anchor_imu = "pelvis" if "pelvis" in self.anchor_name.lower() else "torso"

    # ------------------------------------------------------------------ #
    # frame timeline
    # ------------------------------------------------------------------ #

    # recover the previous action the controller stored in the obs, from the
    # logged command: q_des = action * action_scale + default  ->  invert it.
    def recover_actions(self, command: np.ndarray) -> np.ndarray:
        q_des = command[:, 0:N_JOINTS]
        return (q_des - self.default) / self.action_scale

    # Track-start detector: end of the frame-0 hold.
    #
    # The FSM flow is home -> control -> track. In "control" the commanded pose
    # ramps to motion frame 0 (over frame_pos_duration) and is then HELD at frame
    # 0 (often for many seconds) until the operator triggers "track", when the
    # trajectory starts advancing. So the commanded q_des sits at motion[0] during
    # the hold and departs when tracking begins -- that departure is t_start.
    #
    # We key on the COMMAND (q_des), which equals motion[0] essentially exactly at
    # the ramp end, so this is sharp and works even when the policy fails to track
    # (the reference still sweeps, so q_des still departs). Returns (t_start, info)
    # or (None, reason) if no clean hold is found (caller must supply --t_start).
    #
    # qdes_log: (T,29) logged commanded joint positions (command[:, 0:29]).
    # min_run / gap are in ROWS, not seconds: hardware_time is duplicated across
    # log rows (time updates at 100 Hz, logger samples ~200 Hz) so diff(t) is often
    # 0 and a time-derived window is unreliable.
    #
    # The ramp->policy handoff produces a brief (~60 ms) transient spike in q_des
    # right after frame 0 is reached, which must NOT be mistaken for track start.
    # We therefore require the departure from frame 0 to be SUSTAINED: a run of
    # d0 > depart_tol lasting >= min_run rows (bridging gaps <= gap rows so a
    # split/oscillating run still counts). The real (even briefly-aborted) track
    # sweep lasts many rows; the handoff transient does not.
    #
    # Note: logging stops when the FSM leaves control/track (log.py gates on
    # TARGET_FSM_STATES), so an aborted run's real track sits at the tail of the
    # log. This finds the first sustained departure; pass --t_start to pick another.
    def detect_t_start(self, t, qdes_log, hold_tol: float = 0.10,
                       depart_tol: float = 0.8, min_run: int = 20, gap: int = 8):
        d0 = np.linalg.norm(qdes_log - self.motion_joint_pos[0], axis=1)
        below = np.where(d0 < hold_tol)[0]
        if len(below) == 0:
            return None, "no frame-0 hold found (command never reaches motion[0])"
        r0 = int(below[0])  # command first reaches frame 0 (ramp end)
        above = d0 > depart_tol
        n = len(t)
        i = r0
        while i < n:
            if above[i]:
                # extend the run from i, bridging short gaps (<= gap rows)
                last = i
                k = i + 1
                while k < n and (above[k] or (k - last) <= gap):
                    if above[k]:
                        last = k
                    k += 1
                if last - i + 1 >= min_run:
                    return float(t[i]), (f"frame-0 hold {t[r0]:.2f}->{t[i]:.2f}s, "
                                         f"track starts {t[i]:.2f}s (sustained {last - i + 1} rows)")
                i = k
            else:
                i += 1
        return None, f"command holds frame 0 for the whole log (reached at {t[r0]:.2f}s); no sustained track sweep"

    # map control steps (one per motion frame) to nearest log rows within the window
    def build_steps(self, t: np.ndarray, t_start: float, t_end: float):
        frames, rows = [], []
        for f in range(self.num_frames):
            t_target = t_start + f * self.ctrl_dt
            if t_target > t_end or t_target > t[-1]:
                break
            row = int(np.clip(np.searchsorted(t, t_target), 0, t.shape[0] - 1))
            frames.append(f)
            rows.append(row)
        return np.array(frames, dtype=int), np.array(rows, dtype=int)

    # ------------------------------------------------------------------ #
    # observation reconstruction
    # ------------------------------------------------------------------ #

    # heading-alignment quat captured on the first policy tick (control node lines 314-321).
    # The controller captures this ONCE at the control->policy handoff (ramp end) and
    # reuses it through the hold and the whole track, so pass the anchor quat there.
    def capture_init_quat(self, anchor_quat_start: np.ndarray) -> np.ndarray:
        motion_anchor_quat_0 = self.motion_body_quat_w[0, self.anchor_body_idx]
        q_rel = quat_multiply(anchor_quat_start, quat_conjugate(motion_anchor_quat_0))
        return heading_about_z_world(q_rel)

    # zero_imu_yaw (alternative to init_quat/align_heading): the controller captures a yaw
    # offset from the anchor IMU quat at the same handoff and left-multiplies its conjugate
    # onto every measured anchor quat feeding the obs. Mirrors control_29dof_mimic
    # build_observation. Pass the anchor quat at the handoff, then apply_yaw_offset to the log.
    def capture_yaw_offset(self, anchor_quat_start: np.ndarray) -> np.ndarray:
        return quat_conjugate(heading_about_z_world(anchor_quat_start))

    # left-multiply the captured yaw offset onto every row of an anchor-quat array, so the
    # existing obs methods (replay / obs_blocks_full / anchor_ori_b_forms) can be fed the
    # corrected quats with init_quat = identity, matching the controller exactly.
    def apply_yaw_offset(self, anchor_quat_array: np.ndarray, yaw_offset_conj: np.ndarray) -> np.ndarray:
        out = np.empty_like(anchor_quat_array)
        for i in range(anchor_quat_array.shape[0]):
            out[i] = quat_multiply(yaw_offset_conj, anchor_quat_array[i])
        return out

    # row of the control->policy handoff: where the command first reaches frame 0
    # (ramp end). This is where the controller captures init_quat.
    def find_handoff_row(self, qdes_log, hold_tol: float = 0.10) -> int:
        d0 = np.linalg.norm(qdes_log - self.motion_joint_pos[0], axis=1)
        below = np.where(d0 < hold_tol)[0]
        return int(below[0]) if len(below) else 0

    # motion_anchor_ori_b (6) for one frame given the measured anchor quat
    def anchor_ori_b(self, frame: int, anchor_quat: np.ndarray, init_quat: np.ndarray) -> np.ndarray:
        motion_anchor_quat_w = self.motion_body_quat_w[frame, self.anchor_body_idx]
        ref_quat_corrected = quat_multiply(init_quat, motion_anchor_quat_w)
        rel_quat = quat_multiply(quat_conjugate(anchor_quat), ref_quat_corrected)
        return quat_to_rot6d(rel_quat)

    # motion_anchor_ori_b for a single frame, returned in the three equivalent forms
    # used by the plots: 6D rotation (as the policy sees it), quaternion, and rpy.
    # This is the per-tick core shared by obs_blocks_full (offline) and live_debug.py.
    def anchor_ori_b_forms(self, frame: int, anchor_quat: np.ndarray, init_quat: np.ndarray):
        motion_anchor_quat_w = self.motion_body_quat_w[frame, self.anchor_body_idx]
        ref_quat_corrected = quat_multiply(init_quat, motion_anchor_quat_w)
        rel = quat_multiply(quat_conjugate(anchor_quat), ref_quat_corrected)
        rel = rel / (np.linalg.norm(rel) + 1e-12)
        if rel[0] < 0:                          # resolve q/-q double cover for a continuous plot
            rel = -rel
        return quat_to_rot6d(rel), rel.astype(np.float32), quat_to_rpy(rel)

    # motion_anchor_ori_b over the FULL log: the reference is held at frame 0 before
    # t_start (the controller holds frame 0 during the pre-track hold) and advances
    # 1 frame per ctrl_dt after t_start. init_quat is the handoff capture (reused
    # throughout, matching the controller). Returns the orientation error in three
    # equivalent forms: 6D rotation (as the policy sees it), quaternion, and rpy.
    def obs_blocks_full(self, t, t_start, anchor_quat_full, init_quat):
        T = t.shape[0]
        ori6d = np.empty((T, 6), dtype=np.float32)
        relq = np.empty((T, 4), dtype=np.float32)
        relrpy = np.empty((T, 3), dtype=np.float32)
        for i in range(T):
            if t[i] < t_start:
                frame = 0
            else:
                frame = int(np.clip(round((t[i] - t_start) / self.ctrl_dt), 0, self.num_frames - 1))
            ori6d[i], relq[i], relrpy[i] = self.anchor_ori_b_forms(frame, anchor_quat_full[i], init_quat)
        return ori6d, relq, relrpy

    # ------------------------------------------------------------------ #
    # full replay
    # ------------------------------------------------------------------ #

    # Assemble the per-step obs matrix (S, 154) exactly as the controller/policy sees
    # it, using the LOGGED previous action for the action block. This is the single
    # copy of the step-wise obs math; replay() and the ablation diagnostics both build
    # on it. Blocks follow OBS_BLOCKS.
    #
    # anchor_quat_log: (T,4) torso (or pelvis) IMU quat per log row (already yaw-corrected
    #                  by the caller if zero_imu_yaw is on)
    # pelvis_gyro_log: (T,3) pelvis gyro per log row
    # q_log, dq_log:   (T,29) joint encoders
    # actions_log:     (T,29) recovered previous action (from recover_actions)
    def assemble_obs_steps(self, frames, rows, anchor_quat_log, pelvis_gyro_log,
                           q_log, dq_log, actions_log, init_quat):
        S = len(frames)
        obs = np.empty((S, OBS_SIZE), dtype=np.float32)
        for k in range(S):
            f, row = int(frames[k]), int(rows[k])
            command = np.concatenate([self.motion_joint_pos[f], self.motion_joint_vel[f]])
            ori = self.anchor_ori_b(f, anchor_quat_log[row], init_quat)
            omega = pelvis_gyro_log[row]
            qj = q_log[row] - self.default
            dqj = dq_log[row]
            prev_action = actions_log[rows[k - 1]] if k > 0 else np.zeros(N_JOINTS, np.float32)
            obs[k] = np.concatenate([command, ori, omega, qj, dqj, prev_action]).astype(np.float32)
        return obs

    # Run the policy row-by-row on a (possibly mutated) obs matrix. `frames` supplies
    # the per-row time_step. Returns predicted action + qpos_des. This is what the
    # ablation diagnostic drives (mutate one block, re-run, compare).
    def replay_from_obs(self, obs_matrix, frames):
        S = obs_matrix.shape[0]
        pred_action = np.empty((S, N_JOINTS), dtype=np.float32)
        for k in range(S):
            pred_action[k] = self.policy.inference(obs_matrix[k].astype(np.float32),
                                                   time_step=int(frames[k]))
        return {
            "pred_action": pred_action,
            "pred_qpos_des": pred_action * self.action_scale + self.default,
        }

    # Open-loop replay: at each control step feed the reconstructed obs using the
    # LOGGED previous action, run the policy, and return both the predicted and
    # the logged actions plus the IMU-dependent obs blocks. Thin wrapper over
    # assemble_obs_steps + replay_from_obs (kept for the existing debug windows).
    def replay(self, frames, rows, anchor_quat_log, pelvis_gyro_log,
               q_log, dq_log, actions_log, init_quat):
        obs = self.assemble_obs_steps(frames, rows, anchor_quat_log, pelvis_gyro_log,
                                      q_log, dq_log, actions_log, init_quat)
        pred = self.replay_from_obs(obs, frames)
        b = OBS_BLOCKS
        ori_b = obs[:, b[2][1]:b[2][2]]          # motion_anchor_ori_b (6)
        ang_vel = obs[:, b[3][1]:b[3][2]]        # base_ang_vel (3)
        logged_action = actions_log[rows]
        return {
            "frames": frames,
            "rows": rows,
            "init_quat": init_quat,
            "obs": obs,
            "motion_anchor_ori_b": ori_b,
            "base_ang_vel": ang_vel,
            "pred_action": pred["pred_action"],
            "logged_action": logged_action,
            "pred_qpos_des": pred["pred_qpos_des"],
            "logged_qpos_des": logged_action * self.action_scale + self.default,
        }

    # ------------------------------------------------------------------ #
    # full-log obs (all channels) + sensitivity
    # ------------------------------------------------------------------ #

    # Full-log-timeline obs matrix (T, 154) + per-row motion frame index, mirroring the
    # frame logic of obs_blocks_full (reference held at frame 0 before t_start, advancing
    # 1 frame per ctrl_dt after). Non-reference blocks are per log row; the action block
    # uses the recovered action at that row (what the controller had stored). This is the
    # source for the all-obs overview and the two-log channel comparison (full timeline,
    # including the pre-track hold). anchor_quat_full is the (already yaw-corrected) anchor
    # quat per row; init_quat the handoff capture.
    def assemble_obs_full(self, t, t_start, anchor_quat_full, init_quat,
                          pelvis_gyro, q_log, dq_log, actions_log):
        T = t.shape[0]
        obs = np.empty((T, OBS_SIZE), dtype=np.float32)
        frames = np.empty(T, dtype=int)
        for i in range(T):
            if t[i] < t_start:
                f = 0
            else:
                f = int(np.clip(round((t[i] - t_start) / self.ctrl_dt), 0, self.num_frames - 1))
            frames[i] = f
            command = np.concatenate([self.motion_joint_pos[f], self.motion_joint_vel[f]])
            ori = self.anchor_ori_b(f, anchor_quat_full[i], init_quat)
            qj = q_log[i] - self.default
            obs[i] = np.concatenate([command, ori, pelvis_gyro[i], qj,
                                     dq_log[i], actions_log[i]]).astype(np.float32)
        return obs, frames

    # Finite-difference sensitivity of the policy output to each obs channel, per row.
    # For channel i: J[:,i] = (pi(o+eps*e_i) - pi(o-eps*e_i)) / (2*eps); returns
    #   gain[T,154]        = ||J[:,i]|| (norm over the 29 outputs) -- local policy gain
    #   contribution[T,154]= gain * std over the window of that channel -- gain x how much
    #                        the channel actually moved (first-order effect on the output).
    # Local linearization near the observed trajectory (not a global attribution).
    def sensitivity(self, obs_matrix, frames, eps: float = 1e-2):
        T, D = obs_matrix.shape
        gain = np.empty((T, D), dtype=np.float32)
        for k in range(T):
            o = obs_matrix[k].astype(np.float32)
            f = int(frames[k])
            for i in range(D):
                op = o.copy(); op[i] += eps
                om = o.copy(); om[i] -= eps
                da = self.policy.inference(op, time_step=f) - self.policy.inference(om, time_step=f)
                gain[k, i] = np.linalg.norm(da) / (2.0 * eps)
        std = obs_matrix.std(axis=0, keepdims=True)
        contribution = gain * std
        return gain, contribution

    # Sum a per-channel (…,154) array into the 7 OBS_BLOCKS along the last axis.
    def block_reduce(self, per_channel):
        per_channel = np.asarray(per_channel)
        return np.stack([per_channel[..., s:e].sum(axis=-1) for _, s, e in OBS_BLOCKS], axis=-1)

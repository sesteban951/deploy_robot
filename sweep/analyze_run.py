##
#
# Tracking-error analysis for a single sweep run.
#
# Unlike logs/sim2sim_error.py (which phase-aligns and segments a continuously
# looping log), a sweep run plays the motion exactly once and we know precisely
# when 'track' started (from the FSM driver's sidecar). So we trim the log to
# [track_start, track_start + motion_duration], align frames directly, and
# compute the same pos / vel / orientation error using the shared helpers.
#
##

import os
import sys

import numpy as np
import yaml

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "logs"))  # so sim2sim_error's `from plot import ...` resolves

# reuse the vetted loaders + metrics from the existing report
from sim2sim_error import (
    load_log,
    load_reference,
    resolve_anchor,
    joint_error_stats,
    orientation_error,
    config_from_meta,
)
from utils.math_utils import (quat_conjugate, quat_multiply, heading_about_z_world,
                              quat_to_rotation_matrix)


############################################################################
# BASE (FLOATING-BASE) ERROR
############################################################################

# Ground-truth base error, sim only -- needs the `base_state` dataset
# ([pos(3), quat(4), lin_vel(3), ang_vel(3)], world frame) that sim_headless
# publishes. Returns {} when the log predates it.
#
# The robot spawns at an arbitrary position/heading relative to the reference
# frame, so raw world coordinates are not comparable. Both trajectories are
# expressed RELATIVE TO THEIR OWN FIRST FRAME and the measured one is rotated by
# the heading offset, exactly as the orientation error already does. What
# survives is genuine tracking error, not spawn placement.
#
# Height is the exception: z is absolute (both share the floor plane), so
# base_height_rmse is reported un-differenced -- that one is directly "did it
# get as high off the ground as the reference".
#
# COMBINED BASE POSE ERROR
# Translation and rotation cannot be summed directly -- metres and degrees are
# different units, and their raw magnitudes would set the weighting by accident.
# Each is first divided by a characteristic scale, giving dimensionless terms
# that are ~[0,1] and comparable:
#     pos_term = base_pos_rmse   / BASE_POS_SCALE_M     (fraction of body scale)
#     ori_term = base_ori_rmse   / BASE_ORI_SCALE_DEG   (fraction of max error)
#     base_pose_error = pos_term + ori_term
# The scales are FIXED constants, not per-motion quantities, so the metric stays
# comparable across motions (same convention as combined_error.py).
BASE_POS_SCALE_M = 0.79      # nominal base height (reference frame-0 pelvis z)
BASE_ORI_SCALE_DEG = 180.0   # max possible geodesic orientation error
BASE_REF_KEYS = ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w")


# load_reference (shared with sim2sim_error) only carries joint_pos / joint_vel /
# body_quat_w. The base metrics also need positions and world-frame velocities,
# so pull them from the same npz. Returns False if the motion lacks them.
def augment_reference_for_base(ref, cfg):
    if all(k in ref for k in BASE_REF_KEYS):
        return True
    motion = np.load(os.path.join(ROOT_DIR, "motions", cfg["motion_path"]))
    if not all(k in motion.files for k in BASE_REF_KEYS):
        return False
    for k in BASE_REF_KEYS:
        ref[k] = motion[k].astype(np.float64)
    return True


def base_error_metrics(base_win, ref, frame, anchor_idx=0):
    pos = base_win[:, 0:3].astype(np.float64)
    quat = base_win[:, 3:7].astype(np.float64)
    lin = base_win[:, 7:10].astype(np.float64)
    ang = base_win[:, 10:13].astype(np.float64)

    ref_pos = ref["body_pos_w"][frame, anchor_idx]
    ref_quat = ref["body_quat_w"][frame, anchor_idx]
    ref_lin = ref["body_lin_vel_w"][frame, anchor_idx]
    ref_ang = ref["body_ang_vel_w"][frame, anchor_idx]

    # heading offset captured at the first in-window sample (same convention as
    # orientation_error), as a world-frame rotation about z
    init_quat = heading_about_z_world(quat_multiply(quat[0], quat_conjugate(ref_quat[0])))
    R = quat_to_rotation_matrix(quat_conjugate(init_quat))  # world(run) -> world(ref)

    # position: start-relative, de-rotated, compared to the reference's own displacement
    d_meas = (R @ (pos - pos[0]).T).T
    d_ref = ref_pos - ref_pos[0]
    pos_err = d_meas - d_ref
    pos_norm = np.linalg.norm(pos_err, axis=1)

    # velocities: de-rotated only (no differencing -- velocity has no offset)
    lin_err = (R @ lin.T).T - ref_lin
    ang_err = (R @ ang.T).T - ref_ang

    # orientation of the BASE link (the anchor tables score torso_link instead)
    ori = np.degrees([2.0 * np.arccos(np.clip(abs(quat_multiply(quat_conjugate(quat[i]),
                                                                quat_multiply(init_quat, ref_quat[i]))[0]), 0, 1))
                      for i in range(len(quat))])

    base_pos_rmse = float(np.sqrt(np.mean(pos_norm ** 2)))
    base_ori_rmse = float(np.sqrt(np.mean(ori ** 2)))
    pos_term = base_pos_rmse / BASE_POS_SCALE_M
    ori_term = base_ori_rmse / BASE_ORI_SCALE_DEG

    return {
        "base_pos_rmse": base_pos_rmse,
        "base_pos_max": float(np.max(pos_norm)),
        "base_pos_term": pos_term,
        "base_ori_term": ori_term,
        "base_pose_error": pos_term + ori_term,
        "base_height_rmse": float(np.sqrt(np.mean((pos[:, 2] - ref_pos[:, 2]) ** 2))),
        "base_height_max": float(np.max(np.abs(pos[:, 2] - ref_pos[:, 2]))),
        "base_lin_vel_rmse": float(np.sqrt(np.mean(np.sum(lin_err ** 2, axis=1)))),
        "base_ang_vel_rmse": float(np.sqrt(np.mean(np.sum(ang_err ** 2, axis=1)))),
        "base_ori_rmse_deg": base_ori_rmse,
        "base_ori_max_deg": float(np.max(ori)),
    }


############################################################################
# LANDING DETECTION
############################################################################

# A rep "lands" if, after the motion has played out, the robot is holding the
# pose the motion ended in and has stopped moving. The FSM driver holds the robot
# in 'track' for an extra settle window (land_hold) during which the control node
# freezes on the last motion frame, so this window is exactly "did it stay on its
# feet afterwards".
#
# IMPORTANT: the verdict is orientation error against the REFERENCE's final frame,
# not tilt from vertical. Several of these references never straighten up -- they
# end with the torso pitched -- so an absolute-tilt test flags a perfectly good
# landing as a fall. Absolute tilt is still reported, as a diagnostic only.
#
# NOTE: base position is not logged (the IMU is orientation-only), so height off
# the floor is not observable here -- the verdict is orientation + settling.
#
# CAVEAT on the binary verdict: nothing balances the robot after the motion ends
# (the control node freezes on the last frame), so given long enough almost
# anything topples and `landed` really means "still holding at land_hold
# seconds". Report `land_hold_time_s` alongside it -- that one does not depend
# on where the horizon was drawn.
# Two windows, deliberately decoupled:
#   verdict_s  -- short, right after the motion: "did it arrive on-pose?" This is
#                 the binary `landed`, i.e. did the flip complete.
#   land_hold  -- the whole observation window: how long it then held that pose
#                 before diverging (`land_hold_time_s`). Carries the stability
#                 information the short verdict deliberately ignores.
LAND_VERDICT_S = 0.5       # touchdown window (s) the binary verdict is read from
# 45 deg is CALIBRATED, not guessed. Reps were adjudicated visually in the viewer
# (traj_opt_kino_4, whose touchdown error sits at 27-33 deg, was judged landed),
# and the measured distribution is strongly bimodal: landings top out at 29.8 deg
# while genuine falls start at 75.7 deg -- a 46 deg gap with nothing in it. 45 sits
# in that gap with margin above every observed landing, well below every observed
# fall. An earlier 30 deg value bisected the landing cluster, which made one
# policy's success rate swing 15/15 -> 13/15 -> 3/5 across identical sweeps.
LAND_ORI_ERR_DEG = 45.0    # max geodesic error vs the reference's final orientation
LAND_GYRO_RMS = 2.0        # rad/s, max angular-rate RMS over the verdict window
LAND_JOINT_VEL_RMS = 3.0   # rad/s, max joint-velocity RMS over the verdict window
LAND_FINAL_FRAC = 0.4      # verdict reads the last fraction of the verdict window


# tilt of body +z from world +z, in degrees, for (N, 4) wxyz quaternions
def tilt_deg_from_quat(quat_wxyz):
    q = np.atleast_2d(np.asarray(quat_wxyz, dtype=np.float64))
    cos_tilt = np.clip(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


# geodesic angle (deg) between measured quaternions and one fixed target quat
def _geodesic_deg(quat_meas, quat_target):
    ang = np.zeros(len(quat_meas))
    for i in range(len(quat_meas)):
        err_q = quat_multiply(quat_conjugate(quat_meas[i]), quat_target)
        ang[i] = 2.0 * np.arccos(np.clip(abs(err_q[0]), 0.0, 1.0))
    return np.degrees(ang)


# Score the settle window [motion_end, motion_end + land_hold]. `quat_target` is
# the heading-corrected reference orientation for the motion's LAST frame -- the
# pose the robot is supposed to be holding once the motion is over. The binary
# `landed` is read from the first `verdict_s` of that window (the touchdown);
# `land_hold_time_s` spans the whole window. Returns orientation / tilt / rate
# statistics plus the verdict.
def landing_metrics(t, quat, dq, motion_end, land_hold, quat_target=None,
                    verdict_s=LAND_VERDICT_S, ori_err_max=LAND_ORI_ERR_DEG,
                    gyro_max=LAND_GYRO_RMS, joint_vel_max=LAND_JOINT_VEL_RMS, gyro=None):
    out = {"landed": False, "land_reason": None, "land_n_samples": 0}
    if land_hold <= 0.0:
        out["land_reason"] = "no settle window recorded (land_hold = 0)"
        return out

    win = (t >= motion_end) & (t < motion_end + land_hold)
    n = int(np.sum(win))
    out["land_n_samples"] = n
    if n < 2:
        out["land_reason"] = "no samples in the settle window"
        return out
    if quat_target is None:
        out["land_reason"] = "no reference orientation to compare the final pose against"
        return out

    # diagnostics: absolute tilt from vertical (NOT the verdict -- references
    # that end pitched forward make this large even for a clean landing)
    tilt = tilt_deg_from_quat(quat[win])
    out["land_tilt_max_deg"] = float(np.max(tilt))
    out["land_tilt_mean_deg"] = float(np.mean(tilt))

    ori_err_all = _geodesic_deg(quat[win], quat_target)
    t_win = t[win] - motion_end

    # VERDICT window: the touchdown itself, [motion_end, motion_end + verdict_s],
    # read from its tail so the impact transient is not what decides it.
    v = t_win < max(verdict_s, 0.0)
    n_v = int(np.sum(v))
    if n_v < 2:  # verdict_s shorter than the sample period; fall back to the window
        v = np.ones_like(t_win, dtype=bool)
        n_v = len(t_win)
    n_tail = max(2, int(round(n_v * LAND_FINAL_FRAC)))
    tail = np.flatnonzero(v)[-n_tail:]

    out["land_tilt_final_deg"] = float(np.mean(tilt[tail]))
    ori_err_final = float(np.mean(ori_err_all[tail]))
    out["land_ori_err_deg"] = ori_err_final
    out["land_verdict_s"] = float(verdict_s)

    # How long it held the final pose before diverging, measured from the end of
    # the motion. The binary verdict below depends on where we put the horizon;
    # this does not. Nothing here catches the robot once the motion is over (the
    # control node just freezes on the last frame), so a policy that holds for
    # 2.5s and one that folds at 0.2s are genuinely different results even though
    # both may end up on the floor. Right-CENSORED at land_hold: `land_censored`
    # true means it was still holding when we stopped watching, so the true hold
    # time is >= the reported one -- do not average censored and uncensored
    # values naively.
    diverged = ori_err_all > ori_err_max
    # require 3 consecutive samples so a single noisy frame is not a "fall"
    run3 = diverged[:-2] & diverged[1:-1] & diverged[2:] if len(diverged) >= 3 else np.array([], bool)
    idx = np.flatnonzero(run3)
    if idx.size:
        out["land_hold_time_s"] = float(t_win[idx[0]])
        out["land_censored"] = False
    else:
        out["land_hold_time_s"] = float(land_hold)
        out["land_censored"] = True

    # rate checks, also read from the verdict tail: they separate "arrived and
    # is holding" from "tumbling through roughly the right orientation".
    vel_rms = float(np.sqrt(np.mean(dq[win][tail] ** 2))) if dq is not None else 0.0
    out["land_joint_vel_rms"] = vel_rms

    gyro_rms = float("nan")
    if gyro is not None:
        gyro_rms = float(np.sqrt(np.mean(gyro[win][tail] ** 2)))
    out["land_gyro_rms"] = gyro_rms

    on_pose = ori_err_final <= ori_err_max
    settled_body = (not np.isfinite(gyro_rms)) or (gyro_rms <= gyro_max)
    settled_joints = vel_rms <= joint_vel_max

    out["landed"] = bool(on_pose and settled_body and settled_joints)
    if not out["landed"]:
        why = []
        if not on_pose:
            why.append(f"ori err {ori_err_final:.0f}deg > {ori_err_max:.0f}")
        if not settled_body:
            why.append(f"gyro rms {gyro_rms:.1f} > {gyro_max:.1f}")
        if not settled_joints:
            why.append(f"joint vel rms {vel_rms:.1f} > {joint_vel_max:.1f}")
        out["land_reason"] = "; ".join(why)
    return out


# read the config the run used (motion_path / xml_path / policy_path / control_dt)
def load_config(config_name):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    with open(os.path.join(ROOT_DIR, "deploy", "configs", config_name), "r") as f:
        return yaml.safe_load(f)


# analyze one run. Returns a metrics dict (pos/vel/ori RMSE, coverage, n) plus,
# when the run recorded a post-motion settle window, the landing verdict.
# `lead_in` (s) drops the start of the track window, where the robot is still
# converging onto the trajectory rather than following it. Entering 'track' from
# the home pose leaves a real initial mismatch (joint error ~0.26 rad, base ~6cm
# low against frame 0) that decays over ~0.3-0.5s; scoring it reports
# initialization, not tracking. Frame alignment stays keyed to track_start, so
# only the scored SAMPLES move -- the reference indexing is untouched.
#
# Deliberately a FIXED offset applied identically to every rep, not a per-rep
# convergence detector: auto-detecting "when it caught up" would excuse a
# slow-converging policy from its own worst samples and quietly flatter it.
def analyze_log(log_path, config_name, track_start_time, motion_duration, anchor_arg="auto",
                land_hold=0.0, land_thresholds=None, lead_in=0.0):
    # Prefer the config SNAPSHOT embedded in the log over the named config file.
    # deploy/configs/<sweep>.yaml is rewritten by run_policy for every policy, so
    # after the fact it names whatever ran last -- reading it would score a log
    # against another policy's reference motion. The log is self-describing.
    data, meta = load_log(log_path)
    try:
        cfg = config_from_meta(meta)
    except Exception:
        cfg = load_config(config_name)
    ref = load_reference(cfg)
    control_dt = float(cfg["control_dt"])
    num_frames = ref["num_frames"]
    anchor, imu_key, anchor_idx, anchor_name = resolve_anchor(cfg, ref, anchor_arg)

    if "joint_state" not in data or "time" not in data:
        raise ValueError(f"{log_path}: no joint_state/time data to compare.")

    # datasets are logged in lockstep from a common start, but can end a few rows
    # apart (teardown races). Front-align by truncating everything to the shortest.
    keys = [k for k in ("time", "joint_state", imu_key, "base_state") if k in data]
    min_len = min(data[k].shape[0] for k in keys)
    data = {k: v[:min_len] for k, v in data.items()}

    t = data["time"][:, 0].astype(np.float64)

    # frame index within the single track playback (control node: 1 frame / control_dt)
    frame = np.round((t - track_start_time) / control_dt).astype(np.int64)
    end = track_start_time + motion_duration
    scored_start = track_start_time + max(0.0, lead_in)
    in_window = (t >= scored_start) & (t < end) & (frame >= 0) & (frame <= num_frames - 1)
    if not np.any(in_window):
        raise ValueError(f"{log_path}: no samples fall inside the scored window "
                         f"[{scored_start:.2f}, {end:.2f}] (lead_in={lead_in:.2f}s).")
    frame = np.clip(frame[in_window], 0, num_frames - 1)

    # measured joints: joint_state = [q, dq, ddq, tau_est]
    N = data["joint_state"].shape[1] // 4
    if N != ref["joint_pos"].shape[1]:
        raise ValueError(f"{log_path}: joint count mismatch (log {N}, reference {ref['joint_pos'].shape[1]}).")
    q_meas = data["joint_state"][in_window, 0:N].astype(np.float64)
    dq_meas = data["joint_state"][in_window, N:2 * N].astype(np.float64)

    q_ref = ref["joint_pos"][frame]
    dq_ref = ref["joint_vel"][frame]

    pos = joint_error_stats(q_meas - q_ref)
    vel = joint_error_stats(dq_meas - dq_ref)

    # orientation error against the heading-corrected reference (optional)
    ori = None
    land_quat_target = None
    if imu_key in data:
        quat_meas = data[imu_key][in_window, 3:7].astype(np.float64)
        ref_quat_seq = ref["body_quat_w"][frame, anchor_idx]
        angles, _ = orientation_error(quat_meas, ref_quat_seq)
        ori = joint_error_stats(np.degrees(angles)[:, None])

        # the orientation the robot should be HOLDING once the motion is over:
        # the reference's last frame, yaw-aligned to this run the same way
        # orientation_error aligns it (heading captured at the first sample).
        init_quat = heading_about_z_world(quat_multiply(quat_meas[0], quat_conjugate(ref_quat_seq[0])))
        land_quat_target = quat_multiply(init_quat, ref["body_quat_w"][num_frames - 1, anchor_idx])

    # ground-truth base error (sim-only; absent from logs predating base_state)
    base = {}
    if ("base_state" in data and data["base_state"].shape[1] >= 13
            and augment_reference_for_base(ref, cfg)):
        base = base_error_metrics(data["base_state"][in_window], ref, frame, anchor_idx=0)

    # landing verdict over the settle window that follows the motion
    land = {}
    if land_hold and land_hold > 0.0:
        dq_all = data["joint_state"][:, N:2 * N].astype(np.float64)
        if imu_key in data:
            quat_all = data[imu_key][:, 3:7].astype(np.float64)
            gyro_all = data[imu_key][:, 7:10].astype(np.float64)
        else:
            quat_all, gyro_all = None, None
        if quat_all is None:
            land = {"landed": False, "land_reason": f"no {imu_key} data to judge landing",
                    "land_n_samples": 0}
        else:
            land = landing_metrics(t, quat_all, dq_all, track_start_time + motion_duration,
                                   land_hold, quat_target=land_quat_target, gyro=gyro_all,
                                   **(land_thresholds or {}))

    out = {
        "pos_rmse": pos["rmse"], "pos_mae": pos["mae"], "pos_max": pos["max"],
        "vel_rmse": vel["rmse"], "vel_mae": vel["mae"], "vel_max": vel["max"],
        "ori_rmse_deg": (ori["rmse"] if ori is not None else None),
        "ori_max_deg": (ori["max"] if ori is not None else None),
        "n_samples": int(np.sum(in_window)),
        "lead_in_s": float(max(0.0, lead_in)),
        # coverage is of the SCORED frames, so a lead-in lowers it by design
        "coverage": float(np.unique(frame).size / num_frames),
        "anchor": anchor_name,
    }
    out.update(base)
    out.update(land)
    return out

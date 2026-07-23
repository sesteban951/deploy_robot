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
)


# read the config the run used (motion_path / xml_path / policy_path / control_dt)
def load_config(config_name):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    with open(os.path.join(ROOT_DIR, "deploy", "configs", config_name), "r") as f:
        return yaml.safe_load(f)


# analyze one run. Returns a metrics dict (pos/vel/ori RMSE, coverage, n).
def analyze_log(log_path, config_name, track_start_time, motion_duration, anchor_arg="auto"):
    cfg = load_config(config_name)
    ref = load_reference(cfg)
    control_dt = float(cfg["control_dt"])
    num_frames = ref["num_frames"]
    anchor, imu_key, anchor_idx, anchor_name = resolve_anchor(cfg, ref, anchor_arg)

    data, _meta = load_log(log_path)
    if "joint_state" not in data or "time" not in data:
        raise ValueError(f"{log_path}: no joint_state/time data to compare.")

    # datasets are logged in lockstep from a common start, but can end a few rows
    # apart (teardown races). Front-align by truncating everything to the shortest.
    keys = [k for k in ("time", "joint_state", imu_key) if k in data]
    min_len = min(data[k].shape[0] for k in keys)
    data = {k: v[:min_len] for k, v in data.items()}

    t = data["time"][:, 0].astype(np.float64)

    # frame index within the single track playback (control node: 1 frame / control_dt)
    frame = np.round((t - track_start_time) / control_dt).astype(np.int64)
    end = track_start_time + motion_duration
    in_window = (t >= track_start_time) & (t < end) & (frame >= 0) & (frame <= num_frames - 1)
    if not np.any(in_window):
        raise ValueError(f"{log_path}: no samples fall inside the track window "
                         f"[{track_start_time:.2f}, {end:.2f}].")
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
    if imu_key in data:
        quat_meas = data[imu_key][in_window, 3:7].astype(np.float64)
        ref_quat_seq = ref["body_quat_w"][frame, anchor_idx]
        angles, _ = orientation_error(quat_meas, ref_quat_seq)
        ori = joint_error_stats(np.degrees(angles)[:, None])

    return {
        "pos_rmse": pos["rmse"], "pos_mae": pos["mae"], "pos_max": pos["max"],
        "vel_rmse": vel["rmse"], "vel_mae": vel["mae"], "vel_max": vel["max"],
        "ori_rmse_deg": (ori["rmse"] if ori is not None else None),
        "ori_max_deg": (ori["max"] if ori is not None else None),
        "n_samples": int(np.sum(in_window)),
        "coverage": float(np.unique(frame).size / num_frames),
        "anchor": anchor_name,
    }

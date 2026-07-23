##
#
# Single-command debug tool for diagnosing mimic-policy (e.g. 180-twist) blow-ups.
#
# Produces several windows (and, with --save, writes PNGs to debug/plots/):
#   1. IMU + FK cross-check : pelvis/torso rpy & quat, plus torso orientation
#      predicted from pelvis quat + waist encoders (FK) and the measured-vs-
#      predicted divergence (constant offset = mounting, growing = IMU drift).
#   2. IMU-dependent obs blocks : base_ang_vel (== pelvis gyro) and
#      motion_anchor_ori_b (6D torso-vs-reference orientation error) as the
#      policy actually saw them.
#   3. Policy replay : the ONNX policy re-run open-loop on the reconstructed
#      observations, overlaid against the logged command, to reproduce the blow-up.
#
# Usage:
#   python debug/debug.py                                    # latest log; policy/motion from log's embedded config
#   python debug/debug.py --filename logs/hardware/foo.h5
#   python debug/debug.py --filename <log.h5> --policy kd_to_180_twist_jump.onnx --motion kd_to_twist_180_jump_29dof.npz
#   python debug/debug.py --filename <log.h5> --t_start 12.3 # override auto-detected track start
#   python debug/debug.py --save                             # also write PNGs to debug/plots/
#
# Observation diagnostics (all optional, additive to the windows above):
#   --all-obs                                                # every obs channel, grouped by block
#   --compare <good.h5> --obs joint_vel[:12]                 # one obs, this log vs another (aligned at t_start)
#   --ablate joint_vel --ablate-method smooth --smooth-window 5   # re-run policy with the channel cleaned
#   --ablate joint_vel --ablate-method swap --compare <good.h5>   # ...or swapped in from another log
#   --sensitivity [--compare <good.h5>]                      # rank obs channels by influence on the output
# Obs selector: a block name (motion_joint_pos, motion_joint_vel, motion_anchor_ori_b,
# base_ang_vel, joint_pos, joint_vel, prev_action) or block:idx for a single channel.
#
# --policy / --motion are what vary per run and take precedence over the log's
# embedded config; give them explicitly rather than editing a config file. They
# are required only for older logs that don't embed a config.
#
##

import argparse
import glob
import math
import os
import sys

import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt

# resolve repo root (env or inferred from this file) and make utils/ importable
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from fk_crosscheck import TorsoFK, quat_divergence
from obs_replay import MimicReplay, OBS_BLOCKS
from utils.math_utils import quat_to_rpy

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
FK_MAX_POINTS = 6000  # cap FK mj_forward calls (drift is slow, no need for full-res)
SAVE_PNGS = False  # set from --save; PNGs are only written to plots/ when enabled


############################################################################
# LOADING
############################################################################

def find_latest_log() -> str:
    logs_root = os.path.join(ROOT_DIR, "logs")
    candidates = (glob.glob(os.path.join(logs_root, "simulation", "*.h5"))
                  + glob.glob(os.path.join(logs_root, "hardware", "*.h5")))
    if not candidates:
        raise FileNotFoundError(f"No .h5 logs found under {logs_root}/simulation or /hardware.")
    return max(candidates, key=os.path.getmtime)


def load_log(file_path: str):
    print(f"Loading {file_path}")
    with h5py.File(file_path, "r") as f:
        data = {name: f[name][:] for name in f.keys()}
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}

    # drop stale leading rows where time resets (matches logs/plot.py)
    t_raw = data["time"][:, 0]
    resets = np.where(np.diff(t_raw) < 0)[0]
    start = int(resets[0]) + 1 if len(resets) else 0
    if start > 0:
        print(f"  Dropping {start} stale leading row(s) (time resets at row {start}).")
        data = {name: arr[start:] for name, arr in data.items()}

    t = data["time"][:, 0].astype(np.float64)
    t = t - t[0]
    return data, attrs, t


# Resolve the run parameters. The policy and motion are what actually vary
# between runs, so they can be given explicitly with --policy / --motion (which
# is more reliable than tracking an edited config). Precedence for each:
#   explicit flag  >  config embedded in the log  >  --config file  >  default.
# xml_path and control_dt are effectively constant, so they just default.
def resolve_run(attrs: dict, args) -> dict:
    cfg, src = {}, "(none)"
    if "config_yaml" in attrs:
        raw = attrs["config_yaml"]
        cfg = yaml.safe_load(raw.decode() if isinstance(raw, bytes) else str(raw))
        src = "log attrs (config_yaml)"
    elif args.config:
        name = args.config if args.config.endswith(".yaml") else args.config + ".yaml"
        path = os.path.join(ROOT_DIR, "deploy", "configs", name)
        if os.path.exists(path):
            cfg = yaml.safe_load(open(path))
            src = path

    policy = args.policy or cfg.get("policy_path")
    motion = args.motion or cfg.get("motion_path")
    xml = args.xml or cfg.get("xml_path") or "g1_29dof_scene.xml"
    control_dt = float(cfg.get("control_dt", 0.02))
    # whether the run rotated the motion into the robot's heading (init_quat). Match
    # the run so the reconstructed obs is faithful; --align-heading forces it.
    align_heading = bool(cfg.get("align_heading", True)) if args.align_heading is None else args.align_heading
    # alternative: whether the run zeroed the anchor IMU's yaw (applied to the measured
    # anchor quat). Match the run; --zero-imu-yaw forces it.
    zero_imu_yaw = bool(cfg.get("zero_imu_yaw", False)) if args.zero_imu_yaw is None else args.zero_imu_yaw

    if not policy:
        raise SystemExit("No policy resolved. Pass --policy <name.onnx> (log has no embedded config).")
    if not motion:
        raise SystemExit("No motion resolved. Pass --motion <name.npz> (log has no embedded config).")

    print(f"  Run source: config from {src}")
    print(f"    policy = {policy}{'   (--policy)' if args.policy else ''}")
    print(f"    motion = {motion}{'   (--motion)' if args.motion else ''}")
    print(f"    align_heading = {align_heading}"
          f"{'   (--align-heading)' if args.align_heading is not None else ''}")
    print(f"    zero_imu_yaw  = {zero_imu_yaw}"
          f"{'   (--zero-imu-yaw)' if args.zero_imu_yaw is not None else ''}")
    if align_heading and zero_imu_yaw:
        print("    WARNING: align_heading and zero_imu_yaw both on -- yaw double-corrected "
              "(they are alternatives).")
    return {"policy_path": policy, "motion_path": motion, "xml_path": xml,
            "control_dt": control_dt, "align_heading": align_heading, "zero_imu_yaw": zero_imu_yaw}


# PHASE codes the control node publishes on deploy_robot/control_phase and the logger
# records as the per-row "control_phase" dataset: -1 idle / 0 interp (ramp) / 1 frame-0
# hold / 2 track. When present these are the ground-truth FSM boundaries and replace the
# detect_t_start command-departure heuristic (which exists only for older logs).
PHASE_INTERP, PHASE_HOLD, PHASE_TRACK = 0, 1, 2


def phase_boundaries(data, t):
    """From the logged control_phase dataset return (t_start, handoff_row, info):
    t_start = first 'track' row; handoff_row = first 'frame-0 hold' row (the control->
    policy handoff where init_quat / yaw offset are captured). Returns None when the
    dataset is absent or never reaches 'track' (caller falls back to the heuristic)."""
    if "control_phase" not in data:
        return None
    phase = np.rint(data["control_phase"][:, 0]).astype(int)
    track = np.where(phase == PHASE_TRACK)[0]
    if len(track) == 0:
        return None
    hold = np.where(phase == PHASE_HOLD)[0]
    handoff_row = int(hold[0]) if len(hold) else int(track[0])
    t_start = float(t[track[0]])
    info = (f"hold@row {handoff_row} (t={t[handoff_row]:.2f}s), "
            f"track@row {int(track[0])} (t={t_start:.2f}s)")
    return t_start, handoff_row, info


############################################################################
# PLOT HELPERS
############################################################################

def _joint_grid(N: int, title: str, figsize=(15, 9)):
    cols = min(6, N)
    rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    fig.suptitle(title)
    axes = axes.flatten() if N > 1 else [axes]
    for ax in axes[N:]:
        ax.set_visible(False)
    return fig, axes[:N]


def _savefig(fig, log_path: str, tag: str):
    if not SAVE_PNGS:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(log_path))[0]
    out = os.path.join(PLOTS_DIR, f"{base}_{tag}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  Saved {out}")


# draw the track-start marker on every axis of a figure
def _mark_tstart(axes, t_start: float):
    for ax in np.atleast_1d(np.asarray(axes)).flatten():
        if ax.get_visible():
            ax.axvline(t_start, color="black", linestyle="--", linewidth=1.0, zorder=0)


# parse an obs selector: "joint_vel" -> whole block; "joint_vel:12" -> channel 12 of it.
# Returns (name, start, stop, ch) where start/stop are absolute obs indices (stop
# exclusive) and ch is None (whole block) or the in-block channel index.
def parse_obs_selector(sel: str):
    blocks = {n: (s, e) for n, s, e in OBS_BLOCKS}
    name, _, idx = sel.partition(":")
    if name not in blocks:
        raise SystemExit(f"Unknown obs block '{name}'. Choices: {list(blocks)}")
    s, e = blocks[name]
    if idx == "":
        return name, s, e, None
    ch = int(idx)
    if not (0 <= ch < e - s):
        raise SystemExit(f"Channel {ch} out of range for block '{name}' (0..{e - s - 1}).")
    return name, s, e, ch


# per-column smoothing (numpy only). kind="mean" = edge-padded moving average;
# kind="median" = sliding median (better for spikes). window in samples/steps.
def _smooth(arr, window: int, kind: str = "mean"):
    a = np.asarray(arr, dtype=np.float64)
    if window <= 1:
        return a.copy()
    a2 = a if a.ndim == 2 else a[:, None]
    T = a2.shape[0]
    half = window // 2
    out = np.empty_like(a2)
    if kind == "median":
        for i in range(T):
            lo, hi = max(0, i - half), min(T, i + half + 1)
            out[i] = np.median(a2[lo:hi], axis=0)
    else:
        kern = np.ones(window) / window
        for j in range(a2.shape[1]):
            padded = np.pad(a2[:, j], (half, window - 1 - half), mode="edge")
            out[:, j] = np.convolve(padded, kern, mode="valid")
    return out if a.ndim == 2 else out[:, 0]


# step indices whose time falls in [0, sens_window] s after t_start (the oscillation
# onset). Falls back to the first ~25 steps if the window is empty.
def _onset_steps(R, sens_window: float):
    ts = R["t"][R["rows"]] - R["t_start"]
    idx = np.where((ts >= 0) & (ts <= sens_window))[0]
    if len(idx) == 0:
        idx = np.arange(min(len(R["frames"]), 25))
    return idx


############################################################################
# WINDOWS
############################################################################

# window 1: raw IMUs + FK cross-check
def window_imu_crosscheck(t, pelvis, torso, q_log, fk: TorsoFK, log_path: str, t_start: float):
    # FK-predicted torso orientation (strided for speed) vs measured
    T = t.shape[0]
    stride = max(1, T // FK_MAX_POINTS)
    idx = np.arange(0, T, stride)
    print(f"  Running FK cross-check on {len(idx)} rows (stride {stride})...")
    pelvis_quat = pelvis[:, 3:7].astype(np.float64)
    torso_quat = torso[:, 3:7].astype(np.float64)
    pred_torso = fk.predict(pelvis_quat[idx], q_log[idx])
    angle, div_rpy, div_quat = quat_divergence(torso_quat[idx], pred_torso)
    meas_rpy = np.array([quat_to_rpy(q) for q in torso_quat[idx]])
    pred_rpy = np.array([quat_to_rpy(q) for q in pred_torso])

    qlabels = ["qw", "qx", "qy", "qz"]
    rpylabels = ["roll", "pitch", "yaw"]

    fig, ax = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    fig.suptitle("IMU data + FK cross-check (torso predicted from pelvis + waist encoders)")

    # row 0: raw rpy
    ax[0, 0].plot(t, pelvis[:, 0:3], label=rpylabels)
    ax[0, 0].set_title("pelvis IMU rpy"); ax[0, 0].set_ylabel("[rad]"); ax[0, 0].legend(loc="upper right")
    ax[0, 1].plot(t, torso[:, 0:3], label=rpylabels)
    ax[0, 1].set_title("torso IMU rpy"); ax[0, 1].legend(loc="upper right")

    # row 1: raw quat
    ax[1, 0].plot(t, pelvis[:, 3:7], label=qlabels)
    ax[1, 0].set_title("pelvis IMU quat"); ax[1, 0].set_ylabel("quat"); ax[1, 0].legend(loc="upper right")
    ax[1, 1].plot(t, torso[:, 3:7], label=qlabels)
    ax[1, 1].set_title("torso IMU quat"); ax[1, 1].legend(loc="upper right")

    # row 2: torso measured (solid) vs FK-predicted (dashed) -- quat and rpy
    for c in range(4):
        line, = ax[2, 0].plot(t[idx], torso_quat[idx, c], linewidth=1.2, label=qlabels[c])
        ax[2, 0].plot(t[idx], pred_torso[:, c], "--", color=line.get_color(), linewidth=1.0)
    ax[2, 0].set_title("torso quat: measured (solid) vs FK-predicted (dashed)")
    ax[2, 0].set_ylabel("quat"); ax[2, 0].legend(loc="upper right", ncol=4)
    for c in range(3):
        line, = ax[2, 1].plot(t[idx], meas_rpy[:, c], linewidth=1.2, label=rpylabels[c])
        ax[2, 1].plot(t[idx], pred_rpy[:, c], "--", color=line.get_color(), linewidth=1.0)
    ax[2, 1].set_title("torso rpy: measured (solid) vs FK-predicted (dashed)")
    ax[2, 1].legend(loc="upper right", ncol=3)

    # row 3: divergence (measured relative to predicted) -- quat and rpy
    ax[3, 0].plot(t[idx], div_quat, label=["d_qw", "d_qx", "d_qy", "d_qz"])
    ax[3, 0].set_title("measured - predicted divergence (quat)")
    ax[3, 0].set_ylabel("quat"); ax[3, 0].set_xlabel("time [s]"); ax[3, 0].legend(loc="upper right", ncol=4)
    ax[3, 1].plot(t[idx], div_rpy, label=["d_roll", "d_pitch", "d_yaw"])
    ax[3, 1].set_title("measured - predicted divergence (rpy)")
    ax[3, 1].set_xlabel("time [s]"); ax[3, 1].legend(loc="upper right")

    for a in ax.flatten():
        a.grid(True)
    _mark_tstart(ax, t_start)
    _savefig(fig, log_path, "imu_crosscheck")

    # divergence angle in its own window
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(t[idx], angle, color="tab:red")
    ax2.set_title("torso divergence angle (measured vs FK-predicted)")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("[rad]"); ax2.grid(True)
    _mark_tstart(ax2, t_start)
    _savefig(fig2, log_path, "divergence_angle")


# window 2: the IMU-dependent observation blocks, over the FULL log.
# base_ang_vel is just the pelvis gyro; motion_anchor_ori_b is the torso orientation
# ERROR vs the reference (held at frame 0 before t_start, advancing after), shown in
# three equivalent forms: 6D rotation (as the policy sees it), quaternion, and rpy.
def window_obs_blocks(rep: MimicReplay, t, pelvis_gyro_full, ori6d_full,
                      orierr_quat_full, orierr_rpy_full, t_start, log_path: str):
    fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    fig.suptitle(f"IMU-dependent obs blocks (anchor body = {rep.anchor_name}, "
                 f"orientation from {rep.anchor_imu} IMU)")

    ax[0].plot(t, pelvis_gyro_full, label=["wx", "wy", "wz"])
    ax[0].set_title("base_ang_vel == pelvis gyro (same info as the direct gyro plot)")
    ax[0].set_ylabel("[rad/s]"); ax[0].legend(loc="upper right"); ax[0].grid(True)

    ax[1].plot(t, ori6d_full, label=["r11", "r12", "r21", "r22", "r31", "r32"])
    ax[1].set_title("motion_anchor_ori_b (6D) - torso orientation ERROR vs reference "
                    "(frame 0 before t_start, advancing after; this is what the policy sees)")
    ax[1].set_ylabel("6D rot"); ax[1].legend(loc="upper right", ncol=6); ax[1].grid(True)

    ax[2].plot(t, orierr_quat_full, label=["qw", "qx", "qy", "qz"])
    ax[2].set_title("same orientation error as a quaternion")
    ax[2].set_ylabel("quat"); ax[2].legend(loc="upper right", ncol=4); ax[2].grid(True)

    ax[3].plot(t, orierr_rpy_full, label=["roll", "pitch", "yaw"])
    ax[3].set_title("same orientation error as rpy")
    ax[3].set_ylabel("[rad]"); ax[3].set_xlabel("time [s]")
    ax[3].legend(loc="upper right"); ax[3].grid(True)

    _mark_tstart(ax, t_start)
    _savefig(fig, log_path, "obs_imu_blocks")


# window 3: policy replay vs logged command.
# logged q_des is drawn over the FULL log; the policy prediction is overlaid only
# on the replay window (after the detected t_start).
def window_policy_replay(rep: MimicReplay, result, t, qdes_logged_full, log_path: str, t_start: float):
    ts = t[result["rows"]]
    pred = result["pred_qpos_des"]
    logged = result["logged_qpos_des"]
    N = pred.shape[1]
    fig, axes = _joint_grid(N, "qpos_des: logged over full log (solid) vs policy replay "
                               "after t_start (dashed, frame timeline approximate)")
    for i, ax in enumerate(axes):
        ax.plot(t, qdes_logged_full[:, i], color="tab:blue", linewidth=1.0, label="logged", zorder=2)
        ax.plot(ts, pred[:, i], "--", color="tab:red", linewidth=1.0, label="policy", zorder=3)
        ax.set_title(f"joint {i}"); ax.set_ylabel("[rad]"); ax.grid(True)
    axes[0].legend(loc="upper right")
    _mark_tstart(axes, t_start)

    # divergence summary (over the replay window) as a separate window
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    err = np.linalg.norm(pred - logged, axis=1)
    ax2.plot(ts, err, color="tab:red")
    ax2.set_title("||policy qpos_des - logged qpos_des|| over replay window "
                  "(match near start, divergence = blow-up)")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("[rad]"); ax2.grid(True)
    _mark_tstart(ax2, t_start)
    _savefig(fig, log_path, "policy_replay")
    _savefig(fig2, log_path, "policy_replay_error")


# window 4: commanded joint torque tau = Kp*(q_des - q) + Kd*(dq_des - dq) + tau_ff,
# i.e. what the low-level PD would drive the motors with. Logged (from the logged
# command) over the full log; policy-predicted (from the replayed q_des, with
# dq_des = tau_ff = 0 as the mimic controller sends) overlaid after t_start.
def window_commanded_torque(result, t, tau_logged_full, tau_pred_steps, log_path: str, t_start: float):
    ts = t[result["rows"]]
    N = tau_logged_full.shape[1]
    fig, axes = _joint_grid(N, "commanded torque Kp*(q_des-q)+Kd*(dq_des-dq)+tau_ff: "
                               "logged over full log (solid) vs policy replay after t_start (dashed) "
                               "[pre-saturation; motors clamp to their force limits]")
    for i, ax in enumerate(axes):
        ax.plot(t, tau_logged_full[:, i], color="tab:blue", linewidth=1.0, label="logged", zorder=2)
        ax.plot(ts, tau_pred_steps[:, i], "--", color="tab:red", linewidth=1.0, label="policy", zorder=3)
        ax.set_title(f"joint {i}"); ax.set_ylabel("[Nm]"); ax.grid(True)
    axes[0].legend(loc="upper right")
    _mark_tstart(axes, t_start)
    _savefig(fig, log_path, "commanded_torque")


# all observation channels over the full log, one subplot per block. A fast scan for
# "which block is already oscillating at onset". 29-wide blocks are drawn thin/faint.
def window_all_obs(R, log_path: str):
    t, obs, t_start = R["t"], R["obs_full"], R["t_start"]
    fig, ax = plt.subplots(len(OBS_BLOCKS), 1, figsize=(13, 16), sharex=True)
    fig.suptitle("all observation channels (full log; reference held at frame 0 before t_start, "
                 "advancing after)")
    for k, (name, s, e) in enumerate(OBS_BLOCKS):
        wide = (e - s) > 6
        ax[k].plot(t, obs[:, s:e], linewidth=(0.7 if wide else 1.2), alpha=(0.5 if wide else 1.0))
        ax[k].set_title(f"{name} ({e - s})"); ax[k].set_ylabel(name.split("_")[-1]); ax[k].grid(True)
    ax[-1].set_xlabel("time [s]")
    # default view = the analysis window (the long pre-track hold is elided; pan to see it)
    ax[0].set_xlim(t_start - 2.0, min(R["t_end"], t[-1]))
    _mark_tstart(ax, t_start)
    _savefig(fig, log_path, "all_obs")


# compare one observation (block or single channel) between two logs, aligned at each
# log's own t_start (x = time since t_start): log A on top, log B on bottom.
def window_compare_obs(RA, RB, sel: str):
    name, s, e, ch = parse_obs_selector(sel)
    label = f"{name}[{ch}]" if ch is not None else name
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True)
    fig.suptitle(f"observation '{label}': log A (top) vs log B (bottom), aligned at t_start")
    for row, (R, who) in enumerate([(RA, "A"), (RB, "B")]):
        tt = R["t"] - R["t_start"]
        block = R["obs_full"][:, s:e]
        if ch is not None:
            ax[row].plot(tt, block[:, ch], linewidth=1.2)
        else:
            ax[row].plot(tt, block, linewidth=0.8, alpha=0.6)
        ax[row].set_title(f"[{who}] {os.path.basename(R['log_path'])}")
        ax[row].set_ylabel(label); ax[row].grid(True)
        ax[row].axvline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=0)
    span = min(RA["t_end"] - RA["t_start"], RB["t_end"] - RB["t_start"])
    ax[0].set_xlim(-1.0, span)
    ax[-1].set_xlabel("time since t_start [s]")
    _savefig(fig, RA["log_path"], f"compare_{name}")


# ablation replay: on the loaded (bad) log, replace one obs block/channel with either a
# smoothed version of itself or the good log's values (frame-aligned), re-run the policy
# offline, and compare the predicted qpos_des against the un-ablated baseline. A printed
# oscillation metric quantifies whether cleaning that channel removes the chatter.
def window_ablation(R, RB, args):
    name, s, e, ch = parse_obs_selector(args.ablate)
    cols = np.array([s + ch]) if ch is not None else np.arange(s, e)
    rep, frames = R["rep"], np.asarray(R["frames"])
    baseline = R["result"]["obs"]
    modified = baseline.copy()

    if args.ablate_method == "smooth":
        modified[:, cols] = _smooth(baseline[:, cols], args.smooth_window, args.smooth_kind)
        desc = f"smooth(window={args.smooth_window}, {args.smooth_kind})"
    else:  # swap
        if RB is None:
            raise SystemExit("--ablate-method swap requires --compare <good log>.")
        good_obs, good_frames = RB["result"]["obs"], np.asarray(RB["frames"])
        f2g = {}
        for gi, gf in enumerate(good_frames):
            f2g.setdefault(int(gf), gi)  # first good step at each motion frame
        n = 0
        for k, f in enumerate(frames):
            gi = f2g.get(int(f))
            if gi is not None:
                modified[k, cols] = good_obs[gi, cols]; n += 1
        desc = f"swap from {os.path.basename(RB['log_path'])} ({n}/{len(frames)} frames matched)"

    base_pred = R["result"]["pred_qpos_des"]
    abl_pred = rep.replay_from_obs(modified, frames)["pred_qpos_des"]
    osc = lambda p: float(np.mean(np.std(np.diff(p, axis=0), axis=0)))
    tag = name if ch is None else f"{name}[{ch}]"
    print(f"  Ablation [{tag}] via {desc}:")
    print(f"    oscillation metric (mean_j std_t d/dt qpos_des): "
          f"baseline {osc(base_pred):.4f} -> ablated {osc(abl_pred):.4f} rad")

    ts = R["t"][R["rows"]]
    N = base_pred.shape[1]
    fig, axes = _joint_grid(N, f"ablation of {tag} ({desc}): baseline replay (blue) vs ablated (red)")
    for i, ax in enumerate(axes):
        ax.plot(ts, base_pred[:, i], color="tab:blue", linewidth=1.0, label="baseline", zorder=2)
        ax.plot(ts, abl_pred[:, i], "--", color="tab:red", linewidth=1.0, label="ablated", zorder=3)
        ax.set_title(f"joint {i}"); ax.set_ylabel("[rad]"); ax.grid(True)
    axes[0].legend(loc="upper right")
    _mark_tstart(axes, R["t_start"])
    _savefig(fig, R["log_path"], f"ablation_{name}")


# sensitivity ranking: finite-difference the policy at each obs over a short onset window
# and rank channels by contribution (gain x how much the channel actually moved). Block
# bar chart (A vs B if a compare log is given) + per-channel heatmap for A + printed top-N.
def window_sensitivity(R, RB, args):
    block_names = [n for n, _, _ in OBS_BLOCKS]
    runs = [("A", R)] + ([("B", RB)] if RB is not None else [])
    results = {}
    for who, RR in runs:
        idx = _onset_steps(RR, args.sens_window)
        obs = RR["result"]["obs"][idx]
        frames = np.asarray(RR["frames"])[idx]
        print(f"  [{who}] sensitivity over {len(idx)} steps "
              f"(~{2 * obs.shape[1] * len(idx)} inferences)...")
        gain, contrib = RR["rep"].sensitivity(obs, frames, eps=args.sens_eps)
        results[who] = {"R": RR, "gain": gain, "contrib": contrib}

    # per-block contribution bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(block_names))
    keys = list(results)
    w = 0.8 / len(keys)
    for bi, who in enumerate(keys):
        blk = results[who]["R"]["rep"].block_reduce(results[who]["contrib"].mean(axis=0))
        ax.bar(x + bi * w, blk, width=w,
               label=f"[{who}] {os.path.basename(results[who]['R']['log_path'])}")
    ax.set_xticks(x + w * (len(keys) - 1) / 2)
    ax.set_xticklabels(block_names, rotation=20, ha="right")
    ax.set_ylabel("contribution (gain x std)"); ax.grid(True, axis="y")
    ax.set_title(f"per-block obs contribution to policy output at onset (first {args.sens_window}s)")
    ax.legend(loc="upper right")
    _savefig(fig, R["log_path"], "sensitivity_blocks")

    # per-channel contribution heatmap for the primary (A) log
    cA = results["A"]["contrib"]
    fig2, ax2 = plt.subplots(figsize=(13, 7))
    im = ax2.imshow(cA.T, aspect="auto", origin="lower",
                    extent=[0, cA.shape[0], 0, cA.shape[1]], cmap="viridis")
    for _, s, e in OBS_BLOCKS:
        ax2.axhline(e, color="white", linewidth=0.6)
    ax2.set_yticks([(s + e) / 2 for _, s, e in OBS_BLOCKS]); ax2.set_yticklabels(block_names)
    ax2.set_xlabel("step (from onset)"); ax2.set_title("per-channel contribution heatmap [A]")
    fig2.colorbar(im, ax=ax2)
    _savefig(fig2, R["log_path"], "sensitivity_heatmap")

    # printed top-N channels (primary log)
    meanc = cA.mean(axis=0)
    meang = results["A"]["gain"].mean(axis=0)
    print("  Top obs channels by mean contribution [A]:")
    for i in np.argsort(meanc)[::-1][:12]:
        for n, s, e in OBS_BLOCKS:
            if s <= i < e:
                print(f"    {n}[{i - s}]  contribution={meanc[i]:.4f}  gain={meang[i]:.4f}")
                break
    print("  NOTE: local linearization near the observed trajectory; "
          "ablation replay is the causal cross-check.")


############################################################################
# LOAD + RECONSTRUCT (shared by all modes)
############################################################################

# Load a log and rebuild the full replay + full-timeline obs. Shared by the single-log
# default windows, the two-log compare, ablation, and sensitivity. Mirrors the original
# main() pipeline exactly. t_start_override: use args.t_start for the primary log, None
# (auto-detect) for a compare log. tag: short "A"/"B" label for the console output.
def load_and_reconstruct(log_path: str, args, t_start_override, tag: str = ""):
    pfx = f"[{tag}] " if tag else ""
    print(f"{pfx}--- {os.path.basename(log_path)} ---")
    data, attrs, t = load_log(log_path)
    config = resolve_run(attrs, args)

    pelvis = data["pelvis_imu"]
    torso = data["torso_imu"]
    Nj = data["joint_state"].shape[1] // 4
    q_log = data["joint_state"][:, 0:Nj]
    dq_log = data["joint_state"][:, Nj:2 * Nj]
    if "command" not in data:
        raise SystemExit("Log has no 'command' dataset; cannot recover actions / replay policy.")
    command = data["command"]

    rep = MimicReplay(config, ROOT_DIR)
    print(f"    motion: {rep.num_frames} frames @ {rep.motion_fps} fps")
    print(f"    anchor body: {rep.anchor_name} -> {rep.anchor_imu} IMU")

    anchor_quat_log = (torso[:, 3:7] if rep.anchor_imu == "torso" else pelvis[:, 3:7]).astype(np.float32)
    pelvis_gyro = pelvis[:, 7:10].astype(np.float32)
    actions_log = rep.recover_actions(command)
    q_log32, dq_log32 = q_log.astype(np.float32), dq_log.astype(np.float32)

    # t_start + handoff. Precedence: explicit --t_start > logged control_phase (ground
    # truth) > command-departure heuristic (older logs with no control_phase).
    phase_info = phase_boundaries(data, t)
    if t_start_override is not None:
        t_start = t_start_override
        print(f"  Using provided --t_start = {t_start:.3f} s")
    elif phase_info is not None:
        t_start = phase_info[0]
        print(f"  t_start = {t_start:.3f} s from control_phase [{phase_info[2]}].")
    else:
        t_start, info = rep.detect_t_start(t, command[:, 0:Nj].astype(np.float32))
        if t_start is None:
            raise SystemExit(f"Could not auto-detect track start: {info}. Pass --t_start <seconds>.")
        print(f"  Auto-detected t_start = {t_start:.3f} s [{info}] (no control_phase in log).")
    t_end = args.t_end if args.t_end is not None else t_start + rep.num_frames * rep.ctrl_dt
    print(f"  Analysis window: [{t_start:.3f}, {min(t_end, t[-1]):.3f}] s")

    frames, rows = rep.build_steps(t, t_start, t_end)
    if len(frames) == 0:
        raise SystemExit("No control steps in window; check --t_start / --t_end.")

    # handoff (init_quat / yaw capture point): the frame-0-hold start from control_phase
    # when available, else the command-reaches-frame-0 heuristic.
    handoff_row = (phase_info[1] if phase_info is not None
                   else rep.find_handoff_row(command[:, 0:Nj].astype(np.float32)))
    if config["align_heading"]:
        init_quat = rep.capture_init_quat(anchor_quat_log[handoff_row])
        print(f"  init_quat captured at handoff (t={t[handoff_row]:.2f}s).")
    else:
        init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        print("  Heading alignment OFF: init_quat = identity (motion in its own world frame).")
    if config["zero_imu_yaw"]:
        yaw_off = rep.capture_yaw_offset(anchor_quat_log[handoff_row])
        anchor_for_obs = rep.apply_yaw_offset(anchor_quat_log, yaw_off)
        print(f"  zero_imu_yaw ON: anchor IMU yaw zeroed at handoff (t={t[handoff_row]:.2f}s).")
    else:
        anchor_for_obs = anchor_quat_log

    result = rep.replay(frames, rows, anchor_for_obs, pelvis_gyro,
                        q_log32, dq_log32, actions_log, init_quat)
    _err = np.linalg.norm(result["pred_qpos_des"] - result["logged_qpos_des"], axis=1)
    print(f"  Replay reconstruction error: lead-in {_err[1:15].mean():.3f} rad, "
          f"max {_err.max():.3f} rad.")

    obs_full, frames_full = rep.assemble_obs_full(t, t_start, anchor_for_obs, init_quat,
                                                  pelvis_gyro, q_log32, dq_log32, actions_log)

    return {"log_path": log_path, "data": data, "attrs": attrs, "t": t, "config": config,
            "rep": rep, "Nj": Nj, "command": command, "q_log": q_log, "dq_log": dq_log,
            "q_log32": q_log32, "dq_log32": dq_log32, "anchor_for_obs": anchor_for_obs,
            "pelvis": pelvis, "torso": torso, "pelvis_gyro": pelvis_gyro, "actions_log": actions_log,
            "t_start": t_start, "t_end": t_end, "frames": frames, "rows": rows,
            "handoff_row": handoff_row, "init_quat": init_quat, "result": result,
            "obs_full": obs_full, "frames_full": frames_full}


############################################################################
# MAIN
############################################################################

def main():
    parser = argparse.ArgumentParser(description="Mimic-policy debug tool (IMU / obs / policy replay).")
    parser.add_argument("--filename", type=str, default=None,
                        help="Path to .h5 log. Default: most recent under logs/.")
    parser.add_argument("--policy", type=str, default=None,
                        help="Policy .onnx name under policy/. Overrides the log/config value.")
    parser.add_argument("--motion", type=str, default=None,
                        help="Motion .npz name under motions/. Overrides the log/config value.")
    parser.add_argument("--xml", type=str, default=None,
                        help="Model xml name under models/. Default: g1_29dof_scene.xml.")
    parser.add_argument("--config", type=str, default="g1_29dof_mimic.yaml",
                        help="Config yaml used only for defaults (policy/motion/xml/control_dt) "
                             "when the log has no embedded config and no flags are given.")
    parser.add_argument("--t_start", type=float, default=None,
                        help="Track-start time [s] on the zeroed axis. Default: auto-detect (printed).")
    parser.add_argument("--t_end", type=float, default=None,
                        help="End of analysis window [s]. Default: t_start + one full motion playback.")
    parser.add_argument("--align-heading", dest="align_heading", action="store_true", default=None,
                        help="Force init_quat heading alignment ON (default: match the log's config).")
    parser.add_argument("--no-align-heading", dest="align_heading", action="store_false",
                        help="Force init_quat heading alignment OFF (replay in the motion's world frame).")
    parser.add_argument("--zero-imu-yaw", dest="zero_imu_yaw", action="store_true", default=None,
                        help="Force zero_imu_yaw ON (zero the measured anchor IMU yaw). Default: match config.")
    parser.add_argument("--no-zero-imu-yaw", dest="zero_imu_yaw", action="store_false",
                        help="Force zero_imu_yaw OFF.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive windows.")
    parser.add_argument("--save", action="store_true",
                        help="Save PNGs to debug/plots/ (off by default).")
    # observation diagnostics (all optional; default off => existing behavior unchanged)
    parser.add_argument("--all-obs", dest="all_obs", action="store_true",
                        help="Add an all-observation overview window (every obs channel, by block).")
    parser.add_argument("--compare", type=str, default=None,
                        help="Second .h5 log for --obs comparison and --ablate-method swap.")
    parser.add_argument("--obs", type=str, default=None,
                        help="Observation selector for --compare: block name (e.g. joint_vel) or "
                             "block:idx (e.g. joint_vel:12).")
    parser.add_argument("--ablate", type=str, default=None,
                        help="Ablation replay on this obs block/channel (same selector syntax).")
    parser.add_argument("--ablate-method", dest="ablate_method", choices=["swap", "smooth"],
                        default="smooth", help="Ablation: swap from --compare log, or smooth in place.")
    parser.add_argument("--smooth-window", dest="smooth_window", type=int, default=5,
                        help="Smoothing window in steps for --ablate-method smooth (default 5).")
    parser.add_argument("--smooth-kind", dest="smooth_kind", choices=["mean", "median"],
                        default="mean", help="Smoothing kernel for --ablate-method smooth.")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Rank obs channels by influence on the policy output at onset.")
    parser.add_argument("--sens-window", dest="sens_window", type=float, default=0.5,
                        help="Onset window [s] after t_start for sensitivity (default 0.5).")
    parser.add_argument("--sens-eps", dest="sens_eps", type=float, default=1e-2,
                        help="Finite-difference step for sensitivity (default 1e-2).")
    args = parser.parse_args()

    global SAVE_PNGS
    SAVE_PNGS = args.save

    log_path = args.filename or find_latest_log()
    if args.filename is None:
        print(f"No --filename provided, using latest: {log_path}")

    # primary log (uses --t_start if given). "A" tag only when a compare log is present.
    R = load_and_reconstruct(log_path, args, args.t_start, tag="A" if args.compare else "")
    rep, t, t_start, config = R["rep"], R["t"], R["t_start"], R["config"]
    Nj, command, q_log, dq_log = R["Nj"], R["command"], R["q_log"], R["dq_log"]
    result = R["result"]

    # commanded PD torque: tau = Kp*(q_des - q) + Kd*(dq_des - dq) + tau_ff.
    # logged command layout: [q_des, dq_des, Kp, Kd, tau_ff] (Nj each). gains are
    # read from the log so they match the run exactly.
    qdes_log_full = command[:, 0:Nj]
    dqdes_log_full = command[:, Nj:2 * Nj]
    Kp_log = command[:, 2 * Nj:3 * Nj]
    Kd_log = command[:, 3 * Nj:4 * Nj]
    tauff_log = command[:, 4 * Nj:5 * Nj]
    tau_logged_full = Kp_log * (qdes_log_full - q_log) + Kd_log * (dqdes_log_full - dq_log) + tauff_log
    # policy-predicted commanded torque at replay steps (mimic sends dq_des = tau_ff = 0)
    rows = result["rows"]
    tau_pred_steps = Kp_log[rows] * (result["pred_qpos_des"] - q_log[rows]) - Kd_log[rows] * dq_log[rows]

    # motion_anchor_ori_b over the full log (reference held at frame 0 before t_start),
    # in 6D / quat / rpy forms
    ori6d_full, orierr_quat_full, orierr_rpy_full = rep.obs_blocks_full(
        t, t_start, R["anchor_for_obs"], R["init_quat"])

    # default windows (t_start drawn as a black dashed line on every plot)
    fk = TorsoFK(os.path.join(ROOT_DIR, "models", config["xml_path"]))
    window_imu_crosscheck(t, R["pelvis"], R["torso"], q_log, fk, log_path, t_start)
    window_obs_blocks(rep, t, R["pelvis_gyro"], ori6d_full, orierr_quat_full, orierr_rpy_full,
                      t_start, log_path)
    window_policy_replay(rep, result, t, qdes_log_full, log_path, t_start)
    window_commanded_torque(result, t, tau_logged_full, tau_pred_steps, log_path, t_start)

    # optional observation diagnostics
    if args.all_obs:
        window_all_obs(R, log_path)

    # a compare log is loaded once if any mode needs it (--obs comparison or swap ablation)
    RB = None
    if args.compare:
        RB = load_and_reconstruct(args.compare, args, None, tag="B")

    if args.obs:
        if RB is None:
            print("  --obs needs --compare <logB>; skipping comparison (use --all-obs for one log).")
        else:
            window_compare_obs(R, RB, args.obs)

    if args.ablate:
        window_ablation(R, RB, args)

    if args.sensitivity:
        window_sensitivity(R, RB, args)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

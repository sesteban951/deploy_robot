##
#
# Sim2sim tracking-error report.
#
# Reads an HDF5 log produced by a mimic run (deploy/logger/log.py) and compares
# the performed motion (logged joint_state + IMU) against the motion reference
# (the .npz the policy was tracking) to quantify how well sim followed the ref.
#
# The reference is identified from the log's own experiment metadata: the
# verbatim config_yaml attr names the motion_path / xml_path / control_dt, so a
# log is fully self-describing and no extra arguments are needed.
#
# Note: base *position* is not logged (joint_state carries no base pose, the IMU
# is orientation-only), so root position drift is not computable here -- only
# joint tracking and base/anchor orientation error.
#
##

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt

# reuse the log helpers that already live next to this script
from plot import find_latest_log, print_experiment_info, experiment_label

# repo root (motions/, models/ are resolved relative to it, same as the nodes)
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rpy,
    heading_about_z_world,
)


############################################################################
# LOG + REFERENCE LOADING
############################################################################

# load all datasets + root attrs from an hdf5 log
def load_log(file_path: str):
    with h5py.File(file_path, "r") as f:
        data = {name: f[name][:] for name in f.keys()}
        meta = {k: f.attrs[k] for k in f.attrs.keys()}
    return data, meta


# parse the verbatim config_yaml snapshot recorded by experiment_utils
def config_from_meta(meta: dict) -> dict:
    if "config_yaml" not in meta:
        raise ValueError(
            "Log has no 'config_yaml' attr -- cannot identify the reference motion. "
            "Was this run logged with a control node that broadcasts experiment info?"
        )
    cfg = yaml.safe_load(str(meta["config_yaml"]))
    if not isinstance(cfg, dict):
        raise ValueError("config_yaml did not parse to a mapping.")
    return cfg


# load the reference motion .npz named by the config
def load_reference(cfg: dict):
    motion_path = os.path.join(ROOT_DIR, "motions", cfg["motion_path"])
    motion = np.load(motion_path)
    ref = {
        "path": motion_path,
        "fps": float(motion["fps"]),
        "joint_pos": motion["joint_pos"].astype(np.float64),   # (frames, joints)
        "joint_vel": motion["joint_vel"].astype(np.float64),   # (frames, joints)
        "body_quat_w": motion["body_quat_w"].astype(np.float64),  # (frames, bodies, 4) wxyz, body0 = root
    }
    ref["num_frames"] = ref["joint_pos"].shape[0]
    return ref


# resolve which body the policy anchors to (drives the IMU choice + reference
# body index). "auto" reads the policy's onnx metadata; falls back to pelvis.
def resolve_anchor(cfg: dict, ref: dict, anchor_arg: str):
    anchor_name = None
    if anchor_arg == "auto":
        try:
            from utils.policy import Policy
            policy = Policy(os.path.join(ROOT_DIR, "policy", cfg["policy_path"]))
            anchor_name = policy.metadata.get("anchor_body_name")
        except Exception as e:
            print(f"[WARN] could not read anchor from policy metadata ({e}); defaulting to pelvis.")
    else:
        anchor_name = anchor_arg  # explicit "pelvis" / "torso"

    if anchor_name is None:
        anchor_name = "pelvis"

    # which IMU dataset to read the measured orientation from
    if "pelvis" in anchor_name.lower():
        imu_key, anchor = "pelvis_imu", "pelvis"
    elif "torso" in anchor_name.lower():
        imu_key, anchor = "torso_imu", "torso"
    else:
        raise ValueError(f"Unsupported anchor body name: {anchor_name}")

    # index of the anchor body within the reference body arrays. pelvis is the
    # root (body 0). anything else needs the mujoco body ordering, matching the
    # control node (mj bodies 1..nbody, i.e. world skipped).
    if anchor == "pelvis":
        anchor_idx = 0
    else:
        import mujoco
        mj_model = mujoco.MjModel.from_xml_path(os.path.join(ROOT_DIR, "models", cfg["xml_path"]))
        body_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, mj_model.nbody)]
        anchor_idx = body_names.index(anchor_name)

    return anchor, imu_key, anchor_idx, anchor_name


############################################################################
# ALIGNMENT
############################################################################

# rebuild the per-sample reference frame index the way the control node does:
# 1 frame per control_dt, looping. k0 is an integer phase offset (absorbs the
# unknown policy-start time / startup ramp).
def frame_indices(t: np.ndarray, control_dt: float, num_frames: int, k0: int) -> np.ndarray:
    steps = np.floor((t - t[0]) / control_dt).astype(np.int64)
    return (steps + k0) % num_frames


# search the integer phase offset that best aligns measured joints to the
# reference (mean abs joint-pos error). the reference loops, so we sweep a full
# period, sub-sampling the log to keep the search cheap.
def find_phase_offset(t, q_meas, ref, control_dt):
    num_frames = ref["num_frames"]
    ref_q = ref["joint_pos"]

    # subsample the log for the search (metrics later use the full log)
    stride = max(1, len(t) // 1500)
    ts, qs = t[::stride], q_meas[::stride]
    base_steps = np.floor((ts - ts[0]) / control_dt).astype(np.int64)

    best_k, best_err = 0, np.inf
    for k0 in range(num_frames):
        idx = (base_steps + k0) % num_frames
        err = np.mean(np.abs(qs - ref_q[idx]))
        if err < best_err:
            best_err, best_k = err, k0
    return best_k, best_err


############################################################################
# METRICS
############################################################################

# per-joint + aggregate error stats for a (samples, joints) residual
def joint_error_stats(residual: np.ndarray):
    per_joint_rmse = np.sqrt(np.mean(residual ** 2, axis=0))
    per_joint_mae = np.mean(np.abs(residual), axis=0)
    per_joint_max = np.max(np.abs(residual), axis=0)
    return {
        "per_joint_rmse": per_joint_rmse,
        "per_joint_mae": per_joint_mae,
        "per_joint_max": per_joint_max,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "max": float(np.max(np.abs(residual))),
    }


# geodesic angle (rad) between measured anchor orientation and the
# heading-corrected reference, mirroring the control node's yaw alignment.
def orientation_error(quat_meas, ref_quat_seq):
    # heading offset captured once at the first sample (control node: init_quat)
    q_rel0 = quat_multiply(quat_meas[0], quat_conjugate(ref_quat_seq[0]))
    init_quat = heading_about_z_world(q_rel0)

    angles = np.zeros(len(quat_meas))
    rpy_err = np.zeros((len(quat_meas), 3))
    for i in range(len(quat_meas)):
        corrected_ref = quat_multiply(init_quat, ref_quat_seq[i])
        err_q = quat_multiply(quat_conjugate(quat_meas[i]), corrected_ref)
        angles[i] = 2.0 * np.arccos(np.clip(abs(err_q[0]), 0.0, 1.0))
        rpy_err[i] = quat_to_rpy(quat_meas[i]) - quat_to_rpy(corrected_ref)
    return angles, rpy_err


############################################################################
# REPETITION SEGMENTATION
############################################################################

# Split a looping run into repetitions and score each one. A single log recorded
# in no-joystick mode plays the motion back-to-back (1 frame per control_dt,
# looping), so an unwrapped frame index gives the repetition number directly:
#   global_idx = control_steps + phase_offset ; rep = global_idx // num_frames.
# Returns one dict per repetition with its metrics and loop coverage (fraction
# of the motion's frames it actually spans), plus the metrics helper's outputs.
def segment_reps(run):
    num_frames = run["ref"]["num_frames"]
    global_idx = run["steps"] + run["k0"]
    rep_ids = global_idx // num_frames
    rep_ids = rep_ids - rep_ids.min()   # renumber from 0
    frames_in_rep = global_idx % num_frames

    reps = []
    for r in np.unique(rep_ids):
        mask = rep_ids == r
        coverage = np.unique(frames_in_rep[mask]).size / num_frames
        m = metrics_for(run, mask)
        m.update({
            "rep": int(r),
            "coverage": float(coverage),
            "n": int(np.sum(mask)),
            "t_start": float(run["t"][mask][0]),
            "t_end": float(run["t"][mask][-1]),
            "mask": mask,
        })
        reps.append(m)
    return reps


# compute pos/vel/orientation error over a boolean sample-mask of a run
def metrics_for(run, mask):
    pos = joint_error_stats((run["q_meas"] - run["q_ref"])[mask])
    vel = joint_error_stats((run["dq_meas"] - run["dq_ref"])[mask])
    ori = None
    if run["ori_deg_series"] is not None:
        ori = joint_error_stats(run["ori_deg_series"][mask][:, None])
    return {"pos": pos, "vel": vel, "ori": ori}


############################################################################
# REPORT
############################################################################

# per-log report: reference, phase alignment, per-rep table, and the best rep
def print_log_report(result):
    run, reps, best = result["run"], result["reps"], result["best"]
    ref, bar = run["ref"], "=" * 72
    print("")
    print(bar)
    print(f"  {os.path.basename(result['file'])}")
    print(bar)
    print(f"  reference      {os.path.basename(ref['path'])}  ({ref['num_frames']} frames @ {ref['fps']} fps)")
    print(f"  anchor         {run['anchor_name']}")
    print(f"  phase offset   k0={run['k0']} frames  (align err {run['align_err']:.4f} rad)")
    print(f"  repetitions    {len(reps)} total, {sum(r['coverage'] >= run['min_coverage'] for r in reps)} complete "
          f"(coverage >= {run['min_coverage']:.0%})")
    print("")

    has_ori = reps and reps[0]["ori"] is not None
    header = f"  {'rep':>3}  {'window [s]':>15}  {'cov':>5}  {'pos RMSE':>9}  {'vel RMSE':>9}"
    if has_ori:
        header += f"  {'ori RMSE':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in reps:
        eligible = r["coverage"] >= run["min_coverage"]
        marker = " *BEST" if r is best else ("" if eligible else "  (partial)")
        line = (f"  {r['rep']:>3}  {r['t_start']:>6.1f}-{r['t_end']:<8.1f}  {r['coverage']:>4.0%}"
                f"  {r['pos']['rmse']:>9.4f}  {r['vel']['rmse']:>9.4f}")
        if has_ori:
            line += f"  {r['ori']['rmse']:>9.2f}"
        print(line + marker)
    print("")

    _print_best_detail(best, ref, run["anchor_name"])
    print(bar)
    print("")


# per-joint breakdown for the chosen (best) rep
def _print_best_detail(best, ref, anchor_name):
    pos, vel, ori = best["pos"], best["vel"], best["ori"]
    print(f"  BEST rep {best['rep']}  ({best['n']} samples, {best['coverage']:.0%} coverage)")
    print(f"    joint position   RMSE {pos['rmse']:.4f} rad   MAE {pos['mae']:.4f} rad   max {pos['max']:.4f} rad")
    print(f"    joint velocity   RMSE {vel['rmse']:.4f} rad/s MAE {vel['mae']:.4f} rad/s max {vel['max']:.4f} rad/s")
    if ori is not None:
        print(f"    {anchor_name} orient RMSE {ori['rmse']:.2f} deg    MAE {ori['mae']:.2f} deg   max {ori['max']:.2f} deg")
    worst = np.argsort(pos["per_joint_rmse"])[::-1][:3]
    worst_str = ", ".join("{} ({:.4f})".format(j, pos["per_joint_rmse"][j]) for j in worst)
    print(f"    worst joints (pos RMSE): {worst_str}")


# cross-log comparison table, one row per log, using each log's best rep
def print_comparison(results):
    bar = "=" * 72
    print("")
    print(bar)
    print("  ABLATION COMPARISON  (best repetition per log)")
    print(bar)
    has_ori = all(r["best"]["ori"] is not None for r in results)
    header = f"  {'log':<26}  {'rep':>3}  {'pos RMSE':>9}  {'vel RMSE':>9}"
    if has_ori:
        header += f"  {'ori RMSE':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    ranked = sorted(results, key=lambda r: r["best"]["pos"]["rmse"])
    for r in ranked:
        b = r["best"]
        name = os.path.splitext(os.path.basename(r["file"]))[0]
        line = f"  {name:<26}  {b['rep']:>3}  {b['pos']['rmse']:>9.4f}  {b['vel']['rmse']:>9.4f}"
        if has_ori:
            line += f"  {b['ori']['rmse']:>9.2f}"
        print(line)
    print("  " + "-" * (len(header) - 2))
    print(f"  best overall (pos RMSE): {os.path.basename(ranked[0]['file'])}")
    print(bar)
    print("")


def make_plots(t, q_meas, q_ref, dq_meas, dq_ref, ori_deg_series, label):
    N = q_meas.shape[1]
    cols = min(6, N)
    rows = int(np.ceil(N / cols))

    # joint position: measured vs reference
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9), sharex=True)
    fig.suptitle(f"joint position: measured vs reference\n{label}")
    axes = axes.flatten()
    for j in range(N):
        axes[j].plot(t, q_ref[:, j], color="black", linewidth=0.75, label="ref", zorder=1)
        axes[j].plot(t, q_meas[:, j], color="tab:blue", linewidth=1.2, label="meas", zorder=2)
        axes[j].set_title(f"joint {j}")
        axes[j].grid(True)
    for ax in axes[N:]:
        ax.set_visible(False)
    axes[0].legend(loc="upper right")

    # per-joint position error over time
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9), sharex=True)
    fig.suptitle(f"joint position error (meas - ref)\n{label}")
    axes = axes.flatten()
    for j in range(N):
        axes[j].plot(t, q_meas[:, j] - q_ref[:, j], color="tab:red", linewidth=1.0)
        axes[j].axhline(0.0, color="black", linewidth=0.5)
        axes[j].set_title(f"joint {j}")
        axes[j].set_ylabel("[rad]")
        axes[j].grid(True)
    for ax in axes[N:]:
        ax.set_visible(False)

    # orientation error over time
    if ori_deg_series is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(f"anchor orientation error\n{label}")
        ax.plot(t, ori_deg_series, color="tab:purple")
        ax.set_ylabel("geodesic angle [deg]")
        ax.set_xlabel("time [s]")
        ax.grid(True)

    plt.show()


############################################################################
# MAIN
############################################################################

# load a log, align its phase, and precompute per-sample measured/reference
# series. Returns a "run" dict consumed by segment_reps / metrics_for.
def prepare_run(file_path, anchor_arg, min_coverage):
    print(f"Loading {file_path}")
    data, meta = load_log(file_path)

    if "joint_state" not in data or "time" not in data:
        raise ValueError(f"{file_path}: no joint_state/time data (metadata-only run) -- nothing to compare.")

    cfg = config_from_meta(meta)
    ref = load_reference(cfg)
    control_dt = float(cfg["control_dt"])
    anchor, imu_key, anchor_idx, anchor_name = resolve_anchor(cfg, ref, anchor_arg)

    # drop stale leading rows where sim/fsm time resets (same as plot.py)
    t_raw = data["time"][:, 0]
    resets = np.where(np.diff(t_raw) < 0)[0]
    start = int(resets[0]) + 1 if len(resets) else 0
    if start > 0:
        print(f"  Dropping {start} stale leading row(s) (time resets at row {start}).")
        data = {name: arr[start:] for name, arr in data.items()}

    t = data["time"][:, 0].astype(np.float64)
    t = t - t[0]

    # measured joints: joint_state = [q, dq, ddq, tau_est]
    N = data["joint_state"].shape[1] // 4
    assert N == ref["joint_pos"].shape[1], \
        f"{file_path}: joint count mismatch (log {N}, reference {ref['joint_pos'].shape[1]})."
    q_meas = data["joint_state"][:, 0:N].astype(np.float64)
    dq_meas = data["joint_state"][:, N:2 * N].astype(np.float64)

    # align phase, then build the per-sample reference frame indices.
    # steps = control ticks since the log start; the unwrapped index (steps + k0)
    # feeds both the looped reference lookup and the repetition segmentation.
    k0, align_err = find_phase_offset(t, q_meas, ref, control_dt)
    steps = np.floor((t - t[0]) / control_dt).astype(np.int64)
    frames = (steps + k0) % ref["num_frames"]
    q_ref = ref["joint_pos"][frames]
    dq_ref = ref["joint_vel"][frames]

    # orientation error series (optional -- needs the anchor IMU in the log)
    ori_deg_series = None
    if imu_key in data:
        quat_meas = data[imu_key][:, 3:7].astype(np.float64)  # IMU = [rpy(3), quat(4) wxyz, gyro(3), acc(3)]
        ref_quat_seq = ref["body_quat_w"][frames, anchor_idx]
        angles, _ = orientation_error(quat_meas, ref_quat_seq)
        ori_deg_series = np.degrees(angles)
    else:
        print(f"  [SKIP] '{imu_key}' not in log -- skipping orientation error.")

    return {
        "file": file_path, "meta": meta, "ref": ref, "anchor_name": anchor_name,
        "t": t, "steps": steps, "k0": k0, "align_err": align_err,
        "q_meas": q_meas, "q_ref": q_ref, "dq_meas": dq_meas, "dq_ref": dq_ref,
        "ori_deg_series": ori_deg_series, "min_coverage": min_coverage,
    }


# analyze one log: segment into repetitions and pick the least-error complete rep
def analyze_file(file_path, anchor_arg, min_coverage, do_plot):
    run = prepare_run(file_path, anchor_arg, min_coverage)
    reps = segment_reps(run)

    # only complete reps are eligible for "best" (a partial startup/tail rep can't
    # win unfairly). fall back to all reps if none clear the coverage bar.
    eligible = [r for r in reps if r["coverage"] >= min_coverage]
    if not eligible:
        print(f"  [WARN] no rep reaches {min_coverage:.0%} coverage; ranking all reps.")
        eligible = reps
    best = min(eligible, key=lambda r: r["pos"]["rmse"])

    result = {"file": file_path, "run": run, "reps": reps, "best": best}
    print_log_report(result)

    if do_plot:
        m = best["mask"]
        make_plots(run["t"][m], run["q_meas"][m], run["q_ref"][m],
                   run["dq_meas"][m], run["dq_ref"][m],
                   run["ori_deg_series"][m] if run["ori_deg_series"] is not None else None,
                   f"{experiment_label(run['meta'])}  |  best rep {best['rep']}")
    return result


# every .h5 log under logs/simulation and logs/hardware, oldest first
def find_all_logs():
    logs_root = os.path.dirname(os.path.abspath(__file__))
    candidates = (glob.glob(os.path.join(logs_root, "simulation", "*.h5"))
                  + glob.glob(os.path.join(logs_root, "hardware", "*.h5")))
    if not candidates:
        raise FileNotFoundError(f"No .h5 logs found under {logs_root}/simulation or {logs_root}/hardware.")
    return sorted(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Measure sim2sim tracking error (performed motion vs reference). "
                    "For looping logs, each repetition is scored and the least-error one is reported.")
    parser.add_argument("--filename", type=str, nargs="+", default=None,
                        help="One or more .h5 logs. If omitted, uses the most recent log under "
                             "logs/simulation or logs/hardware. Multiple logs print a comparison table.")
    parser.add_argument("--all", action="store_true",
                        help="Analyze every .h5 log under logs/simulation and logs/hardware and compare them. "
                             "Logs that can't be analyzed (no metadata, wrong joint count, etc.) are skipped.")
    parser.add_argument("--anchor", type=str, default="auto", choices=["auto", "pelvis", "torso"],
                        help="Anchor body for orientation error. 'auto' reads it from the policy metadata.")
    parser.add_argument("--min-coverage", type=float, default=0.9,
                        help="Minimum loop coverage for a repetition to be eligible as 'best'. Default 0.9.")
    parser.add_argument("--plot", action="store_true", help="Plot the best rep (measured vs reference).")
    args = parser.parse_args()

    if args.all:
        files = find_all_logs()
        print(f"--all: found {len(files)} log(s) under logs/simulation and logs/hardware.")
    elif args.filename is not None:
        files = args.filename
        for f in files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"No such file: {f}")
    else:
        files = [find_latest_log()]
        print(f"No --filename provided, using latest: {files[0]}")

    # analyze each log; skip (don't abort) ones that fail so a batch survives a bad log
    results = []
    for f in files:
        try:
            results.append(analyze_file(f, args.anchor, args.min_coverage, args.plot))
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(f)}: {e}")

    if not results:
        print("No logs could be analyzed.")
    elif len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()

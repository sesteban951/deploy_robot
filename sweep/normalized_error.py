##
#
# Normalized, summed tracking error -- one scalar per policy / per motion.
#
# For every tracked state (29 joint positions, 29 joint velocities, and the
# anchor orientation) we take the RMSE over all valid reps and divide it by that
# state's reference RANGE (max-min over the motion). Summing the per-state
# normalized errors gives a single dimensionless number: roughly "how many
# state-ranges of error, totaled across the whole state vector."
#
# States whose reference barely moves (range below a small floor) are excluded:
# range-normalizing a near-static state blows up and is not meaningful.
#
##

import json
import os
import sys

import numpy as np

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "logs"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim2sim_error import load_log, load_reference, resolve_anchor, orientation_error
from utils.math_utils import quat_conjugate, quat_multiply
from run_batch import ALL_POLICIES, motion_for, family_of

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
XML = "g1_29dof_scene.xml"
CONTROL_DT = 0.02
POS_FLOOR = 0.01   # rad     : exclude joints that move less than this in the ref
VEL_FLOOR = 0.05   # rad/s
ORI_FLOOR = np.radians(5.0)


def cfg_for(policy):
    return {"motion_path": motion_for(policy), "xml_path": XML,
            "policy_path": f"ablation/{policy}.onnx", "control_dt": CONTROL_DT}


# reference angular travel range: geodesic angle of ref[frame] vs ref[0], ptp
def ref_ori_range(ref, anchor_idx):
    q = ref["body_quat_w"][:, anchor_idx]
    q0 = q[0]
    ang = np.array([2.0 * np.arccos(np.clip(abs(quat_multiply(quat_conjugate(q0), qi)[0]), 0, 1)) for qi in q])
    return float(ang.ptp())


# gather residuals across all valid reps for one policy; returns stacked
# (samples, 29) joint-pos and joint-vel residuals + measured/ref orientation
def collect(policy):
    sidecar = os.path.join(RESULTS, "sidecars", f"{policy}.json")
    if not os.path.exists(sidecar):
        return None
    sc = json.load(open(sidecar))
    matches = [f for f in os.listdir(os.path.join(ROOT_DIR, "logs", "simulation"))
               if f.startswith(f"sweep_{policy}") and f.endswith(".h5")]
    if not matches:
        return None
    log_path = os.path.join(ROOT_DIR, "logs", "simulation", matches[0])

    cfg = cfg_for(policy)
    ref = load_reference(cfg)
    control_dt = float(cfg["control_dt"])
    num_frames = ref["num_frames"]
    _anchor, imu_key, anchor_idx, _ = resolve_anchor(cfg, ref, "auto")

    data, _ = load_log(log_path)
    keys = [k for k in ("time", "joint_state", imu_key) if k in data]
    ml = min(data[k].shape[0] for k in keys)
    data = {k: v[:ml] for k, v in data.items()}
    t = data["time"][:, 0].astype(np.float64)
    N = data["joint_state"].shape[1] // 4

    pos_res, vel_res, ori_ang = [], [], []
    for rep in sc.get("reps", []):
        if not rep.get("standing_ok") or rep.get("track_start_time") is None:
            continue
        ts = rep["track_start_time"]
        frame = np.round((t - ts) / control_dt).astype(np.int64)
        win = (t >= ts) & (t < ts + sc["motion_duration"]) & (frame >= 0) & (frame <= num_frames - 1)
        if not np.any(win):
            continue
        fr = np.clip(frame[win], 0, num_frames - 1)
        q = data["joint_state"][win, 0:N].astype(np.float64)
        dq = data["joint_state"][win, N:2 * N].astype(np.float64)
        pos_res.append(q - ref["joint_pos"][fr])
        vel_res.append(dq - ref["joint_vel"][fr])
        if imu_key in data:
            quat_meas = data[imu_key][win, 3:7].astype(np.float64)
            ang, _ = orientation_error(quat_meas, ref["body_quat_w"][fr, anchor_idx])
            ori_ang.append(ang)

    if not pos_res:
        return None
    return {
        "policy": policy, "motion": cfg["motion_path"], "ref": ref, "anchor_idx": anchor_idx,
        "pos_res": np.concatenate(pos_res), "vel_res": np.concatenate(vel_res),
        "ori_ang": np.concatenate(ori_ang) if ori_ang else None,
    }


# per-state RMSE / reference range, summed over the state vector
def normalized_summed(c):
    ref = c["ref"]
    pos_rmse = np.sqrt(np.mean(c["pos_res"] ** 2, axis=0))       # (29,)
    vel_rmse = np.sqrt(np.mean(c["vel_res"] ** 2, axis=0))       # (29,)
    pos_rng = ref["joint_pos"].ptp(axis=0)
    vel_rng = ref["joint_vel"].ptp(axis=0)

    pm = pos_rng > POS_FLOOR
    vm = vel_rng > VEL_FLOOR
    npos = (pos_rmse[pm] / pos_rng[pm])
    nvel = (vel_rmse[vm] / vel_rng[vm])

    nori, ori_used = 0.0, 0
    if c["ori_ang"] is not None:
        ori_rmse = float(np.sqrt(np.mean(c["ori_ang"] ** 2)))
        ori_rng = ref_ori_range(ref, c["anchor_idx"])
        if ori_rng > ORI_FLOOR:
            nori = ori_rmse / ori_rng
            ori_used = 1

    n_states = int(pm.sum() + vm.sum() + ori_used)
    total = float(npos.sum() + nvel.sum() + nori)
    return {
        "policy": c["policy"], "motion": c["motion"],
        "norm_pos_sum": float(npos.sum()), "norm_vel_sum": float(nvel.sum()),
        "norm_ori": float(nori), "n_states": n_states,
        "total_norm_error": total,
        # mean per active state: comparable across motions (the sum is not,
        # because motions activate different numbers of states)
        "mean_norm_error": total / n_states if n_states else float("nan"),
    }


def main():
    rows = [normalized_summed(c) for p in ALL_POLICIES for c in [collect(p)] if c]

    hdr = (f"  {'policy':<22} {'#states':>7}  {'norm pos':>9}  {'norm vel':>9}  {'norm ori':>9}"
           f"  {'SUM':>8}  {'MEAN/state':>10}")
    bar = "=" * len(hdr)
    print("\n".join(["", bar, "  NORMALIZED TRACKING ERROR  (per-state RMSE / reference range)",
                     "  SUM = summed over active states (compare only within a motion)",
                     "  MEAN/state = SUM / #states (comparable across motions)",
                     bar, hdr, "  " + "-" * (len(hdr) - 2)]))
    for r in sorted(rows, key=lambda r: r["mean_norm_error"]):
        print(f"  {r['policy']:<22} {r['n_states']:>7}  {r['norm_pos_sum']:>9.3f}  "
              f"{r['norm_vel_sum']:>9.3f}  {r['norm_ori']:>9.3f}  {r['total_norm_error']:>8.3f}  "
              f"{r['mean_norm_error']:>10.4f}")
    print("  " + "-" * (len(hdr) - 2))

    # per-motion (pool seeds)
    from collections import defaultdict
    bym = defaultdict(lambda: {"sum": [], "mean": []})
    for r in rows:
        bym[r["motion"]]["sum"].append(r["total_norm_error"])
        bym[r["motion"]]["mean"].append(r["mean_norm_error"])
    print(f"\n  per motion:  {'sum (within-motion)':>22}   {'mean/state (comparable)':>24}")
    for m, v in bym.items():
        print(f"    {m:<32} {np.mean(v['sum']):>10.3f}            {np.mean(v['mean']):>10.4f}   (n={len(v['sum'])})")
    print(bar + "\n")

    import csv
    out = os.path.join(RESULTS, "normalized_error.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "motion", "n_states",
                                          "norm_pos_sum", "norm_vel_sum", "norm_ori",
                                          "total_norm_error", "mean_norm_error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

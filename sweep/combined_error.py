##
#
# Combined tracking error -- one dimensionless scalar per policy / per motion.
#
# The three raw RMSEs are each divided by a FIXED, motion-independent physical
# scale (so barely-moving joints can't blow up the way per-motion-range
# normalization did), then summed:
#
#   pos_term = mean_j ( pos_rmse_j / ROM_j )      ROM_j = joint range of motion (XML jnt_range)
#   vel_term = mean_j ( vel_rmse_j / V_MAX )      V_MAX = nominal joint speed limit (see note)
#   ori_term = ori_rmse_rad / pi                  pi = max possible geodesic error
#   total    = pos_term + vel_term + ori_term
#
# Each term is a dimensionless "fraction of physical scale" (~[0,1]); the total
# is comparable across motions. NOTE: the model carries joint POSITION limits
# but no VELOCITY limit, so V_MAX is a chosen constant (a uniform rescale of the
# velocity term across all policies) rather than a hardware value.
#
##

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import mujoco

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "logs"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalized_error import collect          # reuses the log/reference gathering
from run_batch import ALL_POLICIES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
XML = "g1_29dof_scene.xml"
V_MAX = 20.0   # rad/s -- nominal joint-speed scale (model has no velocity limit)


# per-actuator joint range of motion (rad), in actuator/motion order
def joint_roms():
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT_DIR, "models", XML))
    roms = np.array([np.ptp(m.jnt_range[m.actuator_trnid[i, 0]]) for i in range(m.nu)])
    return roms


def combined(c, roms):
    pos_rmse = np.sqrt(np.mean(c["pos_res"] ** 2, axis=0))
    vel_rmse = np.sqrt(np.mean(c["vel_res"] ** 2, axis=0))
    pos_term = float(np.mean(pos_rmse / roms))
    vel_term = float(np.mean(vel_rmse / V_MAX))
    ori_term = 0.0
    if c["ori_ang"] is not None:
        ori_term = float(np.sqrt(np.mean(c["ori_ang"] ** 2)) / np.pi)
    return {
        "policy": c["policy"], "motion": c["motion"],
        "pos_term": pos_term, "vel_term": vel_term, "ori_term": ori_term,
        "combined_error": pos_term + vel_term + ori_term,
    }


# compute the combined error for each policy (skips policies with no log/sidecar)
def compute(policies):
    roms = joint_roms()
    return [combined(c, roms) for p in policies for c in [collect(p)] if c]


# build the printable report string + per-motion rollup
def format_report(rows):
    hdr = f"  {'policy':<22}  {'pos':>7}  {'vel':>7}  {'ori':>7}  {'COMBINED':>9}"
    bar = "=" * len(hdr)
    lines = ["", bar, "  COMBINED TRACKING ERROR  (fixed-scale normalized, summed)",
             f"  pos/ROM + vel/{V_MAX:g} + ori/180deg   (lower = better)",
             bar, hdr, "  " + "-" * (len(hdr) - 2)]
    for r in sorted(rows, key=lambda r: r["combined_error"]):
        lines.append(f"  {r['policy']:<22}  {r['pos_term']:>7.4f}  {r['vel_term']:>7.4f}  "
                     f"{r['ori_term']:>7.4f}  {r['combined_error']:>9.4f}")
    lines.append("  " + "-" * (len(hdr) - 2))
    bym = defaultdict(list)
    for r in rows:
        bym[r["motion"]].append(r["combined_error"])
    lines.append("\n  per motion (mean over seeds):")
    for m, vals in sorted(bym.items(), key=lambda kv: np.mean(kv[1])):
        lines.append(f"    {m:<32} {np.mean(vals):>8.4f}   (n={len(vals)})")
    lines += [bar, ""]
    return "\n".join(lines)


# compute + print + write combined_error.csv; returns (report_string, rows)
def report(policies, results_dir=RESULTS):
    rows = compute(policies)
    if not rows:
        return "", []
    rep = format_report(rows)
    print(rep)
    out = os.path.join(results_dir, "combined_error.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "motion", "pos_term", "vel_term", "ori_term", "combined_error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out}")
    return rep, rows


def main():
    report(ALL_POLICIES)


if __name__ == "__main__":
    main()

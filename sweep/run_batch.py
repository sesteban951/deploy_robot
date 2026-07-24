##
#
# Sweep orchestrator: test each ablation policy in simulation and record its
# mimic tracking error, N repetitions per policy.
#
# Per run it:
#   1. writes a sweep config (set_config) pointing at the policy + its motion,
#   2. launches the headless sim (sweep/sim_headless.py),
#   3. launches the FSM driver (sweep/fsm_driver.py) -> connected joystick +
#      init->damp->home->control->track, with a torso-upright start gate,
#   4. launches the mimic control node and the logger,
#   5. waits for the FSM driver to finish (sidecar written), tears everything
#      down, and analyzes the log over the single track window.
#
# Runs whose start gate failed (robot not standing at the first frame) are
# flagged and excluded from the per-policy statistics.
#
# NOTE: recorded runs use real-time headless sim. --view shows the viewer;
# --fast is intentionally NOT wired in here because it desyncs the control node.
#
##

import argparse
import glob
import os
import signal
import subprocess
import sys
import time

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DEPLOY_ROOT_DIR", ROOT_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from set_config import write_sweep_config
from analyze_run import analyze_log

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
SWEEP_CONFIG = "g1_29dof_mimic_sweep.yaml"

# policy family -> motion file (relative to motions/)
MOTION_FOR = {
    "kino_backflip":     "ablation/kino_backflip.npz",
    "srb_ik_backflip":   "ablation/srb_backflip.npz",
    "srb_traj_backflip": "ablation/srb_traj_backflip.npz",
    "traj_opt_kino":     "ablation/traj_kino_backflip.npz",
}

# all pulled ablation policies (basename in policy/ablation/, without .onnx)
ALL_POLICIES = [
    "kino_backflip_1", "kino_backflip_2", "kino_backflip_3",
    "srb_ik_backflip_1", "srb_ik_backflip_2", "srb_ik_backflip_3",
    "srb_traj_backflip_0", "srb_traj_backflip_1", "srb_traj_backflip_2",
    "traj_opt_kino_1", "traj_opt_kino_3", "traj_opt_kino_4",
]


def family_of(policy):
    return policy.rsplit("_", 1)[0]


def motion_for(policy):
    fam = family_of(policy)
    if fam not in MOTION_FOR:
        raise KeyError(f"No motion mapping for policy family '{fam}' (policy {policy}).")
    return MOTION_FOR[fam]


def motion_duration(motion_rel, control_dt=0.02):
    m = np.load(os.path.join(ROOT_DIR, "motions", motion_rel))
    return m["joint_pos"].shape[0] * control_dt


# ---------------------------------------------------------------------------
# subprocess helpers (each child in its own session so we can signal the group)
# ---------------------------------------------------------------------------

def launch(argv, log_file):
    f = open(log_file, "w")
    # unbuffered child stdout so readiness lines land in the log immediately
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(argv, cwd=ROOT_DIR, stdout=f, stderr=subprocess.STDOUT,
                          start_new_session=True, env=env)
    p._logfile = f
    return p


# poll a process log file until `needle` appears (process readiness handshake)
def wait_for_log(path, needle, timeout=25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path, "r") as f:
                if needle in f.read():
                    return True
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    return False


def stop(procs, sig=signal.SIGINT, grace=4.0):
    # signal newest-first (logger last-started flushes first is fine); then wait
    for p in procs:
        if p is None or p.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    for p in procs:
        if p is None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if getattr(p, "_logfile", None):
            p._logfile.close()


# ---------------------------------------------------------------------------
# one policy = one chained session of `reps` back-to-back playbacks
# ---------------------------------------------------------------------------

def run_policy(policy, reps, dirs, view=False, timeout_pad=25.0):
    policy_path = f"ablation/{policy}.onnx"
    motion_rel = motion_for(policy)
    write_sweep_config(policy_path, motion_rel, out=SWEEP_CONFIG)

    sidecar = os.path.join(dirs["sidecars"], f"{policy}.json")
    log_base = f"sweep_{policy}"

    # clear stale artifacts
    for stale in glob.glob(os.path.join(ROOT_DIR, "logs", "simulation", f"{log_base}*.h5")):
        os.remove(stale)
    if os.path.exists(sidecar):
        os.remove(sidecar)

    dur = motion_duration(motion_rel)
    procs = []
    try:
        # ORDER MATTERS: the sim applies zero torque until it receives a command,
        # so if it starts first the robot free-falls. Bring up the FSM driver +
        # control node FIRST, launch the sim LAST so torques apply from step 1.

        # 1. FSM driver (publishes the connected joystick; its schedule waits for
        #    sim time so nothing advances until the sim is up).
        fsm_argv = [PYTHON, os.path.join(SWEEP_DIR, "fsm_driver.py"),
                    "--config", SWEEP_CONFIG, "--sidecar", sidecar,
                    "--policy", policy, "--reps", str(reps)]
        fsm = launch(fsm_argv, os.path.join(dirs["proclogs"], f"{policy}_fsm.log"))
        procs.append(fsm)
        time.sleep(2.0)  # ROS discovery so control sees the joystick

        # 2. control node
        ctrl_log = os.path.join(dirs["proclogs"], f"{policy}_ctrl.log")
        ctrl_argv = [PYTHON, os.path.join(ROOT_DIR, "deploy", "simulation", "control_29dof_mimic.py"),
                     "--config", SWEEP_CONFIG]
        procs.append(launch(ctrl_argv, ctrl_log))
        if not wait_for_log(ctrl_log, "Control node initialized.", timeout=25.0):
            print(f"  [WARN] {policy}: control node not confirmed ready; starting sim anyway.")

        # 3. logger
        log_argv = [PYTHON, os.path.join(ROOT_DIR, "deploy", "logger", "log.py"),
                    "--mode", "sim", "--filename", log_base]
        procs.append(launch(log_argv, os.path.join(dirs["proclogs"], f"{policy}_log.log")))

        # 4. simulation LAST
        sim_argv = [PYTHON, os.path.join(SWEEP_DIR, "sim_headless.py"), "--config", SWEEP_CONFIG]
        if not view:
            sim_argv.append("--headless")
        procs.append(launch(sim_argv, os.path.join(dirs["proclogs"], f"{policy}_sim.log")))

        # wait for the FSM driver to finish all reps
        per_rep = dur + 2.0  # motion + settle/reset margin
        timeout = 15.0 + reps * per_rep + timeout_pad
        try:
            fsm.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] {policy}: FSM driver timed out after {timeout:.0f}s.")

        time.sleep(0.8)  # let the logger flush the tail
    finally:
        stop(procs)
        time.sleep(1.5)  # let ROS discovery settle before the next policy

    return sidecar, log_base


def resolve_log(log_base):
    matches = glob.glob(os.path.join(ROOT_DIR, "logs", "simulation", f"{log_base}*.h5"))
    return max(matches, key=os.path.getmtime) if matches else None


# score every rep window in a chained session -> one row per rep
def analyze(sidecar_path, log_base):
    import json
    if not os.path.exists(sidecar_path):
        return [{"ok": False, "reason": "no sidecar (session did not complete)"}]
    with open(sidecar_path) as f:
        sc = json.load(f)

    log_path = resolve_log(log_base)
    reps = sc.get("reps", [])
    rows = []
    for rep in reps:
        row = {
            "policy": sc.get("policy"), "run_idx": rep.get("rep"),
            "standing_ok": bool(rep.get("standing_ok")),
            "gate_tilt_deg": rep.get("gate_tilt_deg"),
            "motion": sc.get("motion_path"),
        }
        if log_path is None:
            row.update({"ok": False, "reason": "no log file"})
        elif rep.get("track_start_time") is None:
            row.update({"ok": False, "reason": "track never started"})
        else:
            try:
                m = analyze_log(log_path, sc["config"], rep["track_start_time"], sc["motion_duration"])
                row.update(m); row["ok"] = True
            except Exception as e:
                row.update({"ok": False, "reason": f"analysis error: {e}"})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# aggregation / reporting
# ---------------------------------------------------------------------------

def _stat(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std()))


def summarize(rows, policies):
    lines = []
    header = (f"  {'policy':<22} {'valid/total':>11}  {'pos RMSE (mean+/-std)':>24}  "
              f"{'vel RMSE':>16}  {'ori RMSE deg':>16}")
    bar = "=" * len(header)
    lines += ["", bar, "  SWEEP TRACKING-ERROR SUMMARY  (valid = standing at start)", bar, header, "  " + "-" * (len(header) - 2)]

    summary_rows = []
    for pol in policies:
        pr = [r for r in rows if r.get("policy") == pol]
        valid = [r for r in pr if r.get("ok") and r.get("standing_ok")]
        pm, ps = _stat([r["pos_rmse"] for r in valid])
        vm, vs = _stat([r["vel_rmse"] for r in valid])
        om, osd = _stat([r["ori_rmse_deg"] for r in valid])
        lines.append(f"  {pol:<22} {len(valid):>4}/{len(pr):<6}  {pm:>10.4f} +/- {ps:<8.4f}  "
                     f"{vm:>7.3f}+/-{vs:<7.3f}  {om:>7.2f}+/-{osd:<6.2f}")
        summary_rows.append({
            "policy": pol, "valid": len(valid), "total": len(pr),
            "pos_rmse_mean": pm, "pos_rmse_std": ps,
            "vel_rmse_mean": vm, "vel_rmse_std": vs,
            "ori_rmse_deg_mean": om, "ori_rmse_deg_std": osd,
        })
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


# aggregate error per motion: mean over every valid rep of every policy sharing
# the motion (so a motion tested with 3 seeds pools all 45 reps)
def summarize_by_motion(rows):
    lines = []
    header = (f"  {'motion':<32} {'policies':>8} {'reps':>5}  {'pos RMSE':>9}  "
              f"{'vel RMSE':>9}  {'ori RMSE deg':>12}")
    bar = "=" * len(header)
    lines += ["", bar, "  PER-MOTION TRACKING ERROR  (mean over valid reps, all seeds pooled)", bar,
              header, "  " + "-" * (len(header) - 2)]

    # preserve first-seen motion order
    motions = []
    for r in rows:
        m = r.get("motion")
        if m and m not in motions:
            motions.append(m)

    summary_rows = []
    for m in motions:
        mr = [r for r in rows if r.get("motion") == m and r.get("ok") and r.get("standing_ok")]
        n_pol = len({r.get("policy") for r in mr})
        pm, _ = _stat([r["pos_rmse"] for r in mr])
        vm, _ = _stat([r["vel_rmse"] for r in mr])
        om, _ = _stat([r["ori_rmse_deg"] for r in mr])
        lines.append(f"  {m:<32} {n_pol:>8} {len(mr):>5}  {pm:>9.4f}  {vm:>9.3f}  {om:>12.2f}")
        summary_rows.append({
            "motion": m, "policies": n_pol, "valid_reps": len(mr),
            "pos_rmse_mean": pm, "vel_rmse_mean": vm, "ori_rmse_deg_mean": om,
        })
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


def write_csv(path, rows, fields):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policies", nargs="+", default=None, help="Policies to test (default: all pulled).")
    p.add_argument("--runs", type=int, default=15, help="Chained repetitions per policy (default 15).")
    p.add_argument("--dry-run", action="store_true", help="Test just the first policy (unless --policies given).")
    p.add_argument("--view", action="store_true", help="Show the sim viewer (real-time) instead of headless.")
    p.add_argument("--outdir", default=os.path.join(SWEEP_DIR, "results"), help="Where to write results.")
    args = p.parse_args()

    policies = args.policies or ALL_POLICIES
    runs = args.runs
    if args.dry_run and args.policies is None:
        policies = [ALL_POLICIES[0]]
        print(f"[dry-run] policy={policies}")

    dirs = {
        "root": args.outdir,
        "sidecars": os.path.join(args.outdir, "sidecars"),
        "proclogs": os.path.join(args.outdir, "proclogs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"[sweep] python:   {PYTHON}")
    print(f"[sweep] policies: {policies}")
    print(f"[sweep] runs each: {runs}   view: {args.view}")

    rows = []
    for pol in policies:
        print(f"\n[sweep] === {pol}  ({runs} chained reps) ===")
        t0 = time.time()
        sidecar, log_base = run_policy(pol, runs, dirs, view=args.view)
        pol_rows = analyze(sidecar, log_base)
        rows.extend(pol_rows)
        dt = time.time() - t0
        n_ok = sum(r.get("ok") and r.get("standing_ok") for r in pol_rows)
        for r in pol_rows:
            if r.get("ok"):
                flag = "" if r.get("standing_ok") else "  [FLAGGED: not standing]"
                print(f"[sweep]   rep {r['run_idx']:>2}: pos={r['pos_rmse']:.4f} "
                      f"vel={r['vel_rmse']:.3f} ori={r.get('ori_rmse_deg'):.1f} "
                      f"cov={r.get('coverage'):.2f}{flag}")
            else:
                print(f"[sweep]   rep {r.get('run_idx')}: FAILED: {r.get('reason')}")
        print(f"[sweep]   {n_ok}/{len(pol_rows)} valid reps in {dt:.0f}s")

    # write per-run + summary artifacts
    run_fields = ["policy", "run_idx", "ok", "standing_ok", "gate_tilt_deg",
                  "pos_rmse", "pos_mae", "pos_max", "vel_rmse", "vel_mae", "vel_max",
                  "ori_rmse_deg", "ori_max_deg", "coverage", "n_samples", "motion", "reason"]
    write_csv(os.path.join(args.outdir, "runs.csv"), rows, run_fields)

    report, summary_rows = summarize(rows, policies)
    print(report)
    sum_fields = ["policy", "valid", "total", "pos_rmse_mean", "pos_rmse_std",
                  "vel_rmse_mean", "vel_rmse_std", "ori_rmse_deg_mean", "ori_rmse_deg_std"]
    write_csv(os.path.join(args.outdir, "summary.csv"), summary_rows, sum_fields)

    # per-motion rollup (all seeds pooled)
    motion_report, motion_rows = summarize_by_motion(rows)
    print(motion_report)
    write_csv(os.path.join(args.outdir, "summary_by_motion.csv"), motion_rows,
              ["motion", "policies", "valid_reps", "pos_rmse_mean", "vel_rmse_mean", "ori_rmse_deg_mean"])

    # combined single-number metric (fixed-scale normalized + summed), computed
    # from the logs/sidecars this run just produced. Lazy import avoids the
    # combined_error <-> run_batch circular import at module load.
    combined_report = ""
    try:
        import combined_error
        combined_report, _ = combined_error.report(policies, results_dir=args.outdir)
    except Exception as e:
        print(f"[sweep] combined-error step skipped: {e}")

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(report + "\n" + motion_report + "\n" + combined_report + "\n")
    print(f"[sweep] wrote {os.path.join(args.outdir, 'runs.csv')}, summary.csv, "
          f"summary_by_motion.csv, combined_error.csv, summary.txt")


if __name__ == "__main__":
    main()

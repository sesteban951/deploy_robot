##
#
# Sweep orchestrator: test each ablation policy in simulation and record its
# mimic tracking error AND whether it lands the motion, N repetitions per policy.
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
# Each rep is held in 'track' for --land-hold seconds past the end of the motion
# (the control node freezes on the last frame), and that settle window is scored
# for a landing: still holding the reference's FINAL orientation (not "upright" --
# several references end pitched over) + settled base/joint rates (see analyze_run).
# That gives a per-policy success rate with a Wilson 95% confidence interval,
# alongside the tracking-error statistics.
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

import analyze_run
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
    "kino_backflip_1", "kino_backflip_2", "kino_backflip_3", "kino_backflip_4", "kino_backflip_5",
    "srb_ik_backflip_1", "srb_ik_backflip_2", "srb_ik_backflip_3", "srb_ik_backflip_4", "srb_ik_backflip_5",
    "srb_traj_backflip_0", "srb_traj_backflip_1", "srb_traj_backflip_2", "srb_traj_backflip_4", "srb_traj_backflip_5",
    "traj_opt_kino_0", "traj_opt_kino_1", "traj_opt_kino_2", "traj_opt_kino_4", "traj_opt_kino_5",
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

def run_policy(policy, reps, dirs, view=False, timeout_pad=25.0, land_hold=3.0):
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
                    "--policy", policy, "--reps", str(reps),
                    "--land-hold", str(land_hold)]
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
        per_rep = dur + land_hold + 2.0  # motion + landing settle + reset margin
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
def analyze(sidecar_path, log_base, land_thresholds=None, lead_in=0.0):
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
            "end_tilt_deg": rep.get("end_tilt_deg"),
            "motion": sc.get("motion_path"),
            # default verdict for sessions recorded before landing detection existed
            "landed": False, "land_reason": "landing not evaluated (no settle window)",
        }
        if log_path is None:
            row.update({"ok": False, "reason": "no log file"})
        elif rep.get("track_start_time") is None:
            row.update({"ok": False, "reason": "track never started"})
        else:
            try:
                m = analyze_log(log_path, sc["config"], rep["track_start_time"], sc["motion_duration"],
                                land_hold=float(rep.get("land_hold", sc.get("land_hold", 0.0)) or 0.0),
                                land_thresholds=land_thresholds, lead_in=lead_in)
                row.update(m); row["ok"] = True
            except Exception as e:
                row.update({"ok": False, "reason": f"analysis error: {e}"})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# aggregation / reporting
# ---------------------------------------------------------------------------

# Wilson score interval for a binomial proportion -- better behaved than the
# normal approximation at the small n (15 reps) and extreme rates (0/15, 15/15)
# this sweep produces.
def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _stat(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std()))


# Tracking error is scored over the reps that LANDED. A rep where the robot
# never completed the motion is a failure, not a data point about tracking
# quality, so it is counted by the success rate instead. Policies that never
# land therefore have no error to report -- rendered "n/a (0 landed)", never a
# blank or a NaN that could be mistaken for missing data.
def scored_reps(rows):
    return [r for r in rows if r.get("ok") and r.get("standing_ok") and r.get("landed")]


# mean +/- std, or a visible n/a when nothing was scored
def _fmt(mean, std, w=8, p=4):
    if not np.isfinite(mean):
        return f"{'n/a':>{w}}     {'':<{w - 1}}"
    return f"{mean:>{w}.{p}f} +/- {std:<{w - 1}.{p}f}"


def summarize(rows, policies):
    lines = []
    header = (f"  {'policy':<22} {'scored/valid':>12}  {'pos RMSE (mean+/-std)':>24}  "
              f"{'vel RMSE':>19}  {'ori RMSE deg':>19}")
    bar = "=" * len(header)
    lines += ["", bar,
              "  SWEEP TRACKING-ERROR SUMMARY  (scored over the reps that LANDED)",
              "  scored/valid = landed reps / reps that passed the standing gate.",
              "  A policy that never lands has no tracking error to report (n/a).", bar,
              header, "  " + "-" * (len(header) - 2)]

    summary_rows = []
    for pol in policies:
        pr = [r for r in rows if r.get("policy") == pol]
        valid = [r for r in pr if r.get("ok") and r.get("standing_ok")]
        scored = scored_reps(pr)
        pm, ps = _stat([r["pos_rmse"] for r in scored])
        vm, vs = _stat([r["vel_rmse"] for r in scored])
        om, osd = _stat([r["ori_rmse_deg"] for r in scored])
        lines.append(f"  {pol:<22} {len(scored):>5}/{len(valid):<6}  {_fmt(pm, ps, 10, 4)}  "
                     f"{_fmt(vm, vs, 7, 3)}  {_fmt(om, osd, 7, 2)}")
        rate, lo, hi = _wilson(len(scored), len(valid))
        summary_rows.append({
            "policy": pol, "valid": len(valid), "total": len(pr),
            "landed": len(scored), "success_rate": rate,
            "success_ci_low": lo, "success_ci_high": hi,
            "pos_rmse_mean": pm, "pos_rmse_std": ps,
            "vel_rmse_mean": vm, "vel_rmse_std": vs,
            "ori_rmse_deg_mean": om, "ori_rmse_deg_std": osd,
        })
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


# Landing success rate per policy: of the reps that started standing and were
# scored, how many ended the motion upright and settled. Reported with a Wilson
# 95% CI so 15/15 and 3/3 are not read as equally strong evidence.
def summarize_success(rows, policies):
    lines = []
    header = (f"  {'policy':<22} {'landed/valid':>13} {'rate':>7}  {'95% CI (Wilson)':>18}  "
              f"{'hold s (med)':>12} {'ori err deg':>12}  {'gate fails':>10}")
    bar = "=" * len(header)
    lines += ["", bar,
              "  LANDING SUCCESS RATE  (landed = arrived on the reference's final pose at touchdown)",
              "  ori err = vs the reference's LAST frame, not tilt from vertical.",
              "  Nothing balances the robot after the motion, so the verdict scores ARRIVAL only;",
              "  'hold s' = median time it then held the pose ('*' = over half never fell in-window).", bar,
              header, "  " + "-" * (len(header) - 2)]

    summary_rows = []
    for pol in policies:
        pr = [r for r in rows if r.get("policy") == pol]
        valid = [r for r in pr if r.get("ok") and r.get("standing_ok")]
        landed = [r for r in valid if r.get("landed")]
        gate_fails = sum(1 for r in pr if r.get("ok") and not r.get("standing_ok"))
        rate, lo, hi = _wilson(len(landed), len(valid))
        em, _ = _stat([r.get("land_ori_err_deg") for r in valid])
        tm, _ = _stat([r.get("land_tilt_final_deg") for r in valid])
        # median tolerates the censored entries (>= horizon) far better than a
        # mean would; flagged with '*' when over half the reps never fell.
        holds = [r.get("land_hold_time_s") for r in valid if r.get("land_hold_time_s") is not None]
        hold_med = float(np.median(holds)) if holds else float("nan")
        censored = sum(1 for r in valid if r.get("land_censored"))
        mark = "*" if censored * 2 > len(valid) and valid else " "
        lines.append(f"  {pol:<22} {len(landed):>6}/{len(valid):<6} {rate:>7.2f}  "
                     f"[{lo:>6.2f}, {hi:>6.2f}]  {hold_med:>11.2f}{mark} {em:>12.1f}  {gate_fails:>10}")
        summary_rows.append({
            "policy": pol, "valid": len(valid), "landed": len(landed),
            "success_rate": rate, "ci_low": lo, "ci_high": hi,
            "hold_time_median_s": hold_med, "n_censored": censored,
            "ori_err_deg_mean": em, "end_tilt_deg_mean": tm, "gate_fails": gate_fails,
        })
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


# Floating-base error, from the sim-only ground-truth base_state log. Scored over
# landed reps like the other error tables. Position/velocity are heading- and
# origin-corrected (see analyze_run.base_error_metrics); height is absolute.
# Empty when the logs predate base_state logging.
def summarize_base(rows, policies):
    scored_any = [r for r in scored_reps(rows) if r.get("base_pos_rmse") is not None]
    if not scored_any:
        return "", []

    lines = []
    header = (f"  {'policy':<22} {'reps':>5}  {'pos drift m':>17}  {'height m':>17}  "
              f"{'lin vel m/s':>17}  {'ang vel rad/s':>17}  {'base ori deg':>17}  {'POSE ERR':>17}")
    bar = "=" * len(header)
    lines += ["", bar, "  FLOATING-BASE ERROR  (ground truth, sim only; landed reps)",
              "  pos drift = heading/origin-corrected displacement error; height = absolute z error.",
              "  base ori = pelvis orientation (the other tables score the torso_link anchor).", bar,
              header, "  " + "-" * (len(header) - 2)]

    summary_rows = []
    for pol in policies:
        sc = [r for r in scored_reps([r for r in rows if r.get("policy") == pol])
              if r.get("base_pos_rmse") is not None]
        if not sc:
            continue
        pm, ps = _stat([r["base_pos_rmse"] for r in sc])
        hm, hs = _stat([r["base_height_rmse"] for r in sc])
        lm, ls = _stat([r["base_lin_vel_rmse"] for r in sc])
        am, as_ = _stat([r["base_ang_vel_rmse"] for r in sc])
        om, os_ = _stat([r["base_ori_rmse_deg"] for r in sc])
        cm, cs = _stat([r.get("base_pose_error") for r in sc])
        lines.append(f"  {pol:<22} {len(sc):>5}  {_fmt(pm, ps, 7, 4)}  {_fmt(hm, hs, 7, 4)}  "
                     f"{_fmt(lm, ls, 7, 3)}  {_fmt(am, as_, 7, 3)}  {_fmt(om, os_, 7, 2)}  "
                     f"{_fmt(cm, cs, 7, 4)}")
        summary_rows.append({
            "policy": pol, "reps": len(sc),
            "base_pose_error_mean": cm, "base_pose_error_std": cs,
            "base_pos_rmse_mean": pm, "base_pos_rmse_std": ps,
            "base_height_rmse_mean": hm, "base_height_rmse_std": hs,
            "base_lin_vel_rmse_mean": lm, "base_lin_vel_rmse_std": ls,
            "base_ang_vel_rmse_mean": am, "base_ang_vel_rmse_std": as_,
            "base_ori_rmse_deg_mean": om, "base_ori_rmse_deg_std": os_,
        })
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


# Per-motion floating-base error: seeds pooled, mean +/- std over every landed
# rep of every policy tracking that motion (so 5 seeds x 15 reps = 75 samples).
# Motions no policy lands produce no row -- there is no base trajectory to score.
def summarize_base_by_motion(rows):
    scored = [r for r in scored_reps(rows) if r.get("base_pos_rmse") is not None]
    if not scored:
        return "", []

    lines = []
    header = (f"  {'motion':<32} {'seeds':>5} {'reps':>5}  {'pos drift m':>17}  {'height m':>17}  "
              f"{'lin vel m/s':>17}  {'ang vel rad/s':>17}  {'base ori deg':>17}  {'POSE ERR':>17}")
    bar = "=" * len(header)
    lines += ["", bar, "  PER-MOTION FLOATING-BASE ERROR  (landed reps, all seeds pooled)",
              "  +/- is the spread across individual reps (seed-to-seed AND rep-to-rep combined).",
              "  pos drift = heading/origin-corrected displacement error; height = absolute z error.",
              "  POSE ERR = pos/0.79m + ori/180deg: each error divided by a characteristic scale to",
              "  make it dimensionless, then summed (translation and rotation are not addable raw).", bar,
              header, "  " + "-" * (len(header) - 2)]

    motions = []
    for r in scored:
        if r.get("motion") and r["motion"] not in motions:
            motions.append(r["motion"])

    summary_rows = []
    for m in motions:
        mr = [r for r in scored if r.get("motion") == m]
        n_seed = len({r.get("policy") for r in mr})
        pm, ps = _stat([r["base_pos_rmse"] for r in mr])
        hm, hs = _stat([r["base_height_rmse"] for r in mr])
        lm, ls = _stat([r["base_lin_vel_rmse"] for r in mr])
        am, as_ = _stat([r["base_ang_vel_rmse"] for r in mr])
        om, os_ = _stat([r["base_ori_rmse_deg"] for r in mr])
        cm, cs = _stat([r.get("base_pose_error") for r in mr])
        ptm, pts = _stat([r.get("base_pos_term") for r in mr])
        otm, ots = _stat([r.get("base_ori_term") for r in mr])
        lines.append(f"  {m:<32} {n_seed:>5} {len(mr):>5}  {_fmt(pm, ps, 7, 4)}  {_fmt(hm, hs, 7, 4)}  "
                     f"{_fmt(lm, ls, 7, 3)}  {_fmt(am, as_, 7, 3)}  {_fmt(om, os_, 7, 2)}  "
                     f"{_fmt(cm, cs, 7, 4)}")
        summary_rows.append({
            "motion": m, "seeds": n_seed, "reps": len(mr),
            "base_pos_term": ptm, "base_pos_term_std": pts,
            "base_ori_term": otm, "base_ori_term_std": ots,
            "base_pose_error_mean": cm, "base_pose_error_std": cs,
            "base_pos_rmse_mean": pm, "base_pos_rmse_std": ps,
            "base_height_rmse_mean": hm, "base_height_rmse_std": hs,
            "base_lin_vel_rmse_mean": lm, "base_lin_vel_rmse_std": ls,
            "base_ang_vel_rmse_mean": am, "base_ang_vel_rmse_std": as_,
            "base_ori_rmse_deg_mean": om, "base_ori_rmse_deg_std": os_,
        })
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


# aggregate error per motion: mean over every valid rep of every policy sharing
# the motion (so a motion tested with 3 seeds pools all 45 reps)
def summarize_by_motion(rows):
    lines = []
    header = (f"  {'motion':<32} {'policies':>8} {'scored/valid':>13}  {'pos RMSE':>19}  "
              f"{'vel RMSE':>17}  {'ori RMSE deg':>17}   {'rate':>6}  {'95% CI':>16}")
    bar = "=" * len(header)
    lines += ["", bar, "  PER-MOTION TRACKING ERROR  (mean +/- std over the reps that LANDED)",
              "  +/- is the spread across individual reps (seed-to-seed AND rep-to-rep combined).",
              "  rate/CI use ALL valid reps as the denominator -- conditioning those on landing",
              "  would force every rate to 1.00.", bar,
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
        sc = scored_reps(mr)
        pm, ps = _stat([r["pos_rmse"] for r in sc])
        vm, vs = _stat([r["vel_rmse"] for r in sc])
        om, osd = _stat([r["ori_rmse_deg"] for r in sc])
        n_land = len(sc)
        rate, lo, hi = _wilson(n_land, len(mr))
        lines.append(f"  {m:<32} {n_pol:>8} {n_land:>6}/{len(mr):<6}  {_fmt(pm, ps, 8, 4)}  "
                     f"{_fmt(vm, vs, 7, 3)}  {_fmt(om, osd, 7, 2)}   "
                     f"{rate:>6.2f}  [{lo:>5.2f}, {hi:>5.2f}]")
        summary_rows.append({
            "motion": m, "policies": n_pol, "valid_reps": len(mr), "scored_reps": n_land,
            "pos_rmse_mean": pm, "pos_rmse_std": ps,
            "vel_rmse_mean": vm, "vel_rmse_std": vs,
            "ori_rmse_deg_mean": om, "ori_rmse_deg_std": osd,
            "landed": n_land, "success_rate": rate, "ci_low": lo, "ci_high": hi,
        })
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


# Per-motion error kept SPLIT into its three components rather than summed into
# one scalar: joint position, joint velocity, and anchor orientation fail in
# different ways and a single number hides which one moved. Each is normalized by
# a fixed physical scale (see combined_error) so the three are comparable to each
# other and across motions. Seeds are pooled: one row per motion, mean over the
# policies that tracked it.
def summarize_error_by_motion(crows):
    lines = []
    header = (f"  {'motion':<32} {'seeds':>5}  {'joint pos':>18}  {'joint vel':>18}  "
              f"{'anchor ori':>18}")
    bar = "=" * len(header)
    lines += ["", bar, "  PER-MOTION NORMALIZED ERROR  (split by component, mean +/- std over seeds)",
              f"  joint pos = rmse/ROM   joint vel = rmse/{combined_error_v_max():g}   "
              "anchor ori = rmse/180deg",
              "  Scored over LANDED reps only; a motion no policy lands has no row here.",
              "  +/- is the SEED-to-seed spread (n = seeds), not the rep-to-rep spread", bar,
              header, "  " + "-" * (len(header) - 2)]

    by_motion = {}
    for r in crows:
        by_motion.setdefault(r["motion"], []).append(r)

    summary_rows = []
    for m, mr in by_motion.items():
        pm, ps = _stat([r["pos_term"] for r in mr])
        vm, vs = _stat([r["vel_term"] for r in mr])
        om, osd = _stat([r["ori_term"] for r in mr])
        lines.append(f"  {m:<32} {len(mr):>5}  {pm:>8.4f} +/- {ps:<7.4f}  "
                     f"{vm:>8.4f} +/- {vs:<7.4f}  {om:>8.4f} +/- {osd:<7.4f}")
        summary_rows.append({"motion": m, "seeds": len(mr),
                             "joint_pos_term": pm, "joint_pos_std": ps,
                             "joint_vel_term": vm, "joint_vel_std": vs,
                             "anchor_ori_term": om, "anchor_ori_std": osd})
    lines += ["  " + "-" * (len(header) - 2), bar, ""]
    return "\n".join(lines), summary_rows


# V_MAX lives in combined_error; read it lazily so the import stays deferred
def combined_error_v_max():
    import combined_error
    return combined_error.V_MAX


def write_csv(path, rows, fields):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


# Read back a runs.csv written by a previous sweep, restoring types, so the
# reports can be regenerated (new columns, different thresholds) without paying
# for another 40-minute simulation pass. Everything the summaries need is in
# there; only the combined/normalized terms are recomputed from the h5 logs.
_BOOL_FIELDS = ("ok", "standing_ok", "landed", "land_censored")
_STR_FIELDS = ("policy", "motion", "reason", "land_reason", "anchor")


def read_runs_csv(path):
    import csv
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if v is None or v == "":
                    row[k] = None
                elif k in _BOOL_FIELDS:
                    row[k] = (v == "True")
                elif k in _STR_FIELDS:
                    row[k] = v
                elif k == "run_idx":
                    row[k] = int(v)
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policies", nargs="+", default=None, help="Policies to test (default: all pulled).")
    p.add_argument("--runs", type=int, default=15, help="Chained repetitions per policy (default 15).")
    p.add_argument("--dry-run", action="store_true", help="Test just the first policy (unless --policies given).")
    p.add_argument("--view", action="store_true", help="Show the sim viewer (real-time) instead of headless.")
    p.add_argument("--outdir", default=os.path.join(SWEEP_DIR, "results"), help="Where to write results.")
    p.add_argument("--report-only", action="store_true", dest="report_only",
                   help="Skip simulation: rebuild every report/CSV from the runs.csv already in "
                        "--outdir. Use after changing the summary tables.")
    p.add_argument("--reanalyze", action="store_true",
                   help="Skip simulation: re-score the existing h5 logs + sidecars (needed when a "
                        "METRIC changes, e.g. --track-lead-in; --report-only only re-tabulates).")
    p.add_argument("--indir", default=None,
                   help="Where --reanalyze/--report-only READ sidecars and runs.csv from "
                        "(default: --outdir). Set it to score an existing sweep into a fresh "
                        "--outdir without touching the original.")
    p.add_argument("--track-lead-in", type=float, default=0.0, dest="lead_in",
                   help="Seconds to drop from the START of each track window, where the robot is "
                        "still converging onto the trajectory from its home pose (~0.3-0.5s here). "
                        "Applied identically to every rep. Default 0.0 (score the whole motion).")
    # landing detection
    p.add_argument("--land-hold", type=float, default=3.0, dest="land_hold",
                   help="Seconds observed after the motion (0 disables landing detection). Sets how "
                        "long hold-time can be measured before it is censored. Default 3.0.")
    p.add_argument("--land-verdict-s", type=float, default=analyze_run.LAND_VERDICT_S, dest="land_verdict_s",
                   help="Touchdown window (s) the binary landed/fell verdict is read from -- short on "
                        f"purpose: it scores arrival, not stability. Default {analyze_run.LAND_VERDICT_S}.")
    p.add_argument("--land-ori-err-deg", type=float, default=analyze_run.LAND_ORI_ERR_DEG, dest="land_ori_err",
                   help="Max orientation error (deg) vs the reference's FINAL frame at the end of the "
                        f"settle window. Default {analyze_run.LAND_ORI_ERR_DEG}.")
    p.add_argument("--land-gyro-rms", type=float, default=analyze_run.LAND_GYRO_RMS, dest="land_gyro",
                   help=f"Max base angular-rate RMS (rad/s) to count as settled. Default {analyze_run.LAND_GYRO_RMS}.")
    p.add_argument("--land-joint-vel-rms", type=float, default=analyze_run.LAND_JOINT_VEL_RMS, dest="land_joint_vel",
                   help=f"Max joint-velocity RMS (rad/s) to count as settled. Default {analyze_run.LAND_JOINT_VEL_RMS}.")
    args = p.parse_args()

    land_thresholds = {"verdict_s": args.land_verdict_s, "ori_err_max": args.land_ori_err,
                       "gyro_max": args.land_gyro, "joint_vel_max": args.land_joint_vel}

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
    print(f"[sweep] landing:  verdict at touchdown ({args.land_verdict_s}s), observed for {args.land_hold}s")
    print(f"[sweep]           ori_err<={args.land_ori_err}deg (vs ref FINAL frame)  "
          f"gyro_rms<={args.land_gyro}  joint_vel_rms<={args.land_joint_vel}")

    rows = []
    read_dir = args.indir or args.outdir
    if args.reanalyze:
        for pol in policies:
            sidecar = os.path.join(read_dir, "sidecars", f"{pol}.json")
            pol_rows = analyze(sidecar, f"sweep_{pol}", land_thresholds=land_thresholds,
                               lead_in=args.lead_in)
            rows.extend(pol_rows)
            n_sc = sum(1 for r in pol_rows if r.get("ok") and r.get("standing_ok") and r.get("landed"))
            print(f"[sweep] rescored {pol}: {n_sc}/{len(pol_rows)} landed "
                  f"(lead_in={args.lead_in:.2f}s)")
        policies = [p for p in policies
                    if any(r.get("policy") == p for r in rows)]

    if args.report_only:
        runs_csv = os.path.join(read_dir, "runs.csv")
        rows = read_runs_csv(runs_csv)
        # report on whatever that sweep actually covered, in its original order
        seen = []
        for r in rows:
            if r.get("policy") and r["policy"] not in seen:
                seen.append(r["policy"])
        policies = [p for p in policies if p in seen] if args.policies else seen
        print(f"[sweep] report-only: {len(rows)} reps, {len(policies)} policies from {runs_csv}")

    for pol in (policies if not (args.report_only or args.reanalyze) else []):
        print(f"\n[sweep] === {pol}  ({runs} chained reps) ===")
        t0 = time.time()
        sidecar, log_base = run_policy(pol, runs, dirs, view=args.view, land_hold=args.land_hold)
        pol_rows = analyze(sidecar, log_base, land_thresholds=land_thresholds, lead_in=args.lead_in)
        rows.extend(pol_rows)
        dt = time.time() - t0
        n_ok = sum(r.get("ok") and r.get("standing_ok") for r in pol_rows)
        n_land = sum(r.get("ok") and r.get("standing_ok") and r.get("landed") for r in pol_rows)
        for r in pol_rows:
            if r.get("ok"):
                flag = "" if r.get("standing_ok") else "  [FLAGGED: not standing]"
                hold = r.get("land_hold_time_s")
                held = "" if hold is None else f" held={hold:.2f}s{'+' if r.get('land_censored') else ''}"
                if r.get("landed"):
                    land = f" LANDED (ori err={r.get('land_ori_err_deg', float('nan')):.0f}deg){held}"
                else:
                    land = f" FELL ({r.get('land_reason')}){held}"
                print(f"[sweep]   rep {r['run_idx']:>2}: pos={r['pos_rmse']:.4f} "
                      f"vel={r['vel_rmse']:.3f} ori={r.get('ori_rmse_deg'):.1f} "
                      f"cov={r.get('coverage'):.2f}{land}{flag}")
            else:
                print(f"[sweep]   rep {r.get('run_idx')}: FAILED: {r.get('reason')}")
        print(f"[sweep]   {n_ok}/{len(pol_rows)} valid reps, {n_land}/{n_ok} landed, in {dt:.0f}s")

    # write per-run + summary artifacts
    run_fields = ["policy", "run_idx", "ok", "standing_ok", "gate_tilt_deg",
                  "pos_rmse", "pos_mae", "pos_max", "vel_rmse", "vel_mae", "vel_max",
                  "ori_rmse_deg", "ori_max_deg", "coverage", "n_samples",
                  "base_pose_error", "base_pos_term", "base_ori_term",
                  "base_pos_rmse", "base_pos_max", "base_height_rmse", "base_height_max",
                  "base_lin_vel_rmse", "base_ang_vel_rmse", "base_ori_rmse_deg", "base_ori_max_deg",
                  "lead_in_s",
                  "landed", "land_hold_time_s", "land_censored", "land_ori_err_deg",
                  "land_tilt_final_deg", "land_tilt_max_deg", "land_gyro_rms",
                  "land_joint_vel_rms", "land_n_samples", "land_reason", "end_tilt_deg",
                  "motion", "reason"]
    write_csv(os.path.join(args.outdir, "runs.csv"), rows, run_fields)

    # landing success rate (the "can it do the motion at all?" statistic)
    success_report, success_rows = summarize_success(rows, policies)
    print(success_report)
    write_csv(os.path.join(args.outdir, "success.csv"), success_rows,
              ["policy", "valid", "landed", "success_rate", "ci_low", "ci_high",
               "hold_time_median_s", "n_censored", "ori_err_deg_mean", "end_tilt_deg_mean",
               "gate_fails"])

    report, summary_rows = summarize(rows, policies)
    print(report)
    sum_fields = ["policy", "valid", "total", "landed", "success_rate",
                  "success_ci_low", "success_ci_high",
                  "pos_rmse_mean", "pos_rmse_std",
                  "vel_rmse_mean", "vel_rmse_std", "ori_rmse_deg_mean", "ori_rmse_deg_std"]
    write_csv(os.path.join(args.outdir, "summary.csv"), summary_rows, sum_fields)

    # floating-base ground-truth error (sim-only; empty for pre-base_state logs)
    base_fields = ["base_pose_error_mean", "base_pose_error_std",
                   "base_pos_rmse_mean", "base_pos_rmse_std",
                   "base_height_rmse_mean", "base_height_rmse_std",
                   "base_lin_vel_rmse_mean", "base_lin_vel_rmse_std",
                   "base_ang_vel_rmse_mean", "base_ang_vel_rmse_std",
                   "base_ori_rmse_deg_mean", "base_ori_rmse_deg_std"]
    base_report, base_rows = summarize_base(rows, policies)
    base_motion_report, base_motion_rows = summarize_base_by_motion(rows)
    if base_report:
        print(base_report)
        write_csv(os.path.join(args.outdir, "base_error.csv"), base_rows,
                  ["policy", "reps"] + base_fields)
        print(base_motion_report)
        write_csv(os.path.join(args.outdir, "base_error_by_motion.csv"), base_motion_rows,
                  ["motion", "seeds", "reps", "base_pos_term", "base_pos_term_std",
                   "base_ori_term", "base_ori_term_std"] + base_fields)
    else:
        print("[sweep] base-error table skipped (no base_state in these logs)")

    # per-motion rollup (all seeds pooled)
    motion_report, motion_rows = summarize_by_motion(rows)
    print(motion_report)
    write_csv(os.path.join(args.outdir, "summary_by_motion.csv"), motion_rows,
              ["motion", "policies", "valid_reps",
               "pos_rmse_mean", "pos_rmse_std", "vel_rmse_mean", "vel_rmse_std",
               "ori_rmse_deg_mean", "ori_rmse_deg_std",
               "landed", "success_rate", "ci_low", "ci_high"])

    # combined single-number metric (fixed-scale normalized + summed), computed
    # from the logs/sidecars this run just produced. Lazy import avoids the
    # combined_error <-> run_batch circular import at module load.
    combined_report, err_by_motion_report = "", ""
    try:
        import combined_error
        # per_motion=False: its own rollup collapses pos+vel+ori into one scalar.
        # We print the split-by-component version below instead.
        # score only the reps that landed, matching the tables above
        landed_filter = {}
        for r in scored_reps(rows):
            landed_filter.setdefault(r["policy"], set()).add(int(r["run_idx"]))
        combined_report, crows = combined_error.report(policies, results_dir=args.outdir,
                                                       per_motion=False, rep_filter=landed_filter)
        if crows:
            err_by_motion_report, err_rows = summarize_error_by_motion(crows)
            print(err_by_motion_report)
            write_csv(os.path.join(args.outdir, "error_by_motion.csv"), err_rows,
                      ["motion", "seeds", "joint_pos_term", "joint_pos_std",
                       "joint_vel_term", "joint_vel_std", "anchor_ori_term", "anchor_ori_std"])
    except Exception as e:
        print(f"[sweep] combined-error step skipped: {e}")

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(success_report + "\n" + report + "\n" + base_report + "\n" + base_motion_report
                + "\n" + motion_report + "\n" + combined_report + "\n" + err_by_motion_report + "\n")
    print(f"[sweep] wrote {os.path.join(args.outdir, 'runs.csv')}, success.csv, summary.csv, "
          f"base_error.csv, base_error_by_motion.csv, summary_by_motion.csv, combined_error.csv, "
          f"error_by_motion.csv, summary.txt")


if __name__ == "__main__":
    main()

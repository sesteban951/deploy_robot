##
#
# Tabulate velocity-tracking staircase results from simulation_det_omni.py.
#
# For each --profile log, computes the steady-state achieved twist per commanded
# setpoint (averaged over whole gait cycles, dropping the first cycle as transient)
# and prints a commanded-vs-achieved table. With --latex it also writes a booktabs
# LaTeX table (needs \usepackage{booktabs}) next to each log.
#
# Usage:
#   python deploy/simulation/velocity_tracking_table.py            # all logs/veltrack_*.npz
#   python deploy/simulation/velocity_tracking_table.py --log logs/veltrack_fwd.npz --latex
#
##

import argparse
import glob
import os

import numpy as np

AXES = ["vx", "vy", "wz"]
UNITS = ["m/s", "m/s", "rad/s"]
LATEX_SYM = [r"v_x", r"v_y", r"\omega_z"]

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR", ".")

# per-profile caption context (fixed axes held during the sweep)
CAPTIONS = {
    "fwd":  "Forward crawl: commanded vs.\\ achieved steady-state twist.",
    "bwd":  "Backward crawl: commanded vs.\\ achieved steady-state twist.",
    "crab": "Lateral crab (at $v_x^{\\mathrm{cmd}}=0.20$ m/s): commanded vs.\\ achieved twist.",
    "turn": "In-place turn: commanded vs.\\ achieved steady-state twist.",
    "all":  "Omnidirectional crawl: commanded vs.\\ achieved steady-state twist.",
}


# steady-state achieved twist per setpoint: mean/std over whole gait cycles at the
# end of each hold (drop the first cycle as transient)
def analyze(d):
    t, cmd, ach, seg = d["t"], d["cmd"], d["ach"], d["seg"].astype(int)
    period, hold = float(d["period_s"]), float(d["hold_s"])
    steady_dur = max(period, hold - period)
    rows = []
    for i in sorted(set(int(s) for s in seg if s >= 0)):
        idx = np.where(seg == i)[0]
        tail = idx[t[idx] >= t[idx].max() - steady_dur]
        rows.append({"cmd": cmd[idx[0]], "mean": ach[tail].mean(axis=0), "std": ach[tail].std(axis=0)})
    return rows


def swept_axis(rows):
    c = np.array([r["cmd"] for r in rows])
    return int(np.argmax(c.max(axis=0) - c.min(axis=0)))


def print_plaintext(profile, rows, ax, n_cyc):
    print(f"\n=== {profile}  (swept: {AXES[ax]}; steady-state mean over {n_cyc} gait cycles) ===")
    head = f"{'cmd '+AXES[ax]:>9} | {'ach vx':>8} {'ach vy':>8} {'ach wz':>8} | {'err '+AXES[ax]:>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        c, m = r["cmd"], r["mean"]
        print(f"{c[ax]:>+9.2f} | {m[0]:>+8.3f} {m[1]:>+8.3f} {m[2]:>+8.3f} | {m[ax]-c[ax]:>+9.3f}")
    err = np.array([r["mean"][ax] - r["cmd"][ax] for r in rows])
    print(f"RMSE({AXES[ax]}) = {np.sqrt(np.mean(err**2)):.4f} {UNITS[ax]}   "
          f"mean|err| = {np.mean(np.abs(err)):.4f}")


def latex_table(profile, rows, ax, n_cyc):
    cap = (CAPTIONS.get(profile, f"{profile}: commanded vs.\\ achieved twist.")
           + f" Achieved = mean over {n_cyc} steady-state gait cycles.")
    lines = []
    lines.append(r"% requires \usepackage{booktabs}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{cap}}}")
    lines.append(f"  \\label{{tab:veltrack_{profile}}}")
    lines.append(r"  \begin{tabular}{rrrrr}")
    lines.append(r"    \toprule")
    lines.append(f"    ${LATEX_SYM[ax]}^{{\\mathrm{{cmd}}}}$ ({UNITS[ax]}) & "
                 f"$v_x$ (m/s) & $v_y$ (m/s) & $\\omega_z$ (rad/s) & err (${LATEX_SYM[ax]}$) \\\\")
    lines.append(r"    \midrule")
    for r in rows:
        c, m = r["cmd"], r["mean"]
        ach = " & ".join(f"${m[k]:+.3f}$" for k in range(3))
        lines.append(f"    ${c[ax]:+.2f}$ & {ach} & ${m[ax] - c[ax]:+.3f}$ \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Tabulate velocity-tracking staircase results.")
    ap.add_argument("--log", type=str, default=None,
                    help="single npz log. Default: all logs/veltrack_*.npz.")
    ap.add_argument("--latex", action="store_true", help="also write a .tex table next to each log.")
    args = ap.parse_args()

    logs = ([args.log] if args.log else
            sorted(glob.glob(os.path.join(ROOT_DIR, "logs", "veltrack_*.npz"))))
    if not logs:
        ap.error("no logs found; run simulation_det_omni.py --profile <p> first.")

    for log in logs:
        d = np.load(log, allow_pickle=True)
        profile = str(d["profile"])
        rows = analyze(d)
        if not rows:
            print(f"{log}: no setpoint segments."); continue
        ax = swept_axis(rows)
        period, hold = float(d["period_s"]), float(d["hold_s"])
        n_cyc = max(1, int(round((hold - period) / period)))   # steady cycles averaged
        print_plaintext(profile, rows, ax, n_cyc)
        if args.latex:
            tex = os.path.splitext(log)[0] + "_table.tex"
            with open(tex, "w") as f:
                f.write(latex_table(profile, rows, ax, n_cyc) + "\n")
            print(f"  -> wrote {tex}")


if __name__ == "__main__":
    main()

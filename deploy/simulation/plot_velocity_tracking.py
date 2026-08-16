##
#
# Analyze / plot a velocity-tracking staircase log from simulation_det_omni.py.
#
# Reads the npz written by a `--profile` run (commanded vs achieved body-frame
# twist per control step + a setpoint segment id), computes the steady-state
# achieved velocity for each setpoint (averaged over the tail of its hold),
# prints a tracking-error table, and saves a figure:
#   - commanded vs achieved time series for vx, vy, wz
#   - steady-state commanded-vs-achieved scatter for the swept axis (ideal = y=x)
#
# Saves a PNG next to each log AND opens the figure(s) in a window when a display
# is available (use --no-show to only save).
#
# Usage:
#   python deploy/simulation/plot_velocity_tracking.py [--log path.npz] [--no-show]
# With no --log it plots ALL logs/veltrack_*.npz (one window each).
#
##

import argparse
import glob
import os
import sys

import numpy as np

# Use the default (interactive) matplotlib backend when a display is available so
# figures can pop up; fall back to Agg (save-only) when headless.
try:
    import matplotlib
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Fonts: real LaTeX Computer Modern via usetex when a LaTeX toolchain is present;
# otherwise matplotlib's bundled Computer Modern (cmr10) so it still looks CM.
if HAVE_MPL:
    import shutil
    try:
        if shutil.which("latex") and shutil.which("dvipng"):
            matplotlib.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}",
            })
        else:
            matplotlib.rcParams.update({
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["cmr10", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "axes.unicode_minus": False,
            })
    except Exception:
        pass
    # larger fonts for paper readability
    matplotlib.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

AXES = ["vx", "vy", "wz"]
UNITS = ["m/s", "m/s", "rad/s"]

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR", ".")


# per-setpoint steady-state: mean/std of achieved over whole gait cycles at the end
# of each hold (drop the first gait cycle as the transient). Whole-cycle averaging
# cancels the gait's within-cycle velocity oscillation.
def analyze(d):
    t, cmd, ach, seg = d["t"], d["cmd"], d["ach"], d["seg"].astype(int)
    period = float(d["period_s"])
    hold = float(d["hold_s"])
    steady_dur = max(period, hold - period)   # skip 1 cycle; remaining is integer cycles
    rows = []
    for i in sorted(set(int(s) for s in seg if s >= 0)):
        idx = np.where(seg == i)[0]
        seg_end = t[idx].max()
        tail = idx[t[idx] >= seg_end - steady_dur]   # last whole cycle(s)
        rows.append({
            "seg": i,
            "cmd": cmd[idx[0]],                 # constant over the segment
            "mean": ach[tail].mean(axis=0),
            "std": ach[tail].std(axis=0),
        })
    return rows


# the axis the profile sweeps = the one whose commanded value varies the most
def swept_axis(rows):
    cmds = np.array([r["cmd"] for r in rows])
    return int(np.argmax(cmds.max(axis=0) - cmds.min(axis=0)))


# classify a commanded twist into a crawl section + its active axis (None = idle)
def section_of(c):
    if c[0] >= 0.10:
        return "Forward", 0
    if c[0] <= -0.05:
        return "Backward", 0
    if abs(c[2]) > 0.10:
        return "Turn", 2
    return None, None


# print per-section steady-state tracking RMSE for a combo run
def combo_summary(rows):
    print("\ncombo sections (steady-state RMSE on the active axis):")
    order, groups = [], {}
    for r in rows:
        lab, axk = section_of(r["cmd"])
        if lab is None:
            continue
        if lab not in groups:
            order.append((lab, axk)); groups[lab] = []
        groups[lab].append(r)
    for lab, axk in order:
        rs = groups[lab]
        err = np.array([rr["mean"][axk] - rr["cmd"][axk] for rr in rs])
        print(f"  {lab:9s} ({AXES[axk]:>3}): {len(rs)} setpoints  "
              f"RMSE={np.sqrt(np.mean(err**2)):.4f} {UNITS[axk]}  "
              f"mean|err|={np.mean(np.abs(err)):.4f}")


# 3x1 traversal figure: ONE continuous run whose command visits the forward,
# backward, and turn regions. Panels are the twist components vx, vy, wz over the
# shared timeline; the three motion regions are shaded and labeled.
def combo_figure(d, out_png):
    COMMANDED, ACHIEVED, GRID, AXIS = "#0b0b0b", "#2a78d6", "#e1e0d9", "#c3c2b7"
    ACH_COLORS = ["#d81b60", "#1e88e5", "#ffc107"]   # per-component vel color (vx, vy, wz)
    BANDS = ("#e6e4de", "#f3f2ee")   # alternating region shades (adjacent regions stay distinct)
    SYM = [r"$v^x$", r"$v^y$", r"$\omega^z$"]
    t, cmd, ach, seg = d["t"], d["cmd"], d["ach"], d["seg"].astype(int)

    segids = sorted(set(int(s) for s in seg if s >= 0))

    # per-setpoint steady-state = mean of the actual velocity over the last whole
    # gait cycle of each hold (the settled value; whole-cycle averaging cancels the
    # gait's velocity swing -- and its endpoints are reliable, unlike a moving avg)
    period, hold = float(d["period_s"]), float(d["hold_s"])
    steady_dur = max(period, hold - period)

    def steady_mean(seg_id, axk):
        idx = np.where(seg == seg_id)[0]
        tail = idx[t[idx] >= t[idx].max() - steady_dur]
        return float(ach[tail, axk].mean())

    # region spans (label, t_start, t_end) for shading + labels. The leading
    # all-zeros settle (seg -1) is its own "Idle" region; then the motion regions.
    regions = []
    settle_idx = np.where(seg == -1)[0]
    if len(settle_idx):
        regions.append(("Idle", float(t[settle_idx].min()), float(t[settle_idx].max())))
    for i in segids:
        idx = np.where(seg == i)[0]
        lab, _ = section_of(cmd[idx[0]])
        if lab is None:
            continue
        t0i, t1i = float(t[idx].min()), float(t[idx].max())
        if regions and regions[-1][0] == lab:
            regions[-1] = (lab, regions[-1][1], t1i)
        else:
            regions.append((lab, t0i, t1i))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7.5, 6.2))
    try:
        fig.canvas.manager.set_window_title("veltrack: traversal (fwd -> bwd -> turn)")
    except Exception:
        pass

    for k in range(3):
        p = axs[k]
        for ri, (lab, t0, t1) in enumerate(regions):   # shade regions (alternating tone,
            p.axvspan(t0, t1, color=BANDS[ri % 2], alpha=0.6, zorder=0)  # slightly transparent
        # actual (measured) body velocity trace + its per-setpoint steady-state mean
        p.plot(t, ach[:, k], color=ACH_COLORS[k], lw=1.3, alpha=0.40, label="Actual")
        xs, ys = [], []
        for i in sorted(set(int(s) for s in seg)):   # include the idle settle (seg -1)
            idx = np.where(seg == i)[0]
            ma = steady_mean(i, k)
            xs += [float(t[idx].min()), float(t[idx].max())]
            ys += [ma, ma]
        p.plot(xs, ys, color=ACH_COLORS[k], lw=2.4, label="Mean")
        p.plot(t, cmd[:, k], color=COMMANDED, ls="--", lw=1.5, label="Command")
        p.set_ylabel(f"{SYM[k]} ({UNITS[k]})")
        p.grid(True, color=GRID, lw=0.6)
        p.set_axisbelow(True)
        p.margins(x=0.0, y=0.01)   # y-limits = 1% above max / below min of the data
        for sp in p.spines.values():
            sp.set_color(AXIS)
        p.tick_params(colors="#52514e")

    # region names across the top of the figure
    for lab, t0, t1 in regions:
        axs[0].text(0.5 * (t0 + t1), 1.04, lab, transform=axs[0].get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=15, fontweight="bold", color="#52514e")
    axs[-1].set_xlabel("Time (sec)")

    # x-ticks every 10 s including 0 (start the axis at t=0 so the 0 tick shows)
    from matplotlib.ticker import MultipleLocator
    for p in axs:
        p.xaxis.set_major_locator(MultipleLocator(10))
    # round the right limit up to the next 10 s so the final tick (e.g. 50) shows
    axs[0].set_xlim(0.0, float(np.ceil(t.max() / 10.0) * 10.0))   # sharex -> all panels

    fig.tight_layout(rect=[0, 0, 1, 0.95])   # leave the top strip for the region labels
    # legend inside the plot, and draggable: reposition it in the live window, then
    # re-save from the window toolbar (renders in LaTeX Computer Modern via usetex)
    # neutral style legend (color now encodes the component, so the legend keys
    # the line STYLE: Actual = thin, Mean = bold, Command = dashed)
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], color="0.45", lw=1.3, alpha=0.7, label="Actual"),
        Line2D([0], [0], color="0.15", lw=2.4, label="Mean"),
        Line2D([0], [0], color=COMMANDED, lw=1.5, ls="--", label="Command"),
    ]
    leg = axs[0].legend(handles=leg_handles, loc="upper right", ncol=3, framealpha=0.9, fontsize=12)
    try:
        leg.set_draggable(True)
    except Exception:
        pass
    fig.savefig(out_png, dpi=150)
    print(f"saved figure -> {out_png}")
    return fig


def print_table(rows, ax):
    print(f"\n{'seg':>3}  {'commanded [vx, vy, wz]':>24}   {'achieved (mean)':>24}   err[{AXES[ax]}]")
    for r in rows:
        c, m = r["cmd"], r["mean"]
        cs = f"[{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}]"
        ms = f"[{m[0]:+.3f} {m[1]:+.3f} {m[2]:+.3f}]"
        print(f"{r['seg']:3d}  {cs:>24}   {ms:>24}   {m[ax]-c[ax]:+.3f}")

    print("\nper-axis error over all setpoints (achieved - commanded):")
    for k in range(3):
        err = np.array([r["mean"][k] - r["cmd"][k] for r in rows])
        tag = "  <- swept" if k == ax else ("  (should be ~0: cross-axis leak/drift)"
                                            if all(abs(r["cmd"][k]) < 1e-6 for r in rows) else "")
        print(f"  {AXES[k]:>3} ({UNITS[k]:>5}): RMSE={np.sqrt(np.mean(err**2)):.4f}  "
              f"mean|err|={np.mean(np.abs(err)):.4f}{tag}")


def plot(d, rows, ax, out_png):
    profile = str(d["profile"])
    t, cmd, ach = d["t"], d["cmd"], d["ach"]
    # moving average over one gait cycle to show the achieved trend through the wag
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    win = max(1, int(round(float(d["period_s"]) / dt)))
    box = np.ones(win) / win

    def smooth(y):
        return np.convolve(y, box, mode="same") if win > 1 else y

    fig, axs = plt.subplots(4, 1, figsize=(9, 11),
                            gridspec_kw={"height_ratios": [1, 1, 1, 1.5]})
    try:
        fig.canvas.manager.set_window_title(f"veltrack: {profile}")
    except Exception:
        pass
    for k in range(3):
        axs[k].plot(t, cmd[:, k], "k--", lw=1.5, label="commanded")
        axs[k].plot(t, ach[:, k], lw=0.6, alpha=0.35, label="achieved (raw)")
        axs[k].plot(t, smooth(ach[:, k]), lw=1.6, label="achieved (1-cycle avg)")
        axs[k].set_ylabel(f"{AXES[k]} ({UNITS[k]})")
        axs[k].grid(alpha=0.3)
        axs[k].legend(loc="upper right", fontsize=8)
    axs[2].set_xlabel("time (s)")

    # steady-state commanded-vs-achieved (mean points; no error bars -- the run is
    # deterministic, and the per-sample spread is gait oscillation, not uncertainty)
    c = np.array([r["cmd"][ax] for r in rows])
    m = np.array([r["mean"][ax] for r in rows])
    sc = axs[3]
    lo, hi = min(c.min(), m.min()), max(c.max(), m.max())
    pad = 0.05 * (hi - lo + 1e-6)
    sc.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", label="ideal (y=x)")
    sc.plot(c, m, "o", label="steady state")
    sc.set_xlabel(f"commanded {AXES[ax]} ({UNITS[ax]})")
    sc.set_ylabel(f"achieved {AXES[ax]} ({UNITS[ax]})")
    sc.grid(alpha=0.3)
    sc.legend(fontsize=8)
    sc.set_aspect("equal", "box")

    fig.suptitle(f"velocity tracking - profile={profile}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"saved figure -> {out_png}")
    return fig


# build + save the figure for one log; optionally open it right away. Returns
# True if a figure was produced. Importable so the sim can auto-plot after a run.
def render_log(log_path, show=False):
    if not HAVE_MPL:
        print("(matplotlib unavailable; cannot render figure)")
        return False
    d = np.load(log_path, allow_pickle=True)
    rows = analyze(d)
    if not rows:
        print(f"  {log_path}: no setpoint segments (only settle?).")
        return False
    png = os.path.splitext(log_path)[0] + ".png"
    if str(d["profile"]) == "combo":
        combo_summary(rows)        # 3x1 traversal figure (fwd -> bwd -> turn)
        combo_figure(d, png)
    else:
        ax = swept_axis(rows)      # per-axis staircase figure
        print_table(rows, ax)
        plot(d, rows, ax, png)
    if show:
        if matplotlib.get_backend().lower() == "agg":
            print("(no display detected -> saved PNG only)")
        else:
            print("opening figure; close the window to finish.")
            plt.show()
    return True


def main():
    ap = argparse.ArgumentParser(description="Plot velocity-tracking staircase log(s).")
    ap.add_argument("--log", type=str, default=None,
                    help="npz from a --profile run. Default: ALL logs/veltrack_*.npz.")
    ap.add_argument("--no-show", action="store_true",
                    help="save PNGs only; do not open figure windows.")
    args = ap.parse_args()

    if not HAVE_MPL:
        ap.error("matplotlib is not available in this environment.")

    logs = ([args.log] if args.log else
            sorted(glob.glob(os.path.join(ROOT_DIR, "logs", "veltrack_*.npz")),
                   key=os.path.getmtime))
    if not logs:
        ap.error("no --log given and no logs/veltrack_*.npz found.")

    made = 0
    for log in logs:
        print(f"\nlog: {log}")
        if render_log(log, show=False):
            made += 1

    if made and not args.no_show:
        if matplotlib.get_backend().lower() == "agg":
            print("\n(no display detected -> saved PNGs only; open them in an image viewer)")
        else:
            print(f"\nopening {made} figure window(s); close them to exit.")
            plt.show()


if __name__ == "__main__":
    main()

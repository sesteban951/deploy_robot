##
#
# Plot a KCF live trace: reference vs. KCF-MPC plan/commands vs. measured (mocap) pelvis state.
# Usage: python logs/plot_kcf.py [--trace logs/kcf/live_traces/<trace>.npz] [--out fig.png]
# Default: the newest trace; the figure is shown and saved next to the trace as <trace>_plot.png.
#
##

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


# categorical slots 1-3 of the default palette, fixed order: reference, KCF-MPC, mocap
C_REF, C_MPC, C_MOC = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DT = 0.02   # [sec] KCF step


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Plot a KCF-MPC live trace.")
    parser.add_argument("--trace", type=str, default=None, help="trace .npz (default: newest in logs/kcf/live_traces)")
    parser.add_argument("--out", type=str, default=None, help="output image (default: next to the trace)")
    parser.add_argument("--plan-every", type=int, default=25, help="draw every Nth MPC plan on the top view")
    args = parser.parse_args()

    # load the trace (newest by default)
    trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kcf", "live_traces")
    traces = glob.glob(os.path.join(trace_dir, "*.npz"))
    if args.trace is None and not traces:
        raise FileNotFoundError(f"No traces in {trace_dir}.")
    path = args.trace or max(traces, key=os.path.getmtime)
    z = np.load(path, allow_pickle=True)
    print(f"Plotting [{path}].")

    # solves: measured state (mocap pose, body twist), reference at the next step, planned twists, command
    n = len(z["state_time"])
    t = z["state_time"] - z["state_time"][0]
    meas = z["measured_observation"].astype(float)
    ref = z["reference"].astype(float)
    plan = z["optimized_controls"].astype(float)
    cmd = z["command"].astype(float)
    yaw_meas = np.unwrap(np.arctan2(meas[:, 2], meas[:, 3]))
    yaw_ref = np.unwrap(np.arctan2(ref[:, 0, 2], ref[:, 0, 3]))

    # style
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 8.5), layout="constrained", facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.4, 1, 1])
    fig.suptitle(f"KCF-MPC trace {os.path.basename(path)}: {n} solves, {t[-1]:.1f} s", color=INK)

    # top view: every Nth plan, dead-reckoned from the measured pose with its planned body twists
    ax_xy = fig.add_subplot(gs[:, 0])
    for k in range(0, n, args.plan_every):
        x, y, th = meas[k, 0], meas[k, 1], yaw_meas[k]
        pts = [(x, y)]
        for vx, vy, wz in plan[k]:
            x += DT * (np.cos(th) * vx - np.sin(th) * vy)
            y += DT * (np.sin(th) * vx + np.cos(th) * vy)
            th += DT * wz
            pts.append((x, y))
        pts = np.array(pts)
        ax_xy.plot(pts[:, 0], pts[:, 1], color=C_MPC, lw=1.0, alpha=0.6,
                   label="KCF-MPC planned path (20 steps)" if k == 0 else None)
    ax_xy.plot(ref[:, 0, 0], ref[:, 0, 1], color=C_REF, lw=2, label="reference")
    ax_xy.plot(meas[:, 0], meas[:, 1], color=C_MOC, lw=2, label="mocap (measured)")
    ax_xy.plot(meas[0, 0], meas[0, 1], "o", ms=8, color=C_MOC, mec=SURFACE, mew=2)
    ax_xy.plot(meas[-1, 0], meas[-1, 1], "s", ms=8, color=C_MOC, mec=SURFACE, mew=2)
    ax_xy.annotate("start", (meas[0, 0], meas[0, 1]), xytext=(6, 6), textcoords="offset points", color=INK2)
    ax_xy.annotate("end", (meas[-1, 0], meas[-1, 1]), xytext=(6, -12), textcoords="offset points", color=INK2)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [m] (world)")
    ax_xy.set_ylabel("y [m] (world)")
    ax_xy.set_title("pelvis path, top view", color=INK, loc="left")
    ax_xy.legend(loc="best", frameon=False)

    # pose and twist over time, one quantity per axis (commands only exist for the twist)
    panels = [
        (gs[0, 1], "x [m]", ref[:, 0, 0], meas[:, 0], None),
        (gs[1, 1], "y [m]", ref[:, 0, 1], meas[:, 1], None),
        (gs[2, 1], "yaw [deg]", np.degrees(yaw_ref), np.degrees(yaw_meas), None),
        (gs[0, 2], "vx body [m/s]", ref[:, 0, 4], meas[:, 4], cmd[:, 0]),
        (gs[1, 2], "vy body [m/s]", ref[:, 0, 5], meas[:, 5], cmd[:, 1]),
        (gs[2, 2], "wz [rad/s]", ref[:, 0, 6], meas[:, 6], cmd[:, 2]),
    ]
    for i, (spec, label, r, m, c) in enumerate(panels):
        ax = fig.add_subplot(spec)
        ax.plot(t, r, color=C_REF, lw=2, label="reference")
        if c is not None:
            ax.plot(t, c, color=C_MPC, lw=1.5, label="KCF-MPC command")
        ax.plot(t, m, color=C_MOC, lw=1.5, label="mocap (measured)")
        ax.set_ylabel(label)
        if i in (2, 5):
            ax.set_xlabel("time since first solve [s]")
        if i in (0, 3):
            ax.legend(loc="best", frameon=False, fontsize=8)

    # save next to the trace (or --out) and show
    out = args.out or os.path.splitext(path)[0] + "_plot.png"
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"Saved [{out}].")
    plt.show()

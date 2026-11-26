##
#
# Quick plot script for HDF5 logs.
#
##

import argparse
import glob
import json
import math
import os
import sys
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt


# joint names in motor index order (matches models/g1/g1_*dof.xml and the Unitree SDK)
G1_29DOF_JOINT_NAMES = [
    "L hip pitch", "L hip roll", "L hip yaw", "L knee", "L ankle pitch", "L ankle roll",
    "R hip pitch", "R hip roll", "R hip yaw", "R knee", "R ankle pitch", "R ankle roll",
    "waist yaw", "waist roll", "waist pitch",
    "L shoulder pitch", "L shoulder roll", "L shoulder yaw", "L elbow",
    "L wrist roll", "L wrist pitch", "L wrist yaw",
    "R shoulder pitch", "R shoulder roll", "R shoulder yaw", "R elbow",
    "R wrist roll", "R wrist pitch", "R wrist yaw",
]
G1_23DOF_JOINT_NAMES = [
    "L hip pitch", "L hip roll", "L hip yaw", "L knee", "L ankle pitch", "L ankle roll",
    "R hip pitch", "R hip roll", "R hip yaw", "R knee", "R ankle pitch", "R ankle roll",
    "waist yaw",
    "L shoulder pitch", "L shoulder roll", "L shoulder yaw", "L elbow", "L wrist roll",
    "R shoulder pitch", "R shoulder roll", "R shoulder yaw", "R elbow", "R wrist roll",
]


GO2_JOINT_NAMES = [f"{leg} {j}" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]


# subplot title for joint i, e.g. "13: waist roll" (falls back to the index for unknown joint counts)
def joint_title(i: int, N: int) -> str:
    names = {29: G1_29DOF_JOINT_NAMES, 23: G1_23DOF_JOINT_NAMES, 12: GO2_JOINT_NAMES}.get(N)
    return f"{i}: {names[i]}" if names else f"joint {i}"


# search both logs/simulation and logs/hardware for the most recently modified .h5 file
def find_latest_log() -> str:
    logs_root = os.path.dirname(os.path.abspath(__file__))
    candidates = (glob.glob(os.path.join(logs_root, "simulation", "*.h5"))
                + glob.glob(os.path.join(logs_root, "hardware", "*.h5")))
    if not candidates:
        raise FileNotFoundError(f"No .h5 logs found under {logs_root}/simulation or {logs_root}/hardware.")
    return max(candidates, key=os.path.getmtime)


# build a (rows, cols) grid that fits N subplots, and return flat axes
def _joint_grid(N: int, title: str, figsize=(14, 9), suptitle_extra: str = ""):
    cols = min(6, N)
    rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, layout="constrained")
    fig.suptitle(f"{title}\n{suptitle_extra}" if suptitle_extra else title)
    axes = axes.flatten() if N > 1 else [axes]
    # hide any unused axes on the last row
    for ax in axes[N:]:
        ax.set_visible(False)
    return fig, axes[:N]


# print the experiment metadata (root attributes written by the logger)
def print_experiment_info(meta: dict):
    bar = "=" * 72
    sub = "-" * 72
    if not meta:
        print("Experiment: (none recorded -- no control node broadcast for this log)")
        return

    print("")
    print(bar)
    print("  EXPERIMENT")
    print(bar)
    print("")
    for key in ("config", "policy_trained_at", "data_logged_at"):
        if key in meta:
            print(f"  {key:18s} {meta[key]}")

    # hardware sign flip flags (only present for hardware logs recorded with hardware.py broadcasting them)
    if "hardware_info" in meta:
        print("")
        for key, value in json.loads(str(meta["hardware_info"])).items():
            print(f"  {key:18s} {value}")

    # config parameters: parse the yaml snapshot and print the resolved key->value
    # pairs (drops comments and commented-out alternatives). raw snapshot stays in the file.
    if "config_yaml" in meta:
        print("")
        print(sub)
        print(f"  CONFIG  ({meta.get('config', '')})")
        print(sub)
        print("")
        try:
            cfg = yaml.safe_load(str(meta["config_yaml"])) or {}
            width = max((len(str(k)) for k in cfg), default=0)
            for k, v in cfg.items():
                print(f"  {str(k):<{width}}  {v}")
        except Exception:
            for line in str(meta["config_yaml"]).splitlines():
                print(f"  {line}")
    print(bar)
    print("")


# pull the active policy_path out of the config-yaml snapshot (for the title)
def _policy_from_snapshot(meta: dict):
    if meta.get("config_yaml"):
        try:
            return yaml.safe_load(str(meta["config_yaml"])).get("policy_path")
        except Exception:
            return None
    return None


# compact one-line experiment label for figure titles
def experiment_label(meta: dict) -> str:
    bits = []
    if meta.get("config"):       bits.append(str(meta["config"]))
    policy = _policy_from_snapshot(meta)
    if policy:                   bits.append(str(policy))
    if meta.get("data_logged_at"):    bits.append(f"logged {meta['data_logged_at']}")
    if "hardware_info" in meta:
        flipped = [k for k, v in json.loads(str(meta["hardware_info"])).items() if v]
        bits.append(f"flipped: {', '.join(flipped)}" if flipped else "no flips")
    return "   |   ".join(bits)


# load a log: (data, meta), stale leading rows dropped and time zeroed; data is None if no rows
def load_log(file_path: str):

    print(f"Loading {file_path}")
    with h5py.File(file_path, "r") as f:
        data = {name: f[name][:] for name in f.keys()}
        meta = {k: f.attrs[k] for k in f.attrs.keys()}

    # experiment metadata (config / policy / motion) recorded as root attributes
    print_experiment_info(meta)

    # a log may contain only experiment metadata
    if "time" not in data or "joint_state" not in data:
        print("No time-series data in this log (only experiment metadata) -- nothing to show.")
        return None, meta

    # drop stale leading rows of time when fsm is in not in right state
    t_raw = data["time"][:, 0]
    resets = np.where(np.diff(t_raw) < 0)[0]
    start = int(resets[0]) + 1 if len(resets) else 0
    if start > 0:
        print(f"Dropping {start} stale leading row(s) (fsm_time resets to ~0 at row {start}).")
        data = {name: arr[start:] for name, arr in data.items()}

    # time axis (zeroed so runs always start at 0)
    data["time"] = data["time"] - data["time"][0]
    return data, meta


# plot all available data from the log
def plot_log(file_path: str):

    data, meta = load_log(file_path)
    if data is None:
        return
    exp_label = experiment_label(meta)
    t = data["time"][:, 0]

    # infer joint count from joint_state width: [q, dq, ddq, tau_est]
    N = data["joint_state"].shape[1] // 4
    q       = data["joint_state"][:, 0:N]
    dq      = data["joint_state"][:, N:2*N]
    tau_est = data["joint_state"][:, 3*N:4*N]

    # command and joystick are optional: log may not contain them
    has_command  = "command" in data
    has_joystick = "joystick" in data

    # command layout: [q_des, dq_des, Kp, Kd, tau_ff] (N each)
    if has_command:
        q_des  = data["command"][:, 0:N]
        dq_des = data["command"][:, N:2*N]
    else:
        print("[SKIP] 'command' not in log, skipping command figures.")

    # IMU layout: [rpy(3), quat(4), gyro(3), acc(3)]; G1 has pelvis + torso, Go2 a single base imu
    imus = [(name.replace("_imu", ""), data[name]) for name in ("pelvis_imu", "torso_imu", "imu") if name in data]

    # joint positions, one subplot per joint (q blue on top, q_des solid black behind)
    _, axes = _joint_grid(N, f"joint positions ({N} joints)", suptitle_extra=exp_label)
    for i, ax in enumerate(axes):
        if has_command:
            ax.plot(t, q_des[:, i], color="black", linewidth=0.75, label="q_des", zorder=1)
        ax.plot(t, q[:, i], color="tab:blue", linewidth=1.5, label="q", zorder=2)
        ax.set_title(joint_title(i, N))
        ax.set_ylabel("[rad]")
        ax.grid(True)
    axes[0].legend(loc="upper right")

    # joint velocities, one subplot per joint (dq blue on top, dq_des solid black behind)
    _, axes = _joint_grid(N, f"joint velocities ({N} joints)", suptitle_extra=exp_label)
    for i, ax in enumerate(axes):
        if has_command:
            ax.plot(t, dq_des[:, i], color="black", linewidth=0.75, label="dq_des", zorder=1)
        ax.plot(t, dq[:, i], color="tab:blue", linewidth=1.5, label="dq", zorder=2)
        ax.set_title(joint_title(i, N))
        ax.set_ylabel("[rad/s]")
        ax.grid(True)
    axes[0].legend(loc="upper right")

    # estimated torque, one subplot per joint
    _, axes = _joint_grid(N, f"tau_est ({N} joints)", suptitle_extra=exp_label)
    for i, ax in enumerate(axes):
        ax.plot(t, tau_est[:, i], color="tab:blue")
        ax.set_title(joint_title(i, N))
        ax.set_ylabel("[Nm]")
        ax.grid(True)

    # IMUs (rows = rpy/quat/gyro/acc, one column per IMU)
    fig, axes = plt.subplots(4, max(len(imus), 1), figsize=(6 * max(len(imus), 1), 9), sharex=True, squeeze=False)
    if exp_label:
        fig.suptitle(exp_label)
    for col, (name, imu) in enumerate(imus):
        axes[0, col].plot(t, imu[:, 0:3],   label=["roll", "pitch", "yaw"])
        axes[0, col].set_title(f"{name} IMU")
        axes[0, col].legend(loc="upper right")
        axes[1, col].plot(t, imu[:, 3:7],   label=["qw", "qx", "qy", "qz"])
        axes[1, col].legend(loc="upper right")
        axes[2, col].plot(t, imu[:, 7:10],  label=["wx", "wy", "wz"])
        axes[2, col].legend(loc="upper right")
        axes[3, col].plot(t, imu[:, 10:13], label=["ax", "ay", "az"])
        axes[3, col].legend(loc="upper right")
        axes[3, col].set_xlabel("time [s]")
        for row in range(4):
            axes[row, col].grid(True)
    axes[0, 0].set_ylabel("rpy [rad]")
    axes[1, 0].set_ylabel("quat")
    axes[2, 0].set_ylabel("gyro [rad/s]")
    axes[3, 0].set_ylabel("acc [m/s^2]")

    # joystick command (vx, vy, omega stacked) -- only if logged
    if has_joystick:
        js = data["joystick"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        if exp_label:
            fig.suptitle(exp_label)
        axes[0].plot(t, js[:, 1])
        axes[0].set_ylabel("vx [m/s]")
        axes[0].set_title("joystick command")
        axes[1].plot(t, js[:, 2])
        axes[1].set_ylabel("vy [m/s]")
        axes[2].plot(t, js[:, 3])
        axes[2].set_ylabel("omega [rad/s]")
        axes[2].set_xlabel("time [s]")
        for ax in axes:
            ax.grid(True)
    else:
        print("[SKIP] 'joystick' not in log, skipping joystick figure")

    # motor temperatures, one subplot per motor (both sensors overlaid) -- only if logged
    if "motor_temp" in data:
        mt = data["motor_temp"]
        Nt = mt.shape[1] // 2          # layout: [sensor0(Nt), sensor1(Nt)]
        s0 = mt[:, 0:Nt]
        s1 = mt[:, Nt:2*Nt]
        _, axes = _joint_grid(Nt, f"motor temperatures ({Nt} motors)", suptitle_extra=exp_label)
        for i, ax in enumerate(axes):
            ax.plot(t, s0[:, i], color="tab:orange", linewidth=1.0, label="sensor0")
            ax.plot(t, s1[:, i], color="tab:red",    linewidth=1.0, label="sensor1")
            ax.set_title(joint_title(i, Nt))
            ax.set_ylabel("[degC]")
            ax.grid(True)
        axes[0].legend(loc="upper right")
    else:
        print("[SKIP] 'motor_temp' not in log, skipping temperature figure")

    # robot status as the robot reported it -- only if logged
    if "robot_status" in data:
        rs = data["robot_status"]
        Ns = (rs.shape[1] - 2) // 3    # layout: [tick, imu temp, mode(Ns), vol(Ns), motorstate(Ns)]
        mode, vol, status = rs[:, 2:2+Ns], rs[:, 2+Ns:2+2*Ns], rs[:, 2+2*Ns:2+3*Ns]
        new = np.r_[True, np.diff(t) > 0]    # one row per hardware tick (the logger can repeat a tick)

        has_dropout = "dropout" in data
        fig, axes = plt.subplots(6 if has_dropout else 5, 1, figsize=(14, 10), sharex=True, layout="constrained")
        fig.suptitle(f"robot status\n{exp_label}" if exp_label else "robot status")
        axes[0].fill_between(t, vol.min(axis=1), vol.max(axis=1), color="tab:blue", alpha=0.25, linewidth=0,
                             label="lowest to highest motor")
        axes[0].plot(t, vol.mean(axis=1), color="tab:blue", linewidth=1.5, label="mean of all motors")
        axes[0].set_ylabel("motor voltage [V]")
        axes[0].legend(loc="upper right")
        axes[1].plot(t, (mode != 1).sum(axis=1), color="tab:blue", linewidth=1.5)
        axes[1].set_ylabel("motors not in\nmode 1 [count]")
        axes[2].plot(t, (status != 0).sum(axis=1), color="tab:blue", linewidth=1.5)
        axes[2].set_ylabel("motors with a non-zero\nstatus word [count]")
        axes[3].plot(t[new][1:], np.diff(rs[new, 0]), color="tab:blue", linewidth=1.0)
        axes[3].set_ylabel("robot clock step per\nhardware tick [ms]")
        axes[4].plot(t, rs[:, 1], color="tab:blue", linewidth=1.5)
        axes[4].set_ylabel("pelvis IMU\ntemperature")
        # readings the guards filled in; layout: [pelvis IMU lost, torso IMU lost, joint lost(Ns)], 1 = lost
        if has_dropout:
            dr = data["dropout"]
            axes[5].plot(t, dr[:, 2:].sum(axis=1), color="tab:blue", linewidth=1.5, label="joints")
            axes[5].plot(t, dr[:, 0], color="tab:orange", linewidth=1.5, label="pelvis IMU")
            axes[5].plot(t, dr[:, 1], color="tab:green", linewidth=1.5, label="torso IMU")
            axes[5].set_ylabel("readings lost\n[count]")
            axes[5].set_ylim(-0.5, max(1.0, axes[5].get_ylim()[1]) + 0.5)
            axes[5].legend(loc="upper right")
        axes[-1].set_xlabel("time [s]")
        for ax in axes[1:3]:
            ax.set_ylim(-0.5, max(1.0, ax.get_ylim()[1]) + 0.5)
        for ax in axes:
            ax.grid(True)

        # the figure shows counts; name the motors here
        print("\nRobot status:")
        print(f"    motor voltage [V]: min {vol.min():.1f}, mean {vol.mean():.2f}, max {vol.max():.1f}")
        for label, bad in (("not in mode 1", mode != 1), ("with a non-zero status word", status != 0)):
            hits = {joint_title(i, Ns): int(bad[:, i].sum()) for i in range(Ns) if bad[:, i].any()}
            print(f"    log rows per motor {label}: {hits if hits else 'none'}")
        words = np.unique(status[status != 0]).astype(np.int64)
        if len(words):
            print(f"    status words seen: {[hex(w) for w in words]}")
        if has_dropout:
            lost = dr[new] > 0.5
            hits = {joint_title(i, Ns): int(lost[:, 2 + i].sum()) for i in range(Ns) if lost[:, 2 + i].any()}
            print(f"    hardware ticks lost: pelvis IMU {int(lost[:, 0].sum())}, torso IMU {int(lost[:, 1].sum())}, "
                  f"per joint {hits if hits else 'none'}")
    else:
        print("[SKIP] 'robot_status' not in log, skipping robot status figure")

    plt.show()


def main():

    # parse command line args
    parser = argparse.ArgumentParser(description="Plot HDF5 deploy logs.")
    parser.add_argument("--filename", 
                        type=str, 
                        default=None,
                        help="Path to .h5 log file. If omitted, plots the most recent log "
                             "under logs/simulation/ or logs/hardware/.")
    parser.add_argument("--replay", action="store_true",
                        help="Replay the log in a MuJoCo viewer instead of plotting.")
    args = parser.parse_args()

    # if no file provided, search for the most recent .h5 log
    if args.filename is None:
        file_path = find_latest_log()
        print(f"No --filename provided, using latest: {file_path}")
    # otherwise, use the provided file path
    else:
        file_path = args.filename
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: {file_path}")

    # replay in mujoco (imported lazily so plotting does not need mujoco), or plot
    if args.replay:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.log_replay_utils import replay_log
        data, meta = load_log(file_path)
        if data is not None:
            replay_log(data, meta, os.path.basename(file_path))
    else:
        plot_log(file_path)


if __name__ == "__main__":
    main()

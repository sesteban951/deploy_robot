##
#
# Quick plot script for HDF5 logs.
#
##

import argparse
import glob
import math
import os
import h5py
import matplotlib.pyplot as plt


# search both logs/simulation and logs/hardware for the most recently modified .h5 file
def find_latest_log() -> str:
    logs_root = os.path.dirname(os.path.abspath(__file__))
    candidates = (glob.glob(os.path.join(logs_root, "simulation", "*.h5"))
                + glob.glob(os.path.join(logs_root, "hardware", "*.h5")))
    if not candidates:
        raise FileNotFoundError(f"No .h5 logs found under {logs_root}/simulation or {logs_root}/hardware.")
    return max(candidates, key=os.path.getmtime)


# build a (rows, cols) grid that fits N subplots, and return flat axes
def _joint_grid(N: int, title: str, figsize=(14, 9)):
    cols = min(6, N)
    rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    fig.suptitle(title)
    axes = axes.flatten() if N > 1 else [axes]
    # hide any unused axes on the last row
    for ax in axes[N:]:
        ax.set_visible(False)
    return fig, axes[:N]


# load and log all avilable data from the log
def plot_log(file_path: str):

    print(f"Loading {file_path}")
    with h5py.File(file_path, "r") as f:
        data = {name: f[name][:] for name in f.keys()}

    # time axis (zeroed so runs always start at 0)
    t = data["time"][:, 0]
    t = t - t[0]

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

    # IMU layout: [rpy(3), quat(4), gyro(3), acc(3)]
    pelvis = data["pelvis_imu"]
    torso  = data["torso_imu"]

    # joint positions, one subplot per joint (q blue on top, q_des solid black behind)
    _, axes = _joint_grid(N, f"joint positions ({N} joints)")
    for i, ax in enumerate(axes):
        if has_command:
            ax.plot(t, q_des[:, i], color="black", linewidth=0.75, label="q_des", zorder=1)
        ax.plot(t, q[:, i], color="tab:blue", linewidth=1.5, label="q", zorder=2)
        ax.set_title(f"joint {i}")
        ax.set_ylabel("[rad]")
        ax.grid(True)
    axes[0].legend(loc="upper right")

    # joint velocities, one subplot per joint (dq blue on top, dq_des solid black behind)
    _, axes = _joint_grid(N, f"joint velocities ({N} joints)")
    for i, ax in enumerate(axes):
        if has_command:
            ax.plot(t, dq_des[:, i], color="black", linewidth=0.75, label="dq_des", zorder=1)
        ax.plot(t, dq[:, i], color="tab:blue", linewidth=1.5, label="dq", zorder=2)
        ax.set_title(f"joint {i}")
        ax.set_ylabel("[rad/s]")
        ax.grid(True)
    axes[0].legend(loc="upper right")

    # estimated torque, one subplot per joint
    _, axes = _joint_grid(N, f"tau_est ({N} joints)")
    for i, ax in enumerate(axes):
        ax.plot(t, tau_est[:, i], color="tab:blue")
        ax.set_title(f"joint {i}")
        ax.set_ylabel("[Nm]")
        ax.grid(True)

    # IMUs (rows = rpy/quat/gyro/acc, columns = pelvis/torso)
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)
    for col, (name, imu) in enumerate([("pelvis", pelvis), ("torso", torso)]):
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

    plt.show()


def main():

    # parse command line args
    parser = argparse.ArgumentParser(description="Plot HDF5 deploy logs.")
    parser.add_argument("--filename", 
                        type=str, 
                        default=None,
                        help="Path to .h5 log file. If omitted, plots the most recent log "
                             "under logs/simulation/ or logs/hardware/.")
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

    # load and plot the log
    plot_log(file_path)


if __name__ == "__main__":
    main()

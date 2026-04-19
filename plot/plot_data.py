##
#
# Plot an HDF5 log produced by deploy/simulation/log_data.py or
# deploy/hardware/log_data.py.
#
##


# standard imports
import argparse
import glob
import os
import sys

# other imports
import h5py
import numpy as np
import matplotlib.pyplot as plt


# directory imports
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")


############################################################################
# CONSTANTS
############################################################################

G1_NUM_MOTOR = 29


############################################################################
# HELPERS
############################################################################

def find_latest_log() -> str:
    """Find the most recently modified .h5 file under logs/{simulation,hardware}/."""
    base = os.path.join(ROOT_DIR, "logs") if ROOT_DIR else "logs"
    candidates = glob.glob(os.path.join(base, "**", "*.h5"), recursive=True)
    if not candidates:
        print(f"No .h5 files found under [{base}].")
        sys.exit(1)
    return max(candidates, key=os.path.getmtime)


def load_h5(path: str) -> dict:
    """Load all datasets from the HDF5 file into a dict of numpy arrays."""
    data = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
    return data


############################################################################
# PLOTTING
############################################################################

def plot_joint_state(joint_state: np.ndarray):
    n = G1_NUM_MOTOR
    q       = joint_state[:, 0*n:1*n]
    dq      = joint_state[:, 1*n:2*n]
    ddq     = joint_state[:, 2*n:3*n]
    tau_est = joint_state[:, 3*n:4*n]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Joint State (29 dof)")
    for ax, data, label in zip(axes,
                               [q, dq, ddq, tau_est],
                               ["q [rad]", "dq [rad/s]", "ddq [rad/s^2]", "tau_est [Nm]"]):
        ax.plot(data, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True)
    axes[-1].set_xlabel("sample")
    fig.tight_layout()


def plot_command(command: np.ndarray):
    n = G1_NUM_MOTOR
    q_des   = command[:, 0*n:1*n]
    dq_des  = command[:, 1*n:2*n]
    Kp      = command[:, 2*n:3*n]
    Kd      = command[:, 3*n:4*n]
    tau_ff  = command[:, 4*n:5*n]

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Command (29 dof)")
    for ax, data, label in zip(axes,
                               [q_des, dq_des, Kp, Kd, tau_ff],
                               ["q_des [rad]", "dq_des [rad/s]", "Kp", "Kd", "tau_ff [Nm]"]):
        ax.plot(data, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True)
    axes[-1].set_xlabel("sample")
    fig.tight_layout()


def plot_imu(imu: np.ndarray, title: str):
    # [rpy(3), quat(4), gyro(3), acc(3)]
    rpy  = imu[:, 0:3]
    quat = imu[:, 3:7]
    gyro = imu[:, 7:10]
    acc  = imu[:, 10:13]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title)
    axes[0].plot(rpy);  axes[0].set_ylabel("rpy [rad]");       axes[0].legend(["r", "p", "y"])
    axes[1].plot(quat); axes[1].set_ylabel("quat");            axes[1].legend(["w", "x", "y", "z"])
    axes[2].plot(gyro); axes[2].set_ylabel("gyro [rad/s]");    axes[2].legend(["x", "y", "z"])
    axes[3].plot(acc);  axes[3].set_ylabel("acc [m/s^2]");     axes[3].legend(["x", "y", "z"])
    for ax in axes:
        ax.grid(True)
    axes[-1].set_xlabel("sample")
    fig.tight_layout()


def plot_joystick(joystick: np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Joystick")
    ax.plot(joystick[:, 0], label="connected")
    ax.plot(joystick[:, 1], label="vx")
    ax.plot(joystick[:, 2], label="vy")
    ax.plot(joystick[:, 3], label="omega")
    ax.set_xlabel("sample")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


def plot_time(time_arr: np.ndarray, name: str):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.suptitle(name)
    ax.plot(time_arr[:, 0])
    ax.set_xlabel("sample")
    ax.set_ylabel("time [s]")
    ax.grid(True)
    fig.tight_layout()


############################################################################
# MAIN
############################################################################

def main():
    parser = argparse.ArgumentParser(description="Plot an HDF5 log.")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the .h5 file. If omitted, uses the most recent file under logs/.",
    )
    args = parser.parse_args()

    path = args.path or find_latest_log()
    print(f"Loading [{path}]")

    data = load_h5(path)
    for k, v in data.items():
        print(f"    {k}: shape={v.shape}")

    if "joint_state" in data:
        plot_joint_state(data["joint_state"])
    if "command" in data:
        plot_command(data["command"])
    if "pelvis_imu" in data:
        plot_imu(data["pelvis_imu"], "Pelvis IMU")
    if "torso_imu" in data:
        plot_imu(data["torso_imu"], "Torso IMU")
    if "joystick" in data:
        plot_joystick(data["joystick"])
    if "simulation_time" in data:
        plot_time(data["simulation_time"], "Simulation Time")
    if "fsm_time" in data:
        plot_time(data["fsm_time"], "FSM Time")

    plt.show()


if __name__ == "__main__":
    main()

##
#
# Live plotting node for deployment in both simulation and hardware.
#
# Subscribes to the same topics as the logger and renders a rolling-window
# real-time view with pyqtgraph (no file involved). Two windows:
#   - joint angles : q (blue) vs q_des (black), one plot per joint
#   - IMU          : rpy / quat / gyro / acc for pelvis + torso
#
# Run standalone, like the logger:
#   python3 deploy/logger/live_plot.py --mode sim
#   python3 deploy/logger/live_plot.py --mode hw
#
# Requires: pip install pyqtgraph   (PySide6 is already provided by the system)
#
##

# standard imports
import argparse
import os
import sys
import threading
from collections import deque

import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64

# pyqtgraph / Qt imports
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# directory imports
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
if ROOT_DIR:
    sys.path.append(ROOT_DIR)


############################################################################
# CONSTANTS
############################################################################

# per-mode config (mirrors logger): only the time topic differs here
MODE_CONFIG = {
    "sim": {"time_topic": "deploy_robot/simulation_time"},
    "hw":  {"time_topic": "deploy_robot/hardware_time"},
}

# IMU layout: [rpy(3), quat(4), gyro(3), acc(3)] -> (label, lo, hi, channel names)
IMU_SECTIONS = [
    ("rpy [rad]",     0,  3,  ["roll", "pitch", "yaw"]),
    ("quat",          3,  7,  ["qw", "qx", "qy", "qz"]),
    ("gyro [rad/s]",  7, 10,  ["wx", "wy", "wz"]),
    ("acc [m/s^2]",  10, 13,  ["ax", "ay", "az"]),
]

# distinct colors for multi-channel plots (rpy/quat/gyro/acc)
CHANNEL_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]


############################################################################
# LIVE PLOT NODE
############################################################################

class LivePlotNode(Node):
    """
    ROS2 node that caches the latest message per topic, samples them onto a
    common time axis at sample_hz, and exposes rolling buffers for the Qt
    redraw loop. The logger is untouched; this just reads the live topics.
    """

    def __init__(self, mode: str, window_s: float, sample_hz: float):
        super().__init__("live_plot_node")

        self.cfg = MODE_CONFIG[mode]

        # rolling-buffer capacity (number of samples kept on screen)
        self._maxlen = max(2, int(window_s * sample_hz))

        # joint count, discovered from the first joint_state message
        self.N = None

        # whether a 'command' publisher is active (gives us q_des)
        self.has_command = False

        # latest-message cache, keyed by dataset name (filled by callbacks)
        self._latest: dict = {}

        # rolling buffers, shared with the Qt thread -> guard with a lock
        self.lock = threading.Lock()
        self.buf = {
            "t":      deque(maxlen=self._maxlen),
            "q":      deque(maxlen=self._maxlen),
            "q_des":  deque(maxlen=self._maxlen),
            "pelvis": deque(maxlen=self._maxlen),
            "torso":  deque(maxlen=self._maxlen),
        }

        # subscribers (same topics the logger uses)
        self.create_subscription(Float32MultiArray, "deploy_robot/joint_state",      self._joint_cb,  5)
        self.create_subscription(Float32MultiArray, "deploy_robot/command",          self._cmd_cb,    5)
        self.create_subscription(Float32MultiArray, "deploy_robot/pelvis_imu_state", self._pelvis_cb, 5)
        self.create_subscription(Float32MultiArray, "deploy_robot/torso_imu_state",  self._torso_cb,  5)
        self.create_subscription(Float64,           self.cfg["time_topic"],          self._time_cb,   5)

        # sampling timer: snapshot latest messages onto a common time axis
        self.create_timer(1.0 / sample_hz, self._sample_cb)

        print(f"Live plot node initialized.")
        print(f"    Mode:        {mode}")
        print(f"    Time topic:  {self.cfg['time_topic']}")
        print(f"    Window:      {window_s} s")
        print(f"    Sample rate: {sample_hz} Hz  (buffer = {self._maxlen} samples)")

    #################################################################
    # CALLBACKS (just cache the latest message)
    #################################################################

    def _joint_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if self.N is None and data.size > 0:
            self.N = data.size // 4   # joint_state layout: [q, dq, ddq, tau_est]
        self._latest["joint_state"] = data

    def _cmd_cb(self, msg: Float32MultiArray):
        self.has_command = True
        self._latest["command"] = np.array(msg.data, dtype=np.float32)

    def _pelvis_cb(self, msg: Float32MultiArray):
        self._latest["pelvis_imu"] = np.array(msg.data, dtype=np.float32)

    def _torso_cb(self, msg: Float32MultiArray):
        self._latest["torso_imu"] = np.array(msg.data, dtype=np.float32)

    def _time_cb(self, msg: Float64):
        self._latest["time"] = float(msg.data)

    #################################################################
    # SAMPLING (align all signals onto one time axis, like the logger)
    #################################################################

    def _sample_cb(self):
        L = self._latest

        # wait until the essentials have all published at least once
        if self.N is None:
            return
        if not all(k in L for k in ("time", "joint_state", "pelvis_imu", "torso_imu")):
            return

        N = self.N
        q = L["joint_state"][0:N]

        # command layout: [q_des, dq_des, Kp, Kd, tau_ff] -> q_des is the first N
        if self.has_command and "command" in L and L["command"].size >= N:
            q_des = L["command"][0:N]
        else:
            q_des = np.full(N, np.nan, dtype=np.float32)

        with self.lock:
            self.buf["t"].append(L["time"])
            self.buf["q"].append(q.copy())
            self.buf["q_des"].append(q_des.copy())
            self.buf["pelvis"].append(L["pelvis_imu"].copy())
            self.buf["torso"].append(L["torso_imu"].copy())

    # thread-safe snapshot of the buffers as numpy arrays for the Qt redraw
    def snapshot(self):
        with self.lock:
            if not self.buf["t"]:
                return None
            t      = np.fromiter(self.buf["t"], dtype=np.float64)
            q      = np.array(self.buf["q"])
            q_des  = np.array(self.buf["q_des"])
            pelvis = np.array(self.buf["pelvis"])
            torso  = np.array(self.buf["torso"])
        # x axis as "seconds ago" so the latest sample sits at 0 on the right
        t = t - t[-1]
        return t, q, q_des, pelvis, torso


############################################################################
# QT VISUALIZER
############################################################################

class Visualizer:
    """Owns the pyqtgraph windows and the redraw timer (runs in the main thread)."""

    def __init__(self, node: LivePlotNode, redraw_hz: float):
        self.node = node
        self.built = False

        pg.setConfigOptions(antialias=False)   # faster; plenty crisp for monitoring
        self.app = pg.mkQApp("deploy_robot live plot")

        # joint-angle window
        self.win_joint = pg.GraphicsLayoutWidget(title="joint angles  (q vs q_des)")
        self.win_joint.resize(1400, 850)

        # IMU window
        self.win_imu = pg.GraphicsLayoutWidget(title="IMU  (pelvis | torso)")
        self.win_imu.resize(1100, 850)

        # curve handles, populated lazily once the joint count is known
        self.q_curves = []
        self.qd_curves = []
        self.imu_curves = {"pelvis": [], "torso": []}   # list of lists, one per section

        # redraw timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(int(1000.0 / redraw_hz))

    # build all plots/curves once N (joint count) is known
    def _build(self, N: int):
        # ---- joint angles: one plot per joint, up to 6 columns ----
        cols = min(6, N)
        for i in range(N):
            p = self.win_joint.addPlot(row=i // cols, col=i % cols, title=f"joint {i}")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("left", "[rad]")
            qd = p.plot(pen=pg.mkPen("k", width=1))            # q_des behind
            q  = p.plot(pen=pg.mkPen("#1f77b4", width=2))      # q in front
            self.qd_curves.append(qd)
            self.q_curves.append(q)

        # ---- IMU: 4 rows (rpy/quat/gyro/acc) x 2 cols (pelvis/torso) ----
        for col, name in enumerate(("pelvis", "torso")):
            for row, (label, lo, hi, ch_names) in enumerate(IMU_SECTIONS):
                p = self.win_imu.addPlot(row=row, col=col)
                p.showGrid(x=True, y=True, alpha=0.3)
                if row == 0:
                    p.setTitle(f"{name} IMU")
                if col == 0:
                    p.setLabel("left", label)
                p.addLegend(offset=(-1, 1))
                section_curves = []
                for k in range(hi - lo):
                    c = p.plot(pen=pg.mkPen(CHANNEL_COLORS[k % len(CHANNEL_COLORS)], width=1.5),
                               name=ch_names[k])
                    section_curves.append(c)
                self.imu_curves[name].append(section_curves)

        self.win_joint.show()
        self.win_imu.show()
        self.built = True

    def _update(self):
        # build lazily once we know how many joints there are
        if not self.built:
            if self.node.N is None:
                return
            self._build(self.node.N)

        snap = self.node.snapshot()
        if snap is None:
            return
        t, q, q_des, pelvis, torso = snap

        # joint angles
        for i in range(len(self.q_curves)):
            self.q_curves[i].setData(t, q[:, i])
            if self.node.has_command and not np.isnan(q_des[:, i]).all():
                self.qd_curves[i].setData(t, q_des[:, i])

        # IMU
        for name, imu in (("pelvis", pelvis), ("torso", torso)):
            for sec_idx, (label, lo, hi, ch_names) in enumerate(IMU_SECTIONS):
                curves = self.imu_curves[name][sec_idx]
                for k in range(hi - lo):
                    curves[k].setData(t, imu[:, lo + k])

    def run(self):
        self.app.exec()   # blocks until the windows are closed


############################################################################
# MAIN
############################################################################

def main():
    parser = argparse.ArgumentParser(description="Live plot node (sim + hardware).")
    parser.add_argument("--mode", type=str, required=True, choices=["sim", "hw"],
                        help='Deployment mode: "sim" or "hw".')
    parser.add_argument("--window", type=float, default=10.0,
                        help="Rolling window length in seconds. Default: 10.0.")
    parser.add_argument("--sample_hz", type=float, default=100.0,
                        help="Rate at which topics are sampled onto the time axis. Default: 100.0.")
    parser.add_argument("--redraw_hz", type=float, default=50.0,
                        help="Plot redraw rate in Hz. Default: 50.0.")
    args = parser.parse_args()

    # init ROS2 and node
    rclpy.init()
    node = LivePlotNode(args.mode, args.window, args.sample_hz)

    # spin ROS in a background thread so the Qt event loop owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # build + run the Qt visualizer (blocks until windows close)
    viz = Visualizer(node, args.redraw_hz)
    try:
        viz.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)

    print("Live plot shutdown complete.")


if __name__ == "__main__":
    main()

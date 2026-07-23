##
#
# Live version of debug/debug.py figures 1-3, for watching a mimic run in real time.
#
# Runs standalone (like deploy/logger/live_plot.py), in parallel with the deploy
# stack, and subscribes to the live topics. Three windows:
#   1. IMU + FK cross-check : pelvis/torso rpy & quat, torso orientation predicted
#      from pelvis quat x FK(waist encoders), and the measured-vs-predicted divergence.
#   2. Torso divergence angle [rad].
#   3. Obs blocks : base_ang_vel (pelvis gyro) and motion_anchor_ori_b (torso-vs-
#      reference orientation error) in 6D / quat / rpy, as the policy sees it.
#
# These three need NO policy inference (pure sensor + FK + obs reconstruction), so
# they are cheap to compute live. The obs math is reused from debug/obs_replay.py
# and debug/fk_crosscheck.py (kept in sync with control_29dof_mimic.py).
#
# The motion frame index and init_quat are NOT published, so they are reconstructed
# live exactly as the control node does, driven by deploy_robot/control_phase and
# deploy_robot/fsm_time (both hardware-only). The run config (policy/motion/xml) is
# taken from the latched deploy_robot/experiment_info topic (exactly what the control
# node loaded), or from --config as an override / for standalone use.
#
# Run:
#   python debug/live_debug.py --mode hw
#   python debug/live_debug.py --mode hw --config g1_29dof_mimic.yaml
#
# Requires: pip install pyqtgraph   (PySide6 provided by the system)
#
##

import argparse
import json
import os
import sys
import threading
from collections import deque

import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String

# pyqtgraph / Qt imports
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# resolve repo root (env or inferred) and make utils/ + this dir importable
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fk_crosscheck import TorsoFK, quat_divergence
from obs_replay import MimicReplay
from utils.math_utils import quat_to_rpy, quat_multiply
from utils.experiment_utils import EXPERIMENT_INFO_TOPIC, experiment_info_qos


############################################################################
# CONSTANTS
############################################################################

# per-mode config (mirrors live_plot.py): only the time topic differs
MODE_CONFIG = {
    "sim": {"time_topic": "deploy_robot/simulation_time"},
    "hw":  {"time_topic": "deploy_robot/hardware_time"},
}

# control-phase codes published by control_29dof_mimic.py
PHASE_IDLE, PHASE_INTERP, PHASE_HOLD, PHASE_TRACK = -1, 0, 1, 2
PHASE_NAMES = {PHASE_IDLE: "idle", PHASE_INTERP: "interp", PHASE_HOLD: "hold", PHASE_TRACK: "track"}

# channel labels
RPY = ["roll", "pitch", "yaw"]
QUAT = ["qw", "qx", "qy", "qz"]
SIXD = ["r11", "r12", "r21", "r22", "r31", "r32"]
GYRO = ["wx", "wy", "wz"]

# distinct colors (up to 6 channels for the 6D block)
COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b"]

# dashed pen style (enum path differs across Qt bindings)
try:
    _DASH = QtCore.Qt.PenStyle.DashLine
except AttributeError:  # pragma: no cover
    _DASH = QtCore.Qt.DashLine


############################################################################
# LIVE DEBUG NODE
############################################################################

class LiveDebugNode(Node):
    """
    Caches the latest message per topic, and at sample_hz computes the FK cross-check
    and the motion_anchor_ori_b obs block, buffering everything (raw + derived) onto a
    common rolling time axis for the Qt redraw loop. The obs machinery (MimicReplay +
    TorsoFK) is lazily built once a config is available.
    """

    def __init__(self, mode: str, config_arg: str, window_s: float, sample_hz: float):
        super().__init__("live_debug_node")

        self.cfg = MODE_CONFIG[mode]
        self._maxlen = max(2, int(window_s * sample_hz))
        self.N = None  # joint count, discovered from the first joint_state message

        # obs machinery (built lazily from config)
        self.rep = None
        self.fk = None
        self._machinery_ready = False
        self._config_override = config_arg is not None  # --config wins over experiment_info
        self.align_heading = True
        self.zero_imu_yaw = False
        self.anchor_imu = "torso"
        self.num_frames = 1
        self.ctrl_dt = 0.02

        # live-reconstructed frame timeline / heading alignment (mirrors control node)
        self._init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._yaw_offset_conj = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._init_captured = False
        self._cur_phase = None
        self._cur_frame = 0

        # latest-message cache + rolling buffers shared with the Qt thread
        self._latest: dict = {}
        self.lock = threading.Lock()
        self.buf = {k: deque(maxlen=self._maxlen) for k in (
            "t", "pelvis", "torso", "pred_quat", "meas_rpy", "pred_rpy",
            "div_quat", "div_rpy", "div_angle", "ori6d", "oq", "orpy",
        )}

        # subscribers (same live topics the logger/control node use)
        self.create_subscription(Float32MultiArray, "deploy_robot/joint_state",      self._joint_cb,  5)
        self.create_subscription(Float32MultiArray, "deploy_robot/pelvis_imu_state", self._pelvis_cb, 5)
        self.create_subscription(Float32MultiArray, "deploy_robot/torso_imu_state",  self._torso_cb,  5)
        self.create_subscription(Float64,           self.cfg["time_topic"],          self._time_cb,   5)
        self.create_subscription(Float64,           "deploy_robot/fsm_time",         self._fsm_time_cb, 5)
        self.create_subscription(Float64,           "deploy_robot/control_phase",    self._phase_cb,  5)
        self.create_subscription(String, EXPERIMENT_INFO_TOPIC, self._experiment_cb, experiment_info_qos())

        # config: --config takes effect immediately; else wait for experiment_info
        if config_arg is not None:
            cfg = self._load_config_file(config_arg)
            if cfg is not None:
                self._init_machinery(cfg, f"--config {config_arg}")

        # sampling timer
        self.create_timer(1.0 / sample_hz, self._sample_cb)

        print("Live debug node initialized.")
        print(f"    Mode:        {mode}")
        print(f"    Time topic:  {self.cfg['time_topic']}")
        print(f"    Window:      {window_s} s  (buffer = {self._maxlen} samples)")
        print(f"    Config:      {'--config ' + config_arg if config_arg else 'from experiment_info (waiting)'}")

    #################################################################
    # CONFIG / MACHINERY
    #################################################################

    def _load_config_file(self, name: str):
        path = os.path.join(ROOT_DIR, "deploy", "configs", name if name.endswith(".yaml") else name + ".yaml")
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except OSError as e:
            print(f"[live_debug] could not read config {path}: {e}")
            return None

    # build MimicReplay + TorsoFK from a parsed config dict (once)
    def _init_machinery(self, cfg: dict, source: str):
        if self._machinery_ready:
            return
        try:
            rep_cfg = {
                "policy_path": cfg["policy_path"],
                "motion_path": cfg["motion_path"],
                "xml_path": cfg.get("xml_path", "g1_29dof_scene.xml"),
                "control_dt": float(cfg.get("control_dt", 0.02)),
            }
            self.rep = MimicReplay(rep_cfg, ROOT_DIR)
            self.fk = TorsoFK(os.path.join(ROOT_DIR, "models", rep_cfg["xml_path"]))
            self.align_heading = bool(cfg.get("align_heading", True))
            self.zero_imu_yaw = bool(cfg.get("zero_imu_yaw", False))
            self.num_frames = self.rep.num_frames
            self.ctrl_dt = self.rep.ctrl_dt
            self.anchor_imu = self.rep.anchor_imu
            self._machinery_ready = True
            print(f"[live_debug] machinery ready from {source}: policy={rep_cfg['policy_path']}, "
                  f"motion={rep_cfg['motion_path']}, anchor={self.rep.anchor_name} ({self.anchor_imu} IMU), "
                  f"frames={self.num_frames}, align_heading={self.align_heading}, "
                  f"zero_imu_yaw={self.zero_imu_yaw}")
            if self.align_heading and self.zero_imu_yaw:
                print("[live_debug] WARNING: align_heading and zero_imu_yaw both on -- yaw double-corrected.")
        except Exception as e:
            print(f"[live_debug] failed to init machinery from {source}: {e}")

    #################################################################
    # CALLBACKS (cache the latest message)
    #################################################################

    def _joint_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if self.N is None and data.size > 0:
            self.N = data.size // 4   # [q, dq, ddq, tau_est]
        self._latest["joint_state"] = data

    def _pelvis_cb(self, msg: Float32MultiArray):
        self._latest["pelvis_imu"] = np.array(msg.data, dtype=np.float32)

    def _torso_cb(self, msg: Float32MultiArray):
        self._latest["torso_imu"] = np.array(msg.data, dtype=np.float32)

    def _time_cb(self, msg: Float64):
        self._latest["time"] = float(msg.data)

    def _fsm_time_cb(self, msg: Float64):
        self._latest["fsm_time"] = float(msg.data)

    def _phase_cb(self, msg: Float64):
        self._latest["control_phase"] = float(msg.data)

    def _experiment_cb(self, msg: String):
        if self._machinery_ready or self._config_override:
            return
        try:
            info = json.loads(msg.data)
            cfg = yaml.safe_load(info["config_yaml"])
        except Exception as e:
            print(f"[live_debug] could not parse experiment_info config: {e}")
            return
        self._init_machinery(cfg, "experiment_info")

    #################################################################
    # FRAME TIMELINE / INIT_QUAT (mirrors control_29dof_mimic.py)
    #################################################################

    def _capture_init(self, anchor_quat):
        if self._init_captured:
            return
        if self.align_heading:
            self._init_quat = self.rep.capture_init_quat(anchor_quat)
        else:
            self._init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if self.zero_imu_yaw:
            self._yaw_offset_conj = self.rep.capture_yaw_offset(anchor_quat)
        else:
            self._yaw_offset_conj = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._init_captured = True

    def _resolve_frame(self, anchor_quat):
        phase = self._latest.get("control_phase")
        fsm_time = self._latest.get("fsm_time", 0.0)
        if phase is None:
            # no phase topic (e.g. sim): reference held at frame 0, no heading capture
            self._init_captured = False
            self._cur_phase, self._cur_frame = None, 0
            return 0
        phase = int(round(phase))
        if phase in (PHASE_IDLE, PHASE_INTERP):
            self._init_captured = False
            frame = 0
        elif phase == PHASE_HOLD:
            self._capture_init(anchor_quat)
            frame = 0
        elif phase == PHASE_TRACK:
            self._capture_init(anchor_quat)
            frame = min(int(fsm_time / self.ctrl_dt), self.num_frames - 1)
        else:
            frame = 0
        self._cur_phase, self._cur_frame = phase, frame
        return frame

    #################################################################
    # SAMPLING
    #################################################################

    def _sample_cb(self):
        L = self._latest
        if self.N is None:
            return
        if not all(k in L for k in ("time", "joint_state", "pelvis_imu", "torso_imu")):
            return
        pelvis, torso = L["pelvis_imu"], L["torso_imu"]
        if pelvis.size < 13 or torso.size < 13:
            return
        q = L["joint_state"][0:self.N]
        pelvis_quat = pelvis[3:7].astype(np.float64)
        torso_quat = torso[3:7].astype(np.float64)

        if self._machinery_ready:
            # FK cross-check: predict torso quat from pelvis quat + waist encoders
            pred = self.fk.predict(pelvis_quat[None, :], q[None, :].astype(np.float64))  # (1, 4)
            angle, drpy, dquat = quat_divergence(torso_quat[None, :], pred)
            pred_quat = pred[0].astype(np.float32)
            meas_rpy = quat_to_rpy(torso_quat).astype(np.float32)
            pred_rpy = quat_to_rpy(pred[0]).astype(np.float32)
            div_quat = dquat[0].astype(np.float32)
            div_rpy = drpy[0].astype(np.float32)
            div_angle = float(angle[0])
            # motion_anchor_ori_b for the current reconstructed frame + init_quat.
            # _resolve_frame captures init_quat/yaw_offset from the RAW anchor quat; the yaw
            # offset (identity unless zero_imu_yaw) is then applied to the quat feeding the obs.
            anchor_quat = (torso_quat if self.anchor_imu == "torso" else pelvis_quat).astype(np.float32)
            frame = self._resolve_frame(anchor_quat)
            anchor_for_obs = quat_multiply(self._yaw_offset_conj, anchor_quat)
            ori6d, oq, orpy = self.rep.anchor_ori_b_forms(frame, anchor_for_obs, self._init_quat)
            ori6d, oq, orpy = ori6d.astype(np.float32), oq.astype(np.float32), orpy.astype(np.float32)
        else:
            n3, n4, n6 = (np.full(3, np.nan, np.float32), np.full(4, np.nan, np.float32),
                          np.full(6, np.nan, np.float32))
            pred_quat, meas_rpy, pred_rpy = n4, n3, n3
            div_quat, div_rpy, div_angle = n4, n3, np.nan
            ori6d, oq, orpy = n6, n4, n3

        with self.lock:
            b = self.buf
            b["t"].append(L["time"])
            b["pelvis"].append(pelvis.copy())
            b["torso"].append(torso.copy())
            b["pred_quat"].append(pred_quat)
            b["meas_rpy"].append(meas_rpy)
            b["pred_rpy"].append(pred_rpy)
            b["div_quat"].append(div_quat)
            b["div_rpy"].append(div_rpy)
            b["div_angle"].append(div_angle)
            b["ori6d"].append(ori6d)
            b["oq"].append(oq)
            b["orpy"].append(orpy)

    # thread-safe numpy snapshot for the Qt redraw (x-axis = "seconds ago")
    def snapshot(self):
        with self.lock:
            if not self.buf["t"]:
                return None
            out = {}
            for k, v in self.buf.items():
                if k in ("t", "div_angle"):
                    out[k] = np.fromiter(v, dtype=np.float64)
                else:
                    out[k] = np.array(v)
        out["t"] = out["t"] - out["t"][-1]
        return out

    def status_text(self):
        if not self._machinery_ready:
            return "waiting for config (experiment_info / --config) ..."
        ph = PHASE_NAMES.get(self._cur_phase, "n/a" if self._cur_phase is None else str(self._cur_phase))
        cap = "captured" if self._init_captured else "identity"
        return (f"phase: {ph}   frame: {self._cur_frame}/{self.num_frames - 1}   "
                f"anchor: {self.anchor_imu} IMU   capture: {cap}   "
                f"align_heading: {self.align_heading}   zero_imu_yaw: {self.zero_imu_yaw}")


############################################################################
# QT VISUALIZER
############################################################################

def _add_curves(plot, labels, dashed=False, width=1.5, color_off=0):
    curves = []
    for i, lab in enumerate(labels):
        color = COLORS[(i + color_off) % len(COLORS)]
        pen = pg.mkPen(color, width=width, style=_DASH) if dashed else pg.mkPen(color, width=width)
        curves.append(plot.plot(pen=pen, name=lab))
    return curves


class Visualizer:
    """Owns the three pyqtgraph windows and the redraw timer (main thread)."""

    def __init__(self, node: LiveDebugNode, redraw_hz: float):
        self.node = node
        pg.setConfigOptions(antialias=False)
        self.app = pg.mkQApp("live_debug")

        self._build_window1()
        self._build_window2()
        self._build_window3()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(int(1000.0 / redraw_hz))

    # ---- Window 1: IMU + FK cross-check (4 rows x 2 cols) ----
    def _build_window1(self):
        self.win1 = pg.GraphicsLayoutWidget(title="IMU + FK cross-check")
        self.win1.resize(1500, 1000)
        w = self.win1
        self.w1 = {}

        def mkplot(r, c, title, ylabel=None):
            p = w.addPlot(row=r, col=c, title=title)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("left").enableAutoSIPrefix(False)
            if ylabel:
                p.setLabel("left", ylabel)
            p.addLegend(offset=(-1, 1))
            return p

        p = mkplot(0, 0, "pelvis IMU rpy", "[rad]"); self.w1["pelvis_rpy"] = _add_curves(p, RPY)
        p = mkplot(0, 1, "torso IMU rpy");           self.w1["torso_rpy"] = _add_curves(p, RPY)
        p = mkplot(1, 0, "pelvis IMU quat", "quat"); self.w1["pelvis_quat"] = _add_curves(p, QUAT)
        p = mkplot(1, 1, "torso IMU quat");          self.w1["torso_quat"] = _add_curves(p, QUAT)

        p = mkplot(2, 0, "torso quat: measured (solid) vs FK-predicted (dashed)", "quat")
        self.w1["tq_meas"] = _add_curves(p, QUAT)
        self.w1["tq_pred"] = _add_curves(p, [q + " (fk)" for q in QUAT], dashed=True, width=1.0)
        p = mkplot(2, 1, "torso rpy: measured (solid) vs FK-predicted (dashed)")
        self.w1["trpy_meas"] = _add_curves(p, RPY)
        self.w1["trpy_pred"] = _add_curves(p, [r + " (fk)" for r in RPY], dashed=True, width=1.0)

        p = mkplot(3, 0, "measured - predicted divergence (quat)", "quat")
        p.setLabel("bottom", "time [s] (0 = now)")
        self.w1["div_quat"] = _add_curves(p, ["d_" + q for q in QUAT])
        p = mkplot(3, 1, "measured - predicted divergence (rpy)")
        p.setLabel("bottom", "time [s] (0 = now)")
        self.w1["div_rpy"] = _add_curves(p, ["d_" + r for r in RPY])
        self.win1.show()

    # ---- Window 2: torso divergence angle ----
    def _build_window2(self):
        self.win2 = pg.GraphicsLayoutWidget(title="torso divergence angle")
        self.win2.resize(1000, 400)
        p = self.win2.addPlot(row=0, col=0, title="torso divergence angle (measured vs FK-predicted)")
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis("left").enableAutoSIPrefix(False)
        p.setLabel("left", "[rad]")
        p.setLabel("bottom", "time [s] (0 = now)")
        self.w2_curve = p.plot(pen=pg.mkPen("#d62728", width=1.5))
        self.win2.show()

    # ---- Window 3: obs blocks (4 rows x 1 col) + status label ----
    def _build_window3(self):
        self.win3 = QtWidgets.QWidget()
        self.win3.setWindowTitle("obs blocks (base_ang_vel + motion_anchor_ori_b)")
        self.win3.resize(1100, 1000)
        layout = QtWidgets.QVBoxLayout(self.win3)
        self.lbl = QtWidgets.QLabel("waiting for data ...")
        layout.addWidget(self.lbl)
        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw)
        self.w3 = {}

        def mkplot(r, title, ylabel):
            p = glw.addPlot(row=r, col=0, title=title)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.setLabel("left", ylabel)
            p.addLegend(offset=(-1, 1))
            return p

        p = mkplot(0, "base_ang_vel == pelvis gyro", "[rad/s]"); self.w3["angvel"] = _add_curves(p, GYRO)
        p = mkplot(1, "motion_anchor_ori_b (6D) - torso orientation ERROR vs reference", "6D rot")
        self.w3["ori6d"] = _add_curves(p, SIXD)
        p = mkplot(2, "same orientation error as a quaternion", "quat"); self.w3["oq"] = _add_curves(p, QUAT)
        p = mkplot(3, "same orientation error as rpy", "[rad]"); self.w3["orpy"] = _add_curves(p, RPY)
        glw.getItem(3, 0).setLabel("bottom", "time [s] (0 = now)")
        self.win3.show()

    #################################################################
    # REDRAW
    #################################################################

    def _update(self):
        snap = self.node.snapshot()
        if snap is None:
            return
        t = snap["t"]

        # window 1: raw IMU (always) + FK overlays/divergence (once machinery ready)
        for k in range(3):
            self.w1["pelvis_rpy"][k].setData(t, snap["pelvis"][:, k])
            self.w1["torso_rpy"][k].setData(t, snap["torso"][:, k])
        for k in range(4):
            self.w1["pelvis_quat"][k].setData(t, snap["pelvis"][:, 3 + k])
            self.w1["torso_quat"][k].setData(t, snap["torso"][:, 3 + k])

        if self.node._machinery_ready:
            for k in range(4):
                self.w1["tq_meas"][k].setData(t, snap["torso"][:, 3 + k])
                self.w1["tq_pred"][k].setData(t, snap["pred_quat"][:, k])
                self.w1["div_quat"][k].setData(t, snap["div_quat"][:, k])
                self.w3["oq"][k].setData(t, snap["oq"][:, k])
            for k in range(3):
                self.w1["trpy_meas"][k].setData(t, snap["meas_rpy"][:, k])
                self.w1["trpy_pred"][k].setData(t, snap["pred_rpy"][:, k])
                self.w1["div_rpy"][k].setData(t, snap["div_rpy"][:, k])
                self.w3["orpy"][k].setData(t, snap["orpy"][:, k])
            for k in range(6):
                self.w3["ori6d"][k].setData(t, snap["ori6d"][:, k])
            self.w2_curve.setData(t, snap["div_angle"])

        # window 3: base_ang_vel (pelvis gyro) always available
        for k in range(3):
            self.w3["angvel"][k].setData(t, snap["pelvis"][:, 7 + k])

        self.lbl.setText(self.node.status_text())

    def run(self):
        self.app.exec()


############################################################################
# MAIN
############################################################################

def main():
    parser = argparse.ArgumentParser(description="Live debug plotter (debug.py figures 1-3).")
    parser.add_argument("--mode", type=str, default="hw", choices=["sim", "hw"],
                        help='Deployment mode: "hw" (default) or "sim". Selects the time topic.')
    parser.add_argument("--config", type=str, default=None,
                        help="Config yaml name under deploy/configs/. Overrides experiment_info. "
                             "Default: auto-load from the running control node's experiment_info.")
    parser.add_argument("--window", type=float, default=10.0,
                        help="Rolling window length in seconds. Default: 10.0.")
    parser.add_argument("--sample_hz", type=float, default=100.0,
                        help="Rate at which topics are sampled onto the time axis. Default: 100.0.")
    parser.add_argument("--redraw_hz", type=float, default=50.0,
                        help="Plot redraw rate in Hz. Default: 50.0.")
    args = parser.parse_args()

    rclpy.init()
    node = LiveDebugNode(args.mode, args.config, args.window, args.sample_hz)

    # spin ROS in a background thread so the Qt event loop owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    viz = Visualizer(node, args.redraw_hz)
    try:
        viz.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)

    print("Live debug shutdown complete.")


if __name__ == "__main__":
    main()

##
#
# Automated FSM driver -- headless stand-in for deploy/joystick/joystick_ros.py.
#
# Publishes a "connected" joystick (so the control node runs in FSM/single-shot
# mode) and drives the finite state machine through init -> damp -> home ->
# control -> track using the real FiniteStateMachine (fed synthetic button
# presses, exactly as joystick_ros.py would). State transitions are advanced as
# fast as the FSM allows -- the pre-track dwell is only a short settle so the
# motion fires promptly instead of leaving the robot idling.
#
# To run many repetitions cheaply it chains them in ONE session: after each
# playback it resets the sim (robot snapped back to the clean home pose) and
# fires the motion again, so a fall on one rep does not contaminate the next.
#
# Before each 'track' it checks the torso is upright -- a per-rep validity gate,
# so reps that never start standing are flagged and dropped from the stats.
#
# On completion it writes a sidecar JSON with one entry per rep (track start
# time + standing verdict), which the orchestrator uses to align and score each
# tracking-error window.
#
##

import argparse
import json
import os
import sys

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.joystick_utils import JoystickState
from utils.finite_state_machine import FiniteStateMachine


# tilt of body +z from world +z, in degrees, from a wxyz quaternion.
def tilt_deg_from_quat(quat_wxyz):
    x, y = float(quat_wxyz[1]), float(quat_wxyz[2])
    cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))


class FSMDriver(Node):

    def __init__(self, args):
        super().__init__("fsm_driver")

        self.sidecar_path = args.sidecar
        self.tilt_threshold = args.tilt_threshold
        self.reps = args.reps
        self.reset_settle = args.reset_settle
        self.post_settle = args.post_settle
        self.bringup_hold = args.bringup_hold

        # resolve motion duration from the config the run uses
        cfg = self._load_config(args.config)
        self.control_dt = float(cfg["control_dt"])
        motion = np.load(os.path.join(ROOT_DIR, "motions", cfg["motion_path"]))
        self.num_frames = int(motion["joint_pos"].shape[0])
        self.fps = float(motion["fps"])
        self.motion_duration = self.num_frames * self.control_dt

        self.meta = {
            "policy": args.policy,
            "config": os.path.splitext(os.path.basename(args.config))[0],
            "motion_path": cfg["motion_path"],
            "num_frames": self.num_frames,
            "fps": self.fps,
            "control_dt": self.control_dt,
            "motion_duration": self.motion_duration,
            "tilt_threshold": self.tilt_threshold,
        }

        self.fsm = FiniteStateMachine()

        # One rep cycles the FSM fast: damp -> (reset) -> home -> control -> track.
        # Going via damp first makes the control node target frame 0 (not the
        # frozen last frame) before we reset, and re-entering control at the
        # freshly reset home pose gives a clean heading recapture. All dwells are
        # tiny except a short settle before the motion fires.
        #   init/track --LB--> damp --A--> home --LMB--> control --RMB--> track
        self.plan = []
        for r in range(self.reps):
            self.plan += [
                {"kind": "press", "target": "damp",    "button": "LB",  "hold": self.bringup_hold},
                {"kind": "reset"},
                {"kind": "press", "target": "home",    "button": "A",   "hold": self.bringup_hold},
                {"kind": "press", "target": "control", "button": "LMB", "hold": self.reset_settle},
                {"kind": "gate", "rep": r},
                {"kind": "press", "target": "track", "button": "RMB",
                 "hold": self.motion_duration + self.post_settle, "rep": r},
            ]

        self.step_idx = 0
        self.step_reached_at = None

        # sensor / timing state
        self.sim_time = None
        self.torso_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # per-rep records
        self.reps_log = []
        self.cur_rep = None

        # publishers (mirror joystick_ros.py) + the sweep reset channel
        self.command_pub = self.create_publisher(Float32MultiArray, "deploy_robot/joystick", 10)
        self.fsm_pub = self.create_publisher(String, "deploy_robot/fsm", 10)
        self.reset_pub = self.create_publisher(Float64, "deploy_robot/sim_reset", 10)

        self.time_sub = self.create_subscription(Float64, "deploy_robot/simulation_time", self.time_callback, 10)
        self.torso_sub = self.create_subscription(Float32MultiArray, "deploy_robot/torso_imu_state", self.torso_callback, 10)

        self.timer = self.create_timer(0.02, self.tick)
        self.done = False
        print(f"FSM driver: policy={args.policy} reps={self.reps} "
              f"motion_duration={self.motion_duration:.2f}s ({self.num_frames} frames @ {self.fps} fps)")

    def _load_config(self, config_path):
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        with open(os.path.join(ROOT_DIR, "deploy", "configs", config_path), "r") as f:
            return yaml.safe_load(f)

    def time_callback(self, msg):
        self.sim_time = msg.data

    def torso_callback(self, msg):
        self.torso_quat = np.array(msg.data[3:7], dtype=np.float32)

    # publish joystick(connected) + current fsm state, stepping the FSM with an
    # optional button press
    def _publish(self, button=None):
        js = JoystickState(**({button: 1} if button else {}))
        state = self.fsm.step(js)
        cmd = Float32MultiArray(); cmd.data = [1.0, 0.0, 0.0, 0.0]
        self.command_pub.publish(cmd)
        m = String(); m.data = state
        self.fsm_pub.publish(m)
        return state

    def tick(self):
        # wait for the sim to be up (first sim-time) before scheduling
        if self.sim_time is None:
            self._publish()
            return
        if self.step_idx >= len(self.plan):
            self._finish()
            return

        step = self.plan[self.step_idx]
        kind = step["kind"]

        if kind == "reset":
            self.reset_pub.publish(Float64(data=1.0))
            self._publish()  # hold current fsm state
            self._advance()
            return

        if kind == "gate":
            tilt = tilt_deg_from_quat(self.torso_quat)
            self.cur_rep = {
                "rep": step["rep"],
                "gate_tilt_deg": tilt,
                "standing_ok": bool(tilt <= self.tilt_threshold),
            }
            verdict = "OK" if self.cur_rep["standing_ok"] else "FLAGGED"
            print(f"FSM driver: rep {step['rep']} gate tilt={tilt:.1f}deg -> {verdict}")
            self._publish()
            self._advance()
            return

        # kind == "press": press the button until the target state is reached,
        # then hold (releasing the button) for the dwell.
        target, button, hold = step["target"], step["button"], step["hold"]
        if self.fsm.state != target:
            self._publish(button)
            return
        if self.step_reached_at is None:
            self.step_reached_at = self.sim_time
            if target == "track" and self.cur_rep is not None:
                self.cur_rep["track_start_time"] = self.sim_time
                self.reps_log.append(self.cur_rep)
                print(f"FSM driver: rep {step.get('rep')} track started at sim_time={self.sim_time:.3f}")
        self._publish()  # hold in target state, no button
        if (self.sim_time - self.step_reached_at) >= hold:
            self._advance()

    def _advance(self):
        self.step_idx += 1
        self.step_reached_at = None

    def _finish(self):
        if self.done:
            return
        self.done = True
        sidecar = dict(self.meta)
        sidecar["reps"] = self.reps_log
        sidecar["complete"] = True
        os.makedirs(os.path.dirname(os.path.abspath(self.sidecar_path)), exist_ok=True)
        with open(self.sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        n_ok = sum(r.get("standing_ok") for r in self.reps_log)
        print(f"FSM driver: done, {len(self.reps_log)} reps ({n_ok} standing). Wrote {self.sidecar_path}")


def main():
    p = argparse.ArgumentParser(description="Automated FSM driver (chained reps) for sweep experiments.")
    p.add_argument("--config", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--policy", default="")
    p.add_argument("--reps", type=int, default=15, help="Repetitions to chain in one session (default 15).")
    p.add_argument("--tilt-threshold", type=float, default=40.0, dest="tilt_threshold",
                   help="Max torso tilt (deg) at the gate to count as standing. Default 40.")
    # fast transitions: states are only held a couple of ticks; the meaningful
    # dwell is the short per-rep settle after a reset before the motion fires.
    p.add_argument("--bringup-hold", type=float, default=0.02, dest="bringup_hold",
                   help="Dwell in each transition state (s). Default 0.02 (~1 tick).")
    p.add_argument("--reset-settle", type=float, default=0.2, dest="reset_settle",
                   help="Settle time (s) after a reset before firing the motion. Default 0.2.")
    p.add_argument("--post-settle", type=float, default=0.1, dest="post_settle",
                   help="Hold (s) after the motion ends before the next rep (dead time; the "
                        "scoring window is just the motion duration). Default 0.1.")
    args = p.parse_args()

    rclpy.init()
    node = FSMDriver(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print("FSM driver shutdown complete.")


if __name__ == "__main__":
    main()

##
#
# Policy step timing for the hardware controllers (deploy/hardware/go2/control_*.py).
#
# Every policy step publishes one row on deploy_robot/policy_timing (logged by the logger):
#   [seq, step interval, slot offset, imu age, joint age, trigger age, same tick, inference, step]
#   - step interval [ms]: time since the previous policy step (should be ~ctrl_dt)
#   - slot offset   [ms]: fsm_time - step * ctrl_dt, how far the triggering tick is from the step's
#                         ideal time (within +-half a tick with the controllers' TRIGGER_OFFSET)
#   - imu / joint age [ms]: now - read time of the IMU / joint state the policy used
#   - trigger age   [ms]: now - start of the hardware tick whose fsm_time triggered this step
#   - same tick (0/1): the IMU and joint state were read in that triggering tick
#   - inference / step [ms]: policy inference time, whole step (obs + inference + publish)
# Ages need hardware.py's time origin (latched hardware_info); they are NaN until it arrives.
#
##

import json
import time

import numpy as np
from std_msgs.msg import Float32MultiArray, Float64, String

from utils.experiment_utils import HARDWARE_INFO_TOPIC, experiment_info_qos


# print a summary every this many seconds (with print_timing)
TIMING_PRINT_PERIOD = 5.0  # [sec]


class PolicyTiming:

    # create the subscriptions this needs. Call BEFORE creating the fsm_time subscription: the
    # executor handles ready subscriptions in creation order, so hardware_time (published just
    # before fsm_time in the same tick) is then up to date when the policy steps.
    def __init__(self, node, ctrl_dt: float, print_timing: bool = False):
        self.ctrl_dt = ctrl_dt
        self.print_timing = print_timing
        self.time_origin = None      # hardware.py time.monotonic() origin of hardware_time
        self.hw_tick_time = np.nan   # hardware_time of the latest tick
        self.imu_stamp = np.nan      # read time of the latest IMU state [hardware_time sec]
        self.joint_stamp = np.nan    # read time of the latest joint state [hardware_time sec]
        self.seq = 0
        self.prev_step = None
        self.t_step = None
        self.t_infer = None
        self.rows = []
        self.last_print = time.monotonic()
        self.window_start = self.last_print

        self.pub = node.create_publisher(Float32MultiArray, 'deploy_robot/policy_timing', 1)
        node.create_subscription(String, HARDWARE_INFO_TOPIC, self._hardware_info_callback,
                                 experiment_info_qos())
        node.create_subscription(Float64, 'deploy_robot/hardware_time', self._hardware_time_callback, 1)

    def _hardware_info_callback(self, msg):
        try:
            self.time_origin = float(json.loads(msg.data)["time_origin_monotonic"])
        except Exception:
            pass

    def _hardware_time_callback(self, msg):
        self.hw_tick_time = msg.data

    # current time in the hardware_time frame
    def now_hw(self) -> float:
        if self.time_origin is None:
            return np.nan
        return time.monotonic() - self.time_origin

    # stamps appended to imu_state (index 13) / joint_state (index 4n) by hardware.py
    def set_imu_stamp(self, data, n_fields: int):
        if len(data) > n_fields:
            self.imu_stamp = float(data[n_fields])

    def set_joint_stamp(self, data, n_fields: int):
        if len(data) > n_fields:
            self.joint_stamp = float(data[n_fields])

    # call at the very start of the policy step
    def step_start(self):
        self.t_step = time.monotonic()

    # call right before / after policy inference
    def inference_start(self):
        self.t_infer = time.monotonic()

    def inference_end(self):
        self.infer_ms = (time.monotonic() - self.t_infer) * 1e3

    # call after publishing the command
    def step_end(self, fsm_time: float, slot_time: float):
        t_end = time.monotonic()
        now = self.now_hw()
        interval = (self.t_step - self.prev_step) * 1e3 if self.prev_step is not None else np.nan
        self.prev_step = self.t_step
        slot_offset = (fsm_time - slot_time) * 1e3
        # ages at the moment the policy started this step
        now_start = now - (t_end - self.t_step)
        imu_age = (now_start - self.imu_stamp) * 1e3
        joint_age = (now_start - self.joint_stamp) * 1e3
        trigger_age = (now_start - self.hw_tick_time) * 1e3
        same_tick = float(self.imu_stamp >= self.hw_tick_time and self.joint_stamp >= self.hw_tick_time)

        self.seq += 1
        row = [float(self.seq), interval, slot_offset, imu_age, joint_age, trigger_age, same_tick,
               self.infer_ms, (t_end - self.t_step) * 1e3]
        msg = Float32MultiArray()
        msg.data = [float(x) for x in row]
        self.pub.publish(msg)

        if self.print_timing:
            if not self.rows:
                self.window_start = self.t_step
            self.rows.append(row)
            if t_end - self.last_print >= TIMING_PRINT_PERIOD:
                self._print_summary(t_end)

    def _print_summary(self, t_end: float):
        a = np.array(self.rows, dtype=np.float64)
        dt = max(t_end - self.window_start, 1e-6)
        self.rows = []
        self.last_print = t_end

        def mm(col):
            x = a[:, col][np.isfinite(a[:, col])]
            return (x.min(), x.mean(), x.max()) if len(x) else (np.nan,) * 3

        iv, so, ia, ja, ta = mm(1), mm(2), mm(3), mm(4), mm(5)
        print(f"[policy timing] {len(a) / dt:.2f} Hz | interval min/mean/max {iv[0]:.1f}/{iv[1]:.1f}/{iv[2]:.1f} ms"
              f" | slot offset min/mean/max {so[0]:.1f}/{so[1]:.1f}/{so[2]:.1f} ms"
              f" | obs age mean/max imu {ia[1]:.1f}/{ia[2]:.1f} joint {ja[1]:.1f}/{ja[2]:.1f} ms"
              f" | trigger age mean/max {ta[1]:.1f}/{ta[2]:.1f} ms"
              f" | same tick {100.0 * np.nanmean(a[:, 6]):.0f}%"
              f" | inference mean {np.nanmean(a[:, 7]):.2f} ms, step mean {np.nanmean(a[:, 8]):.2f} ms",
              flush=True)

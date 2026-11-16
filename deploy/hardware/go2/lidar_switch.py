##
#
# Turn the Go2's L1 LiDAR off / on (rt/utlidar/switch, std_msgs String "OFF" / "ON"), and test it.
# The switch is not an official Unitree API: community reports say OFF stops the spinning and the
# point cloud, and that it is not persistent (the LiDAR is on again after a reboot).
#
# Uses unitree_sdk2py (cyclonedds, DDS domain 0), NOT ros2 topic pub: Foxy's Fast-DDS on domain 0
# crashes when the Go2 is on the network.
#
# Usage:
#   python deploy/hardware/go2/lidar_switch.py eth0            # test: baseline -> OFF -> hold -> ON
#   python deploy/hardware/go2/lidar_switch.py eth0 --off      # just turn it off
#   python deploy/hardware/go2/lidar_switch.py eth0 --on       # just turn it on
#
# The test measures the point cloud rate on rt/utlidar/cloud (the proxy for "running"); watch the
# cage too, since a stopped cloud does not prove a stopped motor.
#
##

import argparse
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


SWITCH_TOPIC = "rt/utlidar/switch"
CLOUD_TOPIC = "rt/utlidar/cloud"

SEND_REPEATS = 3        # the switch is fire-and-forget: send each command a few times
SEND_PERIOD = 0.2       # [sec]
OFF_MAX_RATE = 0.5      # [Hz] cloud rate below this counts as off
ON_MIN_RATE = 2.0       # [Hz] cloud rate above this counts as on


class CloudCounter:
    """Counts point cloud messages (only the arrival time, the cloud itself is not used)."""

    def __init__(self, topic: str):
        self.lock = threading.Lock()
        self.count = 0
        self.last_time = None
        self.sub = ChannelSubscriber(topic, PointCloud2_)
        self.sub.Init(self._callback, 10)

    def _callback(self, msg):
        with self.lock:
            self.count += 1
            self.last_time = time.monotonic()

    # message rate over the next `duration` seconds
    def rate(self, duration: float) -> float:
        with self.lock:
            c0 = self.count
        time.sleep(duration)
        with self.lock:
            return (self.count - c0) / duration

    # seconds since the last message (None if none yet)
    def silence(self) -> float:
        with self.lock:
            return None if self.last_time is None else time.monotonic() - self.last_time


def send(pub, value: str):
    for _ in range(SEND_REPEATS):
        pub.Write(String_(data=value))
        time.sleep(SEND_PERIOD)
    print(f"sent '{value}' on {SWITCH_TOPIC}")


def wait_for_state(cloud, want_on: bool, timeout: float, window: float = 1.0):
    # poll the cloud rate until it matches, or time out; returns (ok, seconds, last rate)
    t0 = time.monotonic()
    rate = cloud.rate(window)
    while time.monotonic() - t0 < timeout:
        if (rate >= ON_MIN_RATE) if want_on else (rate <= OFF_MAX_RATE):
            return True, time.monotonic() - t0, rate
        rate = cloud.rate(window)
    return False, time.monotonic() - t0, rate


def main():
    parser = argparse.ArgumentParser(description="Turn the Go2 LiDAR off / on, or test the switch.")
    parser.add_argument("network", help='network interface to the robot, e.g. "eth0"')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--off", action="store_true", help="only send OFF")
    mode.add_argument("--on", action="store_true", help="only send ON")
    parser.add_argument("--hold", type=float, default=20.0,
                        help="[sec] test: how long to watch that the LiDAR stays off (default 20)")
    parser.add_argument("--leave-off", action="store_true", help="test: do not turn it back on at the end")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.network)
    pub = ChannelPublisher(SWITCH_TOPIC, String_)
    pub.Init()
    time.sleep(0.5)  # let DDS discovery match the robot before the first write

    if args.off or args.on:
        send(pub, "OFF" if args.off else "ON")
        return

    cloud = CloudCounter(CLOUD_TOPIC)
    results = []

    # 1. baseline: the LiDAR should be running
    print(f"\n[1/4] baseline: measuring {CLOUD_TOPIC} for 3 s ...")
    rate = cloud.rate(3.0)
    print(f"      cloud rate {rate:.1f} Hz")
    if rate < ON_MIN_RATE:
        print(f"      no cloud before the test. Already off, wrong topic, or the robot is not reachable on "
              f"{args.network}? Sending ON first.")
        send(pub, "ON")
        ok, t, rate = wait_for_state(cloud, True, 15.0)
        print(f"      after ON: {rate:.1f} Hz ({'running' if ok else 'still no cloud'})")
        if not ok:
            print("\nFAIL: no point cloud at all, can't test the switch.")
            return
    results.append(("baseline cloud running", True))

    # 2. OFF: the cloud should stop
    print("\n[2/4] sending OFF ...")
    send(pub, "OFF")
    ok, t, rate = wait_for_state(cloud, False, 10.0)
    print(f"      cloud {'stopped' if ok else 'did NOT stop'} ({rate:.1f} Hz after {t:.1f} s)")
    input("      Is the cage still spinning? Look, then press [Enter] ... ")
    results.append(("OFF stops the cloud", ok))

    # 3. hold: does it stay off (reported to restart on its own)?
    if ok:
        print(f"\n[3/4] watching for {args.hold:.0f} s that it stays off ...")
        t0 = time.monotonic()
        restarted = None
        while time.monotonic() - t0 < args.hold:
            r = cloud.rate(1.0)
            if r >= ON_MIN_RATE:
                restarted = time.monotonic() - t0
                break
        if restarted is None:
            print("      stayed off")
        else:
            print(f"      RESTARTED on its own after {restarted:.1f} s")
        results.append((f"stays off for {args.hold:.0f} s", restarted is None))
    else:
        print("\n[3/4] skipped (OFF did not stop the cloud)")

    # 4. ON: the cloud should come back
    if args.leave_off:
        print("\n[4/4] skipped (--leave-off): the LiDAR stays off until ON or a reboot")
    else:
        print("\n[4/4] sending ON ...")
        send(pub, "ON")
        ok, t, rate = wait_for_state(cloud, True, 15.0)
        print(f"      cloud {'back' if ok else 'did NOT come back'} ({rate:.1f} Hz after {t:.1f} s)")
        results.append(("ON restarts the cloud", ok))

    print("\nresults:")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()

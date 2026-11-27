##
#
# Pelvis state estimator -> nav_msgs/Odometry for KCF (the mocap contract in deploy/kcf/kcf_interface.py).
# Both modes: raw mocap position, linear velocity from a least-squares line fit over a short window.
#   --mode robot (default): orientation and body omega from the pelvis IMU, IMU yaw offset to match mocap
#     (offset measured once at startup over YAW_CALIB_SAMPLES ticks; keep the robot still while it runs).
#   --mode mocap: mocap orientation, body omega from a line fit of mocap rotations (no robot needed).
# The raw Motive pose is also published on /deploy_robot/mocap.
# Guards: a lost mocap or IMU stream is held at its last value for HOLD_TIME (with warnings), then Odometry stops.
#
##

# standard imports
import argparse
from pathlib import Path
import sys
import socket
import struct
import threading
import time
from collections import deque

# other imports
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kcf_mpc.profiles import load_profile, resolve_obstacle_config


############################################################################
# CONSTANTS
############################################################################

# NatNet message IDs, ports, and rigid body layout
NAT_REQUEST_MODELDEF = 4
NAT_MODELDEF = 5
NAT_FRAMEOFDATA = 7
NAT_DATASET_RIGIDBODY = 1
MULTICAST_GROUP = "239.255.42.99"
COMMAND_PORT = 1510
DATA_PORT = 1511
RIGID_BODY = struct.Struct("<i3f4ffH")   # id, pos (x, y, z), quat (x, y, z, w), mean error, params

IMU_TIMEOUT = 0.1     # [sec] a pelvis IMU message older than this counts as lost (robot mode)
MOCAP_TIMEOUT = 0.1   # [sec] a tracked mocap pose older than this counts as lost
HOLD_TIME = 0.5       # [sec] guards re-publish the last mocap/IMU value this long, then stop (KCF -> control)
YAW_CALIB_SAMPLES = 100   # robot mode: mocap-IMU yaw samples averaged at startup (1 s at 100 Hz)
ODOM_TOPIC = "/deploy_kcf/mocap"   # same as MOCAP_TOPIC in deploy/kcf/kcf_interface.py (KCF input)
RAW_TOPIC = "/deploy_robot/mocap"  # raw Motive pose: [pos (3), quat (w, x, y, z)]


############################################################################
# MOCAP CLIENT
############################################################################

class MocapClient:
    """
    Keep the latest tracked pose of one Motive rigid body (by name), plus a short position history.
    """

    def __init__(self, server_ip: str, body_name: str, capture_hz: float):

        # Motive capture rate: frame number / capture_hz is the capture time (arrival time is bursty)
        self.capture_hz = capture_hz

        # ask Motive for its model definitions, retrying lost packets (this socket also gives our local IP)
        cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd.connect((server_ip, COMMAND_PORT))
        cmd.settimeout(1.0)
        for _ in range(3):
            cmd.send(struct.pack("<HH", NAT_REQUEST_MODELDEF, 0))
            try:
                data = cmd.recv(65535)
            except (socket.timeout, ConnectionRefusedError):
                continue
            if struct.unpack_from("<H", data)[0] == NAT_MODELDEF:
                break
        else:
            raise TimeoutError(f"No reply from Motive at [{server_ip}]: check streaming is on and the network.")
        local_ip = cmd.getsockname()[0]
        cmd.close()

        # find the body's streaming ID: descriptions are [type, size, data], a rigid body starts with name\0, ID
        self.body_id = None
        off = 8
        for _ in range(struct.unpack_from("<i", data, 4)[0]):
            data_type, size = struct.unpack_from("<ii", data, off)
            off += 8
            if data_type == NAT_DATASET_RIGIDBODY and data[off:].startswith(body_name.encode() + b"\0"):
                self.body_id = struct.unpack_from("<i", data, off + len(body_name) + 1)[0]
                break
            off += size
        if self.body_id is None:
            raise KeyError(f"No rigid body [{body_name}] in Motive.")

        # join the multicast stream on the interface that routes to Motive
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", DATA_PORT))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton(local_ip))

        # latest tracked (pos, quat (w, x, y, z), arrival time) and (capture time, pos, quat) history
        self.pose = None
        self.last_frame = None
        self.history = deque(maxlen=256)
        threading.Thread(target=self.receive_loop, daemon=True).start()

    # receive every frame and keep our body's pose whenever it is tracked
    def receive_loop(self):
        while True:
            data = self.sock.recv(65535)
            t_arrival = time.monotonic()

            # a malformed or truncated packet is skipped, so it can never kill this thread
            try:
                if struct.unpack_from("<H", data)[0] != NAT_FRAMEOFDATA:
                    continue
                t_capture = struct.unpack_from("<i", data, 4)[0] / self.capture_hz
                if t_capture == self.last_frame:
                    continue  # A retransmitted frame must not refresh measurement age.

                # skip header and frame number, then marker sets and loose markers ([count, size, data] each)
                off = 8
                off += 8 + struct.unpack_from("<i", data, off + 4)[0]
                off += 8 + struct.unpack_from("<i", data, off + 4)[0]

                # rigid bodies [count, size, bodies]: read only IDs until ours shows up
                n_bodies = struct.unpack_from("<i", data, off)[0]
                off += 8
                for _ in range(n_bodies):
                    if struct.unpack_from("<i", data, off)[0] == self.body_id:
                        _, px, py, pz, qx, qy, qz, qw, _, params = RIGID_BODY.unpack_from(data, off)
                        if params & 0x01:
                            # a restarted frame counter (Motive restart) starts a fresh history
                            if self.history and t_capture <= self.history[-1][0]:
                                self.history.clear()
                            self.history.append((t_capture, (px, py, pz), (qw, qx, qy, qz)))
                            self.pose = ((px, py, pz), (qw, qx, qy, qz), t_arrival)
                            self.last_frame = t_capture
                        break
                    off += RIGID_BODY.size
            except struct.error:
                continue

    # latest tracked ((x, y, z), (w, x, y, z), age [s]); None if never tracked or older than max_age [s]
    def get_pose(self, max_age: float = 0.1):
        pose = self.pose
        if pose is None:
            return None
        age = time.monotonic() - pose[2]
        return (pose[0], pose[1], age) if age <= max_age else None

    # world linear velocity [m/s]: slope of a least-squares line through the last `window` [s] of positions
    def get_velocity(self, window: float):
        history = list(self.history)
        if len(history) < 2:
            return None
        t = np.array([h[0] for h in history])
        p = np.array([h[1] for h in history])
        recent = t >= t[-1] - window
        t, p = t[recent] - t[recent].mean(), p[recent] - p[recent].mean(axis=0)
        if len(t) < 2:
            return None
        return (t @ p) / (t @ t)

    # body-frame angular velocity [rad/s]: slope of a line fit to the window's rotations relative to the newest
    def get_angular_velocity(self, window: float):
        history = list(self.history)
        if len(history) < 2:
            return None
        t = np.array([h[0] for h in history])
        q = np.array([h[2] for h in history])
        recent = t >= t[-1] - window
        t, q = t[recent], q[recent]
        if len(t) < 2:
            return None

        # relative rotation q_ref^-1 * q_i (w >= 0, so the short way around) as a rotation vector
        w0, v0 = q[-1, 0], -q[-1, 1:]
        w = w0 * q[:, 0] - q[:, 1:] @ v0
        v = w0 * q[:, 1:] + q[:, :1] * v0 + np.cross(v0, q[:, 1:])
        v[w < 0], w[w < 0] = -v[w < 0], -w[w < 0]
        norm = np.linalg.norm(v, axis=1)
        scale = np.where(norm > 1e-9, 2.0 * np.arctan2(norm, w) / np.maximum(norm, 1e-12), 2.0)
        r = v * scale[:, None]

        # line fit of the rotation vectors over capture time
        t, r = t - t.mean(), r - r.mean(axis=0)
        return (t @ r) / (t @ t)


############################################################################
# ESTIMATOR NODE
############################################################################

class EstimatorNode(Node):
    """
    Fuse mocap position/velocity with the pelvis IMU and publish the KCF mocap Odometry.
    """

    def __init__(self, mocap: MocapClient, window: float, rate: float, mode: str = 'robot',
                 *, obstacle=None, obstacle_settings=None):

        super().__init__('mocap_estimator')
        self.mocap = mocap
        self.window = window
        self.mode = mode
        self.obstacle = obstacle
        self.obstacle_settings = resolve_obstacle_config(obstacle_settings)
        self.last_obstacle_arrival = None

        # latest pelvis IMU orientation (w, x, y, z), body-frame gyro, and arrival time
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_omega = np.zeros(3)
        self.imu_time = None

        # robot mode: yaw offset [rad] added to the IMU so its yaw matches mocap (None until calibrated)
        self.yaw_samples = []
        self.yaw_offset = None

        # guards: last good mocap velocities, and the arrival time [s, monotonic] when each stream was lost
        self.last_vel_world = None
        self.last_omega_mocap = None
        self.mocap_lost_at = None
        self.imu_lost_at = None
        self.mocap_expired = self.imu_expired = False

        # ROS publisher, subscriber, and publish timer
        self.odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 1)
        self.raw_pub = self.create_publisher(Float32MultiArray, RAW_TOPIC, 1)
        self.obstacle_pub = (self.create_publisher(PointStamped, self.obstacle_settings['topic'], 1)
                             if obstacle is not None else None)
        self.imu_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.imu_callback, 1)
        self.timer = self.create_timer(1.0 / rate, self.publish_callback)

    def publish_obstacle(self):
        if self.obstacle is None:
            return
        # Use one snapshot: its arrival time identifies the sample and its age.
        pose = self.obstacle.pose
        if pose is None or pose[2] == self.last_obstacle_arrival:
            return
        position, _, arrival = pose
        age = time.monotonic() - arrival
        if not 0. <= age <= self.obstacle_settings['max_age_seconds']:
            return
        if not np.isfinite(position).all():
            return
        msg = PointStamped()
        msg.header.frame_id = 'world'
        msg.header.stamp = (self.get_clock().now() - Duration(seconds=age)).to_msg()
        msg.point.x, msg.point.y, msg.point.z = map(float, position)
        self.obstacle_pub.publish(msg)
        self.last_obstacle_arrival = arrival

    # pelvis IMU data: [rpy(3), quat(4), gyro(3), acc(3)]
    def imu_callback(self, msg):
        data = np.array(msg.data, dtype=np.float64)
        self.imu_quat = data[3:7]
        self.imu_omega = data[7:10]
        self.imu_time = time.monotonic()

    # publish the latest estimate (nothing while the body is not tracked, so KCF sees mocap as stale)
    def publish_callback(self):

        self.publish_obstacle()  # Independent of pelvis/IMU initialization and loss.
        now = time.monotonic()

        # fresh mocap: raw pose, line-fit velocities (kept for the guard), and a recovery warning
        pose = self.mocap.get_pose(MOCAP_TIMEOUT)
        if pose is not None:
            pos, mocap_quat, age = pose
            self.raw_pub.publish(Float32MultiArray(data=[*pos, *mocap_quat]))
            vel_world = self.mocap.get_velocity(self.window)
            if vel_world is None:
                return
            self.last_vel_world = vel_world
            if self.mode == 'mocap':
                self.last_omega_mocap = self.mocap.get_angular_velocity(self.window)
            if self.mocap_lost_at is not None:
                print(f"WARNING: mocap recovered after {now - self.mocap_lost_at:.2f} s.", flush=True)
                self.mocap_lost_at, self.mocap_expired = None, False

        # mocap guard: re-publish the last pose/velocity (stamped now) for up to HOLD_TIME, then stop
        else:
            if self.mocap.pose is None or self.last_vel_world is None:
                return
            pos, mocap_quat, arrival = self.mocap.pose
            if self.mocap_lost_at is None:
                self.mocap_lost_at = arrival
                print(f"WARNING: mocap lost (no tracked pose). Holding the last pose for up to {HOLD_TIME} s.", flush=True)
            if now - self.mocap_lost_at > HOLD_TIME:
                if not self.mocap_expired:
                    print(f"WARNING: mocap lost for over {HOLD_TIME} s: Odometry stopped (KCF falls back to control).", flush=True)
                    self.mocap_expired = True
                return
            age, vel_world = 0.0, self.last_vel_world

        # robot: IMU orientation and gyro; mocap: mocap orientation, line-fit omega
        if self.mode == 'robot':

            # IMU guard: keep the last IMU message for up to HOLD_TIME, then stop
            if self.imu_time is None:
                return
            if now - self.imu_time >= IMU_TIMEOUT:
                if self.imu_lost_at is None:
                    self.imu_lost_at = self.imu_time
                    print(f"WARNING: pelvis IMU lost. Holding the last IMU message for up to {HOLD_TIME} s.", flush=True)
                if now - self.imu_lost_at > HOLD_TIME:
                    if not self.imu_expired:
                        print(f"WARNING: pelvis IMU lost for over {HOLD_TIME} s: Odometry stopped (KCF falls back to control).", flush=True)
                        self.imu_expired = True
                    return
            elif self.imu_lost_at is not None:
                print(f"WARNING: pelvis IMU recovered after {now - self.imu_lost_at:.2f} s.", flush=True)
                self.imu_lost_at, self.imu_expired = None, False
            w, x, y, z = self.imu_quat / np.linalg.norm(self.imu_quat)
            omega = self.imu_omega

            # calibrate once, on fresh samples only: circular mean of (mocap yaw - IMU yaw)
            if self.yaw_offset is None:
                if self.mocap_lost_at is not None or self.imu_lost_at is not None:
                    return
                mw, mx, my, mz = mocap_quat
                yaw_mocap = np.arctan2(2 * (mw * mz + mx * my), 1 - 2 * (my * my + mz * mz))
                yaw_imu = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                self.yaw_samples.append(yaw_mocap - yaw_imu)
                if len(self.yaw_samples) < YAW_CALIB_SAMPLES:
                    return
                self.yaw_offset = np.arctan2(np.mean(np.sin(self.yaw_samples)), np.mean(np.cos(self.yaw_samples)))
                print(f"Yaw calibrated: IMU yaw offset {np.degrees(self.yaw_offset):+.2f} deg to match mocap.")

            # IMU orientation rotated by the yaw offset about world z: q = q_offset * q_imu
            c, s = np.cos(self.yaw_offset / 2), np.sin(self.yaw_offset / 2)
            quat = np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
        else:
            omega = self.last_omega_mocap
            if omega is None:
                return
            quat = np.array(mocap_quat) / np.linalg.norm(mocap_quat)

        # world velocity into the body frame with the published quaternion (KCF rotates it back with it)
        w, x, y, z = quat
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        vel_body = rotation.T @ vel_world

        # stamp with the measurement time on the ROS clock (now minus the mocap pose age; a held pose is stamped now)
        msg = Odometry()
        msg.header.stamp = (self.get_clock().now() - Duration(seconds=age)).to_msg()
        msg.header.frame_id, msg.child_frame_id = 'world', 'pelvis'
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        p.x, p.y, p.z = pos
        o.w, o.x, o.y, o.z = quat.tolist()
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        lin.x, lin.y, lin.z = vel_body.tolist()
        ang.x, ang.y, ang.z = omega.tolist()
        self.odom_pub.publish(msg)


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Publish the pelvis Odometry for KCF from OptiTrack + pelvis IMU.')
    parser.add_argument('--mode', type=str, default='robot', choices=['robot', 'mocap'],
                        help='robot: orientation/omega from the pelvis IMU; mocap: everything from mocap')
    parser.add_argument('--server', type=str, default='192.168.1.15', help='IP of the Motive PC')
    parser.add_argument('--body', type=str, default='kcf', help='rigid body name in Motive')
    parser.add_argument('--capture-hz', type=float, default=240.0, help='Motive capture rate [Hz]')
    parser.add_argument('--window', type=float, default=0.04, help='velocity line-fit window [s]')
    parser.add_argument('--rate', type=float, default=100.0, help='publish rate [Hz]')
    parser.add_argument('--profile', default='walking', help='Existing KCF YAML profile for obstacle settings')
    args = parser.parse_args()

    obstacle_settings = load_profile(args.profile).obstacle
    if obstacle_settings['enabled'] and obstacle_settings['body'] == args.body:
        parser.error('Robot and obstacle must be different Motive rigid bodies')
    # connect to Motive and track the body
    mocap = MocapClient(args.server, args.body, args.capture_hz)
    obstacle = (MocapClient(args.server, obstacle_settings['body'], args.capture_hz)
                if obstacle_settings['enabled'] else None)
    if obstacle is not None:
        print(f"Tracking obstacle [{obstacle_settings['body']}] -> {obstacle_settings['topic']}; "
              "rigid-body origin must be its center in the robot's world frame.", flush=True)
    print(f"Tracking rigid body [{args.body}] (ID {mocap.body_id}) from Motive [{args.server}].")
    print(f"    Mode [{args.mode}]: orientation and omega from the "
          f"{'pelvis IMU, yaw aligned to mocap' if args.mode == 'robot' else 'mocap (line fit)'}.")
    print(f"    Guards: a lost mocap/IMU stream is held at its last value for {HOLD_TIME} s, then Odometry stops.")
    if args.mode == 'robot':
        print(f"    Calibrating the IMU yaw offset over the first {YAW_CALIB_SAMPLES} samples: keep the robot still.")
    print(f"    Publishing [{ODOM_TOPIC}] at {args.rate:g} Hz, velocity window {1000 * args.window:g} ms.")
    print(f"    Publishing raw mocap pose [{RAW_TOPIC}].")

    # run the estimator node
    rclpy.init()
    node = EstimatorNode(mocap, args.window, args.rate, args.mode,
                         obstacle=obstacle, obstacle_settings=obstacle_settings)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

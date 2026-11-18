"""OptiTrack-adapter placeholder: publish simulated pelvis on the shared mocap input.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from kcf_interface import SIM_STATE_TOPIC, MOCAP_TOPIC


class DummyMocap(Node):
    def __init__(self):
        super().__init__('kcf_dummy_mocap')
        self.last_time = None
        self.publisher = self.create_publisher(Odometry, MOCAP_TOPIC, 1)
        self.create_subscription(Float64MultiArray, SIM_STATE_TOPIC, self.callback, 1)

    def callback(self, msg):
        a = np.asarray(msg.data, dtype=float)
        if a.shape != (60,) or not np.isfinite(a).all():
            return
        t = a[0]
        if self.last_time is not None and 0 <= t - self.last_time < 0.02 - 1e-6:
            return
        self.last_time = t
        msg = Odometry()
        msg.header.frame_id, msg.child_frame_id = 'world', 'pelvis'
        ns = round(t * 1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1000000000)
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        p.x, p.y, p.z = a[1:4].tolist()
        o.w, o.x, o.y, o.z = a[4:8].tolist()
        w, x, y, z = a[4:8]
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        velocity = rotation.T @ a[31:34]
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        lin.x, lin.y, lin.z = velocity.tolist()
        ang.x, ang.y, ang.z = a[34:37].tolist()
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = DummyMocap()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

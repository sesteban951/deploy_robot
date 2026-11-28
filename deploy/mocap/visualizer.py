##
#
# Live MuJoCo view of the pose published by estimator.py: the G1 pelvis mesh with an RGB triad (x red, y green, z blue).
# Linear velocity is a magenta arrow, angular velocity a cyan arrow, both from the pelvis origin (world frame).
# Drawn in the estimator's world frame; the triad at the origin (white dot) is that world frame.
# The obstacle (Motive body obstacle_1, from the estimator's obstacle topic) is a 6 inch red ball.
#
##

# standard imports
import argparse
import os
import time

# other imports
import mujoco
import mujoco.viewer
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

# custom imports
from estimator import ODOM_TOPIC
from kcf_mpc.profiles import load_profile


# G1 meshes (the pelvis meshes sit at the pelvis frame origin, as in models/g1/g1_29dof_mode11.xml)
ASSETS_DIR = os.getenv("DEPLOY_ROOT_DIR") + "/models/g1/assets"

OBSTACLE_RADIUS = 0.0762  # [m] 6 inch diameter ball

# world: grid floor and the world origin triad; mocap body: the G1 pelvis mesh with its own triad
SCENE_XML = f"""
<mujoco>
  <compiler meshdir="{ASSETS_DIR}"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.1"/>
    <mesh name="pelvis" file="pelvis.STL"/>
    <mesh name="pelvis_contour" file="pelvis_contour_link.STL"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 0.01" material="grid"/>
    <geom type="sphere" size="0.015" rgba="1 1 1 1"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.006" rgba="1 0 0 1"/>
    <geom type="capsule" fromto="0 0 0 0 0.3 0" size="0.006" rgba="0 1 0 1"/>
    <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.006" rgba="0 0 1 1"/>
    <body name="body" mocap="true">
      <geom name="pelvis" type="mesh" mesh="pelvis" rgba="0.2 0.2 0.2 0.6"/>
      <geom name="pelvis_contour" type="mesh" mesh="pelvis_contour" rgba="0.7 0.7 0.7 0.6"/>
      <geom name="x" type="capsule" fromto="0 0 0 0.2 0 0" size="0.008" rgba="1 0 0 1"/>
      <geom name="y" type="capsule" fromto="0 0 0 0 0.2 0" size="0.008" rgba="0 1 0 1"/>
      <geom name="z" type="capsule" fromto="0 0 0 0 0 0.2" size="0.008" rgba="0 0 1 1"/>
    </body>
    <body name="obstacle" mocap="true">
      <geom name="obstacle" type="sphere" size="{OBSTACLE_RADIUS}" rgba="0.9 0.08 0.08 0"/>
    </body>
  </worldbody>
</mujoco>
"""

STALE_TIME = 0.1   # [sec] the body fades when no Odometry arrived for this long

# velocity arrows: length per unit velocity, shaft width, and color
LIN_SCALE = 0.5    # [m per m/s]
ANG_SCALE = 0.25   # [m per rad/s]
ARROW_WIDTH = 0.01 # [m]
LIN_RGBA = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)   # magenta
ANG_RGBA = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)   # cyan


############################################################################
# VISUALIZER NODE
############################################################################

class VisualizerNode(Node):
    """
    Keep the latest pose from the estimator's Odometry topic.
    """

    def __init__(self, topic: str, obstacle_topic: str):
        super().__init__('mocap_visualizer')
        self.pos = None             # (x, y, z) [m]
        self.quat = None            # (w, x, y, z)
        self.lin = np.zeros(3)      # body-frame linear velocity [m/s]
        self.ang = np.zeros(3)      # body-frame angular velocity [rad/s]
        self.time = None            # arrival time [s, monotonic]
        self.odom_sub = self.create_subscription(Odometry, topic, self.odom_callback, 1)
        self.obstacle_pos = None    # (x, y, z) [m] ball center
        self.obstacle_time = None   # arrival time [s, monotonic]
        self.obstacle_sub = self.create_subscription(PointStamped, obstacle_topic, self.obstacle_callback, 1)

    # latest obstacle center (only published while Motive tracks obstacle_1)
    def obstacle_callback(self, msg):
        self.obstacle_pos = (msg.point.x, msg.point.y, msg.point.z)
        self.obstacle_time = time.monotonic()

    # latest pelvis pose (world -> pelvis) and body-frame twist
    def odom_callback(self, msg):
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        self.pos = (p.x, p.y, p.z)
        self.quat = (o.w, o.x, o.y, o.z)
        self.lin = np.array([lin.x, lin.y, lin.z])
        self.ang = np.array([ang.x, ang.y, ang.z])
        self.time = time.monotonic()


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize the pose published by estimator.py.')
    parser.add_argument('--topic', type=str, default=ODOM_TOPIC, help='nav_msgs/Odometry topic to show')
    parser.add_argument('--profile', default='walking', help='KCF YAML profile for the obstacle topic')
    args = parser.parse_args()
    obstacle_topic = load_profile(args.profile).obstacle['topic']

    # subscribe to the estimator output
    rclpy.init()
    node = VisualizerNode(args.topic, obstacle_topic)
    print(f"Visualizing [{args.topic}] and the obstacle ball on [{obstacle_topic}].")

    # build the scene
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    triad_ids = [model.geom(name).id for name in ("x", "y", "z")]
    mesh_ids = [model.geom(name).id for name in ("pelvis", "pelvis_contour")]
    body_id = model.body("body").id
    obstacle_mocap = model.body_mocapid[model.body("obstacle").id]
    obstacle_geom = model.geom("obstacle").id

    # render at 60 Hz; the body fades while no pose arrives (it keeps its last pose)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.0
            scn = viewer.user_scn
            assert scn is not None, "viewer has no user scene for the velocity arrows"
            while viewer.is_running():
                rclpy.spin_once(node, timeout_sec=0.0)
                if node.pos is not None:
                    data.mocap_pos[0], data.mocap_quat[0] = node.pos, node.quat
                live = node.time is not None and time.monotonic() - node.time < STALE_TIME
                model.geom_rgba[triad_ids, 3] = 1.0 if live else 0.2
                model.geom_rgba[mesh_ids, 3] = 0.6 if live else 0.15

                # obstacle ball: hidden until obstacle_1 is seen, faded while its data is stale
                if node.obstacle_pos is not None:
                    data.mocap_pos[obstacle_mocap] = node.obstacle_pos
                    fresh = time.monotonic() - node.obstacle_time < STALE_TIME
                    model.geom_rgba[obstacle_geom, 3] = 1.0 if fresh else 0.25
                mujoco.mj_forward(model, data)

                # velocity arrows from the pelvis origin, rotated from body to world (hidden while stale)
                scn.ngeom = 0
                if live:
                    origin, rotation = data.mocap_pos[0], data.xmat[body_id].reshape(3, 3)
                    for vec, rgba in ((LIN_SCALE * rotation @ node.lin, LIN_RGBA), (ANG_SCALE * rotation @ node.ang, ANG_RGBA)):
                        if np.linalg.norm(vec) < 1e-3:
                            continue
                        geom = scn.geoms[scn.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH, origin, origin + vec)
                        scn.ngeom += 1
                viewer.sync()
                time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

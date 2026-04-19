##
#
# Data logging node for the Mujoco simulation.
#
##

# standard imports
import argparse
import datetime
import os
import sys
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64

# directory imports
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.logger import Logger


############################################################################
# CONSTANTS
############################################################################

# 29dof G1 vector sizes (matches what simulation.py publishes)
G1_NUM_MOTOR = 29
JOINT_STATE_N = 4 * G1_NUM_MOTOR   # [q, dq, ddq, tau_est]        = 116
COMMAND_N     = 5 * G1_NUM_MOTOR   # [q_des, dq_des, Kp, Kd, tau_ff] = 145
IMU_N         = 13                 # [rpy(3), quat(4), gyro(3), acc(3)]
JOYSTICK_N    = 4                  # [connected, vx, vy, omega]
SIM_TIME_N    = 1                  # [simulation_time]


############################################################################
# LOG NODE
############################################################################

class LogNode(Node):
    """
    Asynchronous logging node that subscribes to simulation topics and writes
    them to an HDF5 file via utils.logger.Logger.
    """

    def __init__(self, output_path: str, log_freq: float, dump_period: float):

        super().__init__('log_node')

        self.output_path = output_path
        self.log_freq = log_freq
        self.log_period = 1.0 / log_freq
        self.dump_period = dump_period

        # one Logger per topic, all writing into the same HDF5 file
        self.joint_logger      = Logger(output_path, JOINT_STATE_N, dataset_name="joint_state")
        self.pelvis_imu_logger = Logger(output_path, IMU_N,         dataset_name="pelvis_imu")
        self.torso_imu_logger  = Logger(output_path, IMU_N,         dataset_name="torso_imu")
        self.command_logger    = Logger(output_path, COMMAND_N,     dataset_name="command")
        self.joystick_logger   = Logger(output_path, JOYSTICK_N,    dataset_name="joystick")
        self.sim_time_logger   = Logger(output_path, SIM_TIME_N,    dataset_name="simulation_time")

        self._loggers = [
            self.joint_logger,
            self.pelvis_imu_logger,
            self.torso_imu_logger,
            self.command_logger,
            self.joystick_logger,
            self.sim_time_logger,
        ]

        # latest message cache (None until first message arrives on the topic)
        self.latest_joint      = None
        self.latest_pelvis_imu = None
        self.latest_torso_imu  = None
        self.latest_command    = None
        self.latest_joystick   = None
        self.latest_sim_time   = None

        # ROS subscribers
        self.joint_sub      = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state',      self.joint_callback,      10)
        self.pelvis_imu_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_callback, 10)
        self.torso_imu_sub  = self.create_subscription(Float32MultiArray, 'deploy_robot/torso_imu_state',  self.torso_imu_callback,  10)
        self.command_sub    = self.create_subscription(Float32MultiArray, 'deploy_robot/command',          self.command_callback,    10)
        self.joystick_sub   = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick',         self.joystick_callback,   10)
        self.sim_time_sub   = self.create_subscription(Float64,           'deploy_robot/simulation_time',  self.sim_time_callback,   10)

        # periodic log timer (snapshots latest messages into the loggers)
        self.log_timer = self.create_timer(self.log_period, self.log_callback)

        # periodic dump timer (flushes in-memory buffers to disk)
        self.dump_timer = self.create_timer(self.dump_period, self.dump_callback)

        print(f"Log node initialized.")
        print(f"    Output file:   {output_path}")
        print(f"    Log frequency: {log_freq} Hz ({self.log_period} s)")
        print(f"    Dump period:   {dump_period} s")


    #################################################################
    # CALLBACKS
    #################################################################

    # each callback just caches the latest message; the log_timer does the writing
    def joint_callback(self, msg: Float32MultiArray):
        self.latest_joint = np.array(msg.data, dtype=np.float32)

    def pelvis_imu_callback(self, msg: Float32MultiArray):
        self.latest_pelvis_imu = np.array(msg.data, dtype=np.float32)

    def torso_imu_callback(self, msg: Float32MultiArray):
        self.latest_torso_imu = np.array(msg.data, dtype=np.float32)

    def command_callback(self, msg: Float32MultiArray):
        self.latest_command = np.array(msg.data, dtype=np.float32)

    def joystick_callback(self, msg: Float32MultiArray):
        self.latest_joystick = np.array(msg.data, dtype=np.float32)

    def sim_time_callback(self, msg: Float64):
        self.latest_sim_time = np.array([msg.data], dtype=np.float32)


    #################################################################
    # LOGGING
    #################################################################

    # snapshot the latest cached messages into the loggers at log_freq
    # only starts logging once EVERY topic has published at least once, so
    # all datasets share the same start/end and the same row index.
    def log_callback(self):
        caches = [
            self.latest_joint,
            self.latest_pelvis_imu,
            self.latest_torso_imu,
            self.latest_command,
            self.latest_joystick,
            self.latest_sim_time,
        ]
        if any(c is None for c in caches):
            return
        self.joint_logger.log(self.latest_joint)
        self.pelvis_imu_logger.log(self.latest_pelvis_imu)
        self.torso_imu_logger.log(self.latest_torso_imu)
        self.command_logger.log(self.latest_command)
        self.joystick_logger.log(self.latest_joystick)
        self.sim_time_logger.log(self.latest_sim_time)


    #################################################################
    # DUMP / SHUTDOWN
    #################################################################

    # periodically flush all buffers to disk
    def dump_callback(self):
        for logger in self._loggers:
            logger.dump()

    # flush remaining buffers and close all loggers
    def close_loggers(self):
        for logger in self._loggers:
            logger.close()


############################################################################
# MAIN FUNCTION
############################################################################

def default_output_path() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(ROOT_DIR, "logs", "simulation") if ROOT_DIR else os.path.join("logs", "simulation")
    return os.path.join(logs_dir, f"{ts}.h5")


def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Asynchronous data logging node (simulation).'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HDF5 file path. Defaults to logs/<YYYYMMDD_HHMMSS>.h5.'
    )
    parser.add_argument(
        '--hz',
        type=float,
        default=200.0,
        help='Logging frequency in Hz (rate at which latest messages are snapshotted). Default: 200.0.'
    )
    parser.add_argument(
        '--dump_period',
        type=float,
        default=1.0,
        help='How often (in seconds) to flush the in-memory buffer to disk. Default: 1.0.'
    )
    cli_args = parser.parse_args()

    output_path = cli_args.output or default_output_path()

    # create the log node
    log_node = LogNode(output_path, cli_args.hz, cli_args.dump_period)

    try:
        # spin the node
        rclpy.spin(log_node)

    except KeyboardInterrupt:
        pass

    finally:
        # flush remaining buffers and close files, then shut down ROS2
        log_node.close_loggers()
        log_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

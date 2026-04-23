##
#
# Data logging node for the Unitree G1 deployment.
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
from std_msgs.msg import Float32MultiArray, Float64, String

# directory imports
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.logger import Logger


############################################################################
# CONSTANTS
############################################################################

# 29dof G1 vector sizes
G1_NUM_MOTOR  = 29
COMMAND_SIZE     = 5 * G1_NUM_MOTOR  # 145 = [q_des, dq_des, Kp, Kd, tau_ff]
JOINT_STATE_SIZE = 4 * G1_NUM_MOTOR  # 116 = [q, dq, ddq, tau_est]
IMU_STATE_SIZE   = 13                # 13  = [rpy, quat, gyro, acc]
JOY_STATE_SIZE   = 4                 # 4   = [connected, vx, vy, omega]
TIME_SIZE        = 1                 # 1   = [time]

# start logging only while in this FSM state (hardware mode only)
TARGET_FSM_STATE = "control"

# per-mode config: everything that differs between sim and hardware
MODE_CONFIG = {
    "sim": {
        "time_topic":   "deploy_robot/simulation_time",
        "use_fsm":      False,
        "logs_subdir":  "simulation",
    },
    "hardware": {
        "time_topic":   "deploy_robot/fsm_time",
        "use_fsm":      True,
        "logs_subdir":  "hardware",
    },
}


############################################################################
# LOG NODE
############################################################################

class LogNode(Node):
    """
    Asynchronous logging node that subscribes to deploy topics and writes
    them to an HDF5 file via utils.logger.Logger.

    mode="sim":       logs always; uses deploy_robot/simulation_time.
    mode="hardware":  gated on fsm_state == 'control'; uses deploy_robot/fsm_time.
    """

    def __init__(self, mode: str, output_path: str, log_freq: float, dump_period: float):

        super().__init__('log_node')

        self.mode = mode
        self.cfg = MODE_CONFIG[mode]
        log_period = 1.0 / log_freq

        # current FSM state (only used in hardware mode)
        self.fsm_state = "init"

        # one Logger per topic, all writing into the same HDF5 file
        self.joint_logger      = Logger(output_path, JOINT_STATE_SIZE, dataset_name="joint_state")
        self.pelvis_imu_logger = Logger(output_path, IMU_STATE_SIZE,   dataset_name="pelvis_imu")
        self.torso_imu_logger  = Logger(output_path, IMU_STATE_SIZE,   dataset_name="torso_imu")
        self.command_logger    = Logger(output_path, COMMAND_SIZE,     dataset_name="command")
        self.joystick_logger   = Logger(output_path, JOY_STATE_SIZE,   dataset_name="joystick")
        self.time_logger       = Logger(output_path, TIME_SIZE,        dataset_name="time")

        self._loggers = [
            self.joint_logger,
            self.pelvis_imu_logger,
            self.torso_imu_logger,
            self.command_logger,
            self.joystick_logger,
            self.time_logger,
        ]

        # latest message cache (None until first message arrives on the topic)
        self.latest_joint      = None
        self.latest_pelvis_imu = None
        self.latest_torso_imu  = None
        self.latest_command    = None
        self.latest_joystick   = None
        self.latest_time       = None

        # ROS subscribers
        self.joint_sub      = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state',      self.joint_callback,      10)
        self.pelvis_imu_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/pelvis_imu_state', self.pelvis_imu_callback, 10)
        self.torso_imu_sub  = self.create_subscription(Float32MultiArray, 'deploy_robot/torso_imu_state',  self.torso_imu_callback,  10)
        self.command_sub    = self.create_subscription(Float32MultiArray, 'deploy_robot/command',          self.command_callback,    10)
        self.joystick_sub   = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick',         self.joystick_callback,   10)
        self.time_sub       = self.create_subscription(Float64,           self.cfg["time_topic"],          self.time_callback,       10)

        # FSM subscription only in hardware mode
        if self.cfg["use_fsm"]:
            self.fsm_sub = self.create_subscription(String, 'deploy_robot/fsm', self.fsm_callback, 10)

        # periodic log timer (snapshots latest messages into the loggers)
        self.log_timer = self.create_timer(log_period, self.log_callback)

        # periodic dump timer (flushes in-memory buffers to disk)
        self.dump_timer = self.create_timer(dump_period, self.dump_callback)

        print(f"Log node initialized.")
        print(f"    Mode:          {mode}")
        print(f"    Output file:   {output_path}")
        print(f"    Log frequency: {log_freq} Hz ({log_period} s)")
        print(f"    Dump period:   {dump_period} s")
        print(f"    Time topic:    {self.cfg['time_topic']}")
        if self.cfg["use_fsm"]:
            print(f"    Logging while fsm_state == '{TARGET_FSM_STATE}'.")


    #################################################################
    # CALLBACKS
    #################################################################

    def fsm_callback(self, msg: String):
        self.fsm_state = msg.data

    def _logging_enabled(self) -> bool:
        if not self.cfg["use_fsm"]:
            return True
        return self.fsm_state == TARGET_FSM_STATE

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

    def time_callback(self, msg: Float64):
        self.latest_time = np.array([msg.data], dtype=np.float32)


    #################################################################
    # LOGGING
    #################################################################

    # snapshot the latest cached messages into the loggers at log_freq
    def log_callback(self):
        if not self._logging_enabled():
            return
        caches = [
            self.latest_joint,
            self.latest_pelvis_imu,
            self.latest_torso_imu,
            self.latest_command,
            self.latest_joystick,
            self.latest_time,
        ]
        if any(c is None for c in caches):
            return
        self.joint_logger.log(self.latest_joint)
        self.pelvis_imu_logger.log(self.latest_pelvis_imu)
        self.torso_imu_logger.log(self.latest_torso_imu)
        self.command_logger.log(self.latest_command)
        self.joystick_logger.log(self.latest_joystick)
        self.time_logger.log(self.latest_time)

    # periodically flush all buffers to disk
    def dump_callback(self):
        for logger in self._loggers:
            logger.dump()


    #################################################################
    # SHUTDOWN
    #################################################################

    # flush remaining buffers and close all loggers
    def close_loggers(self):
        for logger in self._loggers:
            logger.close()


############################################################################
# MAIN FUNCTION
############################################################################

def build_output_path(mode: str, filename: str) -> str:
    # <ROOT_DIR>/logs/<subdir>/<filename>.h5
    subdir = MODE_CONFIG[mode]["logs_subdir"]
    root = ROOT_DIR if ROOT_DIR else ""
    logs_dir = os.path.join(root, "logs", subdir)

    return os.path.join(logs_dir, f"{filename}.h5")


def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Asynchronous data logging node (sim + hardware).'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=["sim", "hardware"],
        help='Deployment mode: "sim" or "hardware".'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Output filename (without .h5 extension). Defaults to <YYYYMMDD_HHMMSS>. Saved under logs/<mode>/.'
    )
    parser.add_argument(
        '--hz',
        type=float,
        default=200.0,
        help='Logging frequency in Hz. Default: 200.0.'
    )
    parser.add_argument(
        '--dump_period',
        type=float,
        default=1.0,
        help='How often (in seconds) to flush the in-memory buffer to disk. Default: 1.0 seconds.'
    )
    args = parser.parse_args()

    # filename: use the user-provided one, or fall back to a timestamp
    if args.filename is not None:
        filename = args.filename
    else:
        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = build_output_path(args.mode, filename)


    # create the log node
    log_node = LogNode(args.mode, output_path, args.hz, args.dump_period)

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

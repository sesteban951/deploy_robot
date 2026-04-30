##
#
# Data logging node for deployment in both simulation and hardware.
#
##

# standard imports
import argparse
import datetime
import os
import sys
import time
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

# start logging only while in this FSM state (hardware mode only)
TARGET_FSM_STATE = "control"

# datasets that MUST have a publisher; logger hard-fails if any are missing at discovery
REQUIRED_DATASETS = ["joint_state", "pelvis_imu", "torso_imu", "time"]

# datasets logged only if a publisher exists at discovery time (e.g. no joystick connected)
OPTIONAL_DATASETS = ["command", "joystick"]

# all candidate datasets (topic discovery picks the active subset)
DATASET_NAMES = REQUIRED_DATASETS + OPTIONAL_DATASETS

# dataset name -> topic name (time is mode-dependent, resolved via MODE_CONFIG)
DATASET_TOPICS = {
    "joint_state": "deploy_robot/joint_state",
    "pelvis_imu":  "deploy_robot/pelvis_imu_state",
    "torso_imu":   "deploy_robot/torso_imu_state",
    "command":     "deploy_robot/command",
    "joystick":    "deploy_robot/joystick",
}

# per-mode config: everything that differs between sim and hardware
MODE_CONFIG = {
    "sim": {
        "time_topic":   "deploy_robot/simulation_time",
        "use_fsm":      False,
        "logs_subdir":  "simulation",
    },
    "hw": {
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
    """

    def __init__(self, mode: str, output_path: str, log_freq: float, dump_period: float):

        super().__init__('log_node')

        self.mode = mode
        self.cfg = MODE_CONFIG[mode]
        log_period = 1.0 / log_freq

        # current FSM state (only used in hardware mode)
        self.fsm_state = "init"

        # lazy Loggers + latest-message caches, both keyed by dataset name.
        # Loggers are created on the first message of each topic
        self.output_path = output_path
        self._loggers: dict = {}
        self._latest: dict = {}

        # topic discovery state: retried until all required topics are up, then the active dataset set is locked in.
        self._discovery_done = False
        self._active_datasets: list = []
        self._discovery_retry_period = 0.5
        self._last_discovery_attempt = 0.0
        self._last_missing_key: tuple = ()

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

    # cache the latest message and (on first arrival) create the Logger sized
    # to this particular topic's dimension
    def _handle_msg(self, dataset_name: str, data: np.ndarray):
        self._latest[dataset_name] = data
        if dataset_name not in self._loggers:
            self._loggers[dataset_name] = Logger(self.output_path, data.shape[0], dataset_name=dataset_name)

    def joint_callback(self, msg: Float32MultiArray):
        self._handle_msg("joint_state", np.array(msg.data, dtype=np.float32))

    def pelvis_imu_callback(self, msg: Float32MultiArray):
        self._handle_msg("pelvis_imu", np.array(msg.data, dtype=np.float32))

    def torso_imu_callback(self, msg: Float32MultiArray):
        self._handle_msg("torso_imu", np.array(msg.data, dtype=np.float32))

    def command_callback(self, msg: Float32MultiArray):
        self._handle_msg("command", np.array(msg.data, dtype=np.float32))

    def joystick_callback(self, msg: Float32MultiArray):
        self._handle_msg("joystick", np.array(msg.data, dtype=np.float32))

    def time_callback(self, msg: Float64):
        self._handle_msg("time", np.array([msg.data], dtype=np.float32))


    #################################################################
    # LOGGING
    #################################################################

    # poll publishers for every candidate topic. retry until all required topics are up, then lock in the active set.
    def _run_topic_discovery(self):
        now = time.monotonic()
        if (now - self._last_discovery_attempt) < self._discovery_retry_period:
            return
        self._last_discovery_attempt = now

        topic_for = dict(DATASET_TOPICS, time=self.cfg["time_topic"])

        missing_required = [
            (name, topic_for[name])
            for name in REQUIRED_DATASETS
            if self.count_publishers(topic_for[name]) == 0
        ]

        # still waiting: print only when the missing set changes
        if missing_required:
            missing_key = tuple(n for n, _ in missing_required)
            if missing_key != self._last_missing_key:
                details = ", ".join(f"{n} ({t})" for n, t in missing_required)
                print(f"Waiting for required topics: {details}")
                self._last_missing_key = missing_key
            return

        print("Topic discovery: all required topics ready.")
        active = list(REQUIRED_DATASETS)
        for name in OPTIONAL_DATASETS:
            topic = topic_for[name]
            n_pubs = self.count_publishers(topic)
            if n_pubs > 0:
                active.append(name)
                print(f"    [OK]   {name:12s} <- {topic}  ({n_pubs} pub)")
            else:
                print(f"    [SKIP] {name:12s} <- {topic}  (optional, no publisher)")

        self._active_datasets = active
        self._discovery_done = True

    # log latest cached messages at log_freq. waits until every active
    # dataset has published at least once so rows stay aligned.
    def log_callback(self):
        if not self._logging_enabled():
            return
        if not self._discovery_done:
            self._run_topic_discovery()
        if any(name not in self._loggers for name in self._active_datasets):
            return
        for name in self._active_datasets:
            self._loggers[name].log(self._latest[name])

    # periodically flush all buffers to disk
    def dump_callback(self):
        for logger in self._loggers.values():
            logger.dump()


    #################################################################
    # SHUTDOWN
    #################################################################

    # flush remaining buffers, close all loggers, then tear down the node
    def destroy_node(self):
        for logger in self._loggers.values():
            logger.close()
        super().destroy_node()


############################################################################
# MAIN FUNCTION
############################################################################

def build_output_path(mode: str, filename: str) -> str:
    # <ROOT_DIR>/logs/<subdir>/<filename>.h5
    subdir = MODE_CONFIG[mode]["logs_subdir"]
    root = ROOT_DIR if ROOT_DIR else ""
    logs_dir = os.path.join(root, "logs", subdir)
    os.makedirs(logs_dir, exist_ok=True)

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
        choices=["sim", "hw"],
        help='Deployment mode: "sim" (simulation) or "hw" (hardware).'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Output filename (without .h5 extension). Default: <YYYY_MM_DD__HH_MM_SS>. Saved under logs/<mode>/.'
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
        help='How often (in seconds) to write the in-memory buffer to disk. Default: 1.0 seconds.'
    )
    args = parser.parse_args()

    # filename: use the user-provided one, or fall back to a timestamp
    if args.filename is not None:
        filename = args.filename
    else:
        filename = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    output_path = build_output_path(args.mode, filename)


    # create the log node
    log_node = LogNode(args.mode, output_path, args.hz, args.dump_period)

    try:
        # spin the node
        rclpy.spin(log_node)

    except KeyboardInterrupt:
        pass

    finally:
        # flush remaining buffers, close files, and shut down ROS2
        log_node.destroy_node()
        rclpy.try_shutdown()

    print("Logger shutdown complete.")


if __name__ == "__main__":
    main()

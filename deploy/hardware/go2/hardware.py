##
#
# Deployment code for Unitree Go2 robot.
#
##

# standard imports
import argparse
import json
import numpy as np
import time
import threading

# other imports
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, String

# directory imports
import os
import sys
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.experiment_utils import HARDWARE_INFO_TOPIC, experiment_info_qos

# Unitree SDK imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactory, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from cyclonedds.qos import Qos, Policy
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

# Unitree's Go2 constants (LegID, LOWLEVEL, PosStopF, VelStopF), next to this script
import unitree_legged_const as go2_const


########################################################################
# GLOBAL VARIABLES (DO NOT CHANGE)
########################################################################

GO2_NUM_MOTOR = 12     # actuated joints
GO2_NUM_MOTOR_CMD = 20 # motor_cmd slots in LowCmd_ (only the first 12 are used)

# MODEL_TO_SDK[i] = SDK motor index of model joint i, built from Unitree's LegID
#   model (MuJoCo / policy): FL(0-2), FR(3-5), RL(6-8), RR(9-11), each hip, thigh, calf
#   SDK (motor_state/cmd):   FR(0-2), FL(3-5), RR(6-8), RL(9-11)
# (verified on hardware 2026-11-13 with per-leg tests: [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
MODEL_TO_SDK = np.array([go2_const.LegID[f"{leg}_{j}"]
                         for leg in ("FL", "FR", "RL", "RR") for j in range(3)])

# LowCmd_ constants
LOWLEVEL = go2_const.LOWLEVEL
PMSM_MODE = 0x01                 # servo (PMSM) mode (not in unitree_legged_const)
POS_STOP_F = go2_const.PosStopF  # "no position target" (used with kp = 0)
VEL_STOP_F = go2_const.VelStopF  # "no velocity target" (used with kd = 0)

# low-level control frequency. 200 Hz, not Unitree's 500 Hz: in Python each tick costs ~1.4 ms
# (decode LowState 0.65 ms, encode LowCmd 0.43 ms, CRC 0.24 ms on the Orin), so 500 Hz overran
# and ran at ~150 Hz with 20-30 ms gaps. The motor drivers run the PD loop themselves.
LOW_LEVEL_CONTROL_DT = 0.005  # [sec]

# warn if the control loop falls behind: checked over this window, against these limits
LOOP_STATS_PERIOD = 5.0                          # [sec]
LOOP_MIN_RATE = 0.9 / LOW_LEVEL_CONTROL_DT       # [Hz]
LOOP_MAX_GAP = 3.0 * LOW_LEVEL_CONTROL_DT        # [sec]

# ROS2 sensor publishing frequency
ROS_SENSOR_PUBLISH_DT = 0.005  # [sec]

# damping gain for safety stops, and once the robot is lying down in [damp]
DAMP_KD = 2.0  # [Nm s / rad]

# [damp]: lying pose (model order: FL, FR, RL, RR, each hip, thigh, calf), from Unitree's
# go2_stand_example.py. Calves sit beyond the XML limits (~-2.65 rad), so no clamping.
LIE_JOINT_POS = np.array([ 0.0, 1.36, -2.65,
                           0.0, 1.36, -2.65,
                           0.2, 1.36, -2.65,
                          -0.2, 1.36, -2.65])  # [rad]

# stiff PD gains for standing up ('home') and lying down ('damp'); the policy gains are too soft
POSE_KP = 60.0  # [Nm / rad]
POSE_KD = 5.0   # [Nm s / rad]

# [damp]: time to move from the current pose to LIE_JOINT_POS
LIE_POS_DURATION = 1.5  # [sec]

# [damp]: after reaching the lying pose, ramp the stiffness down to pure damping over this time
LIE_GAIN_RAMP_DOWN = 1.5  # [sec]

# max allowable base roll/pitch before forcing damp (when you fall)
SAFETY_MAX_TILT = np.radians(60.0)  # [rad]

# max time without a new policy command in [control] before forcing damp
SAFETY_CMD_TIMEOUT = 0.1  # [sec]


# smooth 0 -> 1 interpolation (zero velocity at both ends), x clipped to [0, 1]
def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


########################################################################
# CONTROL
########################################################################

class ControlNode(Node):
    def __init__(self, config_path: str,
                       enable_max_tilt: bool,
                       enable_imu_guard: bool,
                       enable_cmd_timeout: bool,
                       hold_lie: bool = False):

        super().__init__("hardware_node")

        # import config
        self.config = self.load_config(config_path)

        # load parameters
        self.load_params()

        # safety features (on by default)
        self.enable_max_tilt = enable_max_tilt
        self.enable_imu_guard = enable_imu_guard
        self.enable_cmd_timeout = enable_cmd_timeout
        if not self.enable_max_tilt:
            print("WARNING: max tilt safety check is DISABLED.")
        if not self.enable_imu_guard:
            print("WARNING: IMU dropout guard is DISABLED.")
        if not self.enable_cmd_timeout:
            print("WARNING: policy command timeout is DISABLED.")

        # [damp]: keep holding the lying pose stiffly instead of relaxing to pure damping
        # (for handling the robot by its torso)
        self.hold_lie = hold_lie
        if self.hold_lie:
            print("NOTE: --hold-lie is on: 'damp' holds the lying pose stiffly (no relax to damping).")

        # IMU states
        self.imu_rpy = None            # roll, pitch, yaw
        self.imu_quaternion = None     # orientation (w, x, y, z)
        self.imu_gyroscope = None      # angular velocity
        self.imu_accelerometer = None  # linear acceleration

        # IMU dropout counting
        self.imu_lost_count = 0

        # joint states (model order)
        self.q = np.zeros(GO2_NUM_MOTOR)          # joint positions
        self.dq = np.zeros(GO2_NUM_MOTOR)         # joint velocities
        self.ddq = np.zeros(GO2_NUM_MOTOR)        # joint accelerations
        self.tau_est = np.zeros(GO2_NUM_MOTOR)    # estimated joint torques
        self.motor_temp = np.zeros(GO2_NUM_MOTOR) # motor temperatures [degC]

        # other robot states
        self.foot_force = np.zeros(4)  # raw foot contact sensors (SDK order: FR, FL, RR, RL)
        self.power = np.zeros(2)       # battery [voltage, current]

        # command arrays (model order)
        self.q_cmd = np.array(self.home_joint_pos, dtype=np.float64)
        self.dq_cmd = np.zeros(GO2_NUM_MOTOR)
        self.Kp_cmd = np.array(self.Kp, dtype=np.float64)
        self.Kd_cmd = np.array(self.Kd, dtype=np.float64)
        self.tau_ff_cmd = np.zeros(GO2_NUM_MOTOR)
        self.last_cmd_time = None  # time.time() of the last policy command

        # locks for thread safety
        self.fsm_lock = threading.Lock()       # protects FSM state
        self.sensor_lock = threading.Lock()    # protects sensor state arrays
        self.cmd_lock = threading.Lock()       # protects command arrays

        # finite state machine state
        self.fsm_state = "init"
        self.prev_fsm_state = "init"
        self.fsm_start_time = 0.0
        self.fsm_start_q = np.zeros(GO2_NUM_MOTOR)
        self.fsm_time = 0.0

        # safety flags
        self.safety_triggered = False
        self.exit_requested = False  # set on Ctrl+C: lie down before exiting

        # SDK state
        self.time_ = 0.0          # [sec] wall-clock time since the control loop started
        self.time_start = None    # time.monotonic() at the first control loop tick

        # control loop rate monitoring
        self.loop_ticks = 0
        self.loop_max_gap = 0.0
        self.loop_prev = None
        self.loop_stats_start = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.init_low_cmd()

    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it (accept the name with or without the .yaml extension)
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        config_path_full = ROOT_DIR + "/deploy/configs/go2/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print("Config file loaded successfully from: [{}].".format(config_path_full))

        return config


    # load params from config
    def load_params(self):

        # time to interpolate to the standing pose
        self.home_pos_duration = self.config['home_pos_duration']   # float

        # policy default joint positions: the standing pose in 'home', and the initial command before the policy starts
        self.home_joint_pos = self.config['home_joint_pos'] # list

        # policy PD gains
        self.Kp = self.config['Kp']  # list
        self.Kd = self.config['Kd']  # list

        # type checks
        assert type(self.home_pos_duration) in [float], "home_pos_duration must be a float."
        assert type(self.home_joint_pos) == list, "home_joint_pos must be a list."
        assert type(self.Kp) == list, "Kp must be a list."
        assert type(self.Kd) == list, "Kd must be a list."

        # length checks
        assert len(self.Kp) == GO2_NUM_MOTOR, f"Expected {GO2_NUM_MOTOR} Kp values, got {len(self.Kp)}."
        assert len(self.Kd) == GO2_NUM_MOTOR, f"Expected {GO2_NUM_MOTOR} Kd values, got {len(self.Kd)}."

        # value checks
        assert self.home_pos_duration >= 1.0, "home_pos_duration must take at least 1 second."
        assert len(self.home_joint_pos) == GO2_NUM_MOTOR, (f"Expected {GO2_NUM_MOTOR} default joint positions, "
                                                           f"got {len(self.home_joint_pos)}")
        for i in range(GO2_NUM_MOTOR):
            assert self.Kp[i] >= 0.0, f"Kp for joint {i} must be non-negative."
            assert self.Kd[i] >= 0.0, f"Kd for joint {i} must be non-negative."

        print("Config parameters loaded successfully.")


    # constant LowCmd_ fields, and every motor slot passive (kp = kd = 0, no targets)
    def init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = LOWLEVEL
        self.low_cmd.gpio = 0
        for i in range(GO2_NUM_MOTOR_CMD):
            self.low_cmd.motor_cmd[i].mode = PMSM_MODE
            self.low_cmd.motor_cmd[i].q = POS_STOP_F
            self.low_cmd.motor_cmd[i].dq = VEL_STOP_F
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0


    # release sport mode, then create the SDK and ROS2 publishers and subscribers
    def Init(self):

        # sport mode must be released before low-level commands take effect
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        # wait until we have low-level control of the robot before proceeding
        status, result = self.msc.CheckMode()
        while result['name']:
            print(f"Releasing sport mode [{result['name']}] (robot lies down first)...")
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber
        # robot state is read in the control loop (latest sample only) instead of in a callback:
        # a callback decodes every message at 500 Hz, which starves the control loop of the GIL
        participant = ChannelFactory()._ChannelFactory__participant
        self.lowstate_reader = DataReader(participant, Topic(participant, "rt/lowstate", LowState_),
                                          Qos(Policy.History.KeepLast(1)))

        print("Unitree SDK publishers and subscribers initialized successfully.")

        # ROS2 publishers (imu_state / joint_state match the Go2 simulation node)
        self.hardware_time_pub = self.create_publisher(Float64, "deploy_robot/hardware_time", 1)
        self.fsm_time_pub = self.create_publisher(Float64, "deploy_robot/fsm_time", 1)
        self.joint_state_pub = self.create_publisher(Float32MultiArray, "deploy_robot/joint_state", 1)
        self.imu_state_pub = self.create_publisher(Float32MultiArray, "deploy_robot/imu_state", 1)
        self.motor_temp_pub = self.create_publisher(Float32MultiArray, "deploy_robot/motor_temperature", 1)
        self.foot_force_pub = self.create_publisher(Float32MultiArray, "deploy_robot/foot_force", 1)
        self.power_pub = self.create_publisher(Float32MultiArray, "deploy_robot/power", 1)

        # ROS2 subscribers
        self.command_sub = self.create_subscription(Float32MultiArray, "deploy_robot/command", self.command_callback, 1)
        self.fsm_sub = self.create_subscription(String, "deploy_robot/fsm", self.fsm_callback, 1)

        # latched hardware info: which flags this run used (recorded by the logger)
        self.hardware_info_pub = self.create_publisher(String, HARDWARE_INFO_TOPIC, experiment_info_qos())
        hardware_info_msg = String()
        hardware_info_msg.data = json.dumps({"robot": "go2",
                                             "max_tilt": self.enable_max_tilt,
                                             "imu_guard": self.enable_imu_guard,
                                             "cmd_timeout": self.enable_cmd_timeout})
        self.hardware_info_pub.publish(hardware_info_msg)

        # sensor publish timer
        self.pub_timer = self.create_timer(ROS_SENSOR_PUBLISH_DT, self.publish_sensor_data)

        print("ROS2 publishers and subscribers initialized successfully.")


    # create a thread to run the low-level control loop
    def Start(self):
        # create a thread for low-level control loop, but do not start it yet
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=LOW_LEVEL_CONTROL_DT, target=self.LowCmdWrite, name="control"
        )

        # wait until we receive the first low state message
        print("Waiting for the first low state message...")
        while self.low_state is None:
            self.read_low_state()
            time.sleep(0.01)

        # start the low-level control thread
        self.lowCmdWriteThreadPtr.Start()
        print("Low-level robot control thread started successfully.")


    #################################################################
    # ROS PUBLISHING AND CALLBACKS
    #################################################################

    # callback to receive FSM state from joystick
    def fsm_callback(self, msg: String):
        with self.fsm_lock:
            self.fsm_state = msg.data


    # callback to receive command messages from ROS2
    def command_callback(self, msg: Float32MultiArray):

        # expected layout (model order): [q(12), dq(12), Kp(12), Kd(12), tau_ff(12)] = 60 floats
        data = np.array(msg.data, dtype=np.float64)

        # safety check on command length
        if len(data) != 5 * GO2_NUM_MOTOR:
            self.get_logger().warn(f"Expected {5 * GO2_NUM_MOTOR} values in command, got {len(data)}")
            return

        # reject non-finite commands (e.g. NaN from the policy)
        if not np.all(np.isfinite(data)):
            self.get_logger().warn("Non-finite values in command, ignoring it.")
            return

        # update command arrays under lock
        nu = GO2_NUM_MOTOR
        with self.cmd_lock:
            self.q_cmd[:] = data[0*nu : 1*nu]
            self.dq_cmd[:] = data[1*nu : 2*nu]
            self.Kp_cmd[:] = data[2*nu : 3*nu]
            self.Kd_cmd[:] = data[3*nu : 4*nu]
            self.tau_ff_cmd[:] = data[4*nu : 5*nu]
            self.last_cmd_time = time.time()


    # publish sensor data to ROS2 topics
    def publish_sensor_data(self):
        # read sensor data under lock
        with self.sensor_lock:
            imu_rpy = np.array(self.imu_rpy, dtype=np.float64) if self.imu_rpy is not None else np.zeros(3)
            imu_quat = np.array(self.imu_quaternion, dtype=np.float64) if self.imu_quaternion is not None else np.array([1.0, 0.0, 0.0, 0.0])
            imu_gyro = np.array(self.imu_gyroscope, dtype=np.float64) if self.imu_gyroscope is not None else np.zeros(3)
            imu_accel = np.array(self.imu_accelerometer, dtype=np.float64) if self.imu_accelerometer is not None else np.zeros(3)
            q = self.q.copy()
            dq = self.dq.copy()
            ddq = self.ddq.copy()
            tau_est = self.tau_est.copy()
            motor_temp = self.motor_temp.copy()
            foot_force = self.foot_force.copy()
            power = self.power.copy()

        # imu_state: [rpy(3), quaternion(4), gyroscope(3), accelerometer(3)] = 13 floats
        imu_msg = Float32MultiArray()
        imu_msg.data = np.concatenate([imu_rpy, imu_quat, imu_gyro, imu_accel]).tolist()

        # joint_state (model order): [q(12), dq(12), ddq(12), tau_est(12)] = 48 floats
        joint_msg = Float32MultiArray()
        joint_msg.data = np.concatenate([q, dq, ddq, tau_est]).tolist()

        # motor_temperature (model order) [degC] = 12 floats
        motor_temp_msg = Float32MultiArray()
        motor_temp_msg.data = motor_temp.tolist()

        # foot_force: raw sensor values (SDK order: FR, FL, RR, RL) = 4 floats
        foot_force_msg = Float32MultiArray()
        foot_force_msg.data = foot_force.tolist()

        # power: [battery voltage [V], current [A]] = 2 floats
        power_msg = Float32MultiArray()
        power_msg.data = power.tolist()

        # hardware_time: single float
        time_msg = Float64()
        time_msg.data = self.time_

        # fsm_time: time since entering current state
        fsm_time_msg = Float64()
        fsm_time_msg.data = self.fsm_time

        # publish to ROS2 topics
        self.imu_state_pub.publish(imu_msg)
        self.joint_state_pub.publish(joint_msg)
        self.motor_temp_pub.publish(motor_temp_msg)
        self.foot_force_pub.publish(foot_force_msg)
        self.power_pub.publish(power_msg)
        self.hardware_time_pub.publish(time_msg)
        self.fsm_time_pub.publish(fsm_time_msg)


    #################################################################
    # SDK HARDWARE
    #################################################################

    # take the latest low state sample, if a new one arrived (non-blocking)
    def read_low_state(self):
        samples = self.lowstate_reader.take(N=1)
        if samples:
            self.LowStateHandler(samples[0])


    # process a low state message
    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        # IMU guard: a lost packet has a near-zero quaternion (and zeroed rpy, gyro, accel)
        imu = msg.imu_state
        quat_norm = np.linalg.norm(np.array(imu.quaternion, dtype=np.float64))
        imu_lost = self.enable_imu_guard and quat_norm < 0.5
        if imu_lost:
            if self.imu_lost_count == 0:
                print("WARNING: IMU packet lost. Holding last good IMU values.")
            self.imu_lost_count += 1
        elif self.imu_lost_count > 0:
            print(f"WARNING: IMU recovered after [{self.imu_lost_count}] lost packets.")
            self.imu_lost_count = 0

        # update sensor states under lock
        with self.sensor_lock:
            # update IMU states (keep the last good values if this packet was lost)
            if not imu_lost:
                self.imu_rpy = imu.rpy
                self.imu_quaternion = imu.quaternion
                self.imu_gyroscope = imu.gyroscope
                self.imu_accelerometer = imu.accelerometer

            # update joint states (SDK order -> model order)
            for i in range(GO2_NUM_MOTOR):
                m = msg.motor_state[MODEL_TO_SDK[i]]
                self.q[i] = m.q
                self.dq[i] = m.dq
                self.ddq[i] = m.ddq
                self.tau_est[i] = m.tau_est
                self.motor_temp[i] = m.temperature

            # other states
            self.foot_force[:] = msg.foot_force
            self.power[0] = msg.power_v
            self.power[1] = msg.power_a


    # write one motor command (model joint index i -> SDK motor slot)
    def set_motor_cmd(self, i, q, dq, kp, kd, tau):
        c = self.low_cmd.motor_cmd[MODEL_TO_SDK[i]]
        c.mode = PMSM_MODE
        c.q = q
        c.dq = dq
        c.kp = kp
        c.kd = kd
        c.tau = tau


    # on exit: lie down (if the motors are active and no safety stop), then damp, then stop the loop
    def Stop(self):
        thread = getattr(self, "lowCmdWriteThreadPtr", None)
        if thread is None or self.low_state is None:
            return
        if not self.safety_triggered and self.prev_fsm_state != "init":
            print("Lying down before exit...")
            self.exit_requested = True
            time.sleep(LIE_POS_DURATION + LIE_GAIN_RAMP_DOWN + 0.2)
        print("Damping motors before exit...")
        self.safety_triggered = True
        time.sleep(0.2)
        thread.Wait(timeout=1.0)


    # force the damp state until restart
    def trigger_safety(self, reason: str):
        print()
        print("*" * 70)
        print(f"SAFETY: {reason} -> FORCING DAMP. PLEASE RESTART!")
        print("*" * 70)
        self.safety_triggered = True


    # warn (at most every LOOP_STATS_PERIOD) if the control loop runs slow or has long gaps
    def monitor_loop_rate(self, now: float):
        self.loop_ticks += 1
        self.loop_max_gap = max(self.loop_max_gap, now - self.loop_prev)
        self.loop_prev = now
        elapsed = now - self.loop_stats_start
        if elapsed < LOOP_STATS_PERIOD:
            return
        rate = self.loop_ticks / elapsed
        if rate < LOOP_MIN_RATE or self.loop_max_gap > LOOP_MAX_GAP:
            print(f"WARNING: control loop at {rate:.0f} Hz (target {1.0 / LOW_LEVEL_CONTROL_DT:.0f} Hz), "
                  f"longest gap {self.loop_max_gap * 1e3:.1f} ms over the last {elapsed:.0f} s.")
        self.loop_ticks = 0
        self.loop_max_gap = 0.0
        self.loop_stats_start = now


    # main control loop to send low-level commands
    def LowCmdWrite(self):

        # update hardware time from the wall clock. The SDK's timer drops ticks the loop misses
        # (e.g. waiting on the GIL), so counting ticks would make time (and the policy's gait
        # clock, which is built from fsm_time) run slow.
        now = time.monotonic()
        if self.time_start is None:
            self.time_start = now
            self.loop_stats_start = now
            self.loop_prev = now
        self.time_ = now - self.time_start
        self.monitor_loop_rate(now)

        # read the latest robot state
        self.read_low_state()

        # read FSM state under lock
        with self.fsm_lock:
            fsm_state = self.fsm_state

        # exiting: lie down regardless of the joystick
        if self.exit_requested:
            fsm_state = "damp"

        # detect state transition
        if fsm_state != self.prev_fsm_state:
            print(f"FSM: {self.prev_fsm_state} -> {fsm_state}")
            self.fsm_start_time = self.time_
            with self.sensor_lock:
                self.fsm_start_q = self.q.copy()
            self.prev_fsm_state = fsm_state

        # update fsm time
        self.fsm_time = self.time_ - self.fsm_start_time

        # safety: force damp if the base tilts beyond specified threshold
        if self.enable_max_tilt and not self.safety_triggered:
            with self.sensor_lock:
                rpy = self.imu_rpy
            if rpy is not None:
                roll, pitch = abs(rpy[0]), abs(rpy[1])
                if roll > SAFETY_MAX_TILT or pitch > SAFETY_MAX_TILT:
                    self.trigger_safety(f"roll={np.degrees(roll):.2f} pitch={np.degrees(pitch):.2f}")

        # read the policy command under lock
        with self.cmd_lock:
            q_cmd = self.q_cmd.copy()
            dq_cmd = self.dq_cmd.copy()
            Kp_cmd = self.Kp_cmd.copy()
            Kd_cmd = self.Kd_cmd.copy()
            tau_ff_cmd = self.tau_ff_cmd.copy()
            last_cmd_time = self.last_cmd_time

        # safety: force damp if the policy stops sending commands while in control
        is_control = fsm_state in ("control", "track")
        if self.enable_cmd_timeout and is_control and not self.safety_triggered:
            if last_cmd_time is None:
                self.trigger_safety("entered control without any policy command (is the controller running?)")
            elif time.time() - last_cmd_time > SAFETY_CMD_TIMEOUT:
                self.trigger_safety(f"no policy command for {time.time() - last_cmd_time:.3f} s")

        # [safety stop]: pure velocity damping, no position tracking (latched until restart)
        if self.safety_triggered:
            for i in range(GO2_NUM_MOTOR):
                self.set_motor_cmd(i, 0.0, 0.0, 0.0, DAMP_KD, 0.0)

        # [init]: zero out all commands (motors passive)
        elif fsm_state == "init":
            for i in range(GO2_NUM_MOTOR):
                self.set_motor_cmd(i, POS_STOP_F, VEL_STOP_F, 0.0, 0.0, 0.0)

        # [damp]: lie down with the stiff pose gains, then ramp the stiffness down to pure damping
        # (no clamping to the XML limits: lying-down calves sit beyond them, ~-2.8 rad)
        elif fsm_state == "damp":
            ratio = smoothstep(self.fsm_time / LIE_POS_DURATION)
            # 1 while moving, then ramps to 0 over LIE_GAIN_RAMP_DOWN once lying
            stiffness = 1.0 - np.clip((self.fsm_time - LIE_POS_DURATION) / LIE_GAIN_RAMP_DOWN, 0.0, 1.0)
            if self.hold_lie:
                stiffness = 1.0
            kd = stiffness * POSE_KD + (1.0 - stiffness) * DAMP_KD
            for i in range(GO2_NUM_MOTOR):
                q_des = (1.0 - ratio) * self.fsm_start_q[i] + ratio * LIE_JOINT_POS[i]
                self.set_motor_cmd(i, q_des, 0.0, stiffness * POSE_KP, kd, 0.0)

        # [home]: stand up with the stiff pose gains and hold the standing pose
        elif fsm_state == "home":
            ratio = smoothstep(self.fsm_time / self.home_pos_duration)
            for i in range(GO2_NUM_MOTOR):
                q_des = (1.0 - ratio) * self.fsm_start_q[i] + ratio * self.home_joint_pos[i]
                self.set_motor_cmd(i, q_des, 0.0, POSE_KP, POSE_KD, 0.0)

        # [control / track]: latest command from the policy node
        elif is_control:
            for i in range(GO2_NUM_MOTOR):
                self.set_motor_cmd(i, q_cmd[i], dq_cmd[i], Kp_cmd[i], Kd_cmd[i], tau_ff_cmd[i])

        # unknown state: pure damping
        else:
            for i in range(GO2_NUM_MOTOR):
                self.set_motor_cmd(i, 0.0, 0.0, 0.0, DAMP_KD, 0.0)

        # check sum commands for safety and then publish
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


############################################################################
# MAIN FUNCTION
############################################################################

# --name / --no-name boolean flag (argparse.BooleanOptionalAction needs python 3.9+)
def add_bool_flag(parser, name: str, default: bool, help: str):
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f'--{name}', dest=dest, action='store_true', help=help + f' (default: {"on" if default else "off"})')
    group.add_argument(f'--no-{name}', dest=dest, action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Go2 hardware deployment node using Unitree SDK2 for Python."
    )
    # network interface name argument
    parser.add_argument(
        '--network',
        type=str,
        required=True,
        help='Network interface name for robot communication. Example: "eth0".'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file for hardware. Example: "go2_vel.yaml".'
    )
    # safety flags (on by default, turn off with the --no- form)
    add_bool_flag(parser, 'max-tilt', True,
                  'Force damp when the base tilts past the safety threshold.')
    add_bool_flag(parser, 'imu-guard', True,
                  'Hold the last good IMU values when an IMU packet is lost.')
    add_bool_flag(parser, 'cmd-timeout', True,
                  'Force damp when the policy stops sending commands while in control.')
    add_bool_flag(parser, 'hold-lie', False,
                  "In 'damp', keep holding the lying pose stiffly instead of relaxing to pure damping.")
    args = parser.parse_args()

    print()
    print("WARNING: Please ensure there are no obstacles around the robot.")
    while input("Press [Enter] to continue: ") != "":
        pass
    print()

    # initialize the channel factory with the specified network interface
    ChannelFactoryInitialize(0, args.network)

    # instantiate the custom control class
    ctrl_node = ControlNode(args.config,
                            enable_max_tilt=args.max_tilt,
                            enable_imu_guard=args.imu_guard,
                            enable_cmd_timeout=args.cmd_timeout,
                            hold_lie=args.hold_lie)
    ctrl_node.Init()

    # spin ROS2 node in background thread
    ros_running = True
    def spin_ros():
        while ros_running and rclpy.ok():
            try:
                rclpy.spin_once(ctrl_node, timeout_sec=0.1)
            except Exception:
                break
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()

    # start the control loop
    ctrl_node.Start()

    # run normally
    try:
        while True:
            time.sleep(1)
    # ctrl + C
    except KeyboardInterrupt:
        print("\nExiting...")
    # graceful shutdown on any exception
    finally:
        try:
            ctrl_node.Stop()
        except Exception as e:
            print(f"WARNING: failed to damp before exit: {e}")
        ros_running = False
        ros_thread.join(timeout=1.0)
        try:
            ctrl_node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    print("Hardware shutdown complete.")


if __name__ == "__main__":
    main()

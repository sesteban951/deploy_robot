##
#
# Deployment code for Unitree G1 robot. 
#
# Common G1 "hg" topics: https://support.unitree.com/home/en/G1_developer/dds_services_interface
# 
##

# standard imports
import argparse
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
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")

# Unitree SDK imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


########################################################################
# GLOBAL VARIABLES (DO NOT CHANGE)
########################################################################

G1_NUM_MOTOR = 29

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


# low-level control frequency 
LOW_LEVEL_CONTROL_DT = 0.002  # [sec]

# ROS2 sensor publishing frequency
ROS_SENSOR_PUBLISH_DT = 0.02  # [sec]

# safety: max allowable pelvis roll/pitch before forcing damp (when you fall)
SAFETY_MAX_TILT = np.radians(30.0)  # [rad]


########################################################################
# CONTROL
########################################################################

class ControlNode(Node):
    def __init__(self, config_path: str):

        super().__init__("hardware_node")

        # import config
        self.config = self.load_config(config_path)

        # load parameters
        self.load_params()

        # IMU states
        self.pelvis_imu_rpy = None            # roll, pitch, yaw
        self.pelvis_imu_quaternion = None     # orientation
        self.pelvis_imu_gyroscope = None      # angular velocity
        self.pelvis_imu_accelerometer = None  # linear acceleration
        self.torso_imu_rpy = None
        self.torso_imu_quaternion = None
        self.torso_imu_gyroscope = None
        self.torso_imu_accelerometer = None

        # Joint states
        self.q = np.zeros(G1_NUM_MOTOR)        # joint positions
        self.dq = np.zeros(G1_NUM_MOTOR)       # joint velocities
        self.ddq = np.zeros(G1_NUM_MOTOR)      # joint accelerations
        self.tau_est = np.zeros(G1_NUM_MOTOR)  # estimated joint torques

        # command arrays
        self.q_cmd = np.array(self.default_joint_pos, dtype=np.float64)
        self.dq_cmd = np.zeros(G1_NUM_MOTOR)
        self.Kp_cmd = np.array(self.Kp, dtype=np.float64)
        self.Kd_cmd = np.array(self.Kd, dtype=np.float64)
        self.tau_ff_cmd = np.zeros(G1_NUM_MOTOR)

        # locks for thread safety
        self.fsm_lock = threading.Lock()       # protects FSM state
        self.sensor_lock = threading.Lock()    # protects sensor state arrays
        self.cmd_lock = threading.Lock()       # protects command arrays

        # finite state machine state
        self.fsm_state = "init"
        self.prev_fsm_state = "init"
        self.fsm_start_time = 0.0
        self.fsm_start_q = np.zeros(G1_NUM_MOTOR)
        self.fsm_time = 0.0

        # safety flags
        self.safety_triggered = False

        # other stuff from unitree's example
        self.time_ = 0.0
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()

    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print("Config file loaded successfully from: [{}].".format(config_path_full))

        return config


    # load params from config
    def load_params(self):

        # time to interpolate to initial
        self.home_pos_duration = self.config['home_pos_duration']   # float

        # default joint positions
        self.default_joint_pos = self.config['default_joint_pos'] # list

        # PD gains
        self.Kp = self.config['Kp']  # list
        self.Kd = self.config['Kd']  # list

        # type checks
        assert type(self.home_pos_duration) in [float], "home_pos_duration must be a float."
        assert type(self.default_joint_pos) == list, "default_joint_pos must be a list."
        assert type(self.Kp) == list, "Kp must be a list."
        assert type(self.Kd) == list, "Kd must be a list."

        # length checks
        assert len(self.Kp) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kp values, got {len(self.Kp)}."
        assert len(self.Kd) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kd values, got {len(self.Kd)}."

        # value checks
        assert self.home_pos_duration >= 3.0, "home_pos_duration must take at least 3 seconds."
        assert len(self.default_joint_pos) == G1_NUM_MOTOR, (f"Expected {G1_NUM_MOTOR} default joint positions, "
                                                             f"got {len(self.default_joint_pos)}")
        for i in range(G1_NUM_MOTOR):
            assert self.Kp[i] >= 0.0, f"Kp for joint {i} must be non-negative."
            assert self.Kd[i] >= 0.0, f"Kd for joint {i} must be non-negative."

        print("Config parameters loaded successfully.")

    
    # initialize the motion switcher client, publishers, and subscribers
    def Init(self):

        # initialize motion switcher client
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        # wait until we have low-level control of the robot before proceeding
        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscribers
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        self.torso_imu_subscriber = ChannelSubscriber("rt/secondary_imu", IMUState_)
        self.torso_imu_subscriber.Init(self.TorsoIMUHandler, 10)

        print("Unitree SDK publishers and subscribers initialized successfully.")

        # ROS2 publishers
        self.hardware_time_pub = self.create_publisher(Float64, "deploy_robot/hardware_time", 10)
        self.fsm_time_pub = self.create_publisher(Float64, "deploy_robot/fsm_time", 10)
        self.joint_state_pub = self.create_publisher(Float32MultiArray, "deploy_robot/joint_state", 10)
        self.pelvis_imu_state_pub = self.create_publisher(Float32MultiArray, "deploy_robot/pelvis_imu_state", 10)
        self.torso_imu_state_pub = self.create_publisher(Float32MultiArray, "deploy_robot/torso_imu_state", 10)

        # ROS2 subscribers
        self.command_sub = self.create_subscription(Float32MultiArray, "deploy_robot/command", self.command_callback, 10)
        self.fsm_sub = self.create_subscription(String, "deploy_robot/fsm", self.fsm_callback, 10)

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
        while self.update_mode_machine_ == False:
            time.sleep(1)
        
        # start the low-level control thread
        if self.update_mode_machine_ == True:
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

        # expected layout: [q(29), dq(29), Kp(29), Kd(29), tau_ff(29)] = 145 floats
        data = np.array(msg.data, dtype=np.float64)

        # safety check on command length
        if len(data) != 5 * G1_NUM_MOTOR:
            self.get_logger().warn(f"Expected {5 * G1_NUM_MOTOR} values in command, got {len(data)}")
            return
        
        # update command arrays under lock
        nu = G1_NUM_MOTOR
        with self.cmd_lock:
            self.q_cmd[:] =  data[0*nu : 1*nu]
            self.dq_cmd[:] = data[1*nu : 2*nu]
            self.Kp_cmd[:] = data[2*nu : 3*nu]
            self.Kd_cmd[:] = data[3*nu : 4*nu]
            self.tau_ff_cmd[:] = data[4*nu : 5*nu]


    # publish sensor data to ROS2 topics
    def publish_sensor_data(self):
        # read sensor data under lock
        with self.sensor_lock:
            # pelvis IMU state
            pelvis_imu_rpy = np.array(self.pelvis_imu_rpy, dtype=np.float64) if self.pelvis_imu_rpy is not None else np.zeros(3)
            pelvis_imu_quat = np.array(self.pelvis_imu_quaternion, dtype=np.float64) if self.pelvis_imu_quaternion is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            pelvis_imu_gyro = np.array(self.pelvis_imu_gyroscope, dtype=np.float64) if self.pelvis_imu_gyroscope is not None else np.zeros(3)
            pelvis_imu_accel = np.array(self.pelvis_imu_accelerometer, dtype=np.float64) if self.pelvis_imu_accelerometer is not None else np.zeros(3)
            # torso IMU state
            torso_imu_rpy = np.array(self.torso_imu_rpy, dtype=np.float64) if self.torso_imu_rpy is not None else np.zeros(3)
            torso_imu_quat = np.array(self.torso_imu_quaternion, dtype=np.float64) if self.torso_imu_quaternion is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            torso_imu_gyro = np.array(self.torso_imu_gyroscope, dtype=np.float64) if self.torso_imu_gyroscope is not None else np.zeros(3)
            torso_imu_accel = np.array(self.torso_imu_accelerometer, dtype=np.float64) if self.torso_imu_accelerometer is not None else np.zeros(3)
            # joint state
            q = self.q.copy()
            dq = self.dq.copy()
            ddq = self.ddq.copy()
            tau_est = self.tau_est.copy()

        # imu_state: [rpy(3), quaternion(4), gyroscope(3), accelerometer(3)] = 13 floats
        pelvis_imu_msg = Float32MultiArray()
        pelvis_imu_msg.data = np.concatenate([pelvis_imu_rpy, pelvis_imu_quat, pelvis_imu_gyro, pelvis_imu_accel]).tolist()
        torso_imu_msg = Float32MultiArray()
        torso_imu_msg.data = np.concatenate([torso_imu_rpy, torso_imu_quat, torso_imu_gyro, torso_imu_accel]).tolist()

        # joint_state: [q(29), dq(29), ddq(29), tau_est(29)] = 116 floats
        joint_msg = Float32MultiArray()
        joint_msg.data = np.concatenate([q, dq, ddq, tau_est]).tolist()

        # hardware_time: single float
        time_msg = Float64()
        time_msg.data = self.time_

        # fsm_time: time since entering current state
        fsm_time_msg = Float64()
        fsm_time_msg.data = self.fsm_time

        # publish to ROS2 topics
        self.pelvis_imu_state_pub.publish(pelvis_imu_msg)
        self.torso_imu_state_pub.publish(torso_imu_msg)
        self.joint_state_pub.publish(joint_msg)
        self.hardware_time_pub.publish(time_msg)
        self.fsm_time_pub.publish(fsm_time_msg)


    #################################################################
    # SDK HARDWARE
    #################################################################

    # callback to receive low state messages
    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        
        # update sensor states under lock
        with self.sensor_lock:
            # update IMU states
            self.pelvis_imu_rpy = self.low_state.imu_state.rpy
            self.pelvis_imu_quaternion = self.low_state.imu_state.quaternion
            self.pelvis_imu_gyroscope = self.low_state.imu_state.gyroscope
            self.pelvis_imu_accelerometer = self.low_state.imu_state.accelerometer

            # update joint states
            for i in range(G1_NUM_MOTOR):
                self.q[i] = self.low_state.motor_state[i].q
                self.dq[i] = self.low_state.motor_state[i].dq
                self.ddq[i] = self.low_state.motor_state[i].ddq
                self.tau_est[i] = self.low_state.motor_state[i].tau_est


    # callback to receive torso IMU messages
    def TorsoIMUHandler(self, msg: IMUState_):
        with self.sensor_lock:
            self.torso_imu_rpy = msg.rpy
            self.torso_imu_quaternion = msg.quaternion
            self.torso_imu_gyroscope = msg.gyroscope
            self.torso_imu_accelerometer = msg.accelerometer


    # main control loop to send low-level commands
    def LowCmdWrite(self):
        
        # update hardware time
        self.time_ += LOW_LEVEL_CONTROL_DT

        # read FSM state under lock
        with self.fsm_lock:
            fsm_state = self.fsm_state

        # detect state transition
        if fsm_state != self.prev_fsm_state:
            print(f"FSM: {self.prev_fsm_state} -> {fsm_state}")
            self.fsm_start_time = self.time_
            with self.sensor_lock:
                self.fsm_start_q = self.q.copy()
            self.prev_fsm_state = fsm_state

        # update fsm time
        self.fsm_time = self.time_ - self.fsm_start_time

        # safety: force damp if pelvis tilts beyond specified threshold
        if not self.safety_triggered:
            with self.sensor_lock:
                rpy = self.pelvis_imu_rpy
            if rpy is not None:
                roll, pitch = abs(rpy[0]), abs(rpy[1])
                if roll > SAFETY_MAX_TILT or pitch > SAFETY_MAX_TILT:
                    print()
                    print("*" * 70)
                    print(f"SAFETY: roll={np.degrees(roll):.2f} pitch={np.degrees(pitch):.2f} -> FORCING DAMP. PLEASE RESTART!")
                    print("*" * 70)
                    self.safety_triggered = True
        if self.safety_triggered:
            fsm_state = "damp"

        # [init]: zero out all commands
        if fsm_state == "init":
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0
                # if i in [13,14]:  # waist joints, disable
                #     self.low_cmd.motor_cmd[i].mode = 0

        # [damp]: Kd damping, no position tracking
        elif fsm_state == "damp":
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 3.0
                # if i in [13,14]:  # waist joints, disable
                #     self.low_cmd.motor_cmd[i].mode = 0

        # [home]: interpolate to default joint positions and gains
        elif fsm_state == "home":
            ratio = np.clip(self.fsm_time / self.home_pos_duration, 0.0, 1.0)
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.fsm_start_q[i] + ratio * self.default_joint_pos[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = ratio * self.Kp[i]
                self.low_cmd.motor_cmd[i].kd = ratio * self.Kd[i]
                # if i in [13,14]:  # waist joints, disable
                #     self.low_cmd.motor_cmd[i].mode = 0

        # [control]: read from ROS2 command subscriber
        elif fsm_state == "control":
            with self.cmd_lock:
                q_cmd = self.q_cmd.copy()
                dq_cmd = self.dq_cmd.copy()
                Kp_cmd = self.Kp_cmd.copy()
                Kd_cmd = self.Kd_cmd.copy()
                tau_ff_cmd = self.tau_ff_cmd.copy()
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = tau_ff_cmd[i]
                self.low_cmd.motor_cmd[i].q = q_cmd[i]
                self.low_cmd.motor_cmd[i].dq = dq_cmd[i]
                self.low_cmd.motor_cmd[i].kp = Kp_cmd[i]
                self.low_cmd.motor_cmd[i].kd = Kd_cmd[i]
                # if i in [13,14]:  # waist joints, disable
                #     self.low_cmd.motor_cmd[i].mode = 0
        # check sum commands for safety and then publish
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Hardware deployment node using Unitree SDK2 for Python."
    )
    # network interface name argument
    parser.add_argument(
        '--network',
        type=str,
        required=True,
        help='Network interface name for robot communication. Example: "enp8s0".'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file for hardware. Example: "g1_29dof_hardware.yaml".'
    )
    args = parser.parse_args()

    print()
    while input("Press [Enter] to continue: ") != "":
        pass
    print()

    # initialize the channel factory with the specified network interface
    ChannelFactoryInitialize(0, args.network)

    # instantiate the custom control class
    ctrl_node = ControlNode(args.config)
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

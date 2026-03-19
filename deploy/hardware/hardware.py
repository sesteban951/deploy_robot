##
#
# Deployment code for Unitree G1 robot. 
#
##

# standard imports
import argparse
import numpy as np
import time

# other imports 
import yaml

# ROS2 imports
import rclpy
import rclpy.node

# directory imports
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")

# Unitree SDK imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


########################################################################
# GLOBAL VARIABLES
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
low_level_control_dt = 0.002  # [sec]


########################################################################
# CONTROL
########################################################################

class Custom:
    def __init__(self, config_path: str):

        # import config
        self.config = self.load_config(config_path)

        # load parameters
        self.load_params()

        # IMU states
        self.imu_rpy = None            # roll, pitch, yaw
        self.imu_quaternion = None     # orientation
        self.imu_gyroscope = None      # angular velocity
        self.imu_accelerometer = None  # linear acceleration

        # Joint states
        self.q = np.zeros(G1_NUM_MOTOR)        # joint positions
        self.dq = np.zeros(G1_NUM_MOTOR)       # joint velocities
        self.ddq = np.zeros(G1_NUM_MOTOR)      # joint accelerations
        self.tau_est = np.zeros(G1_NUM_MOTOR)  # estimated joint torques

        # other stuff from unitree's example
        self.time_ = 0.0
        self.counter_ = 0
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()

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
        self.default_pos_duration = self.config['default_pos_duration'] # float

        # default joint positions
        self.default_joint_pos = self.config['default_joint_pos'] # list

        # PD gains
        self.Kp = self.config['Kp']  # list
        self.Kd = self.config['Kd']  # list

        # type checks
        assert type(self.default_pos_duration) in [float], "default_pos_duration must be a float."
        assert type(self.Kp) == list, "Kp must be a list."
        assert type(self.Kd) == list, "Kd must be a list."

        # length checks
        assert len(self.Kp) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kp values, got {len(self.Kp)}."
        assert len(self.Kd) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kd values, got {len(self.Kd)}."

        # value checks
        assert self.default_pos_duration >= 2.0, "default_pos_duration must take at least 2 seconds for safe interpolation."
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

        # wait until we have control of the robot before proceeding
        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)


    # create a thread to run the low-level control loop
    def Start(self):
        # create a thread for low-level control loop, but do not start it yet
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=low_level_control_dt, target=self.LowCmdWrite, name="control"
        )

        # wait until we receive the first low state message
        while self.update_mode_machine_ == False:
            time.sleep(1)
        
        # start the low-level control thread
        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

        print("Low-level control thread started successfully.")


    # callback to receive low state messages
    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        
        # update IMU states
        self.imu_rpy = self.low_state.imu_state.rpy
        self.imu_quaternion = self.low_state.imu_state.quaternion
        self.imu_gyroscope = self.low_state.imu_state.gyroscope
        self.imu_accelerometer = self.low_state.imu_state.accelerometer

        # update joint states
        for i in range(G1_NUM_MOTOR):
            self.q[i] = self.low_state.motor_state[i].q
            self.dq[i] = self.low_state.motor_state[i].dq
            self.ddq[i] = self.low_state.motor_state[i].ddq
            self.tau_est[i] = self.low_state.motor_state[i].tau_est

        # print("Low state received. IMU RPY: {}".format(self.imu_rpy))
        # print("Low state received. IMU Quaternion: {}".format(self.imu_quaternion))
        # print("Low state received. IMU Gyroscope: {}".format(self.imu_gyroscope))
        # print("Low state received. IMU Accelerometer: {}".format(self.imu_accelerometer))
        # q_ = np.array(self.q)
        # print with 3 decimal places
        # print("Low state received. Joint positions: [{}]".format(", ".join(f"{v:.3f}" for v in q_)))


    # main control loop to send low-level commands
    def LowCmdWrite(self):

        self.time_ += low_level_control_dt

        # [Stage 1]: interpolate to default joint positions
        if self.time_ < self.default_pos_duration :
            # print("Interpolating to default joint positions...")
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.default_pos_duration, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0. 
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q + ratio * self.default_joint_pos[i]
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = self.Kp[i]
                self.low_cmd.motor_cmd[i].kd = self.Kd[i]

        # [Stage 2]: regular control commands
        else:
            # print("Regular control...")
            # max_P = np.pi * 10.0 / 180.0
            # max_R = np.pi * 10.0 / 180.0
            # t = self.time_ - self.default_pos_duration
            # L_P_des = max_P * np.sin(2.0 * np.pi * t)
            # L_R_des = max_R * np.sin(2.0 * np.pi * t)
            # R_P_des = max_P * np.sin(2.0 * np.pi * t)
            # R_R_des = -max_R * np.sin(2.0 * np.pi * t)
            L_P_des = 0.0
            L_R_des = 0.0
            R_P_des = 0.0
            R_R_des = 0.0

            self.low_cmd.mode_pr = Mode.PR
            self.low_cmd.mode_machine = self.mode_machine_
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = L_P_des
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleRoll].q = L_R_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = R_P_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnkleRoll].q = R_R_des
 
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

    print("#" * 65)
    print("# WARNING: Please ensure there are no obstacles around the robot.")
    print("#" * 65)
    while input("Press [Enter] to continue: ") != "":
        pass

    # initialize the channel factory with the specified network interface
    ChannelFactoryInitialize(0, args.network)

    # instantiate the custom control class
    custom = Custom(args.config)
    custom.Init()
    custom.Start()

    while True:        
        time.sleep(1)


if __name__ == "__main__":
    main()

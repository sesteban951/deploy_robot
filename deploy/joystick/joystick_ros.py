##
#
# Run joystick commands (using ROS2 joy_node)
#
##

# standard imports
import subprocess
import time

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, String

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.joystick_utils import JoystickState, rosjoy_to_joystick_state
from utils.finite_state_machine import FiniteStateMachine


############################################################################
# COMMAND NODE
############################################################################

class JoystickNode(Node):
    """
    Use joystick to publish commands to the simulation and control nodes.
    """

    def __init__(self, config_path: str = None):

        super().__init__('joystick_node')

        # initialize joystick
        self.deadzone = 0.05
        self.joystick_state = JoystickState()
        self.init_joystick()
    
        # finite state machine
        self.fsm = FiniteStateMachine()

        # ROS2 publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/joystick', 10)
        self.fsm_pub = self.create_publisher(String, 'deploy_robot/fsm', 10)

        # create timer to publish commands at a fixed rate
        joystick_dt = 0.02
        self.command_timer = self.create_timer(joystick_dt, self.publish_command)

        print("Command node initialized.")


    # initialize the joystick
    def init_joystick(self):

        # launch the ROS2 joy_node as a subprocess
        self.joy_process = subprocess.Popen(
            [
                'ros2', 'run', 'joy', 'joy_node',
                '--ros-args',
                '-r', 'joy:=/deploy_robot/joy',
                '-p', f'deadzone:={self.deadzone}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print("Launched joy_node as a subprocess.")

        # ROS2 subscribers
        self.joy_sub = self.create_subscription(Joy, 'deploy_robot/joy', self.joy_callback, 10)
        
        # wait for the first joystick message to be received
        self.joy_msg = None
        self.is_connected = 0.0
        print("No joystick found. Waiting for connection...")
        while self.joy_msg is None:
            rclpy.spin_once(self, timeout_sec=0.01)

        # joystick timeout vars to detect disconnections
        self._joy_timeout = 0.2  # seconds
        self._last_joy_time = time.time()
        

    # callback for joystick messages
    def joy_callback(self, msg: Joy):

        self.joy_msg = msg

        # update the last joy message time and connection status
        self._last_joy_time = time.time()
        if self.is_connected == 0.0:
            print("Joystick connected.")
            self.is_connected = 1.0


    # publish command based on the latest joystick message
    def publish_command(self):

        # check for timeout
        if time.time() - self._last_joy_time > self._joy_timeout:
            if self.is_connected == 1.0:
                print("Joystick disconnected.")
            self.is_connected = 0.0
            self.joystick_state = JoystickState()

        # if not connected, publish zero command and return
        if self.is_connected == 0.0:
            cmd_msg = Float32MultiArray()
            cmd_msg.data = [0.0] * 9  # same width as a connected message
            self.command_pub.publish(cmd_msg)
            return

        # update the joystick state
        self.joystick_state = rosjoy_to_joystick_state(self.joy_msg)

        # step the finite state machine if applicable
        fsm_state = self.fsm.step(self.joystick_state)
        fsm_msg = String()
        fsm_msg.data = fsm_state
        self.fsm_pub.publish(fsm_msg)

        # convert the joystick state to a command message
        LS_Y = self.joystick_state.LS_Y
        LS_X = self.joystick_state.LS_X
        RS_X = self.joystick_state.RS_X
        RS_Y = self.joystick_state.RS_Y
        RB = float(self.joystick_state.RB)
        U_DPAD = self.joystick_state.U_DPAD
        D_DPAD = self.joystick_state.D_DPAD
        Y = float(self.joystick_state.Y)

        # publish the command:
        #   [connected, LS_Y (vx), LS_X (vy), RS_X (wz), RS_Y, RB, U_DPAD, D_DPAD, Y]
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [self.is_connected, 
                        LS_Y, LS_X, RS_X, RS_Y, 
                        RB, U_DPAD, D_DPAD, Y]

        self.command_pub.publish(cmd_msg)


    # terminate joy_node on shutdown
    def destroy_node(self):
        if self.joy_process is not None:
            self.joy_process.terminate()
            self.joy_process.wait()
        super().destroy_node()


############################################################################
# MAIN FUNCTION
############################################################################

def main():

    # init ROS2
    rclpy.init()

    # create joystick node
    joystick_node = JoystickNode()

    # run normally
    try:
        while rclpy.ok():
            rclpy.spin_once(joystick_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    # ROS2 shutdown
    finally:
        joystick_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("Joystick shutdown complete.")


if __name__ == "__main__":
    main()
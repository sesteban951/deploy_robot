##
#
# Run joystick commands (using ROS2 joy_node)
#
##

# standard imports
import subprocess

# other imports
import time

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.joystick_utils import JoystickState, rosjoy_to_joystick_state


############################################################################
# COMMAND NODE
############################################################################

class JoystickNode(Node):
    """
    Use joystick to publish commands to the simulation and control nodes.
    """

    def __init__(self, config_path: str = None):

        super().__init__('joystick_node')

        # launch joy_node as a subprocess
        deadzone = 0.05
        self.joy_process = subprocess.Popen(
            [
                'ros2', 'run', 'joy', 'joy_node',
                '--ros-args',
                '-r', 'joy:=/deploy_robot/joy',
                '-p', f'deadzone:={deadzone}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print("Launched joy_node as a subprocess.")

        # joystick state
        self.joystick_state = JoystickState()

        # ROS2 subscribers
        self.joy_sub = self.create_subscription(Joy, 'deploy_robot/joy', self.joy_callback, 10)
        
        # wait for the first joystick message to be received
        self.joy_msg = None
        print("Waiting for the first joystick message...")
        while self.joy_msg is None:
            rclpy.spin_once(self, timeout_sec=0.01)
        self.is_connected = 1.0
        print("Received the first joystick message.")

        # ROS2 publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/joystick', 10)

        # create timer to publish commands at a fixed rate
        joystick_dt = 0.1
        self.command_timer = self.create_timer(joystick_dt, self.publish_command)

        print("Command node initialized.")


    # callback for joystick messages
    def joy_callback(self, msg: Joy):
        self.joy_msg = msg


    # publish command based on the latest joystick message
    def publish_command(self):
        # update the joystick state
        self.joystick_state = rosjoy_to_joystick_state(self.joy_msg)

        # convert the joystick state to a command message
        vx_cmd = self.joystick_state.LS_Y
        vy_cmd = self.joystick_state.LS_X
        omega_cmd = self.joystick_state.RS_X

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [self.is_connected, vx_cmd, vy_cmd, omega_cmd]

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
    cmd_node = JoystickNode()

    # run normally
    try:
        while rclpy.ok():
            rclpy.spin_once(cmd_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    # ROS2 shutdown
    finally:
        cmd_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("Joystick shutdown complete.")


if __name__ == "__main__":
    main()
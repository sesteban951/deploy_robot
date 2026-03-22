##
#
# Run joystick commands (using pygame)
#
##

# other imports
import pygame

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.joystick_utils import JoystickState, pygame_to_joystick_state


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

        # ROS2 publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/joystick', 10)

        # create timer to publish commands at a fixed rate
        command_freq = 100.0
        self.command_timer = self.create_timer(1.0 / command_freq, self.publish_command)

        print("Command node initialized.")


    # initialize the joystick
    def init_joystick(self):

        # initialize pygame and the joystick
        pygame.init()
        pygame.joystick.init()

        # check if a joystick is connected
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick connected: [{self.joystick.get_name()}].")
        else:
            print("No joystick found. Waiting for connection...")

    # update joystick state
    def update_joystick_state(self):

        # check for connect/disconnect events
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED and self.joystick is None:
                self.joystick = pygame.joystick.Joystick(event.device_index)
                self.joystick.init()
                print(f"Joystick connected: [{self.joystick.get_name()}].")
            elif event.type == pygame.JOYDEVICEREMOVED and self.joystick is not None:
                print("Joystick disconnected.")
                self.joystick = None
                self.joystick_state = JoystickState()

        # skip if no joystick connected
        if self.joystick is None:
            return

        # update
        try:
            self.joystick_state = pygame_to_joystick_state(self.joystick)
        except pygame.error:
            print("Joystick error.")
            self.joystick = None
            self.joystick_state = JoystickState()

    # publish the command
    def publish_command(self):
        # update the joystick state
        self.update_joystick_state()

        # if no joystick connected, skip
        if self.joystick is None:
            # default to zero command
            is_connected  = 0.0
            vx_cmd = 0.0
            vy_cmd = 0.0
            omega_cmd = 0.0

        # publish the command
        else:
            # convert the joystick state to a command message
            is_connected = 1.0
            vx_cmd = self.joystick_state.LS_Y
            vy_cmd = self.joystick_state.LS_X
            omega_cmd = self.joystick_state.RS_X

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [is_connected, vx_cmd, vy_cmd, omega_cmd]

        self.command_pub.publish(cmd_msg)

    # graceful shutdown
    def destroy_node(self):
        pygame.quit()           # shutdown pygame
        super().destroy_node()  # destroy the ROS2 node


############################################################################
# MAIN FUNCTION
############################################################################

def main():

    # init ROS2
    rclpy.init()

    # create joystick node
    cmd_node = JoystickNode()

    # execute the commands
    try:
        # spin the node
        rclpy.spin(cmd_node)

    except KeyboardInterrupt:
        pass

    finally:
        # close everything
        cmd_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
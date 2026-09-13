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
from std_msgs.msg import Float32MultiArray, String

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.joystick_utils import JoystickState, pygame_to_joystick_state
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
            # step the finite state machine
            fsm_state = self.fsm.step(self.joystick_state)
            fsm_msg = String()
            fsm_msg.data = fsm_state
            self.fsm_pub.publish(fsm_msg)

            # convert the joystick state to a command message
            #   vx = left stick fore/aft, vy = right stick lateral,
            #   wz = left trigger (turn +) minus right trigger (turn -)
            is_connected = 1.0
            vx_cmd = self.joystick_state.LS_Y
            vy_cmd = self.joystick_state.RS_X
            omega_cmd = self.joystick_state.LT - self.joystick_state.RT

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
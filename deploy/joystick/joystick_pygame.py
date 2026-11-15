##
#
# Run joystick commands (using pygame)
#
##

# standard imports
import argparse

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

    def __init__(self, alpha: float = 0.20, use_filter: bool = True):

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

        # first-order low-pass filter on the sticks: y += alpha * (x - y) every publish
        self.alpha = alpha
        self.use_filter = use_filter
        self.sticks = [0.0, 0.0, 0.0, 0.0]   # filtered LS_Y, LS_X, RS_X, RS_Y
        print(f"Stick filter: {'alpha ' + str(alpha) if use_filter else 'off'}")

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
            # default to zero command (and restart the filter from 0)
            self.sticks = [0.0, 0.0, 0.0, 0.0]
            is_connected  = 0.0
            LS_Y = 0.0
            LS_X = 0.0
            RS_X = 0.0
            RS_Y = 0.0
            RB = 0.0
            U_DPAD = 0.0
            D_DPAD = 0.0
            Y = 0.0

        # publish the command
        else:
            # step the finite state machine
            fsm_state = self.fsm.step(self.joystick_state)
            fsm_msg = String()
            fsm_msg.data = fsm_state
            self.fsm_pub.publish(fsm_msg)

            # convert the joystick state to a command message (sticks low-pass filtered)
            is_connected = 1.0
            raw = [self.joystick_state.LS_Y, self.joystick_state.LS_X, self.joystick_state.RS_X, self.joystick_state.RS_Y]
            if self.use_filter:
                self.sticks = [y + self.alpha * (x - y) for x, y in zip(raw, self.sticks)]
            else:
                self.sticks = raw
            LS_Y, LS_X, RS_X, RS_Y = self.sticks
            RB = float(self.joystick_state.RB)
            U_DPAD = self.joystick_state.U_DPAD
            D_DPAD = self.joystick_state.D_DPAD
            Y = float(self.joystick_state.Y)

        # publish the command:
        #   [connected, LS_Y (vx), LS_X (vy), RS_X (wz), RS_Y, RB, U_DPAD, D_DPAD, Y]
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [is_connected, LS_Y, LS_X, RS_X,
                        RS_Y, RB, U_DPAD, D_DPAD, Y]

        self.command_pub.publish(cmd_msg)

    # graceful shutdown
    def destroy_node(self):
        pygame.quit()           # shutdown pygame
        super().destroy_node()  # destroy the ROS2 node


############################################################################
# MAIN FUNCTION
############################################################################

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='Joystick node (pygame) with an optional stick low-pass filter.')
    parser.add_argument('--alpha', type=float, default=0.20,
                        help='low-pass filter alpha in (0, 1] per 20 ms publish; 1 = no smoothing (default 0.20)')
    parser.add_argument('--no-filter', action='store_true',
                        help='bypass the low-pass filter completely and publish the raw sticks')
    args = parser.parse_args()
    if not 0.0 < args.alpha <= 1.0:
        parser.error(f"--alpha must be in (0, 1], got {args.alpha}.")

    # init ROS2
    rclpy.init()

    # create joystick node
    joystick_node = JoystickNode(alpha=args.alpha, use_filter=not args.no_filter)

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
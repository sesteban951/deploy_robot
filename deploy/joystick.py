##
#
# Run joystick commands
#
##

# standard imports
from dataclasses import dataclass

# other imports
import pygame

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# directory imports
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")


############################################################################
# COMMAND NODE
############################################################################

# Xbox buttons
@dataclass
class JoystickState:
    
    # Buttons
    A: int = 0
    B: int = 0
    X: int = 0
    Y: int = 0
    LB: int = 0
    RB: int = 0
    LMB: int = 0   # left middle button
    RMB: int = 0   # right middle button
    LS: int = 0    # left stick press
    RS: int = 0    # right stick press
    MB: int = 0    # middle button between LMB and RMB

    # Axes
    LS_X: float = 0.0
    LS_Y: float = 0.0
    RS_X: float = 0.0
    RS_Y: float = 0.0
    LT: float = 0.0
    RT: float = 0.0
    L_DPAD: float = 0.0  # D-PAD left
    R_DPAD: float = 0.0  # D-PAD right
    U_DPAD: float = 0.0  # D-PAD up
    D_DPAD: float = 0.0  # D-PAD down

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
        self.command_pub = self.create_publisher(Float32MultiArray, 'joystick', 10)

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
            LS_X = -self.joystick.get_axis(0) # invert x-axis
            LS_Y = -self.joystick.get_axis(1) # invert y-axis
            RS_X = -self.joystick.get_axis(3) # invert x-axis
            RS_Y = -self.joystick.get_axis(4) # invert y-axis
            self.joystick_state.LS_X = LS_X if abs(LS_X) > self.deadzone else 0.0
            self.joystick_state.LS_Y = LS_Y if abs(LS_Y) > self.deadzone else 0.0
            self.joystick_state.RS_X = RS_X if abs(RS_X) > self.deadzone else 0.0
            self.joystick_state.RS_Y = RS_Y if abs(RS_Y) > self.deadzone else 0.0

            # triggers
            LT = 0.5 * self.joystick.get_axis(2) + 0.5
            RT = 0.5 * self.joystick.get_axis(5) + 0.5
            self.joystick_state.LT = LT if LT > self.deadzone else 0.0
            self.joystick_state.RT = RT if RT > self.deadzone else 0.0

            # D-PAD
            DPAD = self.joystick.get_hat(0)
            
            DPAD_X = DPAD[0]
            if DPAD_X <= -0.5:
                self.joystick_state.L_DPAD = 1.0
                self.joystick_state.R_DPAD = 0.0
            elif DPAD_X >= 0.5:
                self.joystick_state.L_DPAD = 0.0
                self.joystick_state.R_DPAD = 1.0
            else:
                self.joystick_state.L_DPAD = 0.0
                self.joystick_state.R_DPAD = 0.0

            DPAD_Y = DPAD[1]
            if DPAD_Y <= -0.5:
                self.joystick_state.U_DPAD = 0.0
                self.joystick_state.D_DPAD = 1.0
            elif DPAD_Y >= 0.5:
                self.joystick_state.U_DPAD = 1.0
                self.joystick_state.D_DPAD = 0.0
            else:
                self.joystick_state.U_DPAD = 0.0
                self.joystick_state.D_DPAD = 0.0

            # buttons
            self.joystick_state.A = self.joystick.get_button(0)
            self.joystick_state.B = self.joystick.get_button(1)
            self.joystick_state.X = self.joystick.get_button(2)
            self.joystick_state.Y = self.joystick.get_button(3)
            self.joystick_state.LB = self.joystick.get_button(4)
            self.joystick_state.RB = self.joystick.get_button(5)
            self.joystick_state.LMB = self.joystick.get_button(6)
            self.joystick_state.RMB = self.joystick.get_button(7)
            self.joystick_state.LS = self.joystick.get_button(9)
            self.joystick_state.RS = self.joystick.get_button(10)
            self.joystick_state.MB = self.joystick.get_button(11)

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
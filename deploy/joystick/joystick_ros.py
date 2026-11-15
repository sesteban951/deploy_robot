##
#
# Run joystick commands (using ROS2 joy_node, or the userspace XInput USB reader)
#
##

# standard imports
import argparse
import atexit
import signal
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
from utils.joystick_utils import JoystickState, XInputUSBJoystick, rosjoy_to_joystick_state
from utils.finite_state_machine import FiniteStateMachine


############################################################################
# COMMAND NODE
############################################################################

class JoystickNode(Node):
    """
    Use joystick to publish commands to the simulation and control nodes.
    """

    def __init__(self, config_path: str = None, source: str = "joy_node"):

        super().__init__('joystick_node')

        # initialize joystick
        self.deadzone = 0.05
        self.source = source
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

        # joystick sources:
        #   joy_node:   ROS2 joy package (kernel joystick driver), launched as a subprocess
        #   xinput_usb: userspace XInput USB reader (e.g. 8BitDo on the Jetson Orin), read in-process
        self.joy_process = None
        self.xinput = None
        self.joy_msg = None
        self.is_connected = 0.0

        # joystick timeout vars to detect disconnections (joy_node only)
        self._joy_timeout = 0.2  # seconds
        self._last_joy_time = time.time()

        if self.source == "xinput_usb":
            self.xinput = XInputUSBJoystick(deadzone=self.deadzone)
            self.xinput.start()
            print("Started xinput_usb reader.")

            # wait for the first joystick report
            print("No joystick found. Waiting for connection...")
            while self.xinput.get_state() is None:
                time.sleep(0.01)
            return

        self.joy_process = subprocess.Popen(
            [
                'ros2', 'run', 'joy', 'joy_node',
                '--ros-args',
                '-r', 'joy:=/deploy_robot/joy',
                '-p', f'deadzone:={self.deadzone}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # own process group, so `ros2 run`'s child (the actual joy_node) is stopped with it
            start_new_session=True,
        )

        # make sure the subprocess is stopped even if we exit before destroy_node() runs
        # (e.g. Ctrl+C while still waiting for the first joystick message)
        atexit.register(self._stop_joy_process)

        print("Launched joy_node as a subprocess.")

        # ROS2 subscribers
        self.joy_sub = self.create_subscription(Joy, 'deploy_robot/joy', self.joy_callback, 10)
        
        # wait for the first joystick message to be received
        print("No joystick found. Waiting for connection...")
        while self.joy_msg is None:
            rclpy.spin_once(self, timeout_sec=0.01)


    # callback for joystick messages
    def joy_callback(self, msg: Joy):

        self.joy_msg = msg
        self._last_joy_time = time.time()


    # publish command based on the latest joystick message
    def publish_command(self):

        # read the latest joystick state
        if self.xinput is not None:
            # reader returns None while the controller is disconnected
            state = self.xinput.get_state()
        elif time.time() - self._last_joy_time <= self._joy_timeout:
            state = rosjoy_to_joystick_state(self.joy_msg)
        else:
            # joy_node timeout
            state = None

        # if not connected, publish zero command and return
        if state is None:
            if self.is_connected == 1.0:
                print("Joystick disconnected.")
            self.is_connected = 0.0
            self.joystick_state = JoystickState()
            cmd_msg = Float32MultiArray()
            cmd_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.command_pub.publish(cmd_msg)
            return

        # update the connection status and joystick state
        if self.is_connected == 0.0:
            print("Joystick connected.")
            self.is_connected = 1.0
        self.joystick_state = state

        # step the finite state machine if applicable
        fsm_state = self.fsm.step(self.joystick_state)
        fsm_msg = String()
        fsm_msg.data = fsm_state
        self.fsm_pub.publish(fsm_msg)

        # convert the joystick state to a command message
        vx_cmd = self.joystick_state.LS_Y
        vy_cmd = self.joystick_state.LS_X
        omega_cmd = self.joystick_state.RS_X

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [self.is_connected, vx_cmd, vy_cmd, omega_cmd]

        self.command_pub.publish(cmd_msg)


    # terminate the joystick subprocess (safe to call more than once)
    def _stop_joy_process(self):
        if self.joy_process is None:
            return
        # signal the whole process group (ros2 run wrapper + joy_node)
        try:
            pgid = os.getpgid(self.joy_process.pid)
        except ProcessLookupError:
            return
        for sig, wait_s in ((signal.SIGINT, 2.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 1.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            # wait until every process in the group has exited, else escalate
            deadline = time.time() + wait_s
            while time.time() < deadline:
                self.joy_process.poll()   # reap the direct child
                try:
                    os.killpg(pgid, 0)    # raises once the group is empty
                except ProcessLookupError:
                    return
                time.sleep(0.05)

    # terminate joy_node / the xinput reader on shutdown
    def destroy_node(self):
        self._stop_joy_process()
        if self.xinput is not None:
            self.xinput.stop()
        super().destroy_node()


############################################################################
# MAIN FUNCTION
############################################################################

def main():

    # parse args
    parser = argparse.ArgumentParser(description="Joystick command node.")
    parser.add_argument("--source", choices=["joy_node", "xinput_usb"], default="joy_node",
                        help="joystick driver: ROS2 joy_node (default) or the userspace XInput USB reader "
                             "(for XInput pads the kernel driver can't read, e.g. 8BitDo on the Jetson Orin)")
    args = parser.parse_args()

    # init ROS2
    rclpy.init()

    # create joystick node (Ctrl+C while waiting for the joystick exits cleanly)
    try:
        joystick_node = JoystickNode(source=args.source)
    except KeyboardInterrupt:
        if rclpy.ok():
            rclpy.shutdown()
        print("Joystick shutdown complete.")
        return

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
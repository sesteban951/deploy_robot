##
#
# Control node for replaying a motion.
#
##

# standard imports
import argparse
import time

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64, Float32MultiArray

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)


########################################################################
# GLOBAL VARIABLES (DO NOT CHANGE)
########################################################################

G1_NUM_MOTOR = 29

########################################################################
# CONTROLLER NODE
########################################################################

class ControlNode(Node):
    """
    Control node that replays a motion trajectory.
    """

    def __init__(self, config_path: str):

        super().__init__('control_node')

        # load config file
        self.config = self.load_config(config_path)

        # load params
        self.init_params()

        # ROS publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'command', 10)

        # ROS subscribers
        self.hardware_time_sub = self.create_subscription(Float64, 'hardware_time', self.hardware_time_callback, 10)
        self.state_machine_sub = self.create_subscription(Int32, 'state_machine', self.state_machine_callback, 10)

        # control timer to run the policy at a fixed frequency
        self.control_timer = self.create_timer(self.ctrl_dt, self.control_publish)

        # current hardware time
        self.hardware_time = 0.0
        self.state_machine = 0

        # control timer for internal use
        self.control_time = 0.0

        print("Control node initialized.")

    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from [{config_path_full}].")

        return config
    
    # load the parameters from the config file
    def init_params(self):

        # load the motion trajectory
        motion_path = self.config['motion_path']
        motion_path_full = ROOT_DIR + "/motions/" + motion_path

        # load the npz motion trajectory
        motion_traj = np.load(motion_path_full)
        print(f"Loaded motion trajectory from: [{motion_path_full}].")

        # extract the default joint positions from the motion file
        self.qpos = motion_traj["joint_pos"]
        self.n_frames = self.qpos.shape[0]
        fps = float(motion_traj["fps"][0])

        # playback scaling
        playback_speed = self.config['playback_speed']

        # control frequency
        self.ctrl_dt = self.config['control_dt']

        # PD gains from config
        self.Kp = np.array(self.config['Kp'], dtype=np.float64)
        self.Kd = np.array(self.config['Kd'], dtype=np.float64)

        # default joint positions from config
        self.default_joint_pos = np.array(self.config['default_joint_pos'], dtype=np.float64)

        # interpolation and hold durations
        self.interp_duration = float(self.config['interp_default_pos_duration'])
        self.hold_duration = float(self.config['hold_default_pos_duration'])

        # type checks
        assert type(fps) == float, "FPS must be a float."
        assert type(playback_speed) == float, "Playback speed must be a float."
        assert type(self.ctrl_dt) == float, "Control dt must be a float."

        # length checks
        assert self.qpos.shape[1] == G1_NUM_MOTOR, "Joint position dimension mismatch."
        assert len(self.Kp) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kp values."
        assert len(self.Kd) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kd values."

        # value checks
        assert self.n_frames > 0, "Motion trajectory must have at least one frame."
        assert fps > 0, "FPS must be positive."
        assert playback_speed > 0, "Playback speed must be positive."
        assert playback_speed <= 1.5, "Playback speed greater than 1.5 is too risky."
        assert self.ctrl_dt > 0, "Control dt must be positive."

        # create time array
        frame_dt = 1.0 / (fps * playback_speed)
        self.time_array = np.arange(self.n_frames) * frame_dt
        self.T = self.time_array[-1]

        # playback state
        self.stage = 0  # 0=interp, 1=hold, 2=motion
        self.t0 = None  # set on first control_publish call
        self.start_joint_pos = None  # captured on first call

        print(f"Motion trajectory loaded with [{self.n_frames} frames] at [{fps} FPS].")
        print(f"     Playback speed scaling: [{playback_speed} x].")
        print(f"     Total trajectory time: [{self.T:.2f} seconds].")

    #################################################################
    # HELPERS
    #################################################################

    # hardware time
    def hardware_time_callback(self, msg):
        self.hardware_time = msg.data

    # state machine from hardware
    def state_machine_callback(self, msg):
        self.state_machine = msg.data

    # linear interpolation between two arrays
    def lerp(self, a, b, alpha):
        return (1.0 - alpha) * a + alpha * b

    # publish position command at control frequency
    def control_publish(self):

        # safety: abort if hardware is not ready for control
        if self.state_machine < 2:
            print(f"\nWarning: Hardware is not ready to take commands (state_machine = {self.state_machine}). "
                  f"Wait until hardware is ready before starting. Shutting down.")
            print("Shutting down control node.")
            sys.exit(1)

        # increment control time
        self.control_time += self.ctrl_dt

        # [Stage 0]: interpolate from default pos to first motion frame
        if self.control_time < self.interp_duration:
            alpha = self.control_time / self.interp_duration
            qpos_des = self.lerp(self.default_joint_pos, self.qpos[0, :], alpha)
            print(f"[interp] t={self.control_time:.2f}/{self.interp_duration:.2f}  alpha={alpha:.3f}\r", end="")

        # [Stage 1]: hold first motion frame
        elif self.control_time < self.interp_duration + self.hold_duration:
            qpos_des = self.qpos[0, :].copy()
            t_hold = self.control_time - self.interp_duration
            print(f"[hold] t={t_hold:.2f}/{self.hold_duration:.2f}\r", end="")

        # [Stage 2]: play motion trajectory with linear interpolation between frames
        else:
            t_motion = self.control_time - self.interp_duration - self.hold_duration
            i = np.searchsorted(self.time_array, t_motion) - 1
            i = np.clip(i, 0, self.n_frames - 2)
            j = i + 1
            alpha = (t_motion - self.time_array[i]) / (self.time_array[j] - self.time_array[i])
            alpha = np.clip(alpha, 0.0, 1.0)
            qpos_des = self.lerp(self.qpos[i, :], self.qpos[j, :], alpha)
            print(f"[motion] t={t_motion:.2f}/{self.T:.2f}  frame={i}-{j}  alpha={alpha:.3f}\r", end="")

        # publish command: [q(29), dq(29), Kp(29), Kd(29), tau_ff(29)]
        dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        tau_ff = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, dq_des, self.Kp, self.Kd, tau_ff]).tolist()
        self.command_pub.publish(cmd_msg)



############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Asynchronous Simulation Node using Mujoco.'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the Mujoco config yaml file. Example: "g1_29dof.yaml".'
    )
    args = parser.parse_args()

    # create the simulation node
    ctrl_node = ControlNode(args.config)

    # spin briefly to receive state_machine updates before prompting
    for _ in range(50):
        rclpy.spin_once(ctrl_node, timeout_sec=0.01)

    print(f"\nHardware state_machine = {ctrl_node.state_machine}")
    while input("Press [Enter] to continue: ") != "":
        pass
    print()

    # TODO: / WARNING: this is really bad design, redesign ASAP
    # spin again to get latest state after user presses Enter
    for _ in range(10):
        rclpy.spin_once(ctrl_node, timeout_sec=0.01)

    # execute the policy
    try:
        # spin the node
        rclpy.spin(ctrl_node)
    
    except KeyboardInterrupt:
        pass

    finally:
        # close everything
        ctrl_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
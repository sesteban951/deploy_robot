##
#
# Control node for the Go2 MjLab velocity + body pose policy (Go2-Velocity-Pose-Custom), hardware.
# Same as deploy/simulation/go2/control_pose.py, but the gait phase clock is the hardware node's
# fsm_time (restarts at 0 when entering a new FSM state), and the policy steps on fsm_time arrival
# (see utils/policy_timing.py and time_callback for the timing details).
#
# Joystick:
#   left stick         vx (fore/aft), vy (lateral)
#   right stick X      wz (yaw)
#   hold RB            right stick Y -> pitch (forward = nose down),
#                      right stick X -> roll (right = right side down), yaw zeroed
#   D-pad up / down    raise / lower the height (held; also while walking)
#   Y                  reset the height to nominal
# Roll and pitch only apply while standing, and low heights only at slow speeds:
# utils/locomotion/body_pose.py applies the rules the policy was trained under.
#
##


# standard imports
import argparse

# other imports
import math
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray

# directory imports
import sys
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

# custom imports
from utils.unitree_utils import get_gravity_orientation
from utils.policy import Policy
from utils.experiment_utils import publish_experiment_info
from utils.policy_timing import PolicyTiming


# hardware.py ticks every 5 ms, and fsm_time restarts exactly on a tick, so later ticks land
# exactly on the ctrl_dt slot boundaries. Shift the slot grid by half a tick so a boundary falls
# between two ticks: tick jitter under 2.5 ms can then never move a step to the next tick.
TRIGGER_OFFSET = 0.0025  # [sec] half of hardware.py's LOW_LEVEL_CONTROL_DT
from utils.locomotion.body_pose import effective_body_pose


############################################################################
# CONTROLLER NODE
############################################################################

class ControlNode(Node):
    """
    Asynchronous control node that runs the policy and sends actions to the simulation.
    """

    def __init__(self, config_path: str, print_timing: bool = False):

        super().__init__('control_node')

        # load config file
        self.config = self.load_config(config_path)

        # load params
        self.init_policy()

        # broadcast which experiment is running so the logger can record it
        self.experiment_info_pub = publish_experiment_info(self, config_path, self.config, self.policy)

        # ROS publishers
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/command', 10)

        # ROS subscribers (sensor depth 1: always act on the freshest state)
        # NOTE: keep fsm_time last. The policy steps inside time_callback, and the executor
        # handles ready subscriptions in creation order, so this tick's IMU and joint state
        # are taken first (hardware.py also publishes fsm_time last).
        self.cmd_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joystick', self.cmd_callback, 10)
        self.imu_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/imu_state', self.imu_sensor_callback, 1)
        self.joint_sensor_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/joint_state', self.joint_sensor_callback, 1)
        self.timing = PolicyTiming(self, self.ctrl_dt, print_timing)   # before fsm_time (see PolicyTiming)
        self.fsm_time_sub = self.create_subscription(Float64, 'deploy_robot/fsm_time', self.time_callback, 1)

        # no free-running timer: the policy steps once per control_dt of the hardware clock,
        # on the first fresh sensor tick of each step (see time_callback). A timer here would
        # add 0-5 ms of unsynchronized wait on top of the sensor age.
        self.last_step = None
        self.slot_time = 0.0  # [sec] step * ctrl_dt: the ideal time of the current policy step

        # sensor state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.omega = np.zeros(3)
        self.qpos_joints = np.array(self.qpos_joints_default.copy())
        self.qvel_joints = np.zeros_like(self.qpos_joints_default)
        self.fsm_time = 0.0

        # initialize command: twist and body pose (raw targets, then the effective pose)
        self.cmd = np.zeros(3)
        self.pose_raw = np.array([self.pose_params["nominal_height"], 0.0, 0.0], dtype=np.float32)
        self.pose_cmd = self.pose_raw.copy()
        self.U_DPAD = 0.0
        self.D_DPAD = 0.0
        self.Y = 0.0

        # initialize the action
        self.action = np.zeros(self.act_size)

        print("Control node initialized.")


    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it (accept the name with or without the .yaml extension)
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        config_path_full = ROOT_DIR + "/deploy/configs/go2/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from [{config_path_full}].")

        return config
    
    # initialize the policy
    def init_policy(self):

        # default_joint_pos and action_scale are loaded from the policy metadata
        self.qpos_joints_default = self.config.get('default_joint_pos')

        # scaling params
        self.action_scale = self.config.get('action_scale')
        self.cmd_range = np.array(self.config["cmd_range"], dtype=np.float32)  # (3, 2): [min, max]

        # PD gains
        self.Kp = np.array(self.config["Kp"], dtype=np.float32)
        self.Kd = np.array(self.config["Kd"], dtype=np.float32)

        # control frequency
        self.ctrl_dt = self.config["control_dt"]

        # gait phase params
        self.gait_period = self.config["gait_period"]
        self.stand_cmd_threshold = self.config["stand_cmd_threshold"]

        # body pose params (must match the training config)
        self.pose_params = self.config["body_pose"]
        self.height_rate = self.config["height_rate"]

        # import the policy
        policy_path = self.config['policy_path']
        policy_path_full = ROOT_DIR + "/policy/go2/" + policy_path
        
        # load the policy
        self.policy = Policy(policy_path_full)

        # alias for convenience
        self.obs_size = self.policy.input_size
        self.act_size = self.policy.output_size

        # deployment params embedded in the policy
        self.qpos_joints_default = self.policy.get_param('default_joint_pos', self.qpos_joints_default)
        self.action_scale = self.policy.get_param('action_scale', self.action_scale)
        assert len(self.qpos_joints_default) == self.act_size, \
            f"default_joint_pos has {len(self.qpos_joints_default)} values, expected {self.act_size}."
        # mjlab exports a single action_scale when every joint shares it
        if len(self.action_scale) == 1:
            self.action_scale = np.full(self.act_size, self.action_scale[0], dtype=np.float32)
        assert len(self.action_scale) == self.act_size, \
            f"action_scale has {len(self.action_scale)} values, expected {self.act_size}."
        for _k in ('default_joint_pos', 'action_scale'):
            print(f"    {_k}: from {'policy metadata' if _k in self.policy.metadata else 'yaml config'}")

        # the yaml's home pose must be the policy's default pose: hardware.py stands up to
        # home_joint_pos in 'home', and a zero action holds default_joint_pos -- they must agree
        home = np.array(self.config["home_joint_pos"], dtype=np.float32)
        gap = float(np.abs(home - np.array(self.qpos_joints_default, dtype=np.float32)).max())
        assert gap < 2e-3, \
            f"home_joint_pos differs from the policy's default_joint_pos by {gap:.4f} rad; " \
            f"they must be the same standing idle pose."

        expected_obs = 3 + 3 + 3 + 3 + 2 + 3 * self.act_size
        assert self.obs_size == expected_obs, \
            f"policy expects {self.obs_size} observations, the pose layout builds {expected_obs}."

        print(f"Loading policy from [{policy_path_full}].")
        print(f"    Policy type: {self.policy._policy_type}")
        print(f"    Input size: {self.obs_size}")
        print(f"    Output size: {self.act_size}")
        print(f"    Control frequency: {1.0 / self.ctrl_dt} Hz")


    #################################################################
    # HELPERS
    #################################################################

    # command from the joystick node:
    #   [connected, LS_Y (vx), LS_X (vy), RS_X (wz), RS_Y, RB, U_DPAD, D_DPAD, Y]
    def cmd_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)

        # joystick
        joystick_is_connected = (data[0] > 0.5)
        vx_cmd = data[1]
        vy_cmd = data[2]
        omega_cmd = data[3]

        # body pose inputs (absent from an older joystick publisher: keep the pose neutral)
        pose_inputs = data[4:9] if len(data) >= 9 else np.zeros(5, dtype=np.float32)
        RS_Y, RB, self.U_DPAD, self.D_DPAD, self.Y = pose_inputs

        # hold RB: the right stick sets the attitude, and yaw is zeroed so the robot stands
        if RB > 0.5:
            self.pose_raw[1] = -np.clip(omega_cmd, -1.0, 1.0) * self.pose_params["roll_range"][1]
            self.pose_raw[2] = np.clip(RS_Y, -1.0, 1.0) * self.pose_params["pitch_range"][1]
            omega_cmd = 0.0
        else:
            self.pose_raw[1:] = 0.0

        # map the stick [-1, 1] onto the command range: +1 -> max, -1 -> min
        stick = np.clip(np.array([vx_cmd, vy_cmd, omega_cmd], dtype=np.float32), -1.0, 1.0)
        self.cmd = np.where(stick >= 0.0, stick * self.cmd_range[:, 1], -stick * self.cmd_range[:, 0])

    # height ramp from the D-pad (Y resets to nominal), integrated once per policy step
    def update_height(self):
        if self.Y > 0.5:
            self.pose_raw[0] = self.pose_params["nominal_height"]
        else:
            direction = float(self.U_DPAD > 0.5) - float(self.D_DPAD > 0.5)
            self.pose_raw[0] += self.height_rate * self.ctrl_dt * direction
        lo, hi = self.pose_params["height_range"]
        self.pose_raw[0] = np.clip(self.pose_raw[0], lo, hi)

    # base IMU data: [rpy(3), quat(4), gyro(3), acc(3)]
    def imu_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.quat = data[3:7]
        self.omega = data[7:10]
        self.timing.set_imu_stamp(data, 13)

    # joint data: [q(n), dq(n), ddq(n), tau_est(n)]
    def joint_sensor_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        n = len(self.qpos_joints_default)
        self.qpos_joints = data[:n]
        self.qvel_joints = data[n:2*n]
        self.timing.set_joint_stamp(data, 4 * n)

    # time since entering the current FSM state; steps the policy at control_dt.
    # Step index = floor((fsm_time + TRIGGER_OFFSET) / control_dt): exactly 1/control_dt steps per
    # second of the hardware clock (no drift), every 4th tick; a dropped tick only delays one step.
    def time_callback(self, msg):
        prev_time = self.fsm_time
        self.fsm_time = msg.data
        step = int((self.fsm_time + TRIGGER_OFFSET) // self.ctrl_dt)
        # new step, or fsm_time restarted (FSM transition)
        if self.last_step is None or step != self.last_step or self.fsm_time < prev_time:
            self.last_step = step
            self.slot_time = step * self.ctrl_dt
            self.control_callback()

    # build the observation vector for the policy
    def build_observation(self):

        # base orientation state
        gravity_orientation = get_gravity_orientation(self.quat)

        # joint position and velocity errors
        qj = (self.qpos_joints - self.qpos_joints_default)
        dqj = self.qvel_joints
        
        # gait phase clock (zeroed when standing), from the step's slot time rather than the raw
        # fsm_time: the phase then advances exactly ctrl_dt per step, as in training
        phase = (self.slot_time % self.gait_period) / self.gait_period
        two_pi = 2.0 * math.pi
        gait_phase = np.array([
            math.sin(two_pi * phase),
            math.cos(two_pi * phase),
        ], dtype=np.float32)
        if np.linalg.norm(self.cmd) < self.stand_cmd_threshold:
            gait_phase[:] = 0.0

        # build the observation vector
        # ['base_ang_vel', 'projected_gravity', 'command', 'body_pose', 'phase', 'joint_pos', 'joint_vel', 'actions']
        n = len(qj)
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[0:3]              = self.omega
        obs[3:6]              = gravity_orientation
        obs[6:9]              = self.cmd
        obs[9:12]             = self.pose_cmd
        obs[12:14]            = gait_phase
        obs[14:14+n]          = qj
        obs[14+n:14+2*n]      = dqj
        obs[14+2*n:14+3*n]    = self.action

        return obs

    # control published at the control frequency
    def control_callback(self):
        self.timing.step_start()

        # body pose: ramp the height, then apply the training-time feasibility rules
        self.update_height()
        self.pose_cmd = effective_body_pose(self.pose_raw, self.cmd, self.pose_params)

        # get the current observation
        obs = self.build_observation()

        # target joint positions (PD control)
        self.timing.inference_start()
        self.action = self.policy.inference(obs)
        self.timing.inference_end()

        # build the command: [q_des, dq_des, Kp, Kd, tau_ff]
        qpos_des = self.action * self.action_scale + self.qpos_joints_default
        qvel_des = np.zeros(self.act_size, dtype=np.float32)
        tau_ff = np.zeros(self.act_size, dtype=np.float32)

        # publish the command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = np.concatenate([qpos_des, qvel_des, self.Kp, self.Kd, tau_ff]).tolist()
        self.command_pub.publish(cmd_msg)
        self.timing.step_end(self.fsm_time, self.slot_time)


############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # init ROS2
    rclpy.init()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Go2 velocity + body pose control node (hardware).'
    )
    # config path argument
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the Mujoco config yaml file. Example: "go2_pose.yaml".'
    )
    parser.add_argument(
        '--print-timing',
        action='store_true',
        help='Print a policy step timing summary (rate, intervals, observation age) every 5 s.'
    )
    args = parser.parse_args()

    # create the control node
    ctrl_node = ControlNode(args.config, print_timing=args.print_timing)

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
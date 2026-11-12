##
#
# Simulation node using Mujoco to simulate the Go2 robot.
#
##

# standard imports
import argparse
import time
from collections import deque

# mujoco imports
import mujoco
import mujoco.viewer

# other imports
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
from utils.math_utils import quat_to_rpy


############################################################################
# SIMULATION SETTINGS
############################################################################

# Physics integrates at --sim-hz; the viewer renders at RENDER_HZ (decoupled).
RENDER_HZ = 50.0   # [Hz] viewer render rate
CAM_MSG_SEC = 2.0  # [s] how long the camera-mode line stays up after a [C] press


############################################################################
# SIMULATION NODE
############################################################################

class SimulationNode(Node):
    """
    Asynchronous simulation node that runs the Mujoco simulation.
    """

    def __init__(self, config_path: str, apply_noise: bool = False, delay_ms: int = 0, sim_hz: int = 500):

        super().__init__('simulation_node')

        # load config file
        self.config = self.load_config(config_path)

        # whether to apply per-sensor Gaussian noise
        self.apply_noise = apply_noise

        # how long a received command is held before it takes effect [ms]
        self.delay_ms = delay_ms

        # physics rate [Hz]
        self.sim_hz = sim_hz

        # load params
        self.init_params()

        # initialize mujoco
        self.init_simulation()

        # ROS publishers
        self.imu_state_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/imu_state', 10)
        self.joint_state_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/joint_state', 10)
        self.simulation_time_pub = self.create_publisher(Float64, 'deploy_robot/simulation_time', 10)

        # ROS subscribers
        self.command_sub = self.create_subscription(Float32MultiArray, 'deploy_robot/command', self.command_callback, 10)

        # initial command state
        self.command_received = False
        self.qpos_des = np.zeros(self.nu)
        self.qvel_des = np.zeros(self.nu)
        self.tau_ff = np.zeros(self.nu)
        self.Kp = np.zeros(self.nu)
        self.Kd = np.zeros(self.nu)

        # create a timer to run the simulation loop
        sim_period = 0.0 # run as fast as possible, real-time sync is handled in the loop
        self.timer = self.create_timer(sim_period, self.step_simulation)

        # create timers for publishing
        imu_state_period = self.sim_dt
        joint_state_period = self.sim_dt
        self.imu_timer = self.create_timer(imu_state_period, self.publish_imu)
        self.joint_timer = self.create_timer(joint_state_period, self.publish_joint_state)

        print("Simulation node initialized.")
        print("    Press [Tab] to toggle the left UI.")
        print("    Press [Shift + Tab] to toggle the right UI.")
        print(f"    Press [C] to toggle camera tracking of the {self.track_body_name}.")


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

        return config


    # load policy params
    def init_params(self):

        # set the default state
        self.default_base = np.array(self.config['home_base_pos'])
        self.home_joints = np.array(self.config['home_joint_pos'])


    # initialize the mujoco simulation
    def init_simulation(self):
        # load the XML path
        models_path = ROOT_DIR + "/models/"
        xml_path = models_path + self.config['xml_path']

        # load the mujoco model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # simulate at sim_hz (overrides the model's <option timestep>)
        self.mj_model.opt.timestep = 1.0 / self.sim_hz

        # load model properties
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu
        self.sim_dt = self.mj_model.opt.timestep

        # command delay in whole sim steps (one step is the finest delay the sim can represent)
        step_ms = 1e3 / self.sim_hz
        delay_exact = self.delay_ms / step_ms
        self.delay_steps = int(delay_exact + 0.5)
        if abs(delay_exact - self.delay_steps) > 1e-6:
            print(f"WARNING: --delay {self.delay_ms} ms is not a multiple of the {step_ms:g} ms sim step; "
                  f"using {self.delay_steps * step_ms:g} ms.")
        self._cmd_queue = deque()   # (step at which to apply, command data), oldest first
        self._step_count = 0
        delay_applied_ms = self.delay_steps * self.sim_dt * 1e3
        self._delay_line = f"\nCmd delay:   {delay_applied_ms:.0f} ms" if self.delay_steps > 0 else ""

        # make sure the default joints are the correct size
        assert len(self.home_joints) == self.nu, (f"Default joint angles must be of size "
                                                     f"{self.nu}, got {len(self.home_joints)}.")

        # assign initial state
        self.mj_data.qpos[:7] = self.default_base
        self.mj_data.qpos[7:7+self.nu] = self.home_joints

        # build list of joint sensor names (matching actuator order)
        self.joint_pos_sensor_names = []
        self.joint_vel_sensor_names = []
        for i in range(self.nu):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.joint_pos_sensor_names.append(f"{joint_name}_pos_sensor")
            self.joint_vel_sensor_names.append(f"{joint_name}_vel_sensor")

        print(f"Loaded Mujoco model from [{xml_path}].")
        print(f"    Sim dt: {self.sim_dt} seconds.")
        print(f"    nq: {self.nq}")
        print(f"    nv: {self.nv}")
        print(f"    nu: {self.nu}")
        print(f"    Command delay: {delay_applied_ms:.1f} ms ({self.delay_steps} sim steps).")

        # [C] camera tracking follows the floating-base body, keeping the user's orbit and zoom
        free_joints = np.flatnonzero(self.mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.track_body_id = int(self.mj_model.jnt_bodyid[free_joints[0]])
        self.track_body_name = self.mj_model.body(self.track_body_id).name
        self.cam_track = False
        self._cam_dirty = False
        self._contact_points = False
        self._cam_msg_until = 0.0   # wall time until which the camera-mode line is shown

        # launch the viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model,
            self.mj_data,
            key_callback=self.key_callback,
            show_left_ui=False,  # disable left tab (use 'Tab' for toggling on/off)
            show_right_ui=False, # disable right tab (use 'Tab + Shift' for toggling on/off)
        )

        # viewer settings
        self._viewer_font_scale = getattr(
            mujoco.mjtFontScale,
            'mjFONTSCALE_250',
            getattr(mujoco.mjtFontScale, 'mjFONTSCALE_200', mujoco.mjtFontScale.mjFONTSCALE_150),
        )

        # camera settings
        self.viewer.cam.azimuth   = 135    # degrees, horizontal rotation
        self.viewer.cam.elevation = -20    # degrees, negative looks down
        self.viewer.cam.distance  = 1.5    # meters from lookat point
        self.viewer.cam.lookat[:] = list(self.default_base[0:3]) # (x, y, z) point to look at

        self.viewer_render_hz = RENDER_HZ
        self._last_viewer_sync = 0.0
        self._real_start_time = time.perf_counter()
        self._next_step_deadline = self._real_start_time + self.sim_dt

        # real-time-rate tracking (sim time vs wall time over a render interval)
        self._rtr_last_sim = 0.0
        self._rtr_last_wall = self._real_start_time


    #################################################################
    # PUBLISHING AND CALLBACKS
    #################################################################

    # viewer key press (runs on the viewer thread): only flag it, the sim loop applies it
    def key_callback(self, keycode):
        if keycode == 67:  # C: toggle camera tracking
            self.cam_track = not self.cam_track
            self._cam_dirty = True


    # command callback: [q_des, dq_des, Kp, Kd, tau_ff] (nu * 5)
    def command_callback(self, msg):
        data = np.array(msg.data)

        # hold the command for delay_steps sim steps before it takes effect
        if self.delay_steps > 0:
            self._cmd_queue.append((self._step_count + self.delay_steps, data))
            return
        self.apply_command(data)


    # unpack a command and make it the active one (same order as hardware)
    def apply_command(self, data):
        self.command_received = True
        self.qpos_des = data[0*self.nu : 1*self.nu]
        self.qvel_des = data[1*self.nu : 2*self.nu]
        self.Kp       = data[2*self.nu : 3*self.nu]
        self.Kd       = data[3*self.nu : 4*self.nu]
        self.tau_ff   = data[4*self.nu : 5*self.nu]


    # publish base IMU: [rpy(3), quat(4), gyro(3), acc(3)]
    def publish_imu(self):
        imu_quat = self.mj_data.sensor('imu_quat_sensor').data.copy()
        imu_gyro = self.mj_data.sensor('imu_gyro_sensor').data.copy()
        imu_acc  = self.mj_data.sensor('imu_acc_sensor').data.copy()
        imu_rpy  = quat_to_rpy(imu_quat)

        imu_msg = Float32MultiArray()
        imu_msg.data = np.concatenate([imu_rpy, imu_quat, imu_gyro, imu_acc]).tolist()
        self.imu_state_pub.publish(imu_msg)


    # publish joint state: [q(nu), dq(nu), ddq(nu), tau_est(nu)]
    def publish_joint_state(self):
        qpos_joints = np.array([self.mj_data.sensor(name).data[0] for name in self.joint_pos_sensor_names])
        qvel_joints = np.array([self.mj_data.sensor(name).data[0] for name in self.joint_vel_sensor_names])
        ddq_joints = np.zeros(self.nu)  # NOTE: no such thing as joint acceleration sensors in Mujoco, so we publish zeros here
        tau_est_joints = self.mj_data.ctrl[:self.nu].copy()

        joint_state_msg = Float32MultiArray()
        joint_state_msg.data = np.concatenate([qpos_joints, qvel_joints, ddq_joints, tau_est_joints]).tolist()

        self.joint_state_pub.publish(joint_state_msg)

    #################################################################
    # SIMULATION
    #################################################################

    # add per-sensor Gaussian noise in place on mj_data.sensordata using the
    # std devs declared in the XML (sensor_noise[i] is the std dev for sensor i)
    def _apply_sensor_noise(self):
        for i in range(self.mj_model.nsensor):
            std = self.mj_model.sensor_noise[i]
            if std <= 0.0:
                continue
            adr = self.mj_model.sensor_adr[i]
            dim = self.mj_model.sensor_dim[i]
            self.mj_data.sensordata[adr:adr+dim] += np.random.normal(0.0, std, size=dim)

    # compute torque using PD control + feedforward
    def compute_torque(self):

        # get current joint positions and velocities
        qpos_joints = self.mj_data.qpos[7:7+self.nu]
        qvel_joints = self.mj_data.qvel[6:6+self.nu]

        # tau = kp * (qpos_des - qpos) + kd * (qvel_des - qvel) + tau_ff
        tau = (self.Kp * (self.qpos_des - qpos_joints)
             + self.Kd * (self.qvel_des - qvel_joints)
             + self.tau_ff)

        return tau


    # step the simulation
    def step_simulation(self):

        # release the delayed commands that are due (oldest first, so the newest due one wins)
        while self._cmd_queue and self._cmd_queue[0][0] <= self._step_count:
            self.apply_command(self._cmd_queue.popleft()[1])

        # compute the torque to apply
        if self.command_received == True:
            tau = self.compute_torque()
            self.mj_data.ctrl[:] = tau
        else:
            self.mj_data.ctrl[:] = 0.0

        # step the simulation
        mujoco.mj_step(self.mj_model, self.mj_data)
        self._step_count += 1

        # inject sensor noise (newer mujoco dropped the sensornoise, so we
        # apply the per-sensor std devs from XML noise="..." attrs by hand)
        if self.apply_noise:
            self._apply_sensor_noise()

        # publish simulation time
        time_msg = Float64()
        time_msg.data = self.mj_data.time
        self.simulation_time_pub.publish(time_msg)

        # sync viewer at viewer_render_hz
        now = time.perf_counter()
        if self.viewer.is_running() and (now - self._last_viewer_sync) >= 1.0 / self.viewer_render_hz:
            # apply a [C] toggle, and undo the viewer's own C shortcut (contact points)
            contact_flag = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
            if self._cam_dirty:
                self._cam_dirty = False
                self._cam_msg_until = now + CAM_MSG_SEC
                with self.viewer.lock():
                    self.viewer.opt.flags[contact_flag] = self._contact_points
                    if self.cam_track:
                        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        self.viewer.cam.trackbodyid = self.track_body_id
                    else:
                        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self._contact_points = bool(self.viewer.opt.flags[contact_flag])

            # update the viewer with the current simulation state
            self.viewer.sync()

            # real-time rate (RTR): sim seconds advanced per wall second over the
            # last render interval (1.0 == running exactly at real-time)
            real_elapsed = now - self._real_start_time
            d_wall = now - self._rtr_last_wall
            rtr = (self.mj_data.time - self._rtr_last_sim) / d_wall if d_wall > 0.0 else 0.0
            self._rtr_last_sim = self.mj_data.time
            self._rtr_last_wall = now

            # camera-mode line, shown only for a moment after a [C] press
            cam_line = ""
            if now < self._cam_msg_until:
                cam_line = f"\nCamera:      {'tracking ' + self.track_body_name if self.cam_track else 'free'}"

            self.viewer.set_texts((
                self._viewer_font_scale,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                f"Sim time:    {self.mj_data.time:.2f}s\n"
                f"Real time:   {real_elapsed:.2f}s\n"
                f"RTR:         {rtr:.2f}\n"
                f"Sim rate:    {self.sim_hz:.0f} Hz\n"
                f"Render rate: {RENDER_HZ:.0f} Hz" + self._delay_line + cam_line,
                "",
            ))

            self._last_viewer_sync = now

        # Real-time sync against an absolute schedule so drift does not accumulate.
        remaining = self._next_step_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

        # Keep a fixed absolute schedule so the loop can catch up after overruns.
        self._next_step_deadline += self.sim_dt


    # shutdown the node and close the viewer
    def destroy_node(self):
        if self.viewer.is_running():
            self.viewer.close()
        super().destroy_node()


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
        help='Path to the config yaml file. Example: "go2_12dof.yaml".'
    )
    # enable sensor noise (off by default)
    parser.add_argument(
        '--noise',
        action='store_true',
        help='Enable per-sensor Gaussian noise injection. Noise is off by default.'
    )
    # delay every received command before applying it (off by default)
    parser.add_argument(
        '--delay',
        type=int,
        default=0,
        help='Command delay in whole milliseconds (default 0). Example: "--delay 20".'
    )
    # physics rate
    parser.add_argument(
        '--sim-hz',
        type=int,
        default=500,
        help='Simulation rate in Hz (default 500). Example: "--sim-hz 1000".'
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be zero or positive.")
    if args.sim_hz <= 0:
        parser.error("--sim-hz must be a positive integer.")

    # create the simulation node
    sim_node = SimulationNode(args.config, apply_noise=args.noise, delay_ms=args.delay, sim_hz=args.sim_hz)

    # run normally
    try:
        while rclpy.ok() and sim_node.viewer.is_running():
            rclpy.spin_once(sim_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    # ROS2 shutdown
    finally:
        sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("Simulation shutdown complete.")


if __name__ == "__main__":
    main()

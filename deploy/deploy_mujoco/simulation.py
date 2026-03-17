##
#
# Simulation node using Mujoco to simulate the robot.
#
##

# standard imports
import argparse
import time

# mujoco imports
import mujoco
import mujoco.viewer

# other imports
import numpy as np
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64, Float32MultiArray

# directory imports
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")


############################################################################
# SIMULATION NODE
############################################################################

class SimulationNode(Node):
    """
    Asynchronous simulation node that runs the Mujoco simulation.
    """

    def __init__(self, config_path: str):

        super().__init__('sim_node')

        # load config file
        self.config = self.load_config(config_path)

        # load params
        self.init_params()

        # initialize mujoco
        self.init_simulation()

        # ROS publishers
        self.imu_sensor_pub = self.create_publisher(Float32MultiArray, 'imu_data', 10)
        self.joint_sensor_pub = self.create_publisher(Float32MultiArray, 'joint_data', 10)
        self.time_pub = self.create_publisher(Float64, 'sim_time', 10)

        # ROS subscribers
        self.action_sub = self.create_subscription(Float32MultiArray, 'action', self.action_callback, 10)
        # self.state_machine_sub = self.create_subscription(Int32, 'state_machine', self.state_machine_callback, 10)

        # initial action and state
        self.qpos_joints_des = None
        self.state = 0

        # create a timer to run the simulation loop 
        sim_period = 0.0 # run as fast as possible, real-time sync is handled in the loop
        self.timer = self.create_timer(sim_period, self.step_simulation)

        # create timers for publishing
        imu_sensor_period = self.sim_dt  # or whatever period you want
        joint_sensor_period = self.sim_dt
        self.imu_timer = self.create_timer(imu_sensor_period, self.publish_imu)             
        self.joint_timer = self.create_timer(joint_sensor_period, self.publish_joint_state)

        print("Simulation node initialized.")

    #################################################################
    # INITIALIZATION
    #################################################################

    # load the config file
    def load_config(self, config_path: str):
        # open the config file and load it
        config_path_full = ROOT_DIR + "/deploy/configs/" + config_path
        with open(config_path_full, 'r') as f:
            config = yaml.safe_load(f)

        return config
    

    # load policy params
    def init_params(self):

        # PD gains
        self.Kp = np.array(self.config['kps'])
        self.Kd = np.array(self.config['kds'])

        # set the default state
        self.default_base = np.array(self.config['default_base_pos'])
        self.default_joints = np.array(self.config['default_joint_pos'])


    # initialize the mujoco simulation
    def init_simulation(self):        
        # load the XML path
        models_path = ROOT_DIR + "/models/"
        xml_path = models_path + self.config['xml_path']

        # load the mujoco model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # load model properties
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu
        self.sim_dt = self.mj_model.opt.timestep

        # make sure the gains are the correct size
        assert len(self.Kp) == self.nu, f"Kp must be of size {self.nu}, got {len(self.Kp)}."
        assert len(self.Kd) == self.nu, f"Kd must be of size {self.nu}, got {len(self.Kd)}."
        assert len(self.default_joints) == self.nu, (f"Default joint angles must be of size"
                                                     f"{self.nu}, got {len(self.default_joints)}.")

        # assign initial state
        self.mj_data.qpos[:7] = self.default_base
        self.mj_data.qpos[7:7+self.nu] = self.default_joints

        print(f"Loaded Mujoco model from [{xml_path}].")
        print(f"    Sim dt: {self.sim_dt} seconds.")
        print(f"    nq: {self.nq}")
        print(f"    nv: {self.nv}")
        print(f"    nu: {self.nu}")

        # launch the viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer_render_hz = 50.0
        self._last_viewer_sync = 0.0
        self._real_start_time = time.perf_counter()
        self._next_step_deadline = self._real_start_time + self.sim_dt

        # adjust viewer font
        self._viewer_font_scale = getattr(
            mujoco.mjtFontScale,
            'mjFONTSCALE_250',
            getattr(mujoco.mjtFontScale, 'mjFONTSCALE_200', mujoco.mjtFontScale.mjFONTSCALE_150),
        )


    #################################################################
    # HELPERS 
    #################################################################

    # get the current state machine state
    def state_machine_callback(self, msg):
        self.state = msg.data

    # get the current action command
    def action_callback(self, msg):
        # array of desired joint positions
        self.qpos_joints_des = np.array(msg.data)

    # compute torque given the action and current state
    def compute_torque(self, qpos_joints_des):

        # get current joint positions and velocities
        qpos_joints = self.mj_data.qpos[7:7+self.nu]
        qvel_joints = self.mj_data.qvel[6:6+self.nu]

        # compute the torque using a PD controller
        tau = self.Kp * (qpos_joints_des - qpos_joints) + self.Kd * (0 - qvel_joints)

        return tau

    # publish sensor data
    def publish_imu(self):
        # get the IMU data from the simulation
        quat = self.mj_data.qpos[3:7]   # quaternion (w, x, y, z)
        omega = self.mj_data.qvel[3:6]  # omega (wx, wy, wz)

        imu_sensor_msg = Float32MultiArray()
        imu_sensor_msg.data = np.concatenate([quat, omega]).tolist()

        self.imu_sensor_pub.publish(imu_sensor_msg)


    # publish joint state data
    def publish_joint_state(self):
        # get the joint data from the simulation
        qpos_joints = self.mj_data.qpos[7:7+self.nu]
        qvel_joints = self.mj_data.qvel[6:6+self.nu]

        joint_state_msg = Float32MultiArray()
        joint_state_msg.data = np.concatenate([qpos_joints, qvel_joints]).tolist()

        self.joint_sensor_pub.publish(joint_state_msg)


    # step the simulation
    def step_simulation(self):
        # compute the action torque
        if self.qpos_joints_des is not None:
            tau = self.compute_torque(self.qpos_joints_des)
            self.mj_data.ctrl[:] = tau
        else:
            self.mj_data.ctrl[:] = 0.0

        # step the simulation
        mujoco.mj_step(self.mj_model, self.mj_data)

        # publish sim time
        time_msg = Float64()
        time_msg.data = self.mj_data.time
        self.time_pub.publish(time_msg)

        # sync viewer at viewer_render_hz
        now = time.perf_counter()
        if self.viewer.is_running() and (now - self._last_viewer_sync) >= 1.0 / self.viewer_render_hz:
            # update the viewer with the current simulation state
            self.viewer.sync()

            # display sim time first and wall-clock elapsed time second
            real_elapsed = now - self._real_start_time
            self.viewer.set_texts((
                self._viewer_font_scale,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                f"Sim time:   {self.mj_data.time:.2f}s\nReal time: {real_elapsed:.2f}s",
                "",
            ))
            
            self._last_viewer_sync = now

        # Real-time sync against an absolute schedule so drift does not accumulate.
        remaining = self._next_step_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

        # Keep a fixed absolute schedule so the loop can catch up after overruns.
        self._next_step_deadline += self.sim_dt



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
    sim_node = SimulationNode(args.config)

    # execute the simulation
    try:
        # spin the node
        rclpy.spin(sim_node)
    
    except KeyboardInterrupt:
        pass

    finally:
        # close everything
        sim_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

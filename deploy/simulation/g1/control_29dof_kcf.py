"""29-joint G1 simulation with the hardware running policy and full KCF feedback.

Only MuJoCo's atomic state topic is read. No SDK or mocap process is used.
Policy inference, observation construction and model-contract checks are inherited
from the existing 29-joint policy controller used for the hardware recordings.
"""

import argparse
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String
import yaml

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('DEPLOY_ROOT_DIR', str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.hardware.g1.control_29dof_kcf import ControlNode as RunningPolicyNode
from deploy.kcf.kcf_interface import CommandGate, Feedback, COMMAND_TOPIC, FEEDBACK_TOPIC, SIM_STATE_TOPIC
from kcf_mpc.profiles import load_profile
from utils.experiment_utils import publish_experiment_info


class ControlNode(RunningPolicyNode):
    def __init__(self, config_path):
        Node.__init__(self, 'simulation_29dof_kcf_policy')
        self.config = self.load_config(config_path)
        self.kcf_profile = load_profile(self.config['kcf_profile'])
        self.kcf_profile.validate_assets()
        self.init_policy()
        self.gate = CommandGate(dt=self.ctrl_dt, limits=self.command_limits, command_bands=self.command_bands)
        self.fsm = 'init'
        self.policy_time = 0.0
        self.last_state_time = None
        self.quat = np.array([1., 0., 0., 0.])
        self.omega = np.zeros(3)
        self.qpos_joints = self.qpos_joints_default.copy()
        self.qvel_joints = np.zeros(29)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        self.command_source = 'initializing'
        self.kcf_state = None
        self.command_pub = self.create_publisher(Float32MultiArray, 'deploy_robot/command', 10)
        self.feedback_pub = self.create_publisher(Float64MultiArray, FEEDBACK_TOPIC, 1)
        self.experiment_info_pub = publish_experiment_info(self, config_path, self.config, self.policy)
        self.create_subscription(String, '/deploy_robot/fsm', self.fsm_callback, 1)
        self.create_subscription(Float32MultiArray, '/deploy_robot/joystick',
                                 lambda msg: self.gate.set_manual(msg.data, monotonic()), 1)
        self.create_subscription(Float64MultiArray, COMMAND_TOPIC,
                                 lambda msg: self.gate.set_mpc(msg.data, monotonic()), 1)
        self.create_subscription(Float64MultiArray, SIM_STATE_TOPIC, self.state_callback, 1)

    def load_config(self, config_path):
        path = Path(config_path)
        if not path.is_absolute():
            path = ROOT / 'deploy/configs/g1' / path
        if path.suffix != '.yaml':
            path = path.with_suffix('.yaml')
        return yaml.safe_load(path.read_text())

    def state_callback(self, msg):
        state = np.asarray(msg.data, dtype=float)
        if (state.shape != (72,) or not np.isfinite(state).all() or state[0] < 0
                or abs(np.linalg.norm(state[4:8]) - 1) > .01):
            self.gate.candidates.clear()
            return
        t = float(state[0])
        if self.last_state_time is not None and abs(t - self.last_state_time) < 1e-8:
            return
        if self.last_state_time is not None and t < self.last_state_time:
            self.action[:] = 0
            self.gate.candidates.clear()
            self.clock.reset()
        elif self.last_state_time is not None and self.fsm in ('control', 'track'):
            for _ in range(max(0, round((t - self.last_state_time) / self.ctrl_dt) - 1)):
                self.clock.advance()
        self.last_state_time = t
        self.kcf_state = state.copy()
        self.policy_time = round(t / self.ctrl_dt) * self.ctrl_dt
        self.quat = state[4:8].copy()
        self.omega = state[40:43].copy()
        self.qpos_joints = state[8:37].copy()
        self.qvel_joints = state[43:72].copy()
        self.control_callback()

    def prepare_kcf_feedback(self):
        t = self.policy_time
        q = self.kcf_state[1:37].copy()
        v = self.kcf_state[37:72].copy()
        self.cmd, self.command_source = self.gate.select(t, monotonic())
        feedback = Feedback(t, t, t, True, q, v, self.action.copy(), self.cmd.copy(), self.clock.state.copy())
        self.feedback_pub.publish(Float64MultiArray(data=feedback.pack()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='g1_29dof_kcf.yaml')
    args = parser.parse_args()
    rclpy.init()
    node = ControlNode(args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

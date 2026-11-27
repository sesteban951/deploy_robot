"""Backend-independent state and command-source contracts; no simulator imports."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RobotState:
    t: float
    q: np.ndarray
    v: np.ndarray
    prev_action: np.ndarray
    policy_clock: np.ndarray = field(default_factory=lambda: np.array([0., 1., 43.], dtype=np.float32))


def state_from_feedback(feedback):
    """Preserve the actor clock and raw previous action in MPC feedback."""
    return RobotState(feedback.time, feedback.q, feedback.v, feedback.previous_action, feedback.policy_clock)


def nominal_state(cfg):
    q = np.concatenate([cfg.default_base_pos, cfg.default_joint_pos]).astype(np.float64)
    return RobotState(0.0, q, np.zeros(6 + len(cfg.default_joint_pos)),
                      np.zeros(len(cfg.default_joint_pos)))


class CommandSource:
    def __init__(self, cmd_scale):
        self.cmd_scale = np.asarray(cmd_scale, dtype=np.float32)
        self.t_start, self.ref = 0.0, None

    def get(self, state):
        raise NotImplementedError

    def close(self):
        pass

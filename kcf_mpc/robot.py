"""Backend-independent state and command-source contracts; no simulator imports."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    t: float
    q: np.ndarray
    v: np.ndarray
    prev_action: np.ndarray
    obstacle_position: np.ndarray | None = None
    obstacle_time: float | None = None


def state_from_feedback(feedback):
    """Walking adapter; future recurrent profiles provide their own adapter."""
    return RobotState(feedback.time, feedback.q, feedback.v, feedback.previous_action,
                      feedback.obstacle_position, feedback.obstacle_time)


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

from dataclasses import dataclass, field
import numpy as np


@dataclass
class G1Config:

    # PD gains per joint
    Kp: np.ndarray = field(default_factory=lambda: np.array([
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179, 28.501, 28.501, 14.251, 14.251, 14.251,
        14.251, 14.251, 16.778, 16.778, 14.251, 14.251,
        14.251, 14.251, 14.251, 16.778, 16.778,
    ]))
    Kd: np.ndarray = field(default_factory=lambda: np.array([
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558, 1.814, 1.814, 0.907, 0.907, 0.907,
        0.907, 0.907, 1.068, 1.068, 0.907, 0.907,
        0.907, 0.907, 0.907, 1.068, 1.068,
    ]))

    # default positions
    default_base_pos: np.ndarray = field(default_factory=lambda: np.array([
        0, 0, 0.78, 1, 0, 0, 0,
    ]))
    default_joint_pos: np.ndarray = field(default_factory=lambda: np.array([
        -0.312, 0.000, 0.000, 0.669, -0.363, 0.000,
        -0.312, 0.000, 0.000, 0.669, -0.363, 0.000,
        0.000, 0.000, 0.000, 0.200, 0.200, 0.000,
        0.600, 0.000, 0.000, 0.000, 0.200, -0.200,
        0.000, 0.600, 0.000, 0.000, 0.000,
    ]))

    # action scale
    action_scale: np.ndarray = field(default_factory=lambda: np.array([
        0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
        0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
        0.548, 0.439, 0.439, 0.439, 0.439, 0.439,
        0.439, 0.439, 0.075, 0.075, 0.439, 0.439,
        0.439, 0.439, 0.439, 0.075, 0.075,
    ]))

    # control
    control_dt: float = 0.02

    # gait
    gait_period: float = 0.6
    stand_cmd_threshold: float = 0.1 # set to 0.0 to always moving feet

    # Physical body command clipping [vx m/s, vy m/s, omega rad/s].
    # Walking collection limits; this simulator does not multiply observations by these.
    cmd_scale: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 1.0]))
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

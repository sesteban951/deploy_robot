"""Running policy deployment parameters and audited command support."""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import numpy as np

POLICY_PATH = Path(__file__).resolve().parents[1] / "policy" / "g1" / "2026-11-19_12-04-23_jogrunsprint_backpack_v0.onnx"
# Audited epoch05460 training bands, narrower than the actor's sprint envelope.
# Commands are already effective body-frame values; do not scale them again.
COMMAND_BANDS = (
    ((0., 0., 0.), (0., 0., 0.)),
    ((.5, 0., -.5), (2., 0., .5)),
    ((-1., 0., -.5), (-.5, 0., .5)),
    ((0., 0., 1.), (0., 0., 2.)),
    ((0., 0., -2.), (0., 0., -1.)),
)

@lru_cache(maxsize=1)
def policy_metadata():
    import onnx
    return {entry.key: entry.value for entry in onnx.load(str(POLICY_PATH)).metadata_props}


def _joint_array(key):
    value = np.fromstring(policy_metadata()[key], sep=",", dtype=np.float64)
    if value.shape != (29,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid running policy metadata: {key}")
    return value


@dataclass
class G1Config:
    Kp: np.ndarray = field(default_factory=lambda: _joint_array("joint_stiffness"))
    Kd: np.ndarray = field(default_factory=lambda: _joint_array("joint_damping"))
    default_joint_pos: np.ndarray = field(default_factory=lambda: _joint_array("default_joint_pos"))
    action_scale: np.ndarray = field(default_factory=lambda: _joint_array("action_scale"))
    default_base_pos: np.ndarray = field(default_factory=lambda: np.array([0., 0., .783030, 1., 0., 0., 0.]))
    control_dt: float = .02
    cmd_scale: np.ndarray = field(default_factory=lambda: np.array([2., 0., 2.]))
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

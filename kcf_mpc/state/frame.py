"""Fixed ground/yaw frame for the native 85-state G1 schema."""

from dataclasses import dataclass
import numpy as np
from kcf_mpc.state.relative_frame import RelativeFrame

G1_SINCOS_STATE_DIM = 85
G1_RELATIVE_SINCOS_SCHEMA = "g1_kcf_relative85_sincos_window_v1"


def _values(values, name, width):
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != width:
        raise ValueError(f"{name} must have shape (..., {width})")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real numeric values")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True)
class G1KCFSinCosFrame:
    """Transform native states; retain continuous heading for diagnostics."""

    frame: RelativeFrame
    yaw_origin_world: float | None = None

    def __post_init__(self):
        if (
            abs(float(self.frame.origin_world[2])) > 1e-5
            or np.max(np.abs(self.frame.orientation_world[1:3])) > 1e-5
        ):
            raise ValueError(
                "the G1 frame must be ground-anchored and contain yaw only"
            )
        yaw = (
            self.frame.heading_world
            if self.yaw_origin_world is None
            else float(self.yaw_origin_world)
        )
        difference = yaw - self.frame.heading_world
        if (
            not np.isfinite(yaw)
            or abs(np.arctan2(np.sin(difference), np.cos(difference))) > 1e-5
        ):
            raise ValueError(
                "continuous heading must match the frame orientation modulo 2*pi"
            )
        object.__setattr__(self, "yaw_origin_world", float(yaw))

    @classmethod
    def from_state(cls, anchor_state_world, *, heading_world=None):
        anchor = _values(anchor_state_world, "anchor_state_world", G1_SINCOS_STATE_DIM)
        if anchor.shape != (G1_SINCOS_STATE_DIM,):
            raise ValueError("anchor_state_world must have shape (85,)")
        if not np.isclose(np.linalg.norm(anchor[2:4]), 1.0, atol=1e-5):
            raise ValueError("heading sine/cosine must have unit norm")
        half_yaw = 0.5 * np.arctan2(anchor[2], anchor[3])
        frame = RelativeFrame.from_pose(
            [anchor[0], anchor[1], 0.0],
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            yaw_only=True,
        )
        return cls(frame, heading_world)

    def _pose(self, pose, *, inverse=False):
        pose = _values(pose, "pose", 4)
        if np.any(np.linalg.norm(pose[..., 2:4], axis=-1) < 1e-12):
            raise ValueError("heading sine/cosine must have nonzero magnitude")
        precise = pose.astype(np.float64, copy=False)
        position = np.zeros((*pose.shape[:-1], 3), dtype=np.float64)
        position[..., :2] = precise[..., :2]
        if inverse:
            position = self.frame.position_to_world(position)
            heading = self.frame.heading_sincos_to_world(precise[..., 2:4])
            heading = np.stack((np.sin(heading), np.cos(heading)), axis=-1)
        else:
            position = self.frame.position_to_local(position)
            heading = self.frame.heading_to_local_sincos(
                np.arctan2(precise[..., 2], precise[..., 3])
            )
        return np.concatenate((position[..., :2], heading), axis=-1).astype(pose.dtype)

    def pose_to_local(self, pose_world):
        return self._pose(pose_world)

    def state_to_local(self, state_world):
        result = _values(state_world, "state_world", G1_SINCOS_STATE_DIM).copy()
        result[..., :4] = self.pose_to_local(result[..., :4])
        return result

    def state_to_world(self, state_local):
        result = _values(state_local, "state_local", G1_SINCOS_STATE_DIM).copy()
        result[..., :4] = self._pose(result[..., :4], inverse=True)
        return result

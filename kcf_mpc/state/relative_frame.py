"""
``RelativeFrame`` represents one fixed local frame in world coordinates.  The
frame is defined by a world-space origin and a unit quaternion describing the
local-to-world rotation.
Quaternions use MuJoCo's ``(w, x, y, z)`` convention.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kcf_mpc.state.math_utils import body_to_world, world_to_body, yaw_from_quat

__all__ = ["RelativeFrame"]

_QUATERNION_EPS = 1e-12


def _vectors(values, size: int, name: str) -> np.ndarray:
    """Return finite floating-point vectors with the requested trailing size."""
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size})")
    if (not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)):
        raise TypeError(f"{name} must contain numeric values")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unit_quaternions(values, name: str) -> np.ndarray:
    """Validate and normalize ``(..., 4)`` wxyz quaternions."""
    quaternion = _vectors(values, 4, name)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= _QUATERNION_EPS):
        raise ValueError(f"{name} must not contain a zero quaternion")
    return quaternion / norm


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[..., 1:] *= -1.0
    return result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of broadcast-compatible wxyz quaternions."""
    left, right = np.broadcast_arrays(left, right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class RelativeFrame:
    """A fixed local coordinate frame represented in world coordinates.

    Args:
        origin_world: World position of the local origin, shape ``(3,)``.
        orientation_world: Local-to-world unit quaternion in wxyz order,
            shape ``(4,)``.
    """

    origin_world: np.ndarray
    orientation_world: np.ndarray

    def __post_init__(self) -> None:
        origin = _vectors(self.origin_world, 3, "origin_world")
        orientation = _unit_quaternions(
            self.orientation_world, "orientation_world"
        )
        if origin.shape != (3,):
            raise ValueError("origin_world must have shape (3,)")
        if orientation.shape != (4,):
            raise ValueError("orientation_world must have shape (4,)")

        origin = np.array(origin, copy=True)
        orientation = np.array(orientation, copy=True)
        origin.setflags(write=False)
        orientation.setflags(write=False)
        object.__setattr__(self, "origin_world", origin)
        object.__setattr__(self, "orientation_world", orientation)

    @classmethod
    def from_pose(
        cls,
        position_world,
        orientation_world,
        *,
        yaw_only: bool = False,
    ) -> "RelativeFrame":
        """Create a frame from a world pose.

        With ``yaw_only=True``, the frame keeps world vertical and uses only
        the pose's heading for its orientation.
        """
        position = _vectors(position_world, 3, "position_world")
        orientation = _unit_quaternions(
            orientation_world, "orientation_world"
        )
        if position.shape != (3,) or orientation.shape != (4,):
            raise ValueError("from_pose requires one position and one orientation")
        if yaw_only:
            half_yaw = 0.5 * yaw_from_quat(orientation)
            orientation = np.array(
                [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
                dtype=np.result_type(orientation.dtype, np.float32),
            )
        return cls(position, orientation)

    @property
    def heading_world(self) -> float:
        """Heading of the local frame's x-axis in world coordinates."""
        return float(yaw_from_quat(self.orientation_world))

    def position_to_local(self, position_world) -> np.ndarray:
        """Transform world positions ``(..., 3)`` into the local frame."""
        position = _vectors(position_world, 3, "position_world")
        return world_to_body(
            self.orientation_world, position - self.origin_world
        )

    def position_to_world(self, position_local) -> np.ndarray:
        """Transform local positions ``(..., 3)`` into the world frame."""
        position = _vectors(position_local, 3, "position_local")
        return body_to_world(self.orientation_world, position) + self.origin_world

    def orientation_to_local(self, orientation_world) -> np.ndarray:
        """Transform world orientations ``(..., 4)`` into the local frame."""
        orientation = _unit_quaternions(
            orientation_world, "orientation_world"
        )
        relative = _quaternion_multiply(
            _quaternion_conjugate(self.orientation_world), orientation
        )
        return _unit_quaternions(relative, "orientation_local")

    def orientation_to_world(self, orientation_local) -> np.ndarray:
        """Transform local orientations ``(..., 4)`` into the world frame."""
        orientation = _unit_quaternions(
            orientation_local, "orientation_local"
        )
        world = _quaternion_multiply(self.orientation_world, orientation)
        return _unit_quaternions(world, "orientation_world")

    def linear_velocity_to_local(self, velocity_world) -> np.ndarray:
        """Rotate world linear velocities ``(..., 3)`` into the local frame."""
        velocity = _vectors(velocity_world, 3, "velocity_world")
        return world_to_body(self.orientation_world, velocity)

    def linear_velocity_to_world(self, velocity_local) -> np.ndarray:
        """Rotate local linear velocities ``(..., 3)`` into the world frame."""
        velocity = _vectors(velocity_local, 3, "velocity_local")
        return body_to_world(self.orientation_world, velocity)

    def heading_to_local_sincos(self, heading_world) -> np.ndarray:
        """Encode world heading as ``[..., sin(delta), cos(delta)]``.

        ``delta`` is relative to this fixed frame's world heading.  The result
        represents physical orientation modulo ``2*pi``.
        """
        heading = np.asarray(heading_world)
        if (not np.issubdtype(heading.dtype, np.number)
                or np.issubdtype(heading.dtype, np.complexfloating)):
            raise TypeError("heading_world must contain numeric values")
        if not np.issubdtype(heading.dtype, np.floating):
            heading = heading.astype(np.float64)
        if not np.isfinite(heading).all():
            raise ValueError("heading_world must contain only finite values")
        relative = heading - self.heading_world
        return np.stack((np.sin(relative), np.cos(relative)), axis=-1)

    def orientation_heading_to_local_sincos(
        self, orientation_world
    ) -> np.ndarray:
        """Encode headings extracted from world quaternions in the local frame."""
        orientation = _unit_quaternions(
            orientation_world, "orientation_world"
        )
        return self.heading_to_local_sincos(
            yaw_from_quat(orientation)
        )

    def heading_sincos_to_world(self, heading_local_sincos) -> np.ndarray:
        """Decode local sine/cosine headings to wrapped world headings.

        The returned angles lie in ``[-pi, pi]``.
        """
        encoded = _vectors(
            heading_local_sincos, 2, "heading_local_sincos"
        )
        if np.any(np.linalg.norm(encoded, axis=-1) <= _QUATERNION_EPS):
            raise ValueError("heading_local_sincos must have nonzero magnitude")
        world = np.arctan2(encoded[..., 0], encoded[..., 1]) + self.heading_world
        return np.arctan2(np.sin(world), np.cos(world))

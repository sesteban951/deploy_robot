##
#
# Useful math utilities for the G1 humanoid simulation and policy.
#
##

import numpy as np


def gravity_from_quat(quat: np.ndarray) -> np.ndarray:
    """Projected gravity in body frame from base quaternion (w,x,y,z)."""
    qw, qx, qy, qz = quat
    gx = 2.0 * (-qz * qx + qw * qy)
    gy = -2.0 * (qz * qy + qw * qx)
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return np.array([gx, gy, gz], dtype=np.float32)


def body_to_world(quat: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Rotate a vector from body to world frame: v_world = R(q) @ v_body.

    Uses the closed-form  R(q) v = v + 2w (q_vec x v) + 2 q_vec x (q_vec x v).
    Broadcasts over arbitrary leading dims.

    Args:
        quat:   (..., 4) unit quaternion in (w, x, y, z) order.
        v_body: (..., 3) vector(s) expressed in the body frame.
    Returns:
        v_world: (..., 3) same vector(s) expressed in the world frame.
    """
    w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    c = np.cross(q_vec, v_body, axis=-1)
    return v_body + 2.0 * w * c + 2.0 * np.cross(q_vec, c, axis=-1)


def world_to_body(quat: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Rotate a vector from world to body frame: v_body = R(q)^T @ v_world.

    Inverse of `body_to_world`. quat is (..., 4) in (w, x, y, z), v_world is (..., 3).
    """
    w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    c = np.cross(q_vec, v_world, axis=-1)
    return v_world - 2.0 * w * c + 2.0 * np.cross(q_vec, c, axis=-1)


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    """Yaw angle (rotation about world z) from quaternion (w, x, y, z).

    Args:
        quat: (..., 4) unit quaternion in (w, x, y, z) order.
    Returns:
        theta: (...,) yaw in radians, range (-pi, pi].
    """
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def q_to_rom(q: np.ndarray) -> np.ndarray:
    """Reduce full qpos to a reduced-order pose [px, py, theta] in the world frame.

    Args:
        q: (..., nq) with q[..., 0:3] = base position, q[..., 3:7] = quat (w, x, y, z).
    Returns:
        q_rom: (..., 3) of [px, py, theta].
    """
    px = q[..., 0]
    py = q[..., 1]
    theta = yaw_from_quat(q[..., 3:7])
    return np.stack([px, py, theta], axis=-1)


def v_to_rom(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reduce full qvel to a reduced-order twist [vx, vy, omega] in the body frame.

    MuJoCo's free-joint qvel stores world-frame linear velocity and body-frame
    angular velocity. Rotate only the linear part; angular velocity must not
    be rotated a second time.

    Args:
        q: (..., nq) full qpos; q[..., 3:7] is the base quaternion.
        v: (..., nv) full qvel; v[..., 0:3] world linear, v[..., 3:6] body angular.
    Returns:
        v_rom: (..., 3) of [vx_body, vy_body, omega_z_body].
    """
    quat = q[..., 3:7]
    v_lin_body = world_to_body(quat, v[..., 0:3])   # (..., 3)
    return np.stack([v_lin_body[..., 0],
                     v_lin_body[..., 1],
                     v[..., 5]], axis=-1)

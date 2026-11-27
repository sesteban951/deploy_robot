"""Encode live MuJoCo feedback in the trained 104-state G1 KCF schema."""

from typing import Protocol
import numpy as np
from kcf_mpc.state.math_utils import gravity_from_quat, q_to_rom, v_to_rom, world_to_body


class G1State(Protocol):
    """State fields needed by the encoder (MuJoCo qpos/qvel conventions)."""

    t: float
    q: np.ndarray
    v: np.ndarray
    policy_clock: np.ndarray
    prev_action: np.ndarray


def g1_closed_loop_state(state: G1State) -> np.ndarray:
    """Build the native 104 channels, with position/heading in world coordinates.

    [px, py, sin(theta), cos(theta)] (4), [vx, vy, wz] (3), pz (1),
    gravity (3), joint q (29),
    [vz, wx, wy] (3), joint v (29), gait [sin, cos, period_frames] (3), previous action (29).
    MPC anchors the complete 104-state vector to one fixed rollout frame.
    """
    q = state.q.astype(np.float32)
    v = state.v.astype(np.float32)
    if q.shape != (36,) or v.shape != (35,) or np.shape(state.prev_action) != (29,):
        raise ValueError("Running feedback requires all 29 joints and previous actions")
    quat = q[3:7]
    planar_pose = q_to_rom(q).astype(np.float32, copy=False)
    pose = np.array(
        [
            planar_pose[0],
            planar_pose[1],
            np.sin(planar_pose[2]),
            np.cos(planar_pose[2]),
        ],
        dtype=np.float32,
    )
    planar_velocity = v_to_rom(q, v).astype(np.float32, copy=False)
    planar_velocity[2] = v[5]
    gravity = gravity_from_quat(quat)
    linear_body = world_to_body(quat, v[:3])
    angular_body = v[3:6]
    remaining_base_velocity = np.array(
        [linear_body[2], angular_body[0], angular_body[1]], dtype=np.float32
    )
    gait = np.asarray(state.policy_clock, dtype=np.float32)
    if gait.shape != (3,) or not np.isfinite(gait).all() or gait[2] not in (37., 43.):
        raise ValueError("Running feedback requires actual gait [sin, cos, period_frames]")
    if not np.isclose(np.linalg.norm(gait[:2]), 1., atol=1e-5):
        raise ValueError("KCF requires the ungated unit gait clock before the outgoing command")
    return np.concatenate(
        [
            pose,
            planar_velocity,
            q[2:3],
            gravity,
            q[7:],
            remaining_base_velocity,
            v[6:],
            gait,
            state.prev_action.astype(np.float32),
        ]
    )


def g1_world_heading(state: G1State, previous: float | None = None) -> float:
    """Continuous measured heading for diagnostic metadata and frame anchoring."""
    heading = float(q_to_rom(state.q.astype(np.float32))[2])
    if previous is not None:
        heading = previous + np.arctan2(
            np.sin(heading - previous), np.cos(heading - previous)
        )
    return float(heading)

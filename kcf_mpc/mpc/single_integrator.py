"""Ideal body-twist comparison using the same native 85-state G1 schema."""

import numpy as np
import jax.numpy as jnp

def _period(initial, controls, sample_period):
    if initial.shape != (85,):
        raise ValueError(
            "single-integrator initial state must contain 85 native channels"
        )
    if controls.ndim != 2 or controls.shape[0] < 1 or controls.shape[1] != 3:
        raise ValueError("single-integrator controls must have shape (horizon, 3)")
    period = float(sample_period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("single-integrator sample_period must be finite and positive")
    return period



def single_integrator_rollout_jax(initial, controls, sample_period):

    dt = _period(initial, controls, sample_period)
    increment = dt * controls[:, 2]
    yaw = jnp.arctan2(initial[2], initial[3]) + jnp.cumsum(increment)
    midpoint = yaw - 0.5 * increment
    dx = dt * (jnp.cos(midpoint) * controls[:, 0] - jnp.sin(midpoint) * controls[:, 1])
    dy = dt * (jnp.sin(midpoint) * controls[:, 0] + jnp.cos(midpoint) * controls[:, 1])
    pose = jnp.stack(
        (
            initial[0] + jnp.cumsum(dx),
            initial[1] + jnp.cumsum(dy),
            jnp.sin(yaw),
            jnp.cos(yaw),
        ),
        axis=-1,
    )
    return jnp.concatenate(
        (pose, controls, jnp.broadcast_to(initial[7:], (len(controls), 78))), axis=-1
    )

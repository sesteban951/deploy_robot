"""Unicycle pose prediction in the native 104-state G1 logging schema.

Only forward speed and yaw rate control motion.
"""

import numpy as np
import jax.numpy as jnp

def _period(initial, controls, sample_period):
    if initial.shape != (104,):
        raise ValueError(
            "unicycle initial state must contain 104 native channels"
        )
    if controls.ndim != 2 or controls.shape[0] < 1 or controls.shape[1] != 3:
        raise ValueError("unicycle controls must have shape (horizon, 3)")
    period = float(sample_period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("unicycle sample_period must be finite and positive")
    return period



def unicycle_rollout_jax(initial, controls, sample_period):
    """Exact zero-order-hold integration of xdot=v*cos(yaw), ydot=v*sin(yaw)."""
    dt = _period(initial, controls, sample_period)
    increment = dt * controls[:, 2]
    yaw = jnp.arctan2(initial[2], initial[3]) + jnp.cumsum(increment)
    midpoint = yaw - 0.5 * increment
    distance = dt * controls[:, 0] * jnp.sinc(increment / (2 * jnp.pi))
    dx = distance * jnp.cos(midpoint)
    dy = distance * jnp.sin(midpoint)
    twist = jnp.stack((controls[:, 0], jnp.zeros_like(increment), controls[:, 2]), axis=-1)
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
        (pose, twist, jnp.broadcast_to(initial[7:], (len(controls), 97))), axis=-1
    )

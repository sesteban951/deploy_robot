"""Physical tracking features and the model-independent obstacle objective."""
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
import math
from numbers import Real

import jax.numpy as jnp

TRACKING_LABELS = (
    "px",
    "py",
    "heading_sin",
    "heading_cos",
    "vx_body",
    "vy_body",
    "wz_body",
    "gravity_body_x",
    "gravity_body_y",
    "gravity_body_z",
)



def tracking_features_jax(states):

    return jnp.concatenate(
        (states[..., :7], states[..., 8:11]), axis=-1
    )


@dataclass(frozen=True)
class InverseDistanceCost:
    """Radius-gated center-distance log penalty; alpha=0 disables it."""

    alpha: float = 1.0
    rho: float = 2.0
    radius: float = 2.0

    def __post_init__(self):
        for name in ('alpha', 'rho', 'radius'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'Obstacle cost {name} must be finite and non-negative')

    def __call__(self, robot_xy, obstacle_xy):
        if self.alpha == 0:
            # Skip the singular distance calculation, including coincident centers.
            shape = jnp.broadcast_shapes(jnp.shape(robot_xy), jnp.shape(obstacle_xy))
            return jnp.zeros(shape[:-2], dtype=jnp.result_type(robot_xy, obstacle_xy, 1.0))

        distance = jnp.linalg.norm(robot_xy - obstacle_xy, axis=-1)
        # Keep coincident centers finite, including while JAX traces both branches.
        safe_distance = jnp.maximum(distance, jnp.finfo(distance.dtype).eps)
        penalty = jnp.where(distance < self.radius,
                            -self.alpha * jnp.log(safe_distance), 0.0)
        return jnp.sum(penalty, axis=-1)


def build_obstacle_cost(settings=None):
    """Build once, omitting zero or trajectory-independent obstacle terms."""
    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise ValueError('obstacle_cost must be a configuration mapping')
    if set(settings) - {'enabled', 'factory', 'parameters'}:
        raise ValueError('Unknown obstacle_cost configuration key')
    enabled = settings.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('obstacle_cost.enabled must be boolean')
    if not enabled:
        return None
    factory = settings.get('factory', 'kcf_mpc.mpc.tracking:InverseDistanceCost')
    if not isinstance(factory, str) or factory.count(':') != 1:
        raise ValueError('obstacle_cost.factory must be module:callable')
    parameters = settings.get('parameters', {})
    if not isinstance(parameters, Mapping):
        raise ValueError('obstacle_cost.parameters must be a mapping')
    module, symbol = factory.split(':')
    penalty = getattr(import_module(module), symbol)(**parameters)
    if not callable(penalty):
        raise ValueError('obstacle_cost.factory must return a callable penalty')
    if isinstance(penalty, InverseDistanceCost) and penalty.alpha == 0:
        return None
    return penalty

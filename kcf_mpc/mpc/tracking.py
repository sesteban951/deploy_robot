"""Ten physical tracking features, retaining the native heading sine/cosine."""
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

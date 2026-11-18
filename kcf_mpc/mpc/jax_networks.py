"""Export checkpoint MLP layers and apply them with matching JAX arithmetic."""

from torch import nn
import jax
import jax.numpy as jnp


def _export_mlp(network, array, *, output_size=None):
    """Export the actual layer order, parameters, and normalization epsilon."""
    if network is None:
        return (), ()
    specifications, parameters = [], []
    for layer in network.network:
        if isinstance(layer, nn.Linear):
            specifications.append(("linear", None))
            weight, bias = layer.weight, layer.bias
            if output_size is not None and layer is network.output_layer:
                if not 0 < output_size <= layer.out_features:
                    raise ValueError("output_size must select existing output rows")
                weight, bias = weight[:output_size], bias[:output_size]
            parameters.append((array(weight.T), array(bias)))
        elif isinstance(layer, nn.LayerNorm):
            if len(layer.normalized_shape) != 1 or not layer.elementwise_affine:
                raise ValueError("JAX MPC supports affine, last-axis LayerNorm")
            specifications.append(("layer_norm", float(layer.eps)))
            parameters.append((array(layer.weight), array(layer.bias)))
        elif isinstance(layer, nn.ELU):
            specifications.append(("elu", float(layer.alpha)))
            parameters.append(())
        elif isinstance(layer, (nn.ReLU, nn.SiLU, nn.Tanh)):
            specifications.append((type(layer).__name__.lower(), None))
            parameters.append(())
        else:
            raise ValueError(f"Unsupported JAX MPC layer: {type(layer).__name__}")
    return tuple(specifications), tuple(parameters)


def _layer_norm(values, weight, bias, epsilon):
    centered = values - jnp.mean(values, axis=-1, keepdims=True)
    inverse = jax.lax.rsqrt(jnp.mean(centered**2, axis=-1, keepdims=True) + epsilon)
    normalized = centered * inverse
    return normalized * weight + bias


def _apply_mlp(values, parameters, specifications):
    # The layer loop is static at tracing time; only arrays enter compiled code.
    for (kind, option), params in zip(specifications, parameters):
        if kind == "linear":
            weight, bias = params
            values = (
                jnp.matmul(values, weight, precision=jax.lax.Precision.HIGHEST) + bias
            )
        elif kind == "layer_norm":
            weight, bias = params
            values = _layer_norm(values, weight, bias, option)
        elif kind == "elu":
            values = jax.nn.elu(values, alpha=option)
        elif kind == "relu":
            values = jax.nn.relu(values)
        elif kind == "silu":
            values = jax.nn.silu(values)
        elif kind == "tanh":
            values = jnp.tanh(values)
    return values

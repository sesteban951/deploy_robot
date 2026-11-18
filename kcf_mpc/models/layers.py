from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        choices = ", ".join(sorted(activations))
        raise ValueError(
            f"unsupported activation {name!r}; choose from {choices}"
        ) from error


class MLP(nn.Module):
    """A configurable feed-forward network with a linear output layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "elu",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if any(width < 1 for width in hidden_dims):
            raise ValueError("all hidden dimensions must be positive")

        layers: list[nn.Module] = []
        current_dim = input_dim

        for width in hidden_dims:
            layers.append(nn.Linear(current_dim, width))
            if layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(_activation(activation))
            current_dim = width

        self.output_layer = nn.Linear(current_dim, output_dim)
        layers.append(self.output_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

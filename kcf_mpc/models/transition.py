from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from .layers import MLP

class ControlTransition(nn.Module):
    """Map each control input to A(u)."""

    def __init__(
        self,
        input_dim: int,
        nH: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "elu",
        layer_norm: bool = True,
        initialize_as_identity: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or nH < 1:
            raise ValueError("input_dim and nH must be positive")

        self.input_dim = int(input_dim)
        self.nH = int(nH)
        self.matrix_network = MLP(
            input_dim=self.input_dim,
            output_dim=self.nH * self.nH,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )
        if initialize_as_identity:
            nn.init.zeros_(self.matrix_network.output_layer.weight)
            with torch.no_grad():
                identity = torch.eye(self.nH).reshape(-1)
                self.matrix_network.output_layer.bias.copy_(identity)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        if control.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input dimension {self.input_dim}, "
                f"got {control.shape[-1]}"
            )
        flat_matrix = self.matrix_network(control)
        return flat_matrix.reshape(
            *control.shape[:-1],
            self.nH,
            self.nH,
        )

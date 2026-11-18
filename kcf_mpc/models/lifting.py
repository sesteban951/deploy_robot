from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from .layers import MLP


class StateInclusiveLifting(nn.Module):
    """Lift a causal observation as ``H(eta) = [x_k, h(eta)]``."""

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        nH: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "elu",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if observation_dim < state_dim:
            raise ValueError("observation_dim must be at least state_dim")
        if nH < state_dim:
            raise ValueError("nH must be at least state_dim")

        self.state_dim = int(state_dim)
        self.observation_dim = int(observation_dim)
        self.nH = int(nH)
        learned_dim = self.nH - self.state_dim
        self.learned_features: nn.Module | None
        if learned_dim == 0:
            self.learned_features = None
        else:
            self.learned_features = MLP(
                input_dim=self.observation_dim,
                output_dim=learned_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                layer_norm=layer_norm,
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected delay-observation dimension {self.observation_dim}, "
                f"got {observation.shape[-1]}"
            )
        current_state = observation[..., : self.state_dim]
        if self.learned_features is None:
            return current_state
        return torch.cat(
            (current_state, self.learned_features(observation)), dim=-1
        )

    def state(self, lifted_state: torch.Tensor) -> torch.Tensor:
        """Recover the modeled state stored at the start of the lift."""
        if lifted_state.shape[-1] != self.nH:
            raise ValueError(
                f"expected H dimension {self.nH}, "
                f"got {lifted_state.shape[-1]}"
            )
        return lifted_state[..., : self.state_dim]

    def physical_state(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return self.state(lifted_state)

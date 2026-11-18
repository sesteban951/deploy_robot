from __future__ import annotations
import torch
from torch import nn
from kcf_mpc.config.model_config import ModelConfig
from .lifting import StateInclusiveLifting
from .normalization import FixedAffineNormalizer
from .transition import ControlTransition


class ControlledLiftedModel(nn.Module):
    """Learn delay-informed H(eta) and A(u) and compose them in open loop."""

    def __init__(self, config: ModelConfig, normalization=None) -> None:
        super().__init__()
        self.config = config
        self.normalizer = FixedAffineNormalizer(
            config.state_dim, config.input_dim, normalization
        )
        self.lifting_network = StateInclusiveLifting(
            state_dim=config.state_dim,
            observation_dim=config.lifting_input_dim,
            nH=config.nH,
            hidden_dims=config.lifting_hidden_dims,
            activation=config.activation,
            layer_norm=config.layer_norm,
        )
        self.transition_network = ControlTransition(
            input_dim=config.input_dim + int(config.command_mode_feature),
            nH=config.nH,
            hidden_dims=config.transition_hidden_dims,
            activation=config.activation,
            layer_norm=config.layer_norm,
            initialize_as_identity=(
                config.initialize_transition_as_identity
            ),
        )

    def lift(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize a physical-unit observation and compute its lift."""
        return self.lifting_network(self.normalize_observation(observation))

    def transition(self, control: torch.Tensor) -> torch.Tensor:
        """Compute A(u), including the checkpoint's derived controller mode."""
        features = self.normalizer.normalize_control(control)
        if self.config.command_mode_feature:
            moving = (torch.norm(control, dim=-1, keepdim=True)
                      >= self.config.stand_cmd_threshold).to(features.dtype)
            features = torch.cat((features, moving), dim=-1)
        return self.transition_network(features)

    def normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize every state/control block in a causal delay observation."""
        if observation.shape[-1] != self.config.lifting_input_dim:
            raise ValueError(
                f"expected delay-observation dimension "
                f"{self.config.lifting_input_dim}, got {observation.shape[-1]}"
            )
        state_width = self.config.state_dim * (self.config.L_embed + 1)
        state_history = observation[..., :state_width].reshape(
            *observation.shape[:-1], self.config.L_embed + 1,
            self.config.state_dim,
        )
        normalized_states = self.normalizer.normalize_state(state_history).reshape(
            *observation.shape[:-1], state_width
        )
        if self.config.L_embed == 0:
            return normalized_states
        control_history = observation[..., state_width:].reshape(
            *observation.shape[:-1], self.config.L_embed, self.config.input_dim
        )
        normalized_controls = self.normalizer.normalize_control(
            control_history
        ).reshape(*observation.shape[:-1], -1)
        return torch.cat((normalized_states, normalized_controls), dim=-1)

    def step_lifted(
        self,
        lifted_state: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one lifted step A(u) H"""
        if lifted_state.shape[-1] != self.config.nH:
            raise ValueError(
                f"expected H dimension {self.config.nH}, "
                f"got {lifted_state.shape[-1]}"
            )
        matrix = self.transition(control)
        if matrix.shape[:-2] != lifted_state.shape[:-1]:
            raise ValueError(
                "control and lifted state must have matching batch shapes"
            )
        return torch.matmul(matrix, lifted_state.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        observation: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        """Lift a causal observation and make one lifted-state prediction."""
        return self.step_lifted(self.lift(observation), control)

    def rollout(
        self,
        initial_observation: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Generate an open-loop lifted rollout.

        Args:
            initial_observation: Tensor shaped ``(batch, lifting_input_dim)``.
            controls: Tensor shaped ``(batch, horizon, input_dim)``.

        Returns:
            Predicted lifted states for steps 1 through ``horizon``, shaped
            ``(batch, horizon, nH)``. Measured intermediate states are
            never inserted into the rollout.
        """
        if initial_observation.ndim != 2:
            raise ValueError(
                "initial_observation must have shape (batch, lifting_input_dim)"
            )
        if controls.ndim != 3:
            raise ValueError(
                "controls must have shape (batch, horizon, input_dim)"
            )
        if initial_observation.shape[0] != controls.shape[0]:
            raise ValueError(
                "initial_observation and controls batch sizes must match"
            )
        if controls.shape[1] < 1:
            raise ValueError("rollout horizon must be at least one")

        # Evaluate the control network for the complete sequence in one
        # batched call. The state propagation is recurrent and
        # preserves exactly A(u_j) ... A(u_0) H(x_0).
        transition_matrices = self.transition(controls)
        lifted_state = self.lift(initial_observation)
        predictions = []
        for step in range(controls.shape[1]):
            lifted_state = torch.matmul(
                transition_matrices[:, step],
                lifted_state.unsqueeze(-1),
            ).squeeze(-1)
            predictions.append(lifted_state)
        return torch.stack(predictions, dim=1)

    def state(self, lifted_state: torch.Tensor) -> torch.Tensor:
        """Recover a modeled state in physical units through the fixed inverse."""
        return self.normalizer.denormalize_state(
            self.state_normalized(lifted_state)
        )

    def state_normalized(self, lifted_state: torch.Tensor) -> torch.Tensor:
        """Recover the normalized state stored in the lifted representation."""
        return self.lifting_network.state(lifted_state)

    def physical_state(self, lifted_state: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for :meth:`state`."""
        return self.state(lifted_state)

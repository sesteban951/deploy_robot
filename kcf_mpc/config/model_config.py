from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:

    state_dim: int
    input_dim: int
    nH: int
    L_embed: int = 0
    lifting_hidden_dims: tuple[int, ...] = (128, 128)
    transition_hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "elu"
    layer_norm: bool = True
    initialize_transition_as_identity: bool = True
    command_mode_feature: bool = False
    stand_cmd_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.input_dim < 1:
            raise ValueError("state_dim and input_dim must be positive")
        if self.nH < self.state_dim:
            raise ValueError("nH must be at least state_dim")
        if self.L_embed < 0:
            raise ValueError("L_embed must be non-negative")
        if any(width < 1 for width in self.lifting_hidden_dims):
            raise ValueError("lifting hidden dimensions must be positive")
        if any(width < 1 for width in self.transition_hidden_dims):
            raise ValueError("transition hidden dimensions must be positive")
        if self.command_mode_feature:
            if self.state_dim != 104 or self.input_dim != 3 or self.L_embed != 0:
                raise ValueError("command mode feature requires the 104-state G1 model with three commands and no delay")
            if self.stand_cmd_threshold != 0.0:
                raise ValueError("command mode feature uses the recorded controller threshold 0.0")

    @property
    def lifting_input_dim(self) -> int:
        """Dimension of ``[x_k..x_k-L, u_k-1..u_k-L]``."""
        return self.state_dim * (self.L_embed + 1) + self.input_dim * self.L_embed


def model_config_from_mapping(config: Mapping[str, Any]) -> ModelConfig:
    state_dim = config.get("state_dim")
    input_dim = config.get("input_dim")
    if state_dim is None or input_dim is None:
        raise ValueError(
            "state_dim and input_dim must be supplied by the dataset or config "
        )

    hidden_dim = int(config["hidden_dim"])
    num_hidden_layers = int(config["num_hidden_layers"])
    hidden_dims = (hidden_dim,) * num_hidden_layers
    return ModelConfig(
        state_dim=int(state_dim),
        input_dim=int(input_dim),
        nH=int(config["nH"]),
        L_embed=int(config.get("L_embed", 0)),
        lifting_hidden_dims=hidden_dims,
        transition_hidden_dims=hidden_dims,
        activation=str(config["activation"]),
        layer_norm=bool(config["layer_norm"]),
        initialize_transition_as_identity=bool(
            config["initialize_transition_as_identity"]
        ),
        command_mode_feature=bool(config.get("command_mode_feature", False)),
        stand_cmd_threshold=float(config.get("stand_cmd_threshold", 0.0)),
    )

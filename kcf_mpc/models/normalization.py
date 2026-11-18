from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


class FixedAffineNormalizer(nn.Module):
    """Checkpoint-persistent, non-trainable state and control standardization."""

    _KEYS = ("state_mean", "state_scale", "input_mean", "input_scale")

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        statistics: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        if statistics is not None:
            self.set_statistics(statistics)

    @torch.no_grad()
    def set_statistics(self, statistics: Mapping[str, np.ndarray]) -> None:
        missing = sorted(set(self._KEYS) - set(statistics))
        if missing:
            raise KeyError(f"normalization is missing keys: {missing}")
        for key in self._KEYS:
            destination = getattr(self, key)
            source = torch.as_tensor(
                statistics[key], dtype=destination.dtype, device=destination.device
            )
            if source.shape != destination.shape:
                raise ValueError(
                    f"{key} has shape {tuple(source.shape)}, expected "
                    f"{tuple(destination.shape)}"
                )
            if not torch.isfinite(source).all():
                raise ValueError(f"{key} contains NaN or infinite values")
            if key.endswith("_scale") and torch.any(source <= 0.0):
                raise ValueError(f"{key} must be strictly positive")
            destination.copy_(source)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_scale

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.state_scale + self.state_mean

    def normalize_control(self, control: torch.Tensor) -> torch.Tensor:
        return (control - self.input_mean) / self.input_scale

    def denormalize_control(self, control: torch.Tensor) -> torch.Tensor:
        return control * self.input_scale + self.input_mean

    def numpy_statistics(self) -> dict[str, np.ndarray]:
        return {
            key: getattr(self, key).detach().cpu().numpy().astype(np.float64)
            for key in self._KEYS
        }

    def assert_matches(
        self,
        statistics: Mapping[str, np.ndarray],
        *,
        rtol: float = 1e-6,
        atol: float = 1e-7,
    ) -> None:
        """Reject evaluation data standardized with a different affine map."""
        embedded = self.numpy_statistics()
        for key in self._KEYS:
            candidate = np.asarray(statistics[key], dtype=np.float64)
            if candidate.shape != embedded[key].shape or not np.allclose(
                candidate, embedded[key], rtol=rtol, atol=atol
            ):
                raise ValueError(
                    "processed-data normalization does not match the fixed "
                    f"checkpoint normalization ({key})"
                )

"""MPC using either the trained KCF predictor or an ideal kinematic baseline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kcf_mpc.config import get_default_mpc_config
from kcf_mpc.checkpoint import load_saved_model
from kcf_mpc.device import resolve_device
from kcf_mpc.state.frame import G1KCFSinCosFrame
from kcf_mpc.models.projected import ProjectedG1KCF, is_projected_g1
from kcf_mpc.mpc.tracking import TRACKING_LABELS
from kcf_mpc.mpc.icem import ICEMConfig, ICEMResult


class KoopmanMPC:
    """MPC that optimizes controls through the selected prediction model."""

    def __init__(
        self,
        model: ProjectedG1KCF,
        config: Mapping[str, Any],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        if "mode_aware" in config:
            raise ValueError(
                "The mode_aware prototype has been removed. Remove this setting "
                "and use the supported bounded iCEM MPC formulation."
            )
        if not is_projected_g1(model):
            raise ValueError("MPC requires the native projected 85-state G1 checkpoint")
        self.model = model.eval().requires_grad_(False)
        self.device = model.normalizer.state_mean.device
        self.dtype = model.normalizer.state_mean.dtype
        self.input_dim = model.config.input_dim
        self.projected_g1 = True
        self.observation_dim = model.config.lifting_input_dim
        self.prediction_state_dim = model.config.state_dim
        self.cost_state_dim = len(TRACKING_LABELS)
        self.horizon = int(config["horizon"])

        if self.horizon < 1:
            raise ValueError("horizon must be positive")

        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        expected_input_shape = (self.input_dim,)
        if (
            self.lower_bounds.shape != expected_input_shape
            or self.upper_bounds.shape != expected_input_shape
        ):
            raise ValueError("input bounds must have shape " f"{expected_input_shape}")
        if not (
            np.isfinite(self.lower_bounds).all()
            and np.isfinite(self.upper_bounds).all()
        ):
            raise ValueError("input bounds must be finite")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("input lower bounds cannot exceed upper bounds")
        self.numpy_dtype = np.float64 if self.dtype == torch.float64 else np.float32

        indices = np.asarray(
            config["tracking_feature_indices"],
            dtype=np.int64,
        )
        if (
            indices.ndim != 1
            or indices.size < 1
            or len(set(indices.tolist())) != indices.size
            or indices.min() < 0
            or indices.max() >= self.cost_state_dim
        ):
            raise ValueError("tracking_feature_indices are invalid")

        weights = np.asarray(
            config["state_weights"],
            dtype=np.float64,
        )
        if (
            weights.shape != indices.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "state_weights must be finite, non-negative, and match "
                "tracking_feature_indices"
            )

        self.tracking_feature_indices = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.device,
        )
        self.state_weights = torch.as_tensor(
            weights,
            dtype=self.dtype,
            device=self.device,
        )
        terminal_weights = np.asarray(
            config.get("terminal_state_weights", np.zeros_like(weights)),
            dtype=np.float64,
        )
        if (
            terminal_weights.shape != indices.shape
            or not np.isfinite(terminal_weights).all()
            or np.any(terminal_weights < 0.0)
        ):
            raise ValueError(
                "terminal_state_weights must be finite, non-negative, and match "
                "tracking_feature_indices"
            )
        self.has_terminal_cost = bool(np.any(terminal_weights))
        self.terminal_state_weights = torch.as_tensor(
            terminal_weights,
            dtype=self.dtype,
            device=self.device,
        )
        self.input_weight = float(config["input_weight"])
        if not np.isfinite(self.input_weight) or self.input_weight < 0.0:
            raise ValueError("input_weight must be finite and non-negative")
        change_weights = np.asarray(
            config.get("command_change_weights", 0.0), dtype=np.float64
        )
        if change_weights.ndim == 0:
            change_weights = np.full(self.input_dim, float(change_weights))
        if (
            change_weights.shape != expected_input_shape
            or not np.isfinite(change_weights).all()
            or np.any(change_weights < 0.0)
        ):
            raise ValueError(
                "command_change_weights must be finite, non-negative, and match input_dim"
            )
        self.has_command_change_cost = bool(np.any(change_weights))
        self.command_change_weights = torch.as_tensor(
            change_weights, dtype=self.dtype, device=self.device
        )

        self.solver = str(config.get("solver", "icem"))
        if self.solver != "icem":
            raise ValueError("Only the iCEM optimizer is supported")
        self.backend = str(config.get("backend", "jax"))
        if self.backend != "jax":
            raise ValueError("iCEM requires the JAX forward-only population backend")
        self.rng = np.random.default_rng(int(config["seed"]))
        self.previous_solution: np.ndarray | None = None
        self.icem_config = ICEMConfig(**config.get("icem", {}))
        self.jax_platform = str(config.get("jax", {}).get("platform", "cpu"))
        self.jax_preallocate = bool(config.get("jax", {}).get("preallocate", False))
        self.jax_pinned_host_output = bool(
            config.get("jax", {}).get("pinned_host_output", False)
        )
        self.jax_compilation_cache_dir = config.get("jax", {}).get(
            "compilation_cache_dir"
        )
        self.jax_compiler_options = dict(
            config.get("jax", {}).get("compiler_options", {})
        )
        self.jax_scan_unroll = int(config.get("jax", {}).get("scan_unroll", 1))
        self.prediction_model = str(config.get("prediction_model", "kcf"))
        if self.prediction_model not in ("kcf", "single_integrator"):
            raise ValueError("prediction_model must be 'kcf' or 'single_integrator'")
        if self.prediction_model == "single_integrator" and self.input_dim != 3:
            raise ValueError(
                "single-integrator prediction requires three body-twist inputs"
            )
        if "pose_prediction" in config:
            raise ValueError(
                "pose_prediction has been removed; KCF predicts the full state directly. "
                "Use prediction_sample_period for the single-integrator timestep."
            )
        self.prediction_sample_period = float(
            config.get("prediction_sample_period", 0.02)
        )
        if (
            not np.isfinite(self.prediction_sample_period)
            or self.prediction_sample_period <= 0.0
        ):
            raise ValueError("prediction_sample_period must be finite and positive")
        if not np.isclose(
            model.dt, self.prediction_sample_period, rtol=0.0, atol=1e-12
        ):
            raise ValueError("MPC timestep must match the projected checkpoint")
        tracked = self.tracking_feature_indices.detach().cpu().tolist()
        if not {0, 1, 2, 3}.issubset(tracked):
            raise ValueError(
                "Fixed-frame tracking requires px, py and heading sine/cosine features"
            )
        self._tracked_pose_columns = tuple(
            tracked.index(index) for index in range(4)
        )
        x_column, y_column, sin_column, cos_column = self._tracked_pose_columns
        if (
            weights[x_column] != weights[y_column]
            or terminal_weights[x_column] != terminal_weights[y_column]
        ):
            raise ValueError(
                "Fixed-frame tracking requires equal x/y stage and terminal weights"
            )
        if (
            weights[sin_column] != weights[cos_column]
            or terminal_weights[sin_column] != terminal_weights[cos_column]
        ):
            raise ValueError(
                "Fixed-frame tracking requires equal sine/cosine stage and terminal weights"
            )
        self._jax_backend = None

    def observation_to_prediction_frame(
        self,
        observation: np.ndarray,
        *,
        heading_world: float | None = None,
    ) -> tuple[np.ndarray, G1KCFSinCosFrame]:
        """Canonicalize one MPC initial state and return its fixed frame.

        The returned frame is immutable and is reused for the complete rollout.
        The native fixed frame is mandatory for every supported solve.
        """
        state = np.asarray(observation)
        if state.shape != (self.observation_dim,):
            raise ValueError(
                "observation must match the MPC feedback-interface dimension"
            )
        if not np.isfinite(state).all():
            raise ValueError("observation contains NaN or infinity")
        state = np.array(state, copy=True)
        frame = G1KCFSinCosFrame.from_state(state, heading_world=heading_world)
        return frame.state_to_local(state), frame

    def reference_to_prediction_frame(
        self,
        reference: np.ndarray,
        frame: G1KCFSinCosFrame,
    ) -> np.ndarray:
        """Express all horizon reference poses in the rollout's fixed frame."""
        target = np.asarray(reference)
        expected = (self.horizon, self.tracking_feature_indices.numel())
        if target.shape != expected:
            raise ValueError(f"reference must have shape {expected}")
        if not np.isfinite(target).all():
            raise ValueError("reference contains NaN or infinity")
        result = np.array(target, copy=True)
        pose = np.stack(
            [result[:, column] for column in self._tracked_pose_columns],
            axis=-1,
        )
        local_pose = frame.pose_to_local(pose)
        for coordinate, column in enumerate(self._tracked_pose_columns):
            result[:, column] = local_pose[:, coordinate]
        return result

    def prediction_from_prediction_frame(
        self,
        predicted_states: np.ndarray,
        frame: G1KCFSinCosFrame,
    ) -> np.ndarray:
        """Return predictions to world coordinates in the model's state schema."""
        states = np.asarray(predicted_states)
        if states.ndim == 0 or states.shape[-1] != self.prediction_state_dim:
            raise ValueError("predicted_states must end with the model state dimension")
        if not np.isfinite(states).all():
            raise ValueError("predicted_states contains NaN or infinity")
        result = np.array(states, copy=True)
        return frame.state_to_world(result)

    def _get_jax_backend(self):
        """Initialize lazily, after checkpoint precision has been configured."""
        if self._jax_backend is None:
            from kcf_mpc.mpc.projected_jax import ProjectedJaxMPCBackend

            self._jax_backend = ProjectedJaxMPCBackend(
                self.model,
                self.horizon,
                self.tracking_feature_indices.cpu().numpy(),
                self.state_weights.detach().cpu().numpy(),
                self.input_weight,
                terminal_state_weights=self.terminal_state_weights.detach()
                .cpu()
                .numpy(),
                command_change_weights=self.command_change_weights.detach()
                .cpu()
                .numpy(),
                platform=self.jax_platform,
                preallocate=self.jax_preallocate,
                pinned_host_output=self.jax_pinned_host_output,
                compilation_cache_dir=self.jax_compilation_cache_dir,
                compiler_options=self.jax_compiler_options,
                scan_unroll=self.jax_scan_unroll,
                prediction_sample_period=self.prediction_sample_period,
                prediction_model=self.prediction_model,
            )
        return self._jax_backend

    def reset(self) -> None:
        """Discard the previous plan and iCEM sampler memory, without reseeding."""
        self.previous_solution = None
        if self._jax_backend is not None:
            self._jax_backend.reset_optimizer()

    def optimizer_state(self):
        """Host-only sampler snapshot, without changing RNG or launching GPU work."""
        if self._jax_backend is None:
            return None
        return self._jax_backend.optimizer_state()

    def restore_optimizer_state(self, state):
        self._get_jax_backend().restore_optimizer_state(state)

    def solve(
        self,
        current_observation: np.ndarray | torch.Tensor,
        reference: np.ndarray | torch.Tensor,
        *,
        preceding_command: np.ndarray | torch.Tensor | None = None,
        heading_world: float | None = None,
        warm_start_steps: int = 1,
    ) -> tuple[np.ndarray, ICEMResult]:
        """Optimize one horizon and return the complete input sequence."""
        def host_array(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=self.numpy_dtype)

        observation = host_array(current_observation)
        target = host_array(reference)
        expected_observation_shape = (self.observation_dim,)
        expected_reference_shape = (
            self.horizon, self.tracking_feature_indices.numel(),
        )
        if observation.shape != expected_observation_shape:
            raise ValueError(f"current_observation must have shape {expected_observation_shape}")
        if target.shape != expected_reference_shape:
            raise ValueError(f"reference must have shape {expected_reference_shape}")
        if not np.isfinite(observation).all():
            raise ValueError("current_observation contains NaN or infinity")
        if not np.isfinite(target).all():
            raise ValueError("reference contains NaN or infinity")
        observation, fixed_frame = self.observation_to_prediction_frame(
            observation, heading_world=heading_world
        )
        target = self.reference_to_prediction_frame(target, fixed_frame)
        if preceding_command is None:
            if self.has_command_change_cost or self.icem_config.command_rate_limits is not None:
                raise ValueError(
                    "preceding_command is required for command-change costs or rate limits"
                )
            preceding = np.zeros(self.input_dim, dtype=self.numpy_dtype)
        else:
            preceding = host_array(preceding_command).copy()
            if preceding.shape != (self.input_dim,) or not np.isfinite(preceding).all():
                raise ValueError("preceding_command must be a finite input_dim vector")

        controls, result = self._get_jax_backend().solve_icem(
            observation, target, preceding, self.rng, self.icem_config,
            self.lower_bounds, self.upper_bounds,
            previous_solution=self.previous_solution, warm_start_steps=warm_start_steps)
        self.previous_solution = controls.copy() if result.success else None
        return controls, result


def create_mpc_controller(
    config: Mapping[str, Any] | None = None,
) -> KoopmanMPC:
    """Load the experiment assets and construct the selected MPC controller."""
    mpc_config = get_default_mpc_config() if config is None else dict(config)
    if not mpc_config.get("results_dir"):
        raise ValueError("MPC requires an explicit results_dir; automatic model selection is unsupported")
    if not all(key in mpc_config for key in ("input_lower_bounds", "input_upper_bounds")):
        raise ValueError("MPC requires both explicit input_lower_bounds and input_upper_bounds")
    device = resolve_device(mpc_config["device"])
    results_dir = Path(mpc_config["results_dir"])
    model, _, _ = load_saved_model(
        results_dir,
        device,
        mpc_config["checkpoint"],
    )
    if not is_projected_g1(model):
        raise ValueError("G1 MPC requires the native 85-state projected KCF checkpoint")

    if "dtype" in mpc_config:
        dtypes = {"float32": torch.float32, "float64": torch.float64}
        if mpc_config["dtype"] not in dtypes:
            raise ValueError("MPC dtype must be 'float32' or 'float64'")
        model.to(dtype=dtypes[mpc_config["dtype"]])

    lower_bounds = np.asarray(mpc_config["input_lower_bounds"], dtype=np.float64)
    upper_bounds = np.asarray(mpc_config["input_upper_bounds"], dtype=np.float64)

    return KoopmanMPC(
        model,
        mpc_config,
        lower_bounds,
        upper_bounds,
    )

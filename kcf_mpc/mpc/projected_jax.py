"""Forward-only projected G1 population prediction and costs for iCEM.

GPU KCF uses the fixed cooperative CUDA forward kernels. CPU equations are a
numerical prediction reference, never an automatic GPU fallback. Both apply
the trained projection and re-lifting at each step with all 85 physical states.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from kcf_mpc.mpc.jax_networks import _apply_mlp, _export_mlp
from kcf_mpc.models.projected import is_projected_g1
from kcf_mpc.mpc.tracking import tracking_features_jax
from kcf_mpc.mpc.single_integrator import single_integrator_rollout_jax


def _unit(value, fallback):
    squared = jnp.sum(jnp.square(value), axis=-1, keepdims=True)
    denominator = jnp.sqrt(jnp.maximum(squared, 1e-16))
    return jnp.where(
        squared > 1e-16, value / denominator, jnp.asarray(fallback, dtype=value.dtype)
    )


def _rotation(state):
    sy, cy = _unit(state[2:4], (0.0, 1.0))
    gravity = _unit(state[8:11], (0.0, 0.0, -1.0))
    sp = jnp.clip(gravity[0], -1.0 + 1e-7, 1.0 - 1e-7)
    cp = jnp.sqrt(1.0 - sp * sp)
    sr, cr = _unit(-gravity[1:3], (0.0, 1.0))
    return jnp.stack(
        (
            jnp.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr)),
            jnp.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr)),
            jnp.stack((-sp, cp * sr, cp * cr)),
        )
    )


def _solve_spd3(matrix, rhs):
    """Exact 3x3 SPD solve using scalar LDLᵀ, without a library-call launch.

    The projection matrix is SPD because its velocity weights are positive.
    No regularization, clipping or approximate inverse is introduced.
    """
    d0 = matrix[0, 0]
    l10, l20 = matrix[1, 0] / d0, matrix[2, 0] / d0
    d1 = matrix[1, 1] - l10 * matrix[1, 0]
    residual = matrix[2, 1] - l20 * matrix[1, 0]
    l21 = residual / d1
    d2 = matrix[2, 2] - l20 * matrix[2, 0] - l21 * residual
    y0 = rhs[0]
    y1 = rhs[1] - l10 * y0
    y2 = rhs[2] - l20 * y0 - l21 * y1
    x2 = y2 / d2
    x1 = y1 / d1 - l21 * x2
    return jnp.stack((y0 / d0 - l10 * x1 - l20 * x2, x1, x2))


def project_state_jax(previous, raw, scale, *, dt, balance):
    """The unchanged weighted trapezoidal projection from models/kinematics.py."""
    raw = jnp.concatenate(
        (
            raw[:2],
            _unit(raw[2:4], (0.0, 1.0)),
            raw[4:8],
            _unit(raw[8:11], (0.0, 0.0, -1.0)),
            raw[11:],
        )
    )
    position, velocity = jnp.array([0, 1, 7]), jnp.array([4, 5, 34])
    a = 0.5 * dt * _rotation(raw)
    b = previous[position] + 0.5 * dt * (_rotation(previous) @ previous[velocity])
    wp = balance / jnp.maximum(scale[position], 1e-4) ** 2
    wv = 1.0 / jnp.maximum(scale[velocity], 1e-4) ** 2
    matrix = a.T @ (wp[:, None] * a) + jnp.diag(wv)
    rhs = wv * raw[velocity] + a.T @ (wp * (raw[position] - b))
    corrected_velocity = _solve_spd3(matrix, rhs)
    corrected_position = b + a @ corrected_velocity
    return jnp.concatenate(
        (
            corrected_position[:2],
            raw[2:4],
            corrected_velocity[:2],
            raw[6:7],
            corrected_position[2:3],
            raw[8:34],
            corrected_velocity[2:3],
            raw[35:],
        )
    )


class ProjectedJaxMPCBackend:
    def __init__(
        self,
        model,
        horizon,
        tracking_feature_indices,
        state_weights,
        input_weight,
        *,
        platform="cpu",
        preallocate=False,
        compiler_options=None,
        scan_unroll=1,
        pinned_host_output=False,
        compilation_cache_dir=None,
        command_change_weights=None,
        terminal_state_weights=None,
        prediction_sample_period=0.02,
        prediction_model="kcf"
    ):
        if not is_projected_g1(model) or prediction_model not in (
            "kcf",
            "single_integrator",
        ):
            raise ValueError("Projected backend requires the projected 85-state G1 KCF")
        if not np.isclose(model.dt, prediction_sample_period, rtol=0.0, atol=1e-12):
            raise ValueError("MPC timestep must match the projected checkpoint")
        self.horizon, self.input_dim, self.observation_dim = int(horizon), 3, 85
        self.tracking_feature_indices = tuple(int(i) for i in tracking_feature_indices)
        self.model_dt = float(model.dt)
        self.tracked_dim = len(tracking_feature_indices)
        if self.horizon < 1 or int(scan_unroll) < 1:
            raise ValueError("JAX horizon and scan_unroll must be positive")
        self.dtype = np.dtype(
            "float64"
            if model.normalizer.state_mean.dtype == torch.float64
            else "float32"
        )
        jax.config.update("jax_enable_x64", True)
        if platform in ("gpu", "cuda"):
            os.environ.setdefault(
                "XLA_PYTHON_CLIENT_PREALLOCATE", str(bool(preallocate)).lower()
            )
        self.device = jax.devices(platform)[0]
        self.pinned_host_output = bool(pinned_host_output)
        if self.pinned_host_output and self.device.platform != "gpu":
            raise ValueError("JAX pinned_host_output requires a GPU")
        self.compilation_cache_dir = None
        if compilation_cache_dir is not None:
            cache = (
                Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", compilation_cache_dir))
                .expanduser()
                .resolve()
            )
            cache.mkdir(mode=0o700, parents=True, exist_ok=True)
            if cache.stat().st_mode & 0o022:
                raise ValueError(
                    "JAX compilation cache must not be group/world-writable"
                )
            self.compilation_cache_dir = str(cache)
            jax.config.update("jax_compilation_cache_dir", str(cache))
        self.compiler_options = dict(compiler_options or {})
        self.scan_unroll = min(int(scan_unroll), self.horizon)
        if self.device.platform == "cpu":
            self.scan_unroll = self.horizon

        def array(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return jax.device_put(
                np.array(value, dtype=self.dtype, copy=True), self.device
            )

        self._array = array
        ls, lp = _export_mlp(model.lifting_network.learned_features, array)
        # Only physical rows survive projection. The remaining lifted rows are
        # discarded before re-lifting, so exporting them cannot affect dynamics.
        ts, tp = _export_mlp(
            model.transition_network.matrix_network, array,
            output_size=85 * model.config.nH,
        )
        mean, scale, umean, uscale = [
            array(getattr(model.normalizer, key))
            for key in ("state_mean", "state_scale", "input_mean", "input_scale")
        ]
        self.parameters = (lp, tp, (mean, scale, umean, uscale))
        self._icem_optimizer = None
        self._icem_state = None
        self._icem_host_state = None
        self._native_rollout_kernel = None
        if self.device.platform == "gpu" and prediction_model == "kcf":
            if self.dtype != np.dtype("float32") or self.horizon > 20 or model.config.nH != 256:
                raise ValueError("Native GPU KCF requires FP32, lift256 and horizon <= 20")
            from .native.runtime import NativeRollout
            self._native_rollout_kernel = NativeRollout(model, array, self.device)
        self.execution_implementation = (
            "single_integrator" if prediction_model == "single_integrator"
            else "native32" if self._native_rollout_kernel is not None
            else "jax_cpu_reference"
        )
        indices = jnp.asarray(tracking_feature_indices, dtype=jnp.int32)
        weights = array(state_weights)
        terminal = array(
            np.zeros(self.tracked_dim)
            if terminal_state_weights is None
            else terminal_state_weights
        )
        change = np.asarray(
            0.0 if command_change_weights is None else command_change_weights
        )
        if change.ndim == 0:
            change = np.full(3, float(change))
        self.has_command_change_cost = bool(np.any(change))
        change = array(change)
        mode_feature = model.config.command_mode_feature
        threshold, dt, balance, nH = (
            model.config.stand_cmd_threshold,
            model.dt,
            model.projection_balance,
            model.config.nH,
        )

        def lift(state, parameters):
            lift_parameters, _, (mean, scale, _, _) = parameters
            normalized = (state - mean) / scale
            if not ls:
                return normalized
            learned = _apply_mlp(normalized, lift_parameters, ls)
            return jnp.concatenate((normalized, learned))

        def transitions(controls, parameters):
            _, transition_parameters, (mean, scale, umean, uscale) = parameters
            features = ((controls - umean) / uscale).reshape(-1, 3)
            if mode_feature:
                # Same physical command threshold as the frozen Torch model;
                # the trained binary feature remains unchanged.
                moving = (
                    jnp.linalg.norm(controls.reshape(-1, 3), axis=-1, keepdims=True) >= threshold
                ).astype(features.dtype)
                features = jnp.concatenate((features, moving), axis=-1)
            return _apply_mlp(features, transition_parameters, ts).reshape(
                controls.shape[:-1] + (85, nH))

        def native_rollout(initial, controls, parameters):
            if prediction_model == "single_integrator":
                return single_integrator_rollout_jax(
                    initial, controls, prediction_sample_period
                )
            _, _, (mean, scale, _, _) = parameters
            matrices = transitions(controls, parameters)

            if self._native_rollout_kernel is not None:
                return self._native_rollout_kernel(initial, matrices)
            if self.device.platform != "cpu":
                raise RuntimeError("Refusing an unoptimized GPU KCF fallback")

            # CPU numerical reference only; GPU KCF never falls back to this scan.
            def step(previous, matrix):
                raw_lift = jnp.matmul(
                    matrix,
                    lift(previous, parameters),
                    precision=jax.lax.Precision.HIGHEST,
                )
                raw = raw_lift * scale + mean
                following = project_state_jax(
                    previous, raw, scale, dt=dt, balance=balance
                )
                return following, following

            return jax.lax.scan(
                step, initial, matrices,
                unroll=min(self.scan_unroll, controls.shape[0]),
            )[1]

        def rollout(initial, controls, parameters):
            return native_rollout(initial, controls, parameters)

        def population_rollout(initial, controls, parameters):
            if prediction_model == "single_integrator" or self.device.platform == "cpu":
                return jax.vmap(lambda u: native_rollout(initial, u, parameters))(controls)
            matrices = transitions(controls, parameters)
            initials = jnp.broadcast_to(initial, (controls.shape[0], 85))
            return self._native_rollout_kernel.batched(initials, matrices)

        def trajectory_cost(states, controls, reference, preceding):
            error = tracking_features_jax(states)[..., indices] - reference
            differences = jnp.concatenate(
                (controls[..., :1, :] - preceding,
                 controls[..., 1:, :] - controls[..., :-1, :]), axis=-2
            )
            return (
                jnp.sum(weights * error**2, axis=(-2, -1))
                + float(input_weight) * jnp.sum(controls**2, axis=(-2, -1))
                + jnp.sum(terminal * error[..., -1, :] ** 2, axis=-1)
                + jnp.sum(change * differences**2, axis=(-2, -1))
            )

        def population_cost(initial, controls, reference, preceding, parameters):
            return trajectory_cost(population_rollout(initial, controls, parameters),
                                   controls, reference, preceding)

        self._population_cost_function = population_cost
        self._population_cost = jax.jit(population_cost, compiler_options=self.compiler_options or None)
        self._population_rollout = jax.jit(population_rollout, compiler_options=self.compiler_options or None)
        options = dict(compiler_options=self.compiler_options or None)
        self._native_rollout = jax.jit(native_rollout, **options)
        objective_options = dict(options)
        if self.pinned_host_output:
            objective_options["out_shardings"] = jax.sharding.SingleDeviceSharding(
                self.device, memory_kind="pinned_host"
            )
        # Committed-prefix prediction is consumed on the host, like the
        # packed solver result. Avoid a separate device-to-host transfer.
        self._rollout = jax.jit(rollout, **objective_options)

    def lift(self, observation):
        """Prepare the physical carry; re-lifting occurs inside every step."""
        state = self._array(observation)
        if state.shape != (85,):
            raise ValueError(
                "Projected backend requires one native 85-state observation"
            )
        return state

    def rollout(self, initial, controls):
        return self._rollout(initial, self._array(controls), self.parameters)

    def native_rollout(self, initial, controls):
        return self._native_rollout(initial, self._array(controls), self.parameters)

    def reset_optimizer(self):
        self._icem_state = self._icem_host_state = None

    def optimizer_state(self):
        if self._icem_host_state is None:
            return None
        return tuple(value.copy() for value in self._icem_host_state)

    def restore_optimizer_state(self, state):
        if state is None:
            self.reset_optimizer()
            return
        mean, elites = (np.asarray(value, dtype=self.dtype) for value in state)
        if (mean.shape != (self.horizon, 3) or elites.ndim != 3
                or elites.shape[1:] != mean.shape or not np.isfinite(mean).all()
                or not np.isfinite(elites).all()):
            raise ValueError('Invalid serialized iCEM warm memory')
        self._icem_host_state = (mean.copy(), elites.copy())
        self._icem_state = (self._array(mean), self._array(elites))

    def solve_icem(self, observation, reference, preceding, rng, options, lower, upper,
                   *, previous_solution=None, warm_start_steps=1):
        from .icem import ICEMResult
        from .icem import build_optimizer, reference_initial_mean
        if self._icem_optimizer is None:
            self._icem_options = options
            lo, hi = self._array(lower), self._array(upper)
            cold_mean = None
            if options.reference_initialization:
                pose_columns = tuple(self.tracking_feature_indices.index(i) for i in range(4))
                cold_mean = lambda initial, target: reference_initial_mean(
                    initial, target, self.horizon, self.model_dt, pose_columns, lo, hi)
            self._icem_optimizer = build_optimizer(self._population_cost_function,
                self.horizon, lo, hi, options,
                compiler_options=self.compiler_options,
                pinned_host_output=self.pinned_host_output, device=self.device, cold_mean=cold_mean,
                dt=self.model_dt)
        if options != self._icem_options:
            raise ValueError('Construct a new controller when changing iCEM parameters')
        if (isinstance(warm_start_steps, bool) or not isinstance(warm_start_steps, (int, np.integer))
                or warm_start_steps < 0):
            raise ValueError('iCEM warm_start_steps must be a nonnegative integer')
        shift = min(int(warm_start_steps), self.horizon)
        has_memory = self._icem_state is not None
        if has_memory:
            mean, elites = self._icem_state
            if elites.shape != (options.reuse_count, self.horizon, 3):
                raise ValueError('iCEM serialized elite memory does not match configuration')
        else:
            mean = self._array(np.broadcast_to((lower + upper) / 2, (self.horizon, 3)))
            elites = self._array(np.broadcast_to((lower + upper) / 2,
                (options.reuse_count, self.horizon, 3)))
        has_previous = previous_solution is not None
        if (options.command_rate_limits is not None
                and (np.any(preceding < lower - 1e-7) or np.any(preceding > upper + 1e-7))):
            raise ValueError('Rate-limited iCEM needs a preceding command inside the command bounds')
        previous = (np.asarray(previous_solution, dtype=self.dtype) if has_previous else
                    np.broadcast_to((lower + upper) / 2, (self.horizon, 3)))
        if (previous.shape != (self.horizon, 3) or not np.isfinite(previous).all()
                or np.any(previous < lower - 1e-7) or np.any(previous > upper + 1e-7)):
            raise ValueError('iCEM previous solution must be a finite bounded horizon plan')
        # NumPy's captured RNG is the sole seed source; warmup/replay can restore it.
        key = jax.device_put(rng.integers(0, 2**32, size=2, dtype=np.uint32), self.device)
        packed, next_mean, next_elites = self._icem_optimizer(self._array(observation),
            self._array(reference), self._array(preceding), mean, elites,
            jax.device_put(np.asarray(has_memory), self.device), self._array(previous),
            jax.device_put(np.asarray(has_previous), self.device), key, self.parameters,
            jax.device_put(np.asarray(shift, dtype=np.int32), self.device))
        host = np.asarray(packed) if self.pinned_host_output else np.asarray(jax.device_get(packed))
        length = self.horizon * 3
        plan = host[:length].reshape(self.horizon, 3).astype(np.float64)
        cost = float(host[length])
        mean_host = host[length + 1:2 * length + 1].reshape(self.horizon, 3).copy()
        memory_end = (2 + options.reuse_count) * length + 1
        elite_host = host[2 * length + 1:memory_end].reshape(options.reuse_count, self.horizon, 3).copy()
        warm_cost, mean_cost, sampled_cost = (float(value) for value in host[memory_end:memory_end + 3])
        warm_feasible = has_previous
        rate_feasible = True
        if options.command_rate_limits is not None:
            warm = previous[np.minimum(np.arange(self.horizon) + shift, self.horizon - 1)]
            warm_feasible = bool(has_previous and np.all(np.abs(np.diff(
                np.vstack((preceding, warm)), axis=0))
                <= np.asarray(options.command_rate_limits) * self.model_dt + 1e-7))
            changes = np.diff(np.vstack((preceding, plan)), axis=0)
            rate_feasible = bool(np.all(np.abs(changes)
                <= np.asarray(options.command_rate_limits) * self.model_dt + 1e-7))
        competitive = (cost <= mean_cost and cost <= sampled_cost
                       and (not warm_feasible or cost <= warm_cost))
        success = bool(host[-1] == 1 and np.isfinite(cost) and np.isfinite(plan).all()
                       and np.all(plan >= lower - 1e-7) and np.all(plan <= upper + 1e-7)
                       and competitive and rate_feasible)
        if success:
            self._icem_state = (next_mean, next_elites)
            self._icem_host_state = (mean_host, elite_host)
        else:
            self.reset_optimizer()
        result = ICEMResult(x=plan.ravel(), fun=cost, success=success,
            status=0 if success else 1, nit=options.iterations, nfev=options.evaluations,
            optimizer='icem', sampling_budget_completed=True,
            population_sizes=options.populations,
            warm_start_evaluated=warm_feasible, warm_start_feasible=warm_feasible,
            warm_start_steps=int(warm_start_steps),
            warm_start_cost=warm_cost if warm_feasible else None,
            final_mean_cost=mean_cost, best_sampled_cost=sampled_cost,
            warm_start_selected=bool(warm_feasible and np.array_equal(
                plan, previous[np.minimum(np.arange(self.horizon) + shift, self.horizon - 1)].astype(np.float64))),
            best_cost_history=host[memory_end + 3:-1].copy(),
            message=('iCEM sampling budget completed'
                     if success else 'iCEM failed: insufficient finite elites, invalid plan or candidate retention'))
        return plan, result

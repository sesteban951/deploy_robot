"""GPU-resident iCEM trajectory optimization (Pinneri et al., CoRL 2020).

Algorithm/reference: https://proceedings.mlr.press/v155/pinneri21a.html
https://github.com/martius-lab/iCEM/blob/main/icem/controllers/icem.py
Independent JAX implementation; no dependency on that environment/controller.
"""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np


class ICEMResult(dict):
    """Solve diagnostics with mapping and attribute access; no solver dependency."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    __setattr__ = dict.__setitem__


@dataclass(frozen=True)
class ICEMConfig:
    population: int = 96
    iterations: int = 3
    elites: int = 10
    decay: float = 1.25
    noise_beta: float = 2.0
    initial_std: float = 0.5  # Multiplier of half the physical command range.
    warm_std: float | None = None  # None retains the same reset width on warm solves.
    momentum: float = 0.1
    reuse_fraction: float = 0.3
    reference_initialization: bool = False
    mean_each_iteration: bool = False
    stopping_candidate: bool = False
    stopping_switch_seconds: tuple[float, ...] = (0.0,)
    idle_recovery_candidates: bool = False
    command_rate_limits: tuple[float, float, float] | None = None
    command_bands: tuple | None = None

    def __post_init__(self):
        for name in ('population', 'iterations', 'elites'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'iCEM {name} must be a positive integer')
        if self.elites < 2 or self.population < 2 * self.elites:
            raise ValueError('iCEM needs at least two elites and population >= 2*elites')
        for name in ('decay', 'noise_beta', 'initial_std', 'momentum', 'reuse_fraction'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'iCEM {name} must be finite')
        if self.decay < 1 or self.noise_beta < 0 or self.initial_std <= 0:
            raise ValueError('Invalid iCEM decay, noise_beta or initial_std')
        if not 0 <= self.momentum < 1 or not 0 <= self.reuse_fraction <= 1:
            raise ValueError('Invalid iCEM momentum or reuse_fraction')
        if self.warm_std is not None and (not math.isfinite(self.warm_std) or self.warm_std <= 0):
            raise ValueError('iCEM warm_std must be finite and positive')
        if not isinstance(self.reference_initialization, bool):
            raise ValueError('iCEM reference_initialization must be boolean')
        if not isinstance(self.mean_each_iteration, bool):
            raise ValueError('iCEM mean_each_iteration must be boolean')
        if not isinstance(self.stopping_candidate, bool):
            raise ValueError("iCEM stopping_candidate must be boolean")
        times = self.stopping_switch_seconds
        if (not times or any(isinstance(t, (bool, np.bool_)) for t in times)
                or any(not math.isfinite(t) or t < 0 for t in times)
                or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("Stopping switch times must start at zero and increase")
        object.__setattr__(self, "stopping_switch_seconds", tuple(map(float, times)))
        if self.stopping_candidate and len(times) >= min(self.populations):
            raise ValueError("Stopping candidates must leave a population slot for the mean")
        if not isinstance(self.idle_recovery_candidates, bool):
            raise ValueError("iCEM idle_recovery_candidates must be boolean")
        if (self.idle_recovery_candidates and self.stopping_candidate
                and self.reference_initialization and self.command_bands is not None
                and len(times) != 4):
            raise ValueError("Idle recovery requires four stopping candidates")
        if self.command_rate_limits is not None:
            rates = np.asarray(self.command_rate_limits)
            if (rates.shape != (3,) or rates.dtype.kind not in 'fiu'
                    or any(isinstance(value, (bool, np.bool_)) for value in self.command_rate_limits)
                    or not np.isfinite(rates).all() or np.any(rates <= 0)):
                raise ValueError('iCEM command_rate_limits must contain three finite positive rates')
            object.__setattr__(self, 'command_rate_limits', tuple(map(float, rates)))

        if self.command_bands is not None:
            boxes = np.asarray(self.command_bands, dtype=float)
            if (boxes.ndim != 3 or boxes.shape[1:] != (2, 3)
                    or not np.isfinite(boxes).all() or np.any(boxes[:, 0] > boxes[:, 1])):
                raise ValueError("command_bands must contain finite lower/upper command boxes")
            if self.command_rate_limits is not None:
                raise ValueError("Disjoint command bands require explicit command steps; disable slew limits")
            object.__setattr__(self, "command_bands", tuple(tuple(tuple(row) for row in box) for box in boxes))

    @property
    def reuse_count(self):
        return int(self.elites * self.reuse_fraction)

    @property
    def populations(self):
        # Match the reference implementation's integer decay at each iteration.
        result = [self.population]
        for _ in range(1, self.iterations):
            result.append(max(2 * self.elites, int(result[-1] / self.decay)))
        return tuple(result)

    @property
    def evaluations(self):
        # Only shifted outer-step elites need fresh dynamics. Inner costs reuse
        # the same initial state/reference/preceding command and stay on GPU.
        # One additional deterministic previous-plan slot is evaluated in the
        # first batch (a masked placeholder on a cold start, for one static JIT).
        return sum(self.populations) + self.reuse_count + 1


def colored_noise_filter(horizon, beta):
    """Linear Gaussian filter with the reference colorednoise spectral law.

    Equivalent in distribution to its independent real/imaginary Fourier draws,
    including DC and Nyquist corrections and theoretical normalization. Do not
    subtract each sample's temporal mean: DC exploration is intentional.
    At H<=20 a precomputed filter avoids FFT launches on every optimizer step.
    """
    if horizon < 1 or not math.isfinite(beta) or beta < 0:
        raise ValueError('Invalid colored-noise horizon/exponent')
    if horizon == 1 or beta == 0:
        return np.eye(horizon, dtype=np.float32)
    frequencies = np.maximum(np.fft.rfftfreq(horizon), 1.0 / horizon)
    amplitude = frequencies ** (-0.5 * beta)
    non_dc = amplitude[1:].copy()
    non_dc[-1] *= (1 + horizon % 2) / 2
    sigma = 2 * np.sqrt(np.sum(non_dc ** 2)) / horizon
    transformed = np.fft.irfft(np.fft.rfft(np.eye(horizon), axis=-1) * amplitude,
                               n=horizon, axis=-1)
    return (transformed * np.sqrt(2.0 / horizon) / sigma).astype(np.float32)


def reference_initial_mean(initial, reference, horizon, dt, pose_columns, lower, upper):
    """Bounded kinematic proposal for cold sampling, never a published policy.

    Aim at the terminal reference pose over the unchanged prediction horizon.
    The learned-model objective, not this proposal, selects every returned plan.
    All coordinates are in the same fixed prediction frame.
    """
    target = reference[-1, jnp.asarray(pose_columns)]
    duration = horizon * dt
    velocity = (target[:2] - initial[:2]) / duration
    angle = jnp.arctan2(target[2] * initial[3] - target[3] * initial[2],
                       target[3] * initial[3] + target[2] * initial[2])
    turn = jnp.clip(angle / duration, lower[2], upper[2])
    yaw = jnp.arctan2(initial[2], initial[3]) + (jnp.arange(horizon) + .5) * dt * turn
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    proposal = jnp.stack((c * velocity[0] + s * velocity[1],
                          -s * velocity[0] + c * velocity[1],
                          jnp.full(horizon, turn)), axis=-1)
    return jnp.clip(proposal, lower, upper)


def rate_limit_controls(controls, preceding, lower, upper, step_limits, command_bands=None):
    """Causally map proposals to feasible commands BEFORE model evaluation.

    Vectorized over candidate populations; the scan dimension is horizon time.
    The preceding command is the last actually committed command, not zero.
    """
    if command_bands is not None:
        boxes = jnp.asarray(command_bands, dtype=controls.dtype)
        lo, hi = jnp.maximum(boxes[:, 0], lower), jnp.minimum(boxes[:, 1], upper)
        choices = jnp.clip(controls[..., None, :], lo, hi)
        distance = jnp.sum((choices - controls[..., None, :])**2, axis=-1)
        nearest = jnp.argmin(jnp.where(jnp.all(lo <= hi, axis=-1), distance, jnp.inf), axis=-1)
        return jnp.take_along_axis(choices, nearest[..., None, None], axis=-2)[..., 0, :]
    if step_limits is None:
        return jnp.clip(controls, lower, upper)
    initial = jnp.broadcast_to(preceding, controls.shape[:-2] + (3,))
    def step(previous, proposed):
        command = jnp.clip(proposed, jnp.maximum(lower, previous - step_limits),
                           jnp.minimum(upper, previous + step_limits))
        return command, command
    return jnp.moveaxis(jax.lax.scan(step, initial, jnp.moveaxis(controls, -2, 0),
                                   unroll=controls.shape[-2])[1], 0, -2)


def build_optimizer(evaluate, horizon, lower, upper, config, *, compiler_options=None,
                    pinned_host_output=False, device=None, cold_mean=None, dt=0.02):
    """Compile all sampling/refitting iterations and costs into one executable.

    evaluate(initial, U[B,H,3], reference, preceding, parameters)->cost[B].
    The only host synchronization is the final packed plan/state/diagnostics.
    """
    dtype = lower.dtype
    midpoint = (lower + upper) * 0.5
    initial_std = (upper - lower) * (0.5 * config.initial_std)
    warm_std = initial_std if config.warm_std is None else (upper - lower) * (0.5 * config.warm_std)
    noise_filter = jax.device_put(np.asarray(colored_noise_filter(horizon, config.noise_beta),
                                            dtype=np.dtype(dtype)), device)
    reuse = config.reuse_count
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('iCEM prediction dt must be finite and positive')
    stop_steps = tuple(sorted({min(horizon, round(t / dt))
                              for t in config.stopping_switch_seconds}))
    idle_recovery = (config.idle_recovery_candidates and config.stopping_candidate
                     and config.reference_initialization and config.command_bands is not None)
    if idle_recovery:
        if cold_mean is None:
            raise ValueError("Idle recovery requires a reference proposal")
        boxes = np.asarray(config.command_bands)
        floors = np.zeros((2, 2))  # forward/backward and pure-pivot minima
        for row, axis in enumerate((0, 2)):
            others = [column for column in range(3) if column != axis]
            zero_others = np.all((boxes[:, 0, others] <= 0)
                                 & (boxes[:, 1, others] >= 0), axis=-1)
            for column, sign in enumerate((-1, 1)):
                bounds = boxes[:, 0 if sign > 0 else 1, axis]
                choices = bounds[zero_others & (sign * bounds > 0)]
                if not choices.size:
                    raise ValueError("Idle recovery requires signed drive and pure-pivot bands")
                floors[row, column] = choices[np.argmin(np.abs(choices))]
        floors = jnp.asarray(floors, dtype=dtype)
        # Keep all four slots even when a short horizon merges switch times.
        stop_steps = tuple(min(horizon, round(t / dt))
                           for t in config.stopping_switch_seconds)
        recovery_steps = tuple(max(1, step) for step in stop_steps[1:3])
    step_limits = (None if config.command_rate_limits is None else
                   jnp.asarray(config.command_rate_limits, dtype=dtype) * dt)

    def optimize(initial, reference, preceding, mean, memory, has_memory,
                 previous, has_previous, key, parameters, shift_steps=1):
        indices = jnp.minimum(jnp.arange(horizon) + shift_steps, horizon - 1)
        shifted_mean = mean[indices]
        start = (jnp.broadcast_to(midpoint, (horizon, 3)) if cold_mean is None else
                 cold_mean(initial, reference))
        mean = jnp.where(has_memory, shifted_mean, start)
        mean = rate_limit_controls(mean, preceding, lower, upper, step_limits, config.command_bands)
        # The last feasible best plan is distinct from the fitted mean/elites.
        # Shift it without noise, repeating its last bounded command as tail.
        warm = previous[indices]
        warm_feasible = has_previous
        if step_limits is not None:
            changes = jnp.concatenate((warm[:1] - preceding, warm[1:] - warm[:-1]))
            warm_feasible = has_previous & jnp.all(jnp.abs(changes) <= step_limits + 1e-7)
        std = jnp.broadcast_to(jnp.where(has_memory, warm_std, initial_std), (horizon, 3))
        history = []
        valid = jnp.asarray(True)
        best_sampled_cost = jnp.asarray(jnp.inf, dtype=dtype)
        for iteration, population in enumerate(config.populations):
            key, draw = jax.random.split(key)
            white = jax.random.normal(draw, (population + reuse, 3, horizon), dtype=dtype)
            noise = jnp.swapaxes(jnp.matmul(white, noise_filter,
                                precision=jax.lax.Precision.HIGHEST), 1, 2)
            samples = jnp.clip(mean + std * noise[:population], lower, upper)
            samples = rate_limit_controls(samples, preceding, lower, upper, step_limits, config.command_bands)
            # Optionally compete with the current mean in every batch. This
            # also evaluates the cold reference proposal, not just its noisy
            # neighbours, without increasing the rollout population/budget.
            inject_mean = config.mean_each_iteration or iteration == config.iterations - 1
            if inject_mean:
                samples = samples.at[0].set(mean)
            if config.stopping_candidate:
                # Predict the full physical transient after a command switches
                # to idle. These compete under the same cost/batch budget.
                # Retain the mean's prefix to also optimize when braking starts.
                for slot, switch in enumerate(stop_steps, start=1):
                    stop = jnp.where((jnp.arange(horizon) < switch)[:, None], mean, 0.)
                    if idle_recovery:
                        # An idle mean otherwise makes every stop proposal zero.
                        # Compete short supported corrections followed by idle;
                        # never publish one without the unchanged model/cost.
                        axis = 0 if slot <= 2 else 2
                        row = 0 if axis == 0 else 1
                        direction = start[0, axis]
                        pulse = jnp.zeros(3, dtype=dtype).at[axis].set(
                            jnp.where(direction == 0, 0.,
                                      floors[row, (direction > 0).astype(jnp.int32)]))
                        recovery = jnp.where(
                            (jnp.arange(horizon) < recovery_steps[(slot - 1) % 2])[:, None],
                            pulse, 0.)
                        stationary = ((jnp.max(jnp.abs(reference - reference[0])) < 1e-6)
                                      & jnp.all(jnp.abs(reference[..., 4:7]) < 1e-6))
                        stop = jnp.where(jnp.all(mean == 0.) & stationary, recovery, stop)
                    stop = rate_limit_controls(stop, preceding, lower, upper,
                                               step_limits, config.command_bands)
                    samples = samples.at[slot].set(stop)
            if iteration == 0:
                tails = jnp.clip(mean + std * noise[population:], lower, upper)
                new_tail = jnp.arange(horizon) >= horizon - shift_steps
                shifted = jnp.where(new_tail[None, :, None], tails, memory[:, indices])
                shifted = rate_limit_controls(shifted, preceding, lower, upper, step_limits, config.command_bands)
                candidates = jnp.concatenate((samples, shifted, warm[None]), axis=0)
                costs = evaluate(initial, candidates, reference, preceding, parameters)
                memory_slice = slice(population, population + reuse)
                costs = costs.at[memory_slice].set(jnp.where(has_memory, costs[memory_slice], jnp.inf))
                # An unperturbed shifted plan competes exactly when feasible.
                # Never publish an infeasible old plan or compare against its cost.
                costs = costs.at[-1].set(jnp.where(warm_feasible, costs[-1], jnp.inf))
            else:
                new_costs = evaluate(initial, samples, reference, preceding, parameters)
                candidates = jnp.concatenate((samples, elite[:reuse]), axis=0)
                costs = jnp.concatenate((new_costs, elite_costs[:reuse]), axis=0)
            costs = jnp.where(jnp.isfinite(costs), costs, jnp.inf)
            if iteration == 0:
                warm_cost = jnp.where(warm_feasible, costs[-1], jnp.nan)
            if iteration == config.iterations - 1:
                final_mean_cost = costs[0]
            # Include newly evaluated candidates (including stop proposals),
            # excluding the deterministic warm plan and injected final mean.
            begin = 1 if inject_mean else 0
            end = population + (reuse if iteration == 0 else 0)
            best_sampled_cost = jnp.minimum(best_sampled_cost, jnp.min(costs[begin:end]))
            _, indices = jax.lax.top_k(-costs, config.elites)
            elite, elite_costs = candidates[indices], costs[indices]
            valid = valid & jnp.all(jnp.isfinite(elite_costs))
            new_mean = jnp.mean(elite, axis=0)
            new_std = jnp.sqrt(jnp.mean((elite - new_mean) ** 2, axis=0))
            mean = config.momentum * mean + (1 - config.momentum) * new_mean
            mean = rate_limit_controls(mean, preceding, lower, upper, step_limits, config.command_bands)
            std = config.momentum * std + (1 - config.momentum) * new_std
            if iteration == 0:
                best, best_cost = elite[0], elite_costs[0]
            else:
                improved = elite_costs[0] < best_cost
                best = jnp.where(improved, elite[0], best)
                best_cost = jnp.minimum(best_cost, elite_costs[0])
            history.append(best_cost)
        # Pack small host diagnostics together, including reproducible warm
        # memory. mean/elites also remain resident for the next optimizer call.
        packed = jnp.concatenate((best.ravel(), best_cost[None], mean.ravel(),
            elite[:reuse].ravel(), jnp.stack((warm_cost, final_mean_cost, best_sampled_cost)),
            jnp.stack(history), valid.astype(dtype)[None]))
        return packed, mean, elite[:reuse]

    options = dict(compiler_options=compiler_options or None)
    if pinned_host_output:
        host = jax.sharding.SingleDeviceSharding(device, memory_kind='pinned_host')
        gpu = jax.sharding.SingleDeviceSharding(device)
        options['out_shardings'] = (host, gpu, gpu)
    return jax.jit(optimize, **options)

"""G1 MPC execution: startup, feedback, solves, command buffering, and tracing.
"""

from collections import deque
from copy import deepcopy
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from kcf_mpc.robot import CommandSource, RobotState, nominal_state
from kcf_mpc.robot_config import G1Config, COMMAND_BANDS
from kcf_mpc.state.encoder import g1_closed_loop_state, g1_world_heading
from kcf_mpc.reference import PlanarReference

KCF_ROOT = Path(__file__).resolve().parent

from kcf_mpc.config.mpc_config import (
    REFERENCE_MODES,
    get_default_mpc_config,
    resolve_mpc_model_config,
)
from kcf_mpc.mpc.mpc import create_mpc_controller
from kcf_mpc.mpc.mpc_runtime import StartupGCFreeze
from kcf_mpc.mpc.mpc_trace import MPCSolveTrace
from kcf_mpc.mpc.tracking import TRACKING_LABELS
from kcf_mpc.mpc.mpc_buffered import (
    AsyncConsole,
    BufferedPlan,
    SingleSolveWorker,
    SolveRequest,
)
from kcf_mpc.mpc.mpc_handover import HandoverTiming, predict_handover_state


def _configure_cpu_affinity(cpus):
    """Pin the control thread before JAX workers start; None keeps OS defaults."""
    if cpus is None:
        return None
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(
            "MPC cpu_affinity is unsupported here; set it to None in mpc_config.py"
        )
    try:
        selected = set(cpus)
    except TypeError as error:
        raise ValueError(
            "MPC cpu_affinity must be a sequence of logical CPU IDs or None"
        ) from error
    if not selected or any(
        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in selected
    ):
        raise ValueError("MPC cpu_affinity must contain non-negative integer CPU IDs")
    if not selected.issubset(os.sched_getaffinity(0)):
        raise ValueError(
            "MPC cpu_affinity selects unavailable CPUs; adjust it or set None in mpc_config.py"
        )
    os.sched_setaffinity(0, selected)
    return tuple(sorted(selected))


def resolve_mpc_settings(
    settings=None,
    *,
    reference_mode: str | None = None,
    prediction_model: str | None = None,
):
    """Load MPC settings with optional process-local predictor/reference overrides.

    Explicit settings are copied, so overrides cannot mutate a saved trial plan.
    An explicit predictor selection also selects that model's gain profile.
    Omitting it preserves resolved settings, including saved per-run gains.
    """
    settings = get_default_mpc_config() if settings is None else deepcopy(settings)
    selected_model = (
        settings.get("prediction_model", "kcf")
        if prediction_model is None
        else prediction_model
    )
    if selected_model not in ("kcf", "unicycle"):
        raise ValueError("prediction_model must be 'kcf' or 'unicycle'")
    if prediction_model is not None:
        settings = resolve_mpc_model_config(settings, prediction_model=selected_model)
    if reference_mode is not None:
        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}")
        settings["reference_mode"] = reference_mode
    return settings


class KCFCommand(CommandSource):
    """Track a configured planar reference using the selected MPC predictor.

    The optimizer, cost and existing input bounds belong to kcf_mpc/mpc/mpc.py.
    iCEM performs forward-only GPU population evaluation.
    """

    # Keep the publisher phase-locked to fresh feedback, not another 50 Hz wall
    # clock that can straddle messages and skip a physics tick. Solves stay async.
    feedback_driven = True

    def __init__(
        self,
        cmd_scale: np.ndarray,
        *,
        settings,
        quiet: bool = False,
        verbose: bool = False,
        feedback_is_synchronous: bool = False,
    ):
        super().__init__(cmd_scale)
        self.cfg = G1Config()
        self.cpu_affinity = _configure_cpu_affinity(settings.get("cpu_affinity"))
        threads = int(settings["num_threads"])
        if threads < 1:
            raise ValueError("MPC num_threads must be positive")
        torch.set_num_threads(threads)
        self.mpc = create_mpc_controller(settings)
        self.backend_label = self.mpc.backend
        if self.mpc.backend == "jax":
            self.backend_label += f"/{self.mpc.jax_platform}"
        self.backend_label += (
            f"/{str(self.mpc.dtype).removeprefix('torch.')}/{self.mpc.solver}"
        )
        self.solver_label = 'iCEM'
        self.solver_description = (
            f"populations={self.mpc.icem_config.populations} elites={self.mpc.icem_config.elites} "
            f"beta={self.mpc.icem_config.noise_beta:g}"
        )

        expected_tracked_indices = tuple(range(len(TRACKING_LABELS)))
        observation_dim = getattr(
            self.mpc, "observation_dim", self.mpc.model.config.state_dim
        )
        if (
            observation_dim != 104
            or self.mpc.model.config.state_dim != 104
            or self.mpc.model.config.L_embed != 0
            or self.mpc.input_dim != 3
            or tuple(self.mpc.tracking_feature_indices.cpu().tolist())
            != expected_tracked_indices
        ):
            raise ValueError(
                "G1 requires the native 104-state model " "with three commands"
            )
        if np.any(self.mpc.lower_bounds < -self.cmd_scale) or np.any(
            self.mpc.upper_bounds > self.cmd_scale
        ):
            raise ValueError("MPC input bounds exceed the simulator's command limits")
        if self.mpc.icem_config.command_bands != COMMAND_BANDS:
            raise ValueError("Running MPC command bands must match policy/config.py")
        self.tracked_dim = len(expected_tracked_indices)
        self.reference_path = PlanarReference(settings)
        self.goal_hold_seconds = settings.get("goal_hold_seconds")
        if self.goal_hold_seconds is not None and not (
            isinstance(self.goal_hold_seconds, (int, float))
            and not isinstance(self.goal_hold_seconds, bool)
            and np.isfinite(self.goal_hold_seconds) and self.goal_hold_seconds >= 0
        ):
            raise ValueError("goal_hold_seconds must be None or a nonnegative number of seconds")
        self.N = self.mpc.horizon
        self.dt = self.cfg.control_dt
        self.standing_hold = settings.get("standing_hold")
        if self.standing_hold is not None:
            hold = self.standing_hold
            if (not all(np.isfinite(value) and value > 0 for value in hold.values())
                    or hold["enter_position_m"] >= hold["exit_position_m"]
                    or hold["enter_heading_rad"] >= hold["exit_heading_rad"]):
                raise ValueError("Standing hold requires positive entry limits below release limits")
        self.quiet, self.verbose = quiet, verbose
        self.solve_count = self.failure_count = self.deadline_miss_count = 0
        self.solver_failure_count = self.stale_rejections = self.reset_discards = 0
        self.solve_time = 0.0
        deadline = settings["live_command_deadline"]
        self.live_command_deadline = None if deadline is None else float(deadline)
        if self.live_command_deadline is not None and (
            not np.isfinite(self.live_command_deadline)
            or self.live_command_deadline <= 0
        ):
            raise ValueError("Live command deadline must be positive or None")
        self.live_plan_max_age = float(settings["live_plan_max_age"])
        self.live_feedback_timeout = float(settings["live_feedback_timeout"])
        if any(
            not np.isfinite(x) or x <= 0
            for x in (self.live_plan_max_age, self.live_feedback_timeout)
        ):
            raise ValueError(
                "Plan age and feedback timeout must be finite and positive"
            )
        self.application_lead_steps = settings["command_application_lead_steps"]
        if (
            isinstance(self.application_lead_steps, bool)
            or not isinstance(self.application_lead_steps, int)
            or self.application_lead_steps < 0
        ):
            raise ValueError(
                "command_application_lead_steps must be a non-negative integer"
            )
        self.handover_timing = HandoverTiming(
            settings["live_handover"],
            dt=self.dt,
            application_lead=self.application_lead_steps,
            horizon=self.N,
        )
        self.wait_for_first_state = bool(settings["wait_for_first_state"])
        self.warmup_enabled = bool(settings["warmup"])
        self.synchronous_feedback = bool(feedback_is_synchronous)
        trace_options = settings["live_trace"]
        self._trace = (
            MPCSolveTrace(
                trace_options,
                settings,
                state_dim=observation_dim,
                horizon=self.N,
                input_dim=self.mpc.input_dim,
                tracked_dim=self.tracked_dim,
            )
            if trace_options["enabled"]
            else None
        )
        self._worker = None
        self._console = None
        self._request_counter = 0
        self.reset()
        if self.warmup_enabled:
            # Compile/warm numerical kernels before the shared loop connects.
            # This is the same KCF solve, and its command is never published.
            rng_state = self.mpc.rng.bit_generator.state
            began = perf_counter()
            try:
                initial = self._encode_state(nominal_state(self.cfg))
                reference = self._tracked_reference(self.dt * np.arange(1, self.N + 1))
                _, result = self.mpc.solve(
                    initial, reference, preceding_command=np.zeros(3)
                )
                if not result.success:
                    raise RuntimeError(f"MPC initialization failed: {result.message}")
                # Variable-length prefixes must never trigger compilation on a
                # live tick. No optimizer/RNG changes occur in these rollouts.
                if not self.synchronous_feedback:
                    for steps in range(self.handover_timing.minimum,
                                       self.handover_timing.maximum + 1):
                        predict_handover_state(
                            self.mpc, initial, np.zeros((steps, 3), np.float32),
                        )
                if not self.quiet:
                    print(
                        f"[MPC {self.solver_label} initialization] backend={self.backend_label} "
                        f"compute={1000*(perf_counter()-began):.1f}ms "
                        f"iterations={result.nit} status=solver_success "
                        f"termination={result.message} (command not published)",
                        flush=True,
                    )
            finally:
                # Warmup must not alter seeded initialization of the real run.
                self.mpc.rng.bit_generator.state = rng_state
                self.mpc.reset()
        if not self.quiet:
            print(
                f"Using {KCF_ROOT / 'mpc' / 'mpc.py'} ({self.solver_label}, {self.backend_label}); "
                f"{self.solver_description}; "
                f"Q_diag={self.mpc.state_weights.detach().cpu().tolist()} "
                f"Qf_diag={self.mpc.terminal_state_weights.detach().cpu().tolist()}; "
                f"R={self.mpc.input_weight:g} "
                f"S_diag={self.mpc.command_change_weights.detach().cpu().tolist()}; "
                f"prediction_model={self.mpc.prediction_model}; "
                "prediction_frame=fixed_initial_ground_yaw; "
                f"reference_mode={self.reference_path.mode}; "
                f"cpu_affinity={self.cpu_affinity}; "
                f"execution={'synchronous' if self.synchronous_feedback else 'committed_handover'}; "
                f"plan_max_age={self.live_plan_max_age:g}s; "
                f"application_lead_steps={self.application_lead_steps}; "
                f"handover_steps={self.handover_timing.minimum}..{self.handover_timing.maximum} "
                f"(initial={self.handover_timing.initial}); "
                f"warm_start=iCEM shifted mean/elites + exact feasible plan; "
                f"settings from {KCF_ROOT / 'config' / 'mpc_config.py'}",
                flush=True,
            )
        self._runtime_gc = StartupGCFreeze(
            settings.get("freeze_gc_after_warmup", False)
        )
        self._worker = SingleSolveWorker(
            self.mpc, trace=self._trace, deadline=self.live_command_deadline
        )
        if not self.quiet or self.verbose:
            self._console = AsyncConsole()

    def close(self):
        # Runs after the publishing loop stops, not between control ticks.
        try:
            worker = getattr(self, "_worker", None)
            if worker is not None:
                worker.close()
                for outcome in (getattr(self, "_ready_outcome", None), worker.poll()):
                    if outcome is not None and self._trace is not None:
                        self._trace.handover(
                            outcome.trace_row,
                            accepted=False,
                            command=np.zeros(3),
                            index=-1,
                            age=np.nan,
                            reason="shutdown",
                        )
            console = getattr(self, "_console", None)
            if console is not None:
                console.close()
                if console.dropped:
                    print(
                        f"Dropped {console.dropped} console lines (solve trace retained).",
                        flush=True,
                    )
            trace = getattr(self, "_trace", None)
            if trace is not None:
                path = trace.close()
                if path is not None:
                    print(
                        f"Saved MPC solve trace: {path} "
                        f"({trace.count} solves, {trace.tick_count} command ticks; "
                        f"dropped={trace.dropped}/{trace.dropped_ticks})",
                        flush=True,
                    )
        finally:
            self._runtime_gc.close()

    def reference(self, t):
        """World pose and twist at scalar/array controller time ``t``."""
        return self.reference_path.reference(t)

    def _tracked_reference(self, times):
        return self.reference_path.track(*self.reference(times))

    def _encode_state(self, state):
        return g1_closed_loop_state(state)

    def reset(self):
        ready = getattr(self, "_ready_outcome", None)
        if ready is not None:
            self.reset_discards += 1
            if self._trace is not None:
                self._trace.handover(
                    ready.trace_row,
                    accepted=False,
                    command=np.zeros(3),
                    index=-1,
                    age=np.nan,
                    reason="reset_discard",
                )
        self.theta_prev = None
        self.ref = np.asarray(self.reference(0.0)[0][:2]).copy()
        self.last_state_time = None
        self.last_observation = None
        self.previous_command = np.zeros(3, dtype=np.float32)
        self._standing_hold_active = False
        self._standing_hold_since = None
        self._plan = None
        self._active_outcome = None
        self._commitment = None
        self._commitment_invalid = None
        self._ready_outcome = None
        self._last_application_time = None
        self._generation = getattr(self, "_generation", 0) + 1
        self._last_feedback_wall = None
        self._command_history = deque(maxlen=self.N + 2)
        self.optimizer_time = 0.0
        self.start_time_seen = self.t_start
        timing = getattr(self, "handover_timing", None)
        path = getattr(self, "reference_path", None)
        if path is not None:
            path.reset_initial_pose()
        if timing is not None:
            timing.reset()
        worker = getattr(self, "_worker", None)
        if worker is None:
            self.mpc.reset()  # Startup only, before any worker exists.
        else:
            worker.invalidate()

    def _request(
        self, state, observation, now, *, plan_start_time=None, prefix_controls=None
    ):
        # The same model timestep, reference conversion and optimization problem.
        t = (state.t if plan_start_time is None else plan_start_time) - self.t_start
        reference = self._tracked_reference(t + self.dt * np.arange(1, self.N + 1))
        self._request_counter += 1
        observation, reference = observation.copy(), reference.copy()
        observation.flags.writeable = reference.flags.writeable = False
        if prefix_controls is not None:
            prefix_controls = np.array(prefix_controls, dtype=np.float32, copy=True)
            prefix_controls.flags.writeable = False
        previous = np.array(self.previous_command, dtype=np.float32, copy=True)
        previous.flags.writeable = False
        heading = self.theta_prev if observation.shape == (104,) else None
        return SolveRequest(
            self._generation,
            self._request_counter,
            float(state.t),
            now,
            observation,
            reference,
            plan_start_time,
            prefix_controls,
            previous,
            heading_world=heading,
        )

    def observe_applied_command(self, application_time, command):
        """Reconcile an external policy's actual input with planned commitments.

        Optional ROS-adapter acknowledgement; the existing simulation path is
        unchanged. A manual override or late packet invalidates predictions
        based on a different prefix instead of silently pretending it applied.
        """
        command = np.asarray(command, dtype=np.float32)
        if (not np.isfinite(application_time) or command.shape != (3,)
                or not np.isfinite(command).all()):
            raise ValueError("Applied command must be a finite timestamp and twist")
        expected = self._issued_at(application_time)
        if expected is not None and not np.allclose(expected, command, atol=1e-6, rtol=0):
            self._cancel_commitment("applied_command_mismatch")
            self._plan = None
        history = [(t, u) for t, u in self._command_history
                   if abs(t - application_time) > 1e-8]
        history.append((float(application_time), command.copy()))
        self._command_history.clear()
        self._command_history.extend(sorted(history, key=lambda item: item[0]))
        if (self._last_application_time is None
                or application_time >= self._last_application_time - 1e-8):
            self.previous_command = command.copy()

    def _issued_at(self, slot_time):
        history = [
            command
            for time, command in self._command_history
            if abs(time - slot_time) <= 1e-8
        ]
        return history[-1] if history else None

    def _prefix_error(self, request):
        errors = []
        for step, expected in enumerate(request.prefix_controls):
            issued = self._issued_at(request.state_time + step * self.dt)
            if issued is not None:
                errors.append(np.abs(issued - expected))
        return (
            np.max(errors, axis=0) if errors else np.full(3, np.nan),
            len(errors) / len(request.prefix_controls),
        )

    def _cancel_commitment(self, reason):
        if self._commitment is not None and self._commitment_invalid is None:
            self._commitment_invalid = reason

    def _select_command(self, application_time, now):
        command, index, age, reason = None, -1, np.nan, "no_plan"
        if self._plan is not None:
            command, index, age, reason = self._plan.select(
                application_time, now, self.dt, self.live_plan_max_age
            )
        request = self._commitment
        if (
            request is not None
            and self._commitment_invalid is None
            and application_time < request.control_start_time - 1e-9
        ):
            step = int(
                np.floor((application_time - request.state_time + 1e-9) / self.dt)
            )
            if 0 <= step < len(request.prefix_controls):
                expected = request.prefix_controls[step]
                current = (
                    command if command is not None else np.zeros(3, dtype=np.float32)
                )
                if not np.array_equal(expected, current):
                    self._cancel_commitment("buffer_changed")
                else:
                    command = expected.copy()
                    reason = "committed"
        if (
            self._last_feedback_wall is None
            or now - self._last_feedback_wall >= self.live_feedback_timeout
        ):
            self._cancel_commitment("feedback_timeout")
            self._plan = None
            command, reason = None, "feedback_timeout"
        if command is None:
            command = np.zeros(3, dtype=np.float32)
        return command, index, age, reason

    def _submit_handover(self, state, observation, now):
        steps = self.handover_timing.steps()
        prefix = np.zeros((steps, 3), np.float32)
        for step in range(steps):
            if step < self.application_lead_steps:
                current = self._issued_at(state.t + step * self.dt)
                if current is None:
                    return
                prefix[step] = current
            elif self._plan is not None:
                value, _, _, _ = self._plan.select(
                    state.t + step * self.dt,
                    now + (step - self.application_lead_steps) * self.dt,
                    self.dt,
                    self.live_plan_max_age,
                )
                if value is not None:
                    prefix[step] = value
        request = self._request(
            state,
            observation,
            now,
            plan_start_time=float(state.t + steps * self.dt),
            prefix_controls=prefix,
        )
        if self._worker.submit(request):
            self._commitment = request
            self._commitment_invalid = None

    def _advance_handover(self, application_time, now, state):
        outcome = (
            self._worker.poll()
        )  # Nonblocking, including while a result waits for handover.
        if outcome is not None:
            if (
                outcome.request.generation != self._generation
                or not outcome.result.success
            ):
                self._consume(outcome, application_time, now, state)
                if (
                    self._commitment is not None
                    and outcome.request.plan_id == self._commitment.plan_id
                ):
                    self._commitment = self._ready_outcome = None
                    self._commitment_invalid = None
            elif (
                self._commitment is None
                or outcome.request.plan_id != self._commitment.plan_id
            ):
                self._consume(
                    outcome,
                    application_time,
                    now,
                    state,
                    rejection="commitment_cancelled",
                )
            else:
                self._ready_outcome = outcome
        request = self._commitment
        if request is None:
            return
        if application_time > request.control_start_time + 1e-9:
            self._cancel_commitment("handover_missed")
        due = application_time >= request.control_start_time - 1e-9
        if due and self._ready_outcome is None:
            self._cancel_commitment("handover_missed")
        if self._ready_outcome is not None and (
            due or self._commitment_invalid is not None
        ):
            error, known = self._prefix_error(request)
            if known < 1 or np.any(error > 1e-7):
                self._cancel_commitment("commitment_mismatch")
            outcome = self._ready_outcome
            rejection = self._commitment_invalid
            self._commitment = self._ready_outcome = None
            self._commitment_invalid = None
            self._consume(outcome, application_time, now, state, rejection=rejection)

    def _consume(self, outcome, application_time, now, state, *, rejection=None):
        result = outcome.result
        self.solve_time, self.optimizer_time = (
            outcome.compute_seconds,
            outcome.optimizer_seconds,
        )
        self.solve_count += 1
        self.deadline_miss_count += int(result.deadline_missed)
        self.latency_miss_count = getattr(self, "latency_miss_count", 0) + int(
            result.get("latency_missed", False)
        )
        self.solver_failure_count += int(not result.success)
        index, age = -1, application_time - outcome.request.control_start_time
        accepted = False
        if outcome.request.generation != self._generation:
            reason = "reset_discard"
            self.reset_discards += 1
        elif not result.success:
            # A lockstep solve owns the current physics tick, so its last
            # successful horizon is still aligned with simulation time.  Keep
            # that bounded plan available for the synchronous failure path;
            # asynchronous execution continues to fail closed because its
            # wall-time alignment can no longer be trusted here.
            if not self.synchronous_feedback:
                self._plan = None
            reason = "solver_failure"
            self.failure_count += 1
        elif rejection is not None:
            reason = rejection
            self.stale_rejections += 1
            self.failure_count += 1
        elif not self.synchronous_feedback and (
            self._last_feedback_wall is None
            or now - self._last_feedback_wall >= self.live_feedback_timeout
        ):
            reason = "feedback_timeout"
            self.stale_rejections += 1
            self.failure_count += 1
        else:
            candidate = BufferedPlan(outcome)
            if self.synchronous_feedback:
                command, index, age, reason = (
                    candidate.controls[0].copy(),
                    0,
                    0.0,
                    "buffered",
                )
            elif (
                outcome.request.prefix_controls is not None
                and abs(application_time - outcome.request.control_start_time) > 1e-9
            ):
                command, reason = None, "handover_missed"
            else:
                command, index, age, reason = candidate.select(
                    application_time, now, self.dt, self.live_plan_max_age
                )
            if command is not None:
                self._plan = candidate
                self._active_outcome = outcome
                accepted = True
            else:
                self.stale_rejections += 1
                self.failure_count += 1
        if outcome.request.generation == self._generation:
            self.handover_timing.observe(outcome)
        result.command_accepted = accepted
        result.command_source = "buffered" if accepted else "zero"
        result.deadline_rejected = bool(
            result.success and not accepted and reason != "reset_discard"
        )
        result.used_fallback = not accepted
        if outcome.request.prefix_controls is not None:
            prefix_error, prefix_known = self._prefix_error(outcome.request)
        else:
            prefix_error, prefix_known = np.zeros(3), 1.0
        result.estimated_prefix_max_abs_error = prefix_error
        result.prefix_known_fraction = prefix_known
        self.mpc_result = result
        if self._trace is not None:
            self._trace.handover(
                outcome.trace_row,
                accepted=accepted,
                command=command if accepted else np.zeros(3),
                index=index,
                age=age,
                reason=reason,
                prefix_error=prefix_error,
                prefix_known_fraction=prefix_known,
            )
        if self._console is not None:
            missed = result.deadline_missed or result.get("latency_missed", False)
            status = (
                "accepted_late"
                if accepted and missed
                else "accepted" if accepted else reason
            )
            self._console.write(
                f"[MPC {self.solver_label}] t={outcome.request.state_time:.3f}s "
                f"backend={self.backend_label} compute={1000*self.solve_time:.1f}ms "
                f"predict={1000*result.get('prediction_seconds',0.):.1f}ms "
                f"solve={1000*self.optimizer_time:.1f}ms iterations={result.get('nit', 'n/a')} "
                f"request_latency={1000*result.get('request_latency_seconds',self.solve_time):.1f}ms "
                f"evaluations={result.get('nfev', 'n/a')} status={status} "
                f"plan_id={outcome.request.plan_id} plan_index={index} plan_age={1000*age:.1f}ms "
                f"handover_t={outcome.request.control_start_time:.3f}s "
                f"prefix_error={np.max(prefix_error):.3g} prefix_known={prefix_known:.2f} "
                f"apply_t={application_time:.3f}s deadline_missed={int(result.deadline_missed)} "
                f"feedback_t={state.t:.3f}s "
                f"solver_failures={self.solver_failure_count} stale_rejections={self.stale_rejections} "
                f"overruns={self.deadline_miss_count}/{self.solve_count} "
                f"latency_overruns={self.latency_miss_count}/{self.solve_count} "
                f"position_error={np.linalg.norm(state.q[:2]-self.ref):.3f}m "
                f"termination={result.message}"
            )

    def get(self, state: RobotState) -> np.ndarray:
        now = perf_counter()
        if not np.isfinite(state.t):
            self.reset()
            return np.zeros(3, dtype=np.float32)
        if self.start_time_seen != self.t_start or (
            self.last_state_time is not None and state.t < self.last_state_time
        ):
            self.reset()
        lead = 0 if self.synchronous_feedback else self.application_lead_steps
        application_time = float(state.t + lead * self.dt)
        if self.reference_path._initial_pose is None and np.isfinite(state.q).all():
            self.reference_path.set_initial_pose(
                [state.q[0], state.q[1], g1_world_heading(state)])
        self.ref = np.asarray(
            self.reference(application_time - self.t_start)[0][:2]
        ).copy()
        if (
            self.wait_for_first_state
            and self.last_state_time is None
            and state.t == 0.0
        ):
            initial_observation = self._encode_state(state)
            if np.isfinite(initial_observation).all():
                self.theta_prev = g1_world_heading(state, self.theta_prev)
            command = np.zeros(3, dtype=np.float32)
            if not self.synchronous_feedback:
                # Retain this actually published next-tick zero commitment, so
                # first fresh feedback need not spend another tick creating it.
                self._command_history.append((application_time, command.copy()))
                self._last_application_time = application_time
            return command

        observation = self._encode_state(state)
        duplicate = (
            self.last_state_time == state.t
            and self.last_observation is not None
            and np.array_equal(observation, self.last_observation)
        )
        if self.last_state_time == state.t and not duplicate:
            self.reset()
            if np.isfinite(state.q).all():
                self.reference_path.set_initial_pose(
                    [state.q[0], state.q[1], g1_world_heading(state)])
            observation = self._encode_state(state)
            self.ref = np.asarray(
                self.reference(application_time - self.t_start)[0][:2]
            ).copy()
        if not np.isfinite(observation).all():
            self.reset()
            return np.zeros(3, dtype=np.float32)
        fresh = not duplicate
        if fresh:
            self._last_feedback_wall = now
            self.theta_prev = g1_world_heading(state, self.theta_prev)
            self.last_state_time = float(state.t)
            self.last_observation = observation.copy()
            reference_time = application_time - self.t_start
            if self.reference_path.needs_anchor(reference_time):
                start = [float(state.q[0]), float(state.q[1]), float(self.theta_prev)]
                self.reference_path.anchor_start(start, reference_time)
                self.ref = np.asarray(self.reference(reference_time)[0][:2]).copy()
                print(f"KCF start pose recorded at {reference_time:.1f} s: ({start[0]:.2f}, {start[1]:.2f}, "
                      f"yaw {np.degrees(start[2]):.1f} deg); tracking relative to it.", flush=True)

        if fresh and self.standing_hold is not None:
            poses, twists = self.reference(
                application_time - self.t_start + self.dt * np.arange(self.N + 1)
            )
            stationary = (np.max(np.abs(twists)) < 1e-6
                          and np.max(np.abs(poses - poses[0])) < 1e-6)
            position_error = np.linalg.norm(observation[:2] - poses[0, :2])
            heading_error = abs(np.arctan2(
                np.sin(self.theta_prev - poses[0, 2]),
                np.cos(self.theta_prev - poses[0, 2]),
            ))
            hold = self.standing_hold
            holding = (stationary and position_error < hold["exit_position_m"]
                       and heading_error < hold["exit_heading_rad"]
                       if self._standing_hold_active else
                       stationary and position_error < hold["enter_position_m"]
                       and heading_error < hold["enter_heading_rad"]
                       and np.linalg.norm(observation[4:6]) < hold["enter_linear_speed_mps"]
                       and abs(observation[6]) < hold["enter_yaw_rate_rps"]
                       and not np.any(self.previous_command))
            if not self._standing_hold_active:
                if not holding:
                    self._standing_hold_since = None
                elif self._standing_hold_since is None:
                    self._standing_hold_since = state.t
                holding = (holding and self._standing_hold_since is not None
                           and state.t - self._standing_hold_since >= hold["settle_seconds"])
            if holding != self._standing_hold_active:
                # Outstanding plans belong to the previous execution state.
                # Reset only on the solver thread; preserve the reference clock.
                self._generation += 1
                self._worker.invalidate()
                if self._ready_outcome is not None:
                    self._consume(self._ready_outcome, application_time, now, state)
                self._plan = self._active_outcome = None
                self._commitment = self._ready_outcome = None
                self._commitment_invalid = None
                self._standing_hold_active = holding
                if not holding:
                    self._standing_hold_since = None
        if self._standing_hold_active:
            self._advance_handover(application_time, now, state)
            command = np.zeros(3, dtype=np.float32)
            index, age, reason = -1, np.nan, "standing_hold"
            if now - self._last_feedback_wall >= self.live_feedback_timeout:
                command = np.zeros(3, dtype=np.float32)
                reason = "feedback_timeout"
        elif self.synchronous_feedback:
            if not fresh:
                return self.previous_command.copy()
            outcome = self._worker.solve_inline(self._request(state, observation, now))
            self._consume(outcome, application_time, now, state)
            if outcome.result.success:
                command = outcome.controls[0].astype(np.float32, copy=True)
                index, age, reason = 0, 0.0, "synchronous"
            else:
                # Wall time is deliberately irrelevant in lockstep: physics
                # has not advanced during the failed solve.  Advance through
                # the previous successful horizon using simulation time only.
                # BufferedPlan still enforces the horizon/max-age bound.
                if self._plan is None:
                    command, index, age = None, -1, np.nan
                else:
                    command, index, age, _ = self._plan.select(
                        application_time,
                        self._plan.request.query_wall_time,
                        self.dt,
                        self.live_plan_max_age,
                    )
                if command is None:
                    self._plan = None
                    command = np.zeros(3, dtype=np.float32)
                    reason = "synchronous_failure_zero"
                else:
                    reason = "synchronous_failure_fallback"
                    if self._console is not None:
                        self._console.write(
                            f"[MPC synchronous fallback] "
                            f"failed_plan_id={outcome.request.plan_id} "
                            f"fallback_plan_id={self._plan.request.plan_id} "
                            f"plan_index={index} plan_age={1000*age:.1f}ms"
                        )
        else:
            if (
                self._last_application_time is not None
                and application_time > self._last_application_time + self.dt + 1e-8
            ):
                self._cancel_commitment("execution_gap")
            if now - self._last_feedback_wall >= self.live_feedback_timeout:
                self._cancel_commitment("feedback_timeout")
            self._advance_handover(application_time, now, state)
            if fresh and not self._worker.busy and self._commitment is None:
                self._submit_handover(state, observation, now)
            command, index, age, reason = self._select_command(application_time, now)
        selected_at = perf_counter()
        callback_seconds = selected_at - now
        if not self.synchronous_feedback and not self._standing_hold_active:
            extra_steps = int(np.floor((callback_seconds + 1e-9) / self.dt))
            application_time += extra_steps * self.dt
            if extra_steps:
                self._cancel_commitment("execution_gap")
                if (
                    self._plan is not None
                    and self._plan.request.prefix_controls is not None
                    and self._plan.request.control_start_time
                    > (
                        self._last_application_time
                        if self._last_application_time is not None
                        else -np.inf
                    )
                    + 1e-9
                ):
                    activated = self._active_outcome
                    if activated is not None:
                        activated.result.command_accepted = False
                        activated.result.command_source = "zero"
                        activated.result.deadline_rejected = True
                        activated.result.used_fallback = True
                        if self._trace is not None:
                            self._trace.handover(
                                activated.trace_row,
                                accepted=False,
                                command=np.zeros(3),
                                index=-1,
                                age=application_time
                                - activated.request.control_start_time,
                                reason="activation_missed",
                            )
                        if self._console is not None:
                            self._console.write(
                                f"[MPC activation rejected] plan_id={activated.request.plan_id} "
                                "reason=activation_missed"
                            )
                    self.stale_rejections += 1
                    self.failure_count += 1
                    self._plan = None
                self.publisher_overruns = getattr(self, "publisher_overruns", 0) + 1
            command, index, age, reason = self._select_command(
                application_time, selected_at
            )
            if extra_steps and self._console is not None:
                self._console.write(
                    f"[MPC publisher overrun] callback={1000*callback_seconds:.1f}ms "
                    f"adjusted_plan_index={index} reason={reason}"
                )
            self.ref = np.asarray(
                self.reference(application_time - self.t_start)[0][:2]
            ).copy()
        self.previous_command = command.copy()
        self._command_history.append((application_time, command.copy()))
        self._last_application_time = application_time
        self.command_plan_index = index
        self.command_plan_age = age
        self.command_reason = reason
        if self._trace is not None:
            self._trace.record_command(
                now=selected_at,
                state_time=state.t,
                application_time=application_time,
                plan=self._plan,
                index=index,
                age=age,
                generation=self._generation,
                pending=self._worker.busy,
                command=command,
                reason=reason,
                callback_seconds=callback_seconds,
                commitment=self._commitment,
            )
        return command.copy()


def create_profile_controller(profile, settings, *, quiet=True):
    if (profile.values['state_dim'], profile.values['joint_count'],
            profile.values['recurrent_state_dim'], profile.values['nH']) != (104, 29, 0, 256):
        raise ValueError('Running requires the native-104, 29-joint, lift-256 profile')
    return KCFCommand(G1Config().cmd_scale, settings=settings, quiet=quiet)

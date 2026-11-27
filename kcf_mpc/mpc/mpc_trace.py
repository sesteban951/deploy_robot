"""Bounded, host-only solve capture; never draws random numbers or evaluates a model.

Inputs are copied before the real solve; initial guesses can subsequently be
recreated using the original solver's RNG state and previous optimal sequence.
Disk output happens only after the controller's publishing loop has stopped.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
from kcf_mpc.mpc.tracking import TRACKING_LABELS


class MPCSolveTrace:
    FORMAT_VERSION = 14
    ACTIONS = {"mpc": 0, "held": 1, "zero": 2, "pending": 3, "buffered": 4}

    def __init__(
        self, options, settings, *, state_dim, horizon, input_dim, tracked_dim
    ):
        capacity = options["capacity"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("live_trace.capacity must be a positive integer")
        self.directory = Path(options["directory"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings = dict(settings)
        self.state_dim = state_dim
        self.count = self.dropped = 0
        self.tick_count = self.dropped_ticks = 0
        self.capacity = capacity
        self.saved_path = None
        # Touch all storage at startup, not on the real-time control path.
        self.arrays = {
            "observation": np.zeros((capacity, state_dim), np.float32),
            "measured_observation": np.zeros((capacity, state_dim), np.float32),
            "heading_world": np.full(capacity, np.nan),
            "measured_heading_world": np.full(capacity, np.nan),
            "plan_start_time": np.zeros(capacity),
            "prefix_controls": np.full(
                (capacity, horizon, input_dim), np.nan, np.float32
            ),
            "prefix_steps": np.zeros(capacity, np.int32),
            "preceding_command": np.full((capacity, input_dim), np.nan),
            "prediction_seconds": np.zeros(capacity),
            "reference": np.zeros((capacity, horizon, tracked_dim), np.float64),
            "obstacle_positions": np.full((capacity, horizon, 2), np.nan),
            "obstacle_present": np.zeros(capacity, bool),
            "obstacle_sample_time": np.full(capacity, np.nan),
            "obstacle_sample_position": np.full((capacity, 2), np.nan),
            "obstacle_velocity": np.full((capacity, 2), np.nan),
            "previous_solution": np.full((capacity, horizon, input_dim), np.nan),
            "has_previous": np.zeros(capacity, bool),
            "warm_start_steps": np.ones(capacity, np.int32),
            "rng_state": np.full(capacity, "", dtype="U256"),
            "optimized_controls": np.full((capacity, horizon, input_dim), np.nan),
            "command": np.zeros((capacity, input_dim), np.float32),
            "state_time": np.zeros(capacity),
            "compute_seconds": np.zeros(capacity),
            "cost": np.full(capacity, np.nan),
            "success": np.zeros(capacity, bool),
            "completed": np.zeros(capacity, bool),
            "deadline_rejected": np.zeros(capacity, bool),
            "action": np.full(capacity, self.ACTIONS["zero"], dtype=np.uint8),
            "iterations": np.full(capacity, -1, dtype=np.int32),
            "evaluations": np.full(capacity, -1, dtype=np.int32),
            "generation": np.zeros(capacity, dtype=np.int32),
            "plan_id": np.full(capacity, -1, dtype=np.int64),
            "query_wall_time": np.full(capacity, np.nan),
            "deadline_missed": np.zeros(capacity, bool),
            "latency_missed": np.zeros(capacity, bool),
            "request_latency_seconds": np.full(capacity, np.nan),
            "completed_wall_time": np.full(capacity, np.nan),
            "plan_accepted": np.zeros(capacity, bool),
            "handover_index": np.full(capacity, -1, dtype=np.int32),
            "handover_age": np.full(capacity, np.nan),
            "handover_reason": np.full(capacity, "pending", dtype="U32"),
            "estimated_prefix_max_abs_error": np.full((capacity, input_dim), np.nan),
            "prefix_known_fraction": np.full(capacity, np.nan),
        }
        if settings.get('solver', 'icem') != 'icem':
            raise ValueError('Only iCEM solve traces are supported')
        from .icem import ICEMConfig
        reuse = ICEMConfig(**settings.get('icem', {})).reuse_count
        self.arrays.update(
            has_sampler_memory=np.zeros(capacity, bool),
            sampler_mean=np.full((capacity, horizon, input_dim), np.nan, np.dtype(settings.get('dtype','float32'))),
            sampler_elites=np.full((capacity, reuse, horizon, input_dim), np.nan, np.dtype(settings.get('dtype','float32'))),
            warm_start_evaluated=np.zeros(capacity, bool),
            warm_start_feasible=np.zeros(capacity, bool),
            warm_start_selected=np.zeros(capacity, bool),
            warm_start_cost=np.full(capacity, np.nan),
            final_mean_cost=np.full(capacity, np.nan),
            best_sampled_cost=np.full(capacity, np.nan),
        )
        # Only the publisher writes these arrays; only the worker writes solve
        # rows until completion. Shutdown joins the worker before reading either.
        self.ticks = {
            "wall_time": np.zeros(capacity),
            "state_time": np.zeros(capacity),
            "application_time": np.zeros(capacity),
            "plan_state_time": np.full(capacity, np.nan),
            "plan_age": np.full(capacity, np.nan),
            "plan_wall_age": np.full(capacity, np.nan),
            "plan_index": np.full(capacity, -1, dtype=np.int32),
            "plan_id": np.full(capacity, -1, dtype=np.int64),
            "generation": np.zeros(capacity, dtype=np.int32),
            "solve_pending": np.zeros(capacity, bool),
            "command": np.zeros((capacity, input_dim), np.float32),
            "reason": np.full(capacity, "", dtype="U32"),
            "callback_seconds": np.zeros(capacity),
            "commitment_id": np.full(capacity, -1, np.int64),
            "commitment_index": np.full(capacity, -1, np.int32),
            "handover_time": np.full(capacity, np.nan),
        }

    def begin(
        self,
        solver,
        observation,
        reference,
        state_time,
        *,
        generation=0,
        plan_id=-1,
        query_wall_time=np.nan,
        plan_start_time=None,
        prefix_controls=None,
        preceding_command=None,
        heading_world=None,
        warm_start_steps=1,
        obstacle_positions=None,
        obstacle_sample_time=None,
        obstacle_sample_position=None,
        obstacle_velocity=None,
    ):
        if self.count == self.capacity:
            self.dropped += 1
            return None
        row = self.count
        rng_state = json.dumps(solver.rng.bit_generator.state)
        if len(rng_state) > 256:
            raise ValueError("RNG state is too large for the live trace format")
        a = self.arrays
        a["observation"][row] = observation
        a["measured_observation"][row] = observation
        if heading_world is not None:
            a["heading_world"][row] = a["measured_heading_world"][row] = heading_world
        a["plan_start_time"][row] = (
            state_time if plan_start_time is None else plan_start_time
        )
        if prefix_controls is not None:
            a["prefix_steps"][row] = len(prefix_controls)
            a["prefix_controls"][row, : len(prefix_controls)] = prefix_controls
        if preceding_command is not None:
            a["preceding_command"][row] = preceding_command
        a["reference"][row] = reference
        if obstacle_positions is not None:
            a['obstacle_positions'][row] = obstacle_positions
            a['obstacle_present'][row] = True
            for key, value in (('obstacle_sample_time', obstacle_sample_time),
                               ('obstacle_sample_position', obstacle_sample_position),
                               ('obstacle_velocity', obstacle_velocity)):
                if value is not None:
                    a[key][row] = value
        a["state_time"][row] = state_time
        a['warm_start_steps'][row] = warm_start_steps
        a["generation"][row] = generation
        a["plan_id"][row] = plan_id
        a["query_wall_time"][row] = query_wall_time
        a["rng_state"][row] = rng_state
        if solver.previous_solution is not None:
            a["has_previous"][row] = True
            a["previous_solution"][row] = solver.previous_solution
        if 'has_sampler_memory' in a:
            state = solver.optimizer_state()
            if state is not None:
                a['has_sampler_memory'][row] = True
                a['sampler_mean'][row], a['sampler_elites'][row] = state
        self.count += 1
        return row

    def projected(self, row, observation, *, heading_world=None):
        if row is not None:
            self.arrays["observation"][row] = observation
            if heading_world is not None:
                self.arrays["heading_world"][row] = heading_world

    def finish(self, row, result, controls, command, compute_seconds):
        if row is None:
            return
        a = self.arrays
        if controls is not None:
            a["optimized_controls"][row] = controls
        a["command"][row] = command
        a["compute_seconds"][row] = compute_seconds
        a["prediction_seconds"][row] = result.get("prediction_seconds", 0.0)
        a["cost"][row] = result.get("fun", np.nan)
        if 'warm_start_cost' in a:
            for name in ('warm_start_evaluated', 'warm_start_selected', 'warm_start_feasible'):
                a[name][row] = result.get(name, False)
            for name in ('warm_start_cost', 'final_mean_cost', 'best_sampled_cost'):
                value = result.get(name)
                a[name][row] = np.nan if value is None else value
        a["success"][row] = bool(result.success)
        a["deadline_rejected"][row] = result.deadline_rejected
        a["deadline_missed"][row] = result.get("deadline_missed", False)
        a["latency_missed"][row] = result.get("latency_missed", False)
        a["request_latency_seconds"][row] = result.get(
            "request_latency_seconds", np.nan
        )
        a["completed_wall_time"][row] = result.get("completed_wall_time", np.nan)
        a["action"][row] = self.ACTIONS[result.command_source]
        a["iterations"][row] = result.get("nit", -1)
        a["evaluations"][row] = result.get("nfev", -1)
        a["completed"][row] = True

    def handover(
        self,
        row,
        *,
        accepted,
        command,
        index,
        age,
        reason,
        prefix_error=None,
        prefix_known_fraction=np.nan,
    ):
        if row is None:
            return
        self.arrays["plan_accepted"][row] = accepted
        self.arrays["handover_index"][row] = index
        self.arrays["handover_age"][row] = age
        self.arrays["handover_reason"][row] = reason
        self.arrays["deadline_rejected"][row] = (
            self.arrays["success"][row]
            and not accepted
            and reason not in ("reset_discard", "shutdown")
        )
        if prefix_error is not None:
            self.arrays["estimated_prefix_max_abs_error"][row] = prefix_error
        self.arrays["prefix_known_fraction"][row] = prefix_known_fraction
        self.arrays["command"][row] = command
        self.arrays["action"][row] = self.ACTIONS["buffered" if accepted else "zero"]

    def record_command(
        self,
        *,
        now,
        state_time,
        application_time,
        plan,
        index,
        age,
        generation,
        pending,
        command,
        reason,
        callback_seconds=0.0,
        commitment=None,
    ):
        if self.tick_count == self.capacity:
            self.dropped_ticks += 1
            return
        row = self.tick_count
        a = self.ticks
        a["wall_time"][row] = now
        a["state_time"][row] = state_time
        a["application_time"][row] = application_time
        a["plan_age"][row] = age
        a["plan_index"][row] = index
        a["generation"][row] = generation
        a["solve_pending"][row] = pending
        a["command"][row] = command
        a["reason"][row] = reason
        a["callback_seconds"][row] = callback_seconds
        if plan is not None:
            a["plan_id"][row] = plan.request.plan_id
            a["plan_state_time"][row] = plan.request.control_start_time
            a["plan_wall_age"][row] = now - plan.request.query_wall_time
        if commitment is not None:
            a["commitment_id"][row] = commitment.plan_id
            a["commitment_index"][row] = int(
                np.floor(
                    (application_time - commitment.state_time + 1e-9)
                    / (
                        (commitment.control_start_time - commitment.state_time)
                        / len(commitment.prefix_controls)
                    )
                )
            )
            a["handover_time"][row] = commitment.control_start_time
        self.tick_count += 1

    def close(self):
        if self.saved_path is not None or (self.count == 0 and self.tick_count == 0):
            return self.saved_path
        metadata = dict(
            format_version=self.FORMAT_VERSION,
            settings=self.settings,
            state_dim=self.state_dim,
            tracking_features=TRACKING_LABELS if self.state_dim == 103 else None,
            heading_representation="sine_cosine" if self.state_dim == 103 else "generic",
            heading_cost="sincos_chordal" if self.state_dim == 103 else "generic",
            dropped=self.dropped,
            dropped_ticks=self.dropped_ticks,
            actions=self.ACTIONS,
        )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="mpc_trace_",
            suffix=".npz",
            dir=self.directory,
            delete=False,
        ) as stream:
            np.savez(
                stream,
                metadata=np.asarray(json.dumps(metadata)),
                **{key: value[: self.count] for key, value in self.arrays.items()},
                **{
                    f"tick_{key}": value[: self.tick_count]
                    for key, value in self.ticks.items()
                },
            )
            self.saved_path = Path(stream.name)
        return self.saved_path

"""Timestamped execution of existing MPC sequences, outside the optimizer.

One worker owns the solver, RNG, history and solve-trace rows. The publisher
only polls completed futures and reads immutable plan copies; it never waits
for a solve. Simulation time chooses the ZOH slot, while monotonic wall time
also bounds freshness when simulation feedback stalls.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Full, Queue
from threading import Thread
from time import perf_counter

import numpy as np
from kcf_mpc.mpc.icem import ICEMResult
from kcf_mpc.mpc.mpc_handover import predict_handover_state


@dataclass(frozen=True)
class SolveRequest:
    generation: int
    plan_id: int
    state_time: float
    query_wall_time: float
    observation: np.ndarray
    reference: np.ndarray
    plan_start_time: float | None = None
    prefix_controls: np.ndarray | None = None
    previous_command: np.ndarray | None = None
    heading_world: float | None = None
    obstacle_positions: np.ndarray | None = None  # world XY centers at the solve horizon's times
    obstacle_sample_time: float | None = None
    obstacle_sample_position: np.ndarray | None = None
    obstacle_velocity: np.ndarray | None = None

    @property
    def control_start_time(self):
        return self.state_time if self.plan_start_time is None else self.plan_start_time

    @property
    def preceding_command(self):
        """Last committed slot before handover; previous command only for immediate solves."""
        if self.prefix_controls is not None and len(self.prefix_controls):
            return self.prefix_controls[-1]
        return self.previous_command


@dataclass(frozen=True)
class SolveOutcome:
    request: SolveRequest
    controls: np.ndarray | None
    result: ICEMResult
    compute_seconds: float
    optimizer_seconds: float
    trace_row: int | None


class BufferedPlan:
    def __init__(self, outcome):
        self.request = outcome.request
        self.controls = outcome.controls.copy()
        self.controls.flags.writeable = False

    def select(self, application_time, now, dt, max_age):
        age = float(application_time - self.request.control_start_time)
        wall_age = float(now - self.request.query_wall_time)
        reservation = self.request.control_start_time - self.request.state_time
        # Round only sub-nanosecond boundary error from simulation integration.
        eps = 1e-9
        index = int(np.floor((age + eps) / dt))
        if age < -eps or wall_age < -eps:
            return None, index, age, "future"
        if age >= max_age - eps or wall_age >= reservation + max_age - eps:
            return None, index, age, "expired"
        if not 0 <= index < len(self.controls):
            return None, index, age, "exhausted"
        return (
            self.controls[index].astype(np.float32, copy=True),
            index,
            age,
            "buffered",
        )


class SingleSolveWorker:
    """No job backlog, no concurrent solves, and no publisher-thread RNG access."""

    def __init__(self, solver, *, trace=None, deadline=0.02, clock=perf_counter):
        self.solver, self.trace, self.deadline, self.clock = (
            solver,
            trace,
            deadline,
            clock,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kcf-mpc-solve"
        )
        self._future = None
        self._reset_next = False
        self._last_control_start_time = None
        self._closed = False

    @property
    def busy(self):
        # Even a completed result must be consumed before another job starts.
        return self._future is not None

    def invalidate(self):
        # Never reset solver history concurrently with an in-flight solve.
        self._reset_next = True

    def submit(self, request):
        if self._closed or self.busy:
            return False
        reset = self._reset_next
        self._reset_next = False
        self._future = self._executor.submit(self.solve_inline, request, reset)
        return True

    def poll(self):
        if self._future is None or not self._future.done():
            return None
        future, self._future = self._future, None
        if future.cancelled():
            return None
        return future.result()  # Already completed: never waits for optimization.

    def solve_inline(self, request, reset=None):
        if reset is None:  # Synchronous/offline adapter, with no concurrent job.
            reset, self._reset_next = self._reset_next, False
        if reset:
            self.solver.reset()
            self._last_control_start_time = None
        began = self.clock()
        controls = None
        row = None
        optimizer_began = None
        optimizer_seconds = 0.0
        prediction_seconds = 0.0
        try:
            warm_steps = 1
            if getattr(self.solver, 'solver', None) == 'icem':
                if (self._last_control_start_time is not None
                        and self.solver.previous_solution is not None):
                    elapsed = request.control_start_time - self._last_control_start_time
                    if elapsed < -1e-8:
                        self.solver.reset()
                        self._last_control_start_time = None
                    else:
                        warm_steps = max(0, int(round(elapsed / self.solver.prediction_sample_period)))
            preceding = request.preceding_command
            preceding = None if preceding is None else np.array(preceding, copy=True)
            heading = request.heading_world
            if self.trace is not None:
                trace_arguments = {} if heading is None else {"heading_world": heading}
                if request.obstacle_positions is not None:
                    trace_arguments['obstacle_positions'] = request.obstacle_positions
                    for key in ('obstacle_sample_time', 'obstacle_sample_position', 'obstacle_velocity'):
                        value = getattr(request, key)
                        if value is not None:
                            trace_arguments[key] = value
                row = self.trace.begin(
                    self.solver,
                    request.observation,
                    request.reference,
                    request.state_time,
                    generation=request.generation,
                    plan_id=request.plan_id,
                    query_wall_time=request.query_wall_time,
                    plan_start_time=request.control_start_time,
                    prefix_controls=request.prefix_controls,
                    preceding_command=preceding,
                    warm_start_steps=warm_steps,
                    **trace_arguments,
                )
            observation = request.observation.copy()
            if request.prefix_controls is not None:
                prediction_began = self.clock()
                if (
                    heading is not None
                    and getattr(self.solver, "projected_g1", False) is True
                ):
                    observation, heading = predict_handover_state(
                        self.solver,
                        observation,
                        request.prefix_controls,
                        heading_world=heading,
                        return_heading_world=True,
                    )
                else:
                    observation = predict_handover_state(
                        self.solver, observation, request.prefix_controls
                    )
                observation = np.asarray(observation, dtype=request.observation.dtype)
                prediction_seconds = self.clock() - prediction_began
                if self.trace is not None:
                    trace_arguments = (
                        {} if heading is None else {"heading_world": heading}
                    )
                    self.trace.projected(row, observation, **trace_arguments)
            optimizer_began = self.clock()
            arguments = {} if preceding is None else {"preceding_command": preceding}
            if heading is not None:
                arguments["heading_world"] = heading
            if request.obstacle_positions is not None:
                arguments['obstacle_positions'] = request.obstacle_positions.copy()
            if getattr(self.solver, 'solver', None) == 'icem':
                arguments['warm_start_steps'] = warm_steps
            controls, result = self.solver.solve(
                observation, request.reference.copy(), **arguments
            )
            optimizer_seconds = self.clock() - optimizer_began
            controls = np.asarray(controls, dtype=np.float64)
            expected = (self.solver.horizon, self.solver.input_dim)
            if (
                controls.shape != expected
                or not np.isfinite(controls).all()
                or np.any(controls < self.solver.lower_bounds - 1e-7)
                or np.any(controls > self.solver.upper_bounds + 1e-7)
            ):
                raise RuntimeError("KCF MPC returned an invalid or out-of-bounds plan")
            if not result.success:
                self.solver.reset()
                self._last_control_start_time = None
            else:
                self._last_control_start_time = request.control_start_time
        except Exception as error:
            # Unexpected worker errors must also surface as a failed solve,
            # rather than repeatedly escaping the publisher and leaving a command alive.
            if optimizer_began is not None:
                optimizer_seconds = self.clock() - optimizer_began
            result = ICEMResult(
                success=False, message=f"{type(error).__name__}: {error}"
            )
            controls = None
            self.solver.reset()
            self._last_control_start_time = None
        finished = self.clock()
        elapsed = finished - began
        result.request_latency_seconds = max(0.0, finished - request.query_wall_time)
        result.latency_missed = bool(
            self.deadline is not None and result.request_latency_seconds > self.deadline
        )
        result.completed_wall_time = finished
        result.prediction_seconds = prediction_seconds
        result.deadline_missed = bool(
            self.deadline is not None and elapsed > self.deadline
        )
        result.deadline_rejected = (
            False  # Missing the budget alone no longer discards a usable plan.
        )
        result.command_accepted = (
            False  # Annotated when the publisher consumes the result.
        )
        result.command_source = "pending" if result.success else "zero"
        if self.trace is not None:
            self.trace.finish(
                row, result, controls, np.zeros(self.solver.input_dim), elapsed
            )
        return SolveOutcome(request, controls, result, elapsed, optimizer_seconds, row)

    def close(self):
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


class AsyncConsole:
    """Keep potentially blocking terminal/tee writes off the command publisher."""

    def __init__(self, capacity=128):
        self.queue = Queue(maxsize=capacity)
        self.dropped = 0
        self._closed = False
        self._thread = Thread(target=self._run, name="kcf-mpc-log", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            message = self.queue.get()
            try:
                if message is None:
                    return
                try:
                    print(message, flush=True)
                except (BrokenPipeError, OSError):
                    pass
            finally:
                self.queue.task_done()

    def write(self, message):
        if self._closed:
            return
        try:
            self.queue.put_nowait(message)
        except Full:
            self.dropped += 1

    def close(self):
        if not self._closed:
            self._closed = True
            # Shutdown only: no more commands will be published.
            self.queue.put(None)
            self._thread.join()

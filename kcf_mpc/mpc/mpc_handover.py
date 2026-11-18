"""Delay alignment around the selected predictor, not a new MPC objective."""

from collections import deque
import math

import numpy as np


class HandoverTiming:
    """Reserve a future policy tick before solving; never redate a result."""

    def __init__(self, options, *, dt, application_lead, horizon):
        self.dt, self.application_lead = float(dt), application_lead
        for key in (
            "initial_steps",
            "minimum_steps",
            "maximum_steps",
            "latency_window",
        ):
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"live_handover.{key} must be a positive integer")
        self.initial = options["initial_steps"]
        self.minimum, self.maximum = options["minimum_steps"], options["maximum_steps"]
        self.margin = float(options["margin_seconds"])
        if (
            not self.application_lead
            < self.minimum
            <= self.initial
            <= self.maximum
            <= horizon
        ):
            raise ValueError(
                "live_handover steps must exceed application lead and fit the horizon"
            )
        if not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError(
                "live_handover.margin_seconds must be finite and nonnegative"
            )
        self.latencies = deque(maxlen=options["latency_window"])

    def steps(self):
        if not self.latencies:
            return self.initial
        ticks = math.ceil((max(self.latencies) + self.margin) / self.dt)
        return min(self.maximum, max(self.minimum, ticks + self.application_lead))

    def observe(self, outcome):
        latency = float(
            outcome.result.get("request_latency_seconds", outcome.compute_seconds)
        )
        if np.isfinite(latency) and latency >= 0:
            self.latencies.append(latency)

    def reset(self):
        self.latencies.clear()


def predict_handover_state(
    solver,
    observation,
    prefix_controls,
    *,
    heading_world=None,
    return_heading_world=False,
):
    """Propagate the measured state through committed inputs with this predictor.

    No optimizer/RNG/history call occurs here. The unchanged optimizer will lift
    the predicted physical state exactly as it lifts any other initial state.
    Only committed steps are predicted. All configured prefix lengths are
    compiled during controller startup, never first compiled on a live tick.
    """
    controls = np.asarray(prefix_controls)
    if (
        controls.ndim != 2
        or controls.shape[1] != solver.input_dim
        or not 0 < len(controls) <= solver.horizon
        or not np.isfinite(controls).all()
    ):
        raise ValueError("Invalid committed input prefix")
    to_prediction_frame = getattr(solver, "observation_to_prediction_frame", None)
    if to_prediction_frame is None:
        model_observation, fixed_frame = np.array(observation, copy=True), None
    else:
        arguments = {} if heading_world is None else {"heading_world": heading_world}
        model_observation, fixed_frame = to_prediction_frame(observation, **arguments)
    backend = solver._get_jax_backend()
    rollout = np.asarray(backend.rollout(backend.lift(model_observation), controls))
    predicted = rollout[-1]
    continuous_heading = None
    if return_heading_world:
        if fixed_frame is None:
            raise ValueError("Continuous heading requires the native fixed-frame predictor")
        # Continuous heading is diagnostic metadata, not an angular-state cost.
        angles = np.arctan2(rollout[:, 2], rollout[:, 3])
        relative_heading = np.unwrap(np.r_[0.0, angles])[-1]
        continuous_heading = float(fixed_frame.yaw_origin_world + relative_heading)
    from_prediction_frame = getattr(solver, "prediction_from_prediction_frame", None)
    if from_prediction_frame is not None:
        predicted = from_prediction_frame(predicted, fixed_frame)
    predicted = np.array(predicted, copy=True)
    if predicted.shape != np.shape(observation) or not np.isfinite(predicted).all():
        raise RuntimeError("Handover prediction is nonfinite or has the wrong shape")
    return (predicted, continuous_heading) if return_heading_world else predicted

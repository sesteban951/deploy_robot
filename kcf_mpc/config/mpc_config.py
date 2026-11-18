"""Controller, iCEM, execution and trajectory settings shared by KCF and SI.

Select ``reference_mode`` below for pose holding, figure-eight, waypoints,
or the sharp-turn maneuvers.
KCF uses the top-level cost gains; SI uses ``single_integrator_gains``.
"""

from pathlib import Path
from copy import deepcopy
import math
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MODES = (
    "fixed", "lissajous", "waypoints", "rounded_square", "hairpin", "slalom", "stop_turn_go"
)


def resolve_mpc_model_config(
    config: dict[str, Any], *, prediction_model: str | None = None
) -> dict[str, Any]:
    """Select a predictor and its gains before applying explicit run overrides.

    KCF keeps the top-level gains. SI uses its dedicated gain block when present.
    Saved, fully resolved run settings must not be passed through this again:
    they may contain intentional per-run gain overrides.
    """
    resolved = deepcopy(config)
    model = (
        resolved.get("prediction_model", "kcf")
        if prediction_model is None
        else prediction_model
    )
    if model not in ("kcf", "single_integrator"):
        raise ValueError("prediction_model must be 'kcf' or 'single_integrator'")
    resolved["prediction_model"] = model
    if model == "single_integrator" and "single_integrator_gains" in resolved:
        gains = resolved["single_integrator_gains"]
        keys = {
            "state_weights", "terminal_state_weights", "input_weight", "command_change_weights"
        }
        if not isinstance(gains, dict) or set(gains) != keys:
            raise ValueError(
                "single_integrator_gains must contain only state_weights, "
                "terminal_state_weights, input_weight and command_change_weights"
            )
        resolved.update(deepcopy(gains))
    return resolved


def resolve_mpc_horizon_config(
    config: dict[str, Any], *, control_dt: float = 0.02
) -> dict[str, Any]:
    resolved = deepcopy(config)
    horizon = resolved["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    runtime = resolved.get("jax", {})
    if "scan_unroll" in runtime:
        unroll = runtime["scan_unroll"]
        if isinstance(unroll, bool) or not isinstance(unroll, int) or unroll < 1:
            raise ValueError("JAX scan_unroll must be a positive integer")
        runtime["scan_unroll"] = min(unroll, horizon)
    if "live_plan_max_age" in resolved:
        age = resolved["live_plan_max_age"]
        if not math.isfinite(age) or age <= 0:
            raise ValueError("live_plan_max_age must be finite and positive")
        resolved["live_plan_max_age"] = min(age, horizon * control_dt)
    if "live_handover" in resolved:
        handover = resolved["live_handover"]
        lead = resolved["command_application_lead_steps"]
        counts = {
            "command_application_lead_steps": lead,
            **{
                key: handover[key]
                for key in ("initial_steps", "minimum_steps", "maximum_steps")
            },
        }
        for key, value in counts.items():
            minimum = 0 if key == "command_application_lead_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if (
            not lead
            < handover["minimum_steps"]
            <= handover["initial_steps"]
            <= handover["maximum_steps"]
        ):
            raise ValueError(
                "live_handover must satisfy application lead < minimum <= initial <= maximum"
            )
        # A replacement is requested after an application slot has advanced:
        # leave that lead out of the old plan's available reservation capacity.
        capacity = horizon - lead
        if capacity < handover["minimum_steps"]:
            raise ValueError(
                "horizon is too short for the live handover minimum and application lead"
            )
        handover["maximum_steps"] = min(handover["maximum_steps"], capacity)
        handover["initial_steps"] = min(
            handover["initial_steps"], handover["maximum_steps"]
        )
    return resolved


def get_default_mpc_config(
    *, horizon: int | None = None, prediction_model: str | None = None
) -> dict[str, Any]:
    settings = {
        # Saved Trained Model checkpoint
        "results_dir": str(PROJECT_ROOT / "policy" / "g1" / "kcf" / "walking"),
        "checkpoint": "final",
        "device": "cpu",  # Checkpoint loading; JAX execution device is selected below.
        "backend": "jax",
        "solver": "icem",
        "icem": {
            "population": 42,
            "iterations": 2,
            "elites": 10,
            "decay": 6.0,
            "noise_beta": 3.0,
            "initial_std": 0.25,
            "warm_std": 0.1,
            "reference_initialization": True,
            "momentum": 0.1,
            "reuse_fraction": 0.3,
            # Body command rates: m/s², m/s², rad/s². Enforced inside iCEM.
            "command_rate_limits": [1.5, 1.5, 2.0],
        },
        "jax": {
            "platform": "gpu",
            "preallocate": False,
            "compilation_cache_dir": str(
                PROJECT_ROOT / ".cache" / "kcf" / "jax"
            ),
            "pinned_host_output": True,
            # Preferred compiler unrolling; capped to the selected horizon below.
            "scan_unroll": 20,
            "compiler_options": {
                "xla_gpu_enable_command_buffer": "FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE",
                "xla_gpu_graph_min_graph_size": 1,
                "xla_gpu_command_buffer_unroll_loops": True,
            },
        },
        "dtype": "float32",
        "num_threads": 4,
        "cpu_affinity": (0, 2, 4, 6),
        "warmup": True,
        "freeze_gc_after_warmup": True,
        "wait_for_first_state": True,
        "live_command_deadline": 0.02,  # Solve-budget telemetry
        # Execution scheduling only; these are NOT optimization constraints.
        "live_plan_max_age": 0.40,  # Capped to horizon x 20 ms below.
        "live_feedback_timeout": 0.10,
        # simulation.py sends state before polling commands: expect next-tick application.
        "command_application_lead_steps": 1,
        "live_handover": {
            "initial_steps": 2,  # Selected iCEM startup reservation.
            "minimum_steps": 2,
            "maximum_steps": 8,
            "latency_window": 8,
            "margin_seconds": 0.002,
        },
        "live_trace": {
            "enabled": True,
            "capacity": 15000,  # First five minutes at 50 Hz; bounded memory.
            "directory": str(PROJECT_ROOT / "logs" / "kcf" / "live_traces"),
        },
        # Receding-horizon problem
        "horizon": 20,
        "tracking_feature_indices": tuple(range(10)),
        # KCF gains; SI selects the independent block below.
        # [px, py, sin(theta), cos(theta), vx_body, vy_body, wz_body, gravity_xyz].
        "state_weights": (200.0, 200.0, 20.0, 20.0, 5.0, 5.0, 5.0, 100.0, 100.0, 20.0),
        "terminal_state_weights": (
            600.0,
            600.0,
            200.0,
            200.0,
            10.0,
            10.0,
            10.0,
            300.0,
            300.0,
            60.0,
        ),
        "input_weight": 0.2,  # S smooths command changes.
        "prediction_model": "kcf",
        "prediction_sample_period": 0.02,
        # S = diag(s_vx, s_vy, s_wz).
        "command_change_weights": (10.0, 10.0, 10.0),
        # Q/Q_terminal order: [px, py, sin(theta), cos(theta), vx, vy, wz, gx, gy, gz].
        "single_integrator_gains": {
            "state_weights": (150.0, 150.0, 40.0, 40.0, 5.0, 5.0, 5.0, 100.0, 100.0, 20.0),
            "terminal_state_weights": (
                450.0, 450.0, 300.0, 300.0, 10.0, 10.0, 10.0, 300.0, 300.0, 60.0,
            ),
            "input_weight": 0.2,
            "command_change_weights": (10.0, 10.0, 10.0),
        },
        "reference_mode": "fixed",  # fixed waypoints lissajous hairpin slalom stop_turn_go rounded_square
        # Public target: [px, py, yaw, vx_body, vy_body, wz_body, gravity_body_xyz].
        "reference": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        "reference_startup": {
            "hold_seconds": 0.5,
            "transition_seconds": 3.0,
            "maximum_linear_speed": 0.4,
            "maximum_yaw_rate": 0.6,
            # Bind to the first measured pose; leave None for live operation.
            "initial_pose": None,
        },
        "lissajous_reference": {
            "half_width": 1.5,
            "half_height": 0.75,
            "omega": 0.25,
        },
        # Smooth start/stop path. Repeated poses create standing dwells;
        # each pose is [world X (m), world Y (m), yaw (rad)].
        "waypoint_reference": {
            "times_s": [0.0, 2.0, 6.0, 8.0, 12.0, 14.0, 18.0, 20.0],
            "poses": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.6, 0.4, 0.4],
                [0.6, 0.4, 0.4],
                [0.0, 0.4, 0.0],
                [0.0, 0.4, 0.0],
            ],
        },
        # Continuous-forward paths: turn_radius is the MINIMUM instantaneous
        # radius, not a circular-arc radius. Peak yaw rate = speed / turn_radius.
        # Quintic heading ramps make yaw rate and acceleration zero at joins.
        "rounded_square_reference": {
            "start_pose": [0.0, 0.0, 0.0],
            "speed": 0.3,
            "turn_radius": 0.5,
            "straight_length": 1.0,
            "clockwise": False,
        },
        "hairpin_reference": {
            "start_pose": [0.0, 0.0, 0.0],
            "speed": 0.3,
            "turn_radius": 0.5,
            "straight_length": 1.2,
            "clockwise": False,
        },
        "slalom_reference": {
            "start_pose": [0.0, 0.0, 0.0],
            "speed": 0.3,
            "turn_radius": 0.5,
            "straight_length": 0.5,
            "heading_amplitude": math.pi / 4.0,
            "clockwise": False,
        },
        # Translate, stand, turn in place, stand, then resume. Radians for turns.
        # speed and yaw_rate are peak reference values for the quintic segments.
        "stop_turn_go_reference": {
            "start_pose": [0.0, 0.0, 0.0],
            "speed": 0.3,
            "distance": 0.6,
            "yaw_rate": 0.6,
            "dwell_seconds": 1.0,
            "turn_angles": [math.pi / 2.0, -math.pi / 2.0, math.pi],
        },
        # Preserve the existing MPC constraints when changing the predictor.
        "input_lower_bounds": (
            -0.7433253526687622,
            -0.7437153458595276,
            -0.744255006313324,
        ),
        "input_upper_bounds": (
            0.7444727420806885,
            0.74131178855896,
            0.7436532974243164,
        ),
        "seed": 42,
    }
    if horizon is not None:
        settings["horizon"] = horizon
    return resolve_mpc_horizon_config(
        resolve_mpc_model_config(settings, prediction_model=prediction_model)
    )

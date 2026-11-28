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
    "fixed", "lissajous", "waypoints", "rounded_square", "hairpin", "slalom", "stop_turn_go", "star"
)


def resolve_mpc_model_config(
    config: dict[str, Any], *, prediction_model: str | None = None
) -> dict[str, Any]:
    """Select a predictor and its tuning before applying explicit run overrides.

    KCF keeps the top-level gains. SI uses its dedicated gain block when present.
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
        if (not isinstance(gains, dict) or not keys.issubset(gains)
                or set(gains) - keys - {"icem", "standing_hold"}):
            raise ValueError(
                "single_integrator_gains must contain only state_weights, "
                "terminal_state_weights, input_weight and command_change_weights, "
                "with optional iCEM sampling and standing-hold overrides"
            )
        resolved.update({key: deepcopy(gains[key]) for key in keys})
        resolved["icem"].update(deepcopy(gains.get("icem", {})))
        if resolved.get("standing_hold") is not None:
            resolved["standing_hold"].update(deepcopy(gains.get("standing_hold", {})))
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
            "iterations": 3,
            "elites": 10,
            "decay": 6.0,
            "noise_beta": 3.0,
            "initial_std": 0.25,
            "warm_std": 0.03,
            "reference_initialization": True,
            "momentum": 0.1,
            "reuse_fraction": 0.3,
            # Tuned body command rates: m/s², m/s², rad/s²; enforced inside iCEM.
            # These are controller choices, not limits inferred from training data.
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
        "state_weights": (400.0, 400.0, 180.0, 180.0, 5.0, 5.0, 5.0, 100.0, 100.0, 20.0),
        "terminal_state_weights": (
            1200.0,
            1200.0,
            1800.0,
            1800.0,
            10.0,
            10.0,
            15.0,
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
            "state_weights": (150.0, 150.0, 80.0, 80.0, 20.0, 20.0, 5.0, 100.0, 100.0, 20.0),
            "terminal_state_weights": (
                450.0, 450.0, 600.0, 600.0, 50.0, 50.0, 10.0, 300.0, 300.0, 60.0,
            ),
            "input_weight": 0.2,
            "command_change_weights": (10.0, 10.0, 10.0),
            "icem": {"iterations": 2, "warm_std": 0.1},
            "standing_hold": {"heading_gain": 1.5},
        },
        # Stop correcting small errors once a stationary goal has settled.
        "standing_hold": {
            "enter_position_m": 0.025,
            "exit_position_m": 0.045,
            "enter_heading_rad": math.radians(2.0),
            "exit_heading_rad": math.radians(3.0),
            "enter_linear_speed_mps": 0.10,
            "enter_yaw_rate_rps": 0.20,
            "settle_seconds": 0.02,
            # Correct residual yaw during long holds without restarting the gait.
            "heading_gain": 3.0,
            "yaw_rate_damping": 0.2,
        },
        "reference_mode": "star",  # See REFERENCE_MODES above.
        # World-frame room dimensions. Keep pelvis references 0.5 m inside
        # the walls; this validates paths, not the robot's physical footprint.
        "reference_workspace": {
            "center_xy": [1.0, 0.0],
            "size_xy": [2.0, 2.0],
            "margin": 0.0,
        },
        # Public target: [px, py, yaw, vx_body, vy_body, wz_body, gravity_body_xyz].
        "reference": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        # Stand at the final pose this long, then track -> control (zero twist).
        "goal_hold_seconds": 10.0,
        "reference_startup": {
            "hold_seconds": 0.5,
            "transition_seconds": 3.0,
            "maximum_linear_speed": 0.4,
            "maximum_yaw_rate": 0.6,
            # Bind to the first measured pose; leave None for live operation.
            "initial_pose": None,
            # Walk to the start and dwell before the task: fixed starts here, paths at
            # their own first pose. None skips staging.
            "start_pose": (0.0, 0.0, 0.0),
            "start_dwell_seconds": 10.0,
        },
        # Repeating paths run `cycles` laps, then ease to a stop over stop_seconds.
        "lissajous_reference": {
            "center_xy": [1.0, 0.0],
            "half_width": 0.9,
            "half_height": 0.45,
            "omega": 0.20, # peak yaw rate 0.63 rad/s, independent of size
            "cycles": 2,
            "stop_seconds": 2.0,
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
            "start_pose": [0.9, -0.88, 0.0],
            "speed": 0.27,
            "turn_radius": 0.45,
            "straight_length": 0.2,
            "clockwise": False,
            "cycles": 2,
            "stop_seconds": 2.0,
        },
        "hairpin_reference": {
            "start_pose": [0.8, -0.36, 0.0],
            "speed": 0.18,
            "turn_radius": 0.3,
            "straight_length": 0.4,
            "clockwise": False,
            "cycles": 2,
            "stop_seconds": 2.0,
        },
        "slalom_reference": {
            "start_pose": [0.13, -0.23, 0.0],
            "speed": 0.15,
            "turn_radius": 0.25,
            "straight_length": 0.2,
            "heading_amplitude": math.pi / 4.0,
            "clockwise": False,
            # A repeating slalom advances forever; finish and hold instead.
            "cycles": 1,
            "stop_seconds": 1.0,
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
        # Five crossing legs with separate stand/pivot phases at the tips.
        "star_reference": {
            "center_xy": [1.0, 0.0],
            "radius": 0.9,
            "rotation": math.pi / 2.0,
            "speed": 0.3,
            "yaw_rate": 0.6,
            "dwell_seconds": 0.5,
            "clockwise": False,
        },
        # Physical body commands: walking epoch-3240 training split 0.5/99.5
        # Collection limits are +/-[0.5 m/s, 0.5 m/s, 1.0 rad/s].
        # The checkpoint applies its own normalization
        "input_lower_bounds": (
            -0.459801,
            -0.462779,
            -0.944198,
        ),
        "input_upper_bounds": (
            0.487012,
            0.461249,
            0.955684,
        ),
        "seed": 42,
    }
    if horizon is not None:
        settings["horizon"] = horizon
    return resolve_mpc_horizon_config(
        resolve_mpc_model_config(settings, prediction_model=prediction_model)
    )

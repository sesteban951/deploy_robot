"""G1 full-state KCF with physical projection and corrected-state re-lifting."""

import math
import torch

from kcf_mpc.models.model import ControlledLiftedModel
from kcf_mpc.models.kinematics import project_state


def is_projected_g1(model):
    return (
        model is not None
        and model.config.state_dim == 104
        and model.config.input_dim == 3
        and model.config.L_embed == 0
        and hasattr(model, "projection_balance")
    )


class ProjectedG1KCF(ControlledLiftedModel):
    """Hybrid projected Koopman model; each next lift uses the corrected state."""

    def __init__(self, config, normalization=None, *, dt=0.02, projection_balance=1.0):
        super().__init__(config, normalization)
        if config.state_dim != 104 or config.input_dim != 3 or config.L_embed != 0:
            raise ValueError("Projected G1 requires 104 states, 3 inputs, and L_embed=0")
        if (
            not math.isfinite(dt)
            or dt <= 0
            or not math.isfinite(projection_balance)
            or projection_balance <= 0
        ):
            raise ValueError("projection dt and balance must be finite and positive")
        self.dt, self.projection_balance = float(dt), float(projection_balance)

    def rollout_details(self, initial_observation, controls):
        if (
            initial_observation.ndim != 2
            or initial_observation.shape[-1] != 104
            or controls.ndim != 3
            or controls.shape[-1] != 3
            or controls.shape[0] != initial_observation.shape[0]
            or controls.shape[1] < 1
        ):
            raise ValueError("Expected observations (B,104) and controls (B,H,3), H>=1")
        previous = initial_observation
        lifted = self.lift(previous)
        raw_states, corrected_states, raw_lifts, corrected_lifts = [], [], [], []
        # Evaluate transitions step by step to bound memory at H=100; storing
        # every (B,H,nH,nH) transition matrix at once is unnecessary.
        for control in controls.unbind(1):
            raw_lift = torch.bmm(
                self.transition(control), lifted.unsqueeze(-1)
            ).squeeze(-1)
            raw = self.state(raw_lift)
            corrected = project_state(
                previous,
                raw,
                self.normalizer.state_scale,
                dt=self.dt,
                balance=self.projection_balance,
            )
            lifted = self.lift(corrected)
            raw_states.append(raw)
            corrected_states.append(corrected)
            raw_lifts.append(raw_lift)
            corrected_lifts.append(lifted)
            previous = corrected
        return {
            "raw_states": torch.stack(raw_states, 1),
            "states": torch.stack(corrected_states, 1),
            "raw_lifted": torch.stack(raw_lifts, 1),
            "lifted": torch.stack(corrected_lifts, 1),
        }

    def rollout(self, initial_observation, controls):
        return self.rollout_details(initial_observation, controls)["lifted"]

    def forward(self, observation, control):
        return self.rollout(observation, control.unsqueeze(1))[:, 0]

    def step_lifted(self, lifted_state, control):
        return self.forward(self.state(lifted_state), control)

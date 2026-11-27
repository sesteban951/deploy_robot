"""Load an explicitly selected native-104 projected checkpoint for inference."""

import json
from pathlib import Path
from typing import Any
import torch
import numpy as np
from kcf_mpc.config.model_config import model_config_from_mapping
from kcf_mpc.models.projected import ProjectedG1KCF

def load_saved_model(
    results_dir: str | Path,
    device: torch.device,
    checkpoint_choice: str = "final",
) -> tuple[ProjectedG1KCF, dict[str, Any], dict[str, Any]]:
    """Load a trained model and its saved training configuration."""
    results_dir = Path(results_dir)
    config_path = results_dir / "config.json"
    checkpoint_names = {
        "best": "checkpoint_best.pt",
        "last": "checkpoint_last.pt",
        "final": "model_final.pt",
    }

    if checkpoint_choice not in checkpoint_names:
        raise ValueError(
            f"unsupported checkpoint choice {checkpoint_choice!r}"
        )

    checkpoint_path = results_dir / checkpoint_names[checkpoint_choice]
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{results_dir} must contain config.json and "
            f"{checkpoint_path.name}"
        )

    with config_path.open(encoding="utf-8") as stream:
        train_config = json.load(stream)

    if train_config.get("model_type") != "projected_g1_kcf_v1":
        raise ValueError("Only the native projected G1 checkpoint is supported")
    normalization_path = results_dir / "normalization.npz"
    normalization = None
    if normalization_path.is_file():
        with np.load(normalization_path, allow_pickle=False) as archive:
            normalization = {key: archive[key].copy() for key in archive.files}
    model = ProjectedG1KCF(model_config_from_mapping(train_config), normalization,
                          dt=train_config["sample_dt"],
                          projection_balance=train_config["projection_balance"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    checkpoint_metadata = {
        "choice": checkpoint_choice,
        "filename": checkpoint_path.name,
        "selected_by": (
            checkpoint.get(
                "checkpoint_selection_criterion",
                "minimum_validation_loss_at_maximum_rollout_horizon",
            )
            if checkpoint_choice == "best"
            else "diagnostic_override"
        ),
        "epoch": checkpoint.get(
            "completed_epoch",
            checkpoint.get(
                "selected_epoch",
                checkpoint.get("final_epoch"),
            ),
        ),
        "rollout_horizon": checkpoint.get(
            "rollout_horizon",
            checkpoint.get(
                "selected_horizon",
                checkpoint.get("final_horizon"),
            ),
        ),
    }
    return model, train_config, checkpoint_metadata

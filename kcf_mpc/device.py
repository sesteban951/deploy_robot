from __future__ import annotations
import torch


def resolve_device(
    requested: str | torch.device | None = None,
) -> torch.device:
    """Resolve ``auto``, CPU, CUDA, or MPS to an available PyTorch device."""
    requested_name = "auto" if requested is None else str(requested).lower()

    if requested_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device

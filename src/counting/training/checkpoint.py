"""Checkpoint save/load with config snapshot for resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int,
    best_metric: float,
    config_snapshot: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler if scheduler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config_snapshot": dict(config_snapshot),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        "epoch": payload["epoch"],
        "best_metric": payload["best_metric"],
        "config_snapshot": payload.get("config_snapshot", {}),
        "scheduler": payload.get("scheduler"),
    }

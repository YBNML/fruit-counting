"""Training callbacks: LR schedule, early stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class CosineWithWarmup:
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr: float = 0.0

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.base_lr
            return self.base_lr * (step + 1) / (self.warmup_steps + 1)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class EarlyStopping:
    def __init__(self, *, patience: int, mode: Literal["min", "max"]) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0

    def update(self, value: float) -> bool:
        improved = (
            self.best is None
            or (self.mode == "min" and value < self.best)
            or (self.mode == "max" and value > self.best)
        )
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

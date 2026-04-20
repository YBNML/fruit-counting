"""Generic training loop scaffold with TensorBoard support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from counting.training.callbacks import CosineWithWarmup, EarlyStopping
from counting.training.checkpoint import save_checkpoint


log = logging.getLogger("counting.training")


@dataclass
class RunnerState:
    step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")


class Runner:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWithWarmup,
        early_stopping: EarlyStopping,
        device: str,
        run_dir: Path,
        log_every_n_steps: int,
        save_every_n_epochs: int,
        config_snapshot: dict[str, Any],
        tensorboard: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.log_every = log_every_n_steps
        self.save_every = save_every_n_epochs
        self.config_snapshot = config_snapshot
        self.state = RunnerState()
        self.tb = (
            SummaryWriter(self.run_dir / "tensorboard")
            if tensorboard
            else None
        )

    def _set_lr(self) -> float:
        lr = self.scheduler.lr_at(self.state.step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def train_epoch(
        self,
        train_loader: Iterable[Any],
        step_fn: Callable[[Any, torch.nn.Module], dict[str, torch.Tensor]],
    ) -> None:
        self.model.train()
        for batch in train_loader:
            lr = self._set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            losses = step_fn(batch, self.model)
            losses["total"].backward()
            self.optimizer.step()

            if self.tb is not None and self.state.step % self.log_every == 0:
                for k, v in losses.items():
                    self.tb.add_scalar(f"train/{k}", float(v.detach().cpu()), self.state.step)
                self.tb.add_scalar("train/lr", lr, self.state.step)
            self.state.step += 1

    @torch.no_grad()
    def validate(
        self,
        val_loader: Iterable[Any],
        eval_fn: Callable[[Any, torch.nn.Module], dict[str, float]],
    ) -> dict[str, float]:
        self.model.eval()
        agg: dict[str, float] = {}
        n = 0
        for batch in val_loader:
            metrics = eval_fn(batch, self.model)
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
        if n > 0:
            agg = {k: v / n for k, v in agg.items()}
        if self.tb is not None:
            for k, v in agg.items():
                self.tb.add_scalar(f"val/{k}", v, self.state.step)
        return agg

    def maybe_save(self, val_metrics: dict[str, float], metric_key: str) -> None:
        metric = val_metrics.get(metric_key, float("inf"))
        if metric < self.state.best_metric:
            self.state.best_metric = metric
            save_checkpoint(
                path=self.run_dir / "checkpoints" / "best.ckpt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.state.epoch,
                best_metric=self.state.best_metric,
                config_snapshot=self.config_snapshot,
            )
        if (self.state.epoch + 1) % self.save_every == 0:
            save_checkpoint(
                path=self.run_dir / "checkpoints" / "last.ckpt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.state.epoch,
                best_metric=self.state.best_metric,
                config_snapshot=self.config_snapshot,
            )

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()

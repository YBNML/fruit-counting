"""Training configuration schema (sibling to AppConfig)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["fsc147", "coco", "custom"] = "fsc147"
    root: str
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = Field(default=1024, gt=0, le=4096)


class TrainModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sam_checkpoint: str
    init_decoder: str = ""
    init_mlp: str = ""


class TrainCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    dir: str
    dtype: Literal["float16", "float32"] = "float16"
    augment_variants: int = Field(default=1, ge=1, le=16)


class TrainEarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patience: int = Field(default=5, ge=1)
    metric: str = "val_mae"
    mode: Literal["min", "max"] = "min"


class TrainLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(default=8, ge=1)
    epochs: int = Field(default=30, ge=1)
    lr: float = Field(default=1.0e-4, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    warmup_steps: int = Field(default=500, ge=0)
    scheduler: Literal["cosine", "constant"] = "cosine"
    loss_weights: dict[str, float] = Field(default_factory=lambda: {"cls": 1.0, "count": 0.1})
    early_stopping: TrainEarlyStoppingConfig = Field(default_factory=TrainEarlyStoppingConfig)


class TrainLoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tensorboard: bool = True
    log_every_n_steps: int = Field(default=20, ge=1)
    save_every_n_epochs: int = Field(default=1, ge=1)


class TrainAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 42
    output_dir: str = "./runs"
    data: TrainDataConfig
    model: TrainModelConfig
    cache: TrainCacheConfig
    train: TrainLoopConfig = Field(default_factory=TrainLoopConfig)
    logging: TrainLoggingConfig = Field(default_factory=TrainLoggingConfig)

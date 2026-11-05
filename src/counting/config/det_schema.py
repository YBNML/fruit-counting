"""Pydantic configuration schema for the Faster R-CNN detector (training and inference)."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt, field_validator, model_validator,
)

Crop = Literal["pear", "apple"]


class FrcnnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    arch: Literal["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn"] = "fasterrcnn_resnet50_fpn_v2"
    anchor_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [16, 32, 64, 128, 256], min_length=5, max_length=5
    )
    aspect_ratios: list[PositiveFloat] = Field(default_factory=lambda: [0.5, 1.0, 2.0], min_length=3, max_length=3)
    train_size: PositiveInt = 1024
    infer_min_size: PositiveInt = 1080
    infer_max_size: PositiveInt = 1920
    max_dets: PositiveInt = 1000
    score_min: float = Field(default=0.05, ge=0, le=1)
    rpn_pre_nms_train: PositiveInt = 2000
    rpn_post_nms_train: PositiveInt = 2000
    rpn_pre_nms_test: PositiveInt = 2000
    rpn_post_nms_test: PositiveInt = 2000
    box_batch_per_image: PositiveInt = 512
    trainable_backbone_layers: int = Field(default=3, ge=0, le=5)
    freeze_backbone_bn: bool = True

    @model_validator(mode="after")
    def _check_infer_size_order(self) -> FrcnnConfig:
        if self.infer_min_size > self.infer_max_size:
            raise ValueError("model.infer_min_size must be <= model.infer_max_size")
        return self


class DetDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    crops: list[Crop] = Field(default_factory=lambda: ["apple", "pear"], min_length=1)
    root: str = "~/datasets/orchard_synth/{crop}/train_s{seed}"
    seeds: list[NonNegativeInt] | None = None

    @field_validator("crops")
    @classmethod
    def _check_crops_distinct(cls, v: list[Crop]) -> list[Crop]:
        if len(set(v)) != len(v):
            raise ValueError(f"data.crops repeats a crop: {v}")
        return v

    @field_validator("root")
    @classmethod
    def _check_root_has_placeholders(cls, v: str) -> str:
        if "{crop}" not in v or "{seed}" not in v:
            raise ValueError(f"data.root needs the {{crop}} and {{seed}} fields, got {v}")
        try:
            v.format(crop="apple", seed=0)
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"data.root may use only the {{crop}} and {{seed}} fields, got {v}") from None
        return v


class AugConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hflip_p: float = Field(default=0.5, ge=0, le=1)
    brightness: float = Field(default=0.2, ge=0)
    contrast: float = Field(default=0.2, ge=0)
    saturation: float = Field(default=0.2, ge=0)
    hue: float = Field(default=0.02, ge=0, le=0.5)


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    epochs: int = Field(default=12, ge=1)
    batch_size: int = Field(default=2, ge=1)
    lr: float | None = Field(default=None, gt=0)
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = Field(default=500, ge=0)
    min_lr: float = 0.0
    clip_grad_norm: float | None = 10.0

    def base_lr(self) -> float:
        return self.lr if self.lr is not None else 0.02 * self.batch_size / 16


class DetTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run_name: str = "frcnn_synth"
    model: FrcnnConfig = Field(default_factory=FrcnnConfig)
    data: DetDataConfig = Field(default_factory=DetDataConfig)
    aug: AugConfig = Field(default_factory=AugConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    pretrained: bool = True
    seed: NonNegativeInt = 0
    num_workers: int = Field(default=4, ge=0)
    log_every: int = Field(default=20, ge=1)
    device: str = "cuda"
    output_dir: str = "runs/train"


class DetInferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: FrcnnConfig = Field(default_factory=FrcnnConfig)
    device: str = "cuda"
    output_dir: str = "runs/detect"
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


# Model settings that change what a set of weights detects; the inference config must match the training one.
INFER_KEYS = ("arch", "anchor_sizes", "aspect_ratios", "freeze_backbone_bn")


def model_mismatch(a: FrcnnConfig, b: FrcnnConfig) -> list[str]:
    return [k for k in INFER_KEYS if getattr(a, k) != getattr(b, k)]

"""Pydantic configuration schema for the counting pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DeblurStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    weights: str = ""


class PseCoStageConfig(BaseModel):
    """Counting stage (PseCo).

    Note: the default ``enabled=True`` requires all three checkpoint paths to be
    non-empty. When constructing an instance directly (e.g. in tests), either
    pass the three checkpoint paths OR set ``enabled=False`` to skip the check.
    Consumers that only need a placeholder should use ``enabled=False``.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    prompt: str = "protective fruit bag"
    sam_checkpoint: str = ""
    decoder_checkpoint: str = ""
    mlp_checkpoint: str = ""

    @model_validator(mode="after")
    def _require_checkpoints_when_enabled(self):
        if self.enabled:
            missing = [
                name
                for name in ("sam_checkpoint", "decoder_checkpoint", "mlp_checkpoint")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"PseCo enabled but missing: {missing}")
        return self


class SRStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    scale: float = Field(default=2.0, gt=1.0, le=8.0)
    max_crop_side: int = Field(default=500, gt=0)


class ClassifierStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    checkpoint: str = ""
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_ckpt_when_enabled(self):
        if self.enabled and not self.checkpoint:
            raise ValueError("Classifier enabled but checkpoint path is empty")
        return self


class StageSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deblur: DeblurStageConfig = Field(default_factory=DeblurStageConfig)
    pseco: PseCoStageConfig = Field(default_factory=lambda: PseCoStageConfig(enabled=False))
    sr: SRStageConfig = Field(default_factory=SRStageConfig)
    classifier: ClassifierStageConfig = Field(default_factory=ClassifierStageConfig)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stages: StageSet = Field(default_factory=StageSet)


class IOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_format: Literal["json", "csv", "both"] = "json"
    save_visualizations: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 42
    output_dir: str = "./runs"
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    io: IOConfig = Field(default_factory=IOConfig)

    @field_validator("output_dir")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("output_dir must be a non-empty path")
        return v

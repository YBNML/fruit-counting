"""Pydantic configuration schema for the counting eval tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Crop = Literal["pear", "apple"]
Variant = Literal["count1", "any", "strict"]


class TruthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: Variant = "count1"


class PassesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pitches: list[float] | None = None
    sides: list[Literal["A", "B"]] = Field(default=["A", "B"], min_length=1)
    step_m: float | None = Field(default=None, gt=0)


class StagesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    det: Literal["oracle", "oracle_noisy", "cached"] = "oracle"
    track: Literal["oracle", "landmark", "last_box"] = "landmark"
    center: Literal["truth", "measured"] = "measured"
    dedup: Literal["oracle", "self_side", "merge", "hybrid"] = "merge"
    assign: Literal["truth", "grid"] = "grid"


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positives: Variant = "count1"
    cache: str | None = None
    noise: dict | None = None


class LiftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    depth: Literal["det_depth", "box_median"] = "box_median"
    box_frac: float = Field(default=0.5, gt=0, le=1)
    center_offset_m: dict[Literal["fruit", "bag"], float] = Field(
        default_factory=lambda: {"fruit": 0.02, "bag": 0.03}
    )

    def offset(self, label: int) -> float:
        key = "bag" if label == 1 else "fruit"
        return self.center_offset_m.get(key, 0.0)


class NoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose_trans_sigma_m: float = Field(default=0.0, ge=0)
    pose_rot_sigma_deg: float = Field(default=0.0, ge=0)
    pass_bias_m: float = Field(default=0.0, ge=0)
    depth_rel_sigma: float = Field(default=0.0, ge=0)
    depth_dropout: float = Field(default=0.0, ge=0, le=1)
    seeds: list[int] = Field(default_factory=lambda: [0])

    def is_zero(self) -> bool:
        return (
            self.pose_trans_sigma_m == 0
            and self.pose_rot_sigma_deg == 0
            and self.pass_bias_m == 0
            and self.depth_rel_sigma == 0
            and self.depth_dropout == 0
        )


class TrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predict: Literal["pose"] = "pose"
    gate_px_frac: float = Field(default=0.5, gt=0)
    gate_m: float = Field(default=0.05, gt=0)
    max_gap: int | None = Field(default=None, ge=1)
    repair_m: float = Field(default=0.04, ge=0)
    min_obs: int = Field(default=1, ge=1)
    score_hi: float = Field(default=0.5, ge=0, le=1)
    score_lo: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _check_score_order(self) -> TrackerConfig:
        if self.score_lo > self.score_hi:
            raise ValueError("score_lo must be <= score_hi")
        return self


class DedupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    radius_m: float = Field(default=0.04, gt=0)
    band_m: float = Field(default=0.10, ge=0)


class EvalAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str
    crop: Crop
    split: Literal["dev", "val", "test", "exp"]
    tuning: bool = True
    runs: list[str] = Field(min_length=1)
    truth: TruthConfig = Field(default_factory=TruthConfig)
    passes: PassesConfig = Field(default_factory=PassesConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    lift: LiftConfig = Field(default_factory=LiftConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    seed: int = 0
    stop_after: Literal["y1", "y6"] = "y6"
    output_dir: str = "runs/eval"

    @model_validator(mode="after")
    def _check_stage_combinations(self) -> EvalAppConfig:
        if self.stages.det == "cached" and self.stages.track == "oracle":
            raise ValueError("cached detector cannot use track=oracle")
        if self.stages.det == "cached" and self.lift.depth == "det_depth":
            raise ValueError("det_depth needs oracle depth; use box_median with det=cached")
        if self.split == "test" and self.tuning:
            raise ValueError("split=test requires tuning=false")
        return self


class SplitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dev: list[int]
    val: list[int]
    test: list[int]
    exp: list[int]
    train: list[int]

    @model_validator(mode="after")
    def _check_no_overlap(self) -> SplitsConfig:
        seen: dict[int, str] = {}
        overlap: set[int] = set()
        for name in ("dev", "val", "test", "exp", "train"):
            for seed in getattr(self, name):
                if seed in seen:
                    overlap.add(seed)
                else:
                    seen[seed] = name
        if overlap:
            raise ValueError(f"splits overlap: {sorted(overlap)}")
        return self

    def split_of(self, seed: int) -> str | None:
        for name in ("dev", "val", "test", "exp", "train"):
            if seed in getattr(self, name):
                return name
        return None

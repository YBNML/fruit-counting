"""Pydantic configuration schema for the counting eval tool."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

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


class DetNoise(BaseModel):
    """Detection noise of the noisy oracle; every factor at 0 with score fixed gives the oracle's detections.

    Factors must be finite (an infinite false-positive rate would never finish drawing).
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    miss_p: float = Field(default=0.0, ge=0, le=1)
    fp_lambda: float = Field(default=0.0, ge=0)
    jitter_frac: float = Field(default=0.0, ge=0)
    split_p: float = Field(default=0.0, ge=0, le=1)
    split_visib_max: float = Field(default=0.6, gt=0, le=1)
    score: Literal["fixed", "uniform"] = "fixed"
    score_min: float = Field(default=0.3, ge=0, le=1)


def _unset(value: object) -> bool:
    return value is None


class CachePostConfig(BaseModel):
    """Post-processing of cached detections before matching."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    nms_iou: float = Field(default=0.5, gt=0, le=1)
    edge_snap_px: float = Field(default=2.0, ge=0)
    min_short_px: float = Field(default=0.0, ge=0)


# Cached-detector settings are read under stages.det=cached only; unset, they stay out of the dump and the config hash.
class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positives: Variant = "count1"
    cache: str | None = None
    noise: DetNoise | None = None
    post: CachePostConfig | None = Field(default=None, exclude_if=_unset)
    digests: dict[str, str] | None = Field(default=None, exclude_if=_unset)

    @field_validator("digests")
    @classmethod
    def _check_digest_format(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return v
        for key, value in v.items():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"detector.digests[{key}] must be a 64-character lowercase hex digest")
        return v

    def post_settings(self) -> CachePostConfig:
        return self.post if self.post is not None else CachePostConfig()


class LiftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    depth: Literal["det_depth", "box_median", "box_far"] = "box_median"
    # box_median only: side ratio of the centre rectangle of the box.
    box_frac: float = Field(default=0.5, gt=0, le=1)
    # box_far only (far_*): scale of the ellipse inscribed in the box, depth gap (m) that splits
    # clusters, and the least share of the in-ellipse pixels a cluster needs to be picked as the far one.
    far_scale: float = Field(default=1.0, gt=0, le=1)
    far_gap_m: float = Field(default=0.03, gt=0)
    far_min_frac: float = Field(default=0.25, gt=0, le=1)
    center_offset_m: dict[Literal["fruit", "bag"], float] = Field(
        default_factory=lambda: {"fruit": 0.02, "bag": 0.03}
    )

    def offset(self, label: int) -> float:
        key = "bag" if label == 1 else "fruit"
        return self.center_offset_m.get(key, 0.0)


class NoiseConfig(BaseModel):
    """Pose and depth noise, and the noise seeds; factors must be finite."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pose_trans_sigma_m: float = Field(default=0.0, ge=0)
    pose_rot_sigma_deg: float = Field(default=0.0, ge=0)
    pass_bias_m: float = Field(default=0.0, ge=0)
    depth_rel_sigma: float = Field(default=0.0, ge=0)
    depth_dropout: float = Field(default=0.0, ge=0, le=1)
    seeds: list[NonNegativeInt] = Field(default_factory=lambda: [0], min_length=1)

    @field_validator("seeds")
    @classmethod
    def _check_seeds_distinct(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError(f"noise.seeds repeats a seed: {v}")
        return v

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
    depth_gate: Literal["hard", "soft"] = "hard"
    # depth_gate=soft only: extra cost of a pair inside the screen gate that fails the depth gate.
    soft_cost: float = Field(default=0.5, ge=0)
    max_gap: int | None = Field(default=None, ge=1)
    repair_m: float = Field(default=0.04, ge=0)
    min_obs: int = Field(default=1, ge=1)
    min_obs_cached: int | None = Field(default=None, ge=1, exclude_if=_unset)
    score_hi: float = Field(default=0.5, ge=0, le=1)
    score_lo: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _check_score_order(self) -> TrackerConfig:
        if self.score_lo > self.score_hi:
            raise ValueError("score_lo must be <= score_hi")
        return self

    def min_obs_for(self, det: str) -> int:
        if det == "cached" and self.min_obs_cached is not None:
            return self.min_obs_cached
        return self.min_obs


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
    # Pass subsets (names from the pass-subset table) whose rung L5 row is judged too; read by eval ladder only.
    verdict_subsets: list[str] | None = Field(default=None, min_length=1, exclude_if=_unset)

    @model_validator(mode="after")
    def _check_stage_combinations(self) -> EvalAppConfig:
        if self.stages.det == "oracle_noisy" and self.detector.noise is None:
            raise ValueError("stages.det=oracle_noisy needs a detector.noise section")
        if self.detector.noise is not None and self.stages.det != "oracle_noisy":
            raise ValueError(f"detector.noise needs stages.det=oracle_noisy, got stages.det={self.stages.det}")
        if self.stages.det == "cached" and self.stages.track == "oracle":
            raise ValueError("cached detector cannot use track=oracle")
        if self.stages.det == "cached" and self.lift.depth == "det_depth":
            raise ValueError("det_depth needs oracle depth; use a box depth method (box_median or box_far) "
                             "with det=cached")
        if self.stages.det == "cached" and self.detector.cache is None:
            raise ValueError("stages.det=cached needs detector.cache (the folder of detection caches)")
        if self.detector.digests is not None and self.detector.cache is None:
            raise ValueError("detector.digests needs detector.cache")
        if self.split == "test" and self.tuning:
            raise ValueError("split=test requires tuning=false")
        return self

    def is_noisy(self) -> bool:
        """True when the config adds detection, pose or depth noise; noise.seeds alone is not noise."""
        return self.stages.det == "oracle_noisy" or not self.noise.is_zero()


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

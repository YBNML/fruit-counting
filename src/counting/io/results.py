"""Inference result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageTiming:
    stage: str
    ms: float


@dataclass
class CropMeta:
    bbox: tuple[float, float, float, float]     # x1,y1,x2,y2
    score: float | None = None
    is_bag: bool | None = None                  # filled if classifier ran


@dataclass
class CountingResult:
    image_path: str
    raw_count: int                              # PseCo raw count
    verified_count: int                         # after classifier if enabled, else == raw_count
    points: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    crops: list[CropMeta] = field(default_factory=list)
    timings_ms: list[StageTiming] = field(default_factory=list)
    device: str = "cpu"
    config_hash: str = ""
    error: Optional[str] = None

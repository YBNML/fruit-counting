"""Pipeline assembly: runs stages in order with per-stage timing and error isolation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from counting.io.results import CountingResult, CropMeta, StageTiming
from counting.models.base import Stage

_REQUIRED_COUNTING_STAGE = "pseco"


@dataclass
class Pipeline:
    stages: list[Stage]
    device: str
    config_hash: str

    @classmethod
    def from_stages(cls, stages: list[Stage], *, device: str, config_hash: str) -> "Pipeline":
        return cls(stages=stages, device=device, config_hash=config_hash)

    def run_numpy(self, image: np.ndarray, *, image_path: str) -> CountingResult:
        timings: list[StageTiming] = []
        raw_count = 0
        points: list[tuple[float, float]] = []
        boxes: list[tuple[float, float, float, float]] = []
        crops_meta: list[CropMeta] = []
        error: str | None = None

        has_counting = any(s.name == _REQUIRED_COUNTING_STAGE for s in self.stages)
        if not has_counting:
            raise ValueError(f"Pipeline must include a '{_REQUIRED_COUNTING_STAGE}' stage")

        current: Any = image
        verified_count: int | None = None

        for stage in self.stages:
            t0 = time.perf_counter()
            try:
                result = stage.process(current)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                timings.append(StageTiming(stage=stage.name, ms=round(elapsed_ms, 3)))
                error = f"[{stage.name}] {type(exc).__name__}: {exc}"
                break

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(StageTiming(stage=stage.name, ms=round(elapsed_ms, 3)))

            if stage.name == "pseco":
                out = result.output
                raw_count = int(out.get("count", 0))
                points = [tuple(p) for p in out.get("points", [])]
                boxes = [tuple(b) for b in out.get("boxes", [])]
                crops_meta = [CropMeta(bbox=tuple(b)) for b in boxes]
                current = {"image": image, "crops": out.get("crops", [])}
            elif stage.name == "classifier":
                # expects: {"verified_count": int, "per_crop": [bool, ...]}
                out = result.output
                verified_count = int(out.get("verified_count", raw_count))
                for i, is_bag in enumerate(out.get("per_crop", [])):
                    if i < len(crops_meta):
                        crops_meta[i] = CropMeta(
                            bbox=crops_meta[i].bbox,
                            score=crops_meta[i].score,
                            is_bag=bool(is_bag),
                        )
            else:
                current = result.output

        if verified_count is None:
            verified_count = raw_count

        return CountingResult(
            image_path=image_path,
            raw_count=raw_count,
            verified_count=verified_count,
            points=points,
            boxes=boxes,
            crops=crops_meta,
            timings_ms=timings,
            device=self.device,
            config_hash=self.config_hash,
            error=error,
        )


def build_pipeline(cfg, *, device: str | None = None):
    """Construct a Pipeline from an AppConfig, honoring enabled flags."""
    from counting.config.hashing import config_hash
    from counting.models.classification.inference import ClassifierStage
    from counting.models.deblur import DeblurStage
    from counting.models.pseco.inference import PseCoStage
    from counting.models.sr import SRStage
    from counting.utils.device import resolve_device

    resolved = resolve_device(device or cfg.device)
    stages: list[Stage] = []

    s = cfg.pipeline.stages
    if s.deblur.enabled:
        st = DeblurStage(weights=s.deblur.weights)
        st.prepare(cfg)
        stages.append(st)
    if s.pseco.enabled:
        st = PseCoStage(
            prompt=s.pseco.prompt,
            sam_ckpt=s.pseco.sam_checkpoint,
            decoder_ckpt=s.pseco.decoder_checkpoint,
            mlp_ckpt=s.pseco.mlp_checkpoint,
        )
        st.prepare(cfg)
        stages.append(st)
    if s.sr.enabled:
        st = SRStage(scale=s.sr.scale, max_crop_side=s.sr.max_crop_side)
        st.prepare(cfg)
        stages.append(st)
    if s.classifier.enabled:
        st = ClassifierStage(checkpoint=s.classifier.checkpoint, threshold=s.classifier.threshold)
        st.prepare(cfg)
        stages.append(st)

    return Pipeline(stages=stages, device=resolved, config_hash=config_hash(cfg))

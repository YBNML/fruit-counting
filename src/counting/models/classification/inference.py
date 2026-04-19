"""ResNet-based bagged-fruit classifier Stage (inference)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_pil

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class ClassifierStage(Stage):
    name = "classifier"

    def __init__(self, *, checkpoint: str, threshold: float) -> None:
        self.checkpoint = checkpoint
        self.threshold = threshold
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from classification.core.classification_inference import ClassificationInference
        except ImportError as exc:
            raise RuntimeError(
                "Classification source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        self._impl = ClassificationInference(ckpt_path=self.checkpoint, threshold=self.threshold)

    def process(self, payload: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("ClassifierStage used before prepare()")
        crops = payload.get("crops", []) if isinstance(payload, dict) else []
        per_crop: list[bool] = []
        for crop in crops:
            is_bag, _p_bag, _label, _scores = self._impl.verify_bagged_fruit(ensure_pil(crop))
            per_crop.append(bool(is_bag))
        return StageResult(output={
            "verified_count": int(sum(per_crop)),
            "per_crop": per_crop,
        })

    def cleanup(self) -> None:
        self._impl = None

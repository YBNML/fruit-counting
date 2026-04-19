"""DeblurGANv2 inference wrapper (read-only; no training in Plan 1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_np_rgb

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[3]      # repo root
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class DeblurStage(Stage):
    name = "deblur"

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from DeblurGANv2.core.deblur_inference import DeblurInference
        except ImportError as exc:
            raise RuntimeError(
                "DeblurGANv2 source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        self._impl = DeblurInference()
        try:
            self._impl.model.eval()
        except Exception:
            pass

    def process(self, image: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("DeblurStage used before prepare()")
        arr = ensure_np_rgb(image)
        out = self._impl.predict(arr)
        return StageResult(output=np.asarray(out))

    def cleanup(self) -> None:
        self._impl = None

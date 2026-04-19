"""Super-Resolution Neural Operator wrapper (per-crop, inference only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_pil

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class SRStage(Stage):
    name = "sr"

    def __init__(self, *, scale: float, max_crop_side: int) -> None:
        self.scale = scale
        self.max_crop_side = max_crop_side
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from Super_Resolution_Neural_Operator.core.super_resolution_inference import (
                SuperResolutionModule,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SR-NO source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        self._impl = SuperResolutionModule(scale=self.scale)

    def process(self, payload: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("SRStage used before prepare()")
        crops = payload.get("crops", []) if isinstance(payload, dict) else []
        upscaled: list[Image.Image] = []
        for crop in crops:
            pil = ensure_pil(crop)
            if pil.size[0] > self.max_crop_side or pil.size[1] > self.max_crop_side:
                upscaled.append(pil)
                continue
            upscaled.append(self._impl.predict(pil))
        payload = dict(payload) if isinstance(payload, dict) else {}
        payload["crops"] = upscaled
        return StageResult(output=payload)

    def cleanup(self) -> None:
        self._impl = None

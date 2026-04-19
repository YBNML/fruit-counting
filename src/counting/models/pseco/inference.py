"""PseCo counting inference wrapper."""

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
    root = Path(__file__).resolve().parents[4]      # repo root
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class PseCoStage(Stage):
    name = "pseco"

    def __init__(self, *, prompt: str, sam_ckpt: str, decoder_ckpt: str, mlp_ckpt: str) -> None:
        self.prompt = prompt
        self.sam_ckpt = sam_ckpt
        self.decoder_ckpt = decoder_ckpt
        self.mlp_ckpt = mlp_ckpt
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from PseCo.core.PseCo_inference import CountingInference
        except ImportError as exc:
            raise RuntimeError(
                "PseCo source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        # Upstream constructor signature kept as in original_code/ai_modules/main.py.
        self._impl = CountingInference(prompt=self.prompt)

    def process(self, image: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("PseCoStage used before prepare()")
        arr = ensure_np_rgb(image)
        crops, count = self._impl.get_count_images(arr, None)
        points = list(getattr(self._impl, "last_points", []) or [])
        boxes = list(getattr(self._impl, "last_boxes", []) or [])
        return StageResult(output={
            "count": int(count),
            "crops": list(crops or []),
            "points": [tuple(p) for p in points],
            "boxes": [tuple(b) for b in boxes],
        })

    def cleanup(self) -> None:
        self._impl = None

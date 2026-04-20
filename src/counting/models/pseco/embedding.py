"""SAM ViT-H image embedding extractor for the caching pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external" / "PseCo"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class SAMImageEmbedder:
    """Wraps upstream SAM predictor for (image -> embedding) forward.

    Output: torch.Tensor of shape (256, 64, 64) in fp16 on CPU.
    """

    def __init__(self, sam_checkpoint: str, device: str = "cuda") -> None:
        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self._predictor: Any = None

    def prepare(self) -> None:
        _ensure_external_on_path()
        from ops.foundation_models.segment_anything import (
            SamPredictor,
            build_sam_vit_h,
        )

        sam = build_sam_vit_h(checkpoint=self.sam_checkpoint)
        sam.to(self.device).eval()
        self._predictor = SamPredictor(sam)

    @torch.no_grad()
    def embed(self, image_rgb_uint8: np.ndarray) -> torch.Tensor:
        """Return (256, 64, 64) fp16 CPU tensor for the given RGB uint8 image."""
        if self._predictor is None:
            raise RuntimeError("SAMImageEmbedder used before prepare()")
        self._predictor.set_image(image_rgb_uint8)
        feats = self._predictor.features  # (1, 256, 64, 64), fp32 on device
        return feats.squeeze(0).detach().to(dtype=torch.float16, device="cpu").contiguous()

    def cleanup(self) -> None:
        self._predictor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

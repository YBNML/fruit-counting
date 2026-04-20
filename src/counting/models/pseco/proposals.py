"""Frozen PointDecoder proposal generation for cached SAM embeddings."""

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


class PointDecoderProposer:
    """Loads upstream PointDecoder + SAM backbone and emits proposals per image."""

    def __init__(
        self,
        *,
        sam_checkpoint: str,
        point_decoder_checkpoint: str,
        device: str = "cuda",
    ) -> None:
        self.sam_checkpoint = sam_checkpoint
        self.point_decoder_checkpoint = point_decoder_checkpoint
        self.device = device
        self._sam: Any = None
        self._decoder: Any = None

    def prepare(self) -> None:
        _ensure_external_on_path()
        from ops.foundation_models.segment_anything import build_sam_vit_h
        from models import PointDecoder

        self._sam = build_sam_vit_h(checkpoint=self.sam_checkpoint).to(self.device).eval()
        self._decoder = PointDecoder(self._sam).to(self.device).eval()
        state = torch.load(self.point_decoder_checkpoint, map_location="cpu")
        if "model" in state:
            state = state["model"]
        self._decoder.load_state_dict(state, strict=False)

    @torch.no_grad()
    def propose(self, cached_embedding_fp16: np.ndarray) -> dict[str, torch.Tensor]:
        """Return proposals for a single image.

        cached_embedding_fp16 shape: (256, 64, 64)
        Output keys: pred_heatmaps, pred_points, pred_points_score (CPU tensors).
        """
        if self._decoder is None:
            raise RuntimeError("PointDecoderProposer used before prepare()")
        feats = torch.from_numpy(cached_embedding_fp16).to(self.device).to(torch.float32).unsqueeze(0)
        out = self._decoder(feats)
        return {k: v.detach().cpu() for k, v in out.items()}

    def cleanup(self) -> None:
        self._sam = None
        self._decoder = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

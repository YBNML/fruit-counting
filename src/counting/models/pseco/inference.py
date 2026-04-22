"""PseCo counting inference Stage.

Pipeline per image:
    SAM ViT-H encoder  →  PointDecoder (candidate points)  →
    SAM.forward_sam_with_embeddings (SAM-predicted boxes + anchor-prompted boxes)
    →  ROIHeadMLP (CLIP-aligned classification)  →  NMS  →  score threshold
    →  integer count + per-proposal boxes/crops.

This is the production inference flow. It replaces an earlier stub that tried
to import `PseCo.core.PseCo_inference.CountingInference`, a module that does
not exist in the upstream `external/PseCo` submodule. See
`docs/superpowers/specs/2026-04-22-plan4-direction.md` for background.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import torchvision.ops as vision_ops
import torchvision.transforms as T
from PIL import Image

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_np_rgb

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

_IMAGE_SIZE = 1024


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external" / "PseCo"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


def _load_cls_head_state(path: str):
    """Load ROIHeadMLP weights tolerating both our 'model'/'state_dict' wrapped
    format (Runner.save_checkpoint) and upstream's 'cls_head' wrapped format
    (MLP_small_box_w1_zeroshot.tar)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for wrapper_key in ("cls_head", "model", "state_dict"):
            inner = state.get(wrapper_key)
            if isinstance(inner, dict) and any(
                isinstance(k, str) and "." in k for k in inner.keys()
            ):
                return inner
    return state


class PseCoStage(Stage):
    name = "pseco"

    def __init__(
        self,
        *,
        prompt: str,
        sam_ckpt: str,
        decoder_ckpt: str,
        mlp_ckpt: str,
        clip_features_cache: str = "",
        point_threshold: float = 0.05,
        max_points: int = 1000,
        anchor_size: int = 8,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.10,
    ) -> None:
        self.prompt = prompt
        self.sam_ckpt = sam_ckpt
        self.decoder_ckpt = decoder_ckpt
        self.mlp_ckpt = mlp_ckpt
        self.clip_features_cache = clip_features_cache
        self.point_threshold = point_threshold
        self.max_points = max_points
        self.anchor_size = anchor_size
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

        self._device: str = "cpu"
        self._sam: Any = None
        self._point_decoder: Any = None
        self._cls_head: Any = None
        self._text_feature: torch.Tensor | None = None
        self._preprocess: A.Compose | None = None
        self._to_tensor: T.Compose | None = None

    def prepare(self, cfg: Any) -> None:
        _ensure_external_on_path()

        # Local imports so the module is importable without external/ present.
        from models import PointDecoder, ROIHeadMLP
        from ops.foundation_models.segment_anything import build_sam_vit_h

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available() else "cpu"
        )
        self._device = device

        sam = build_sam_vit_h(checkpoint=self.sam_ckpt)
        sam.to(device).eval()
        self._sam = sam

        self._point_decoder = PointDecoder(sam).to(device).eval()
        pd_state = torch.load(
            self.decoder_ckpt, map_location="cpu", weights_only=False
        )
        if isinstance(pd_state, dict) and "model" in pd_state and isinstance(
            pd_state["model"], dict
        ):
            pd_state = pd_state["model"]
        self._point_decoder.load_state_dict(pd_state, strict=False)

        self._cls_head = ROIHeadMLP().to(device).eval()
        self._cls_head.load_state_dict(
            _load_cls_head_state(self.mlp_ckpt), strict=True
        )

        self._text_feature = self._resolve_prompt_feature(device)

        self._preprocess = A.Compose([
            A.LongestMaxSize(_IMAGE_SIZE),
            A.PadIfNeeded(
                _IMAGE_SIZE, _IMAGE_SIZE,
                border_mode=0, position="top_left",
            ),
        ])
        self._to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def _resolve_prompt_feature(self, device: str) -> torch.Tensor:
        """Return a (1, 512) CLIP text feature for self.prompt. Looks up the
        configured cache first; falls back to on-the-fly open_clip encoding."""
        from counting.models.pseco.clip_features import (
            encode_class_names,
            load_text_features,
        )

        if self.clip_features_cache:
            cache_path = Path(self.clip_features_cache)
            if cache_path.exists():
                cache = load_text_features(cache_path)
                if self.prompt in cache:
                    return cache[self.prompt].view(1, 512).to(device)
        # Not in cache — encode now (slower, but functional)
        encoded = encode_class_names([self.prompt], device="cpu")
        return encoded[self.prompt].view(1, 512).to(device)

    @torch.no_grad()
    def process(self, image: Any) -> StageResult:
        if self._sam is None:
            raise RuntimeError("PseCoStage used before prepare()")

        arr = ensure_np_rgb(image)
        orig_h, orig_w = arr.shape[:2]
        scale = _IMAGE_SIZE / float(max(orig_h, orig_w))

        assert self._preprocess is not None
        padded_arr = self._preprocess(image=arr)["image"]
        padded = Image.fromarray(padded_arr)

        assert self._to_tensor is not None
        batched = self._to_tensor(padded).unsqueeze(0).to(self._device)
        features = self._sam.image_encoder(batched)

        self._point_decoder.max_points = self.max_points
        self._point_decoder.point_threshold = self.point_threshold
        self._point_decoder.nms_kernel_size = 3
        outputs = self._point_decoder(features)
        pred_points = outputs["pred_points"].squeeze().reshape(-1, 2)

        if pred_points.numel() == 0:
            return StageResult(output={
                "count": 0, "crops": [], "points": [], "boxes": [],
            })

        text_feature = self._text_feature
        assert text_feature is not None

        all_pred_boxes: list[torch.Tensor] = []
        cls_logits_chunks: list[torch.Tensor] = []
        for chunk in torch.arange(len(pred_points)).split(128):
            outputs_points = self._sam.forward_sam_with_embeddings(
                features, points=pred_points[chunk]
            )
            pred_boxes = outputs_points["pred_boxes"]
            pred_ious = outputs_points["pred_ious"]

            anchor = torch.tensor(
                [[-self.anchor_size, -self.anchor_size,
                  self.anchor_size, self.anchor_size]],
                device=self._device, dtype=pred_boxes.dtype,
            )
            anchor_boxes = pred_points[chunk].repeat(1, 2) + anchor
            anchor_boxes = anchor_boxes.clamp(0.0, float(_IMAGE_SIZE))
            outputs_boxes = self._sam.forward_sam_with_embeddings(
                features, points=pred_points[chunk], boxes=anchor_boxes,
            )
            pred_ious = torch.cat(
                [pred_ious, outputs_boxes["pred_ious"][:, 1].unsqueeze(1)],
                dim=1,
            )
            pred_boxes = torch.cat(
                [pred_boxes, outputs_boxes["pred_boxes"][:, 1].unsqueeze(1)],
                dim=1,
            )

            all_pred_boxes.append(pred_boxes)
            cls_outs = self._cls_head(
                features, [pred_boxes], [text_feature] * len(chunk)
            )
            cls_outs = cls_outs.sigmoid().view(-1, 1, 5).mean(1)
            cls_logits_chunks.append(cls_outs * pred_ious)

        all_boxes = torch.cat(all_pred_boxes)
        all_cls = torch.cat(cls_logits_chunks)
        best_idx = torch.argmax(all_cls, dim=1)
        boxes_best = all_boxes[torch.arange(len(all_boxes)), best_idx]
        scores_best = all_cls.max(1).values

        keep = vision_ops.nms(boxes_best, scores_best, self.nms_threshold)
        boxes_best = boxes_best[keep]
        scores_best = scores_best[keep]

        above = scores_best > self.score_threshold
        boxes_final = boxes_best[above].cpu()
        scores_final = scores_best[above].cpu()

        # Map boxes back to original image coords and build crops
        inv_scale = 1.0 / scale
        boxes_orig: list[tuple[float, float, float, float]] = []
        crops: list[Image.Image] = []
        orig_pil = Image.fromarray(arr)
        for (x1, y1, x2, y2) in boxes_final.tolist():
            ox1 = float(max(0.0, x1 * inv_scale))
            oy1 = float(max(0.0, y1 * inv_scale))
            ox2 = float(min(orig_w, x2 * inv_scale))
            oy2 = float(min(orig_h, y2 * inv_scale))
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            boxes_orig.append((ox1, oy1, ox2, oy2))
            crops.append(orig_pil.crop((ox1, oy1, ox2, oy2)))

        # Points: map top-K (in 1024 space) back to original coords
        pts_np = pred_points.detach().cpu().numpy()
        points_orig = [
            (float(x * inv_scale), float(y * inv_scale))
            for (x, y) in pts_np.tolist()
        ]

        return StageResult(output={
            "count": int(len(boxes_orig)),
            "crops": crops,
            "points": points_orig,
            "boxes": boxes_orig,
        })

    def cleanup(self) -> None:
        self._sam = None
        self._point_decoder = None
        self._cls_head = None
        self._text_feature = None
        if self._device == "cuda":
            torch.cuda.empty_cache()

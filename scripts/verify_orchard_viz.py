"""Orchard zero-shot inference with:
  (A) visualization — save image overlays of predicted boxes for eyeballing
  (B) prompt ablation — same image × multiple prompts, see which aligns best

Writes results under scripts/orchard_viz_out/.
Run from repo root: python scripts/verify_orchard_viz.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision.ops as vision_ops
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT = REPO_ROOT / "external" / "PseCo"
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models import PointDecoder, ROIHeadMLP
from ops.foundation_models.segment_anything import build_sam_vit_h

from counting.models.pseco.clip_features import (
    encode_class_names,
    load_text_features,
)

MODELS = REPO_ROOT / "models" / "PseCo"
SAMPLES = REPO_ROOT / "orchard_samples"
OUT = REPO_ROOT / "scripts" / "orchard_viz_out"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 1024
POINT_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.10
MAX_POINTS = 1000

# (image, list of prompts to compare)
CASES = [
    ("apple_1.jpeg", ["apples", "apple", "apple in paper bag", "protective fruit bag"]),
    ("apple_2.jpeg", ["apples", "apple in paper bag"]),
    ("pear_image_1751534814_2000.0.jpg", ["pears", "pear in paper bag"]),
    ("deploy_1.jpg", [
        "protective fruit bag",
        "apple in paper bag",
        "fruit wrapped in paper",
        "white bag hanging on tree",
    ]),
    ("deploy_2.jpg", [
        "protective fruit bag",
        "apple in paper bag",
        "white bag hanging on tree",
    ]),
]


def read_image(path: Path) -> tuple[Image.Image, float, tuple[int, int]]:
    """Return (padded_1024_image, scale_factor, (orig_w, orig_h))."""
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    scale = IMAGE_SIZE / float(max(orig_w, orig_h))
    transform = A.Compose([
        A.LongestMaxSize(IMAGE_SIZE),
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0,
                      position="top_left"),
    ])
    arr = transform(image=np.array(img))["image"]
    return Image.fromarray(arr), scale, (orig_w, orig_h)


def load_cls_head(path: Path) -> ROIHeadMLP:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for wrapper_key in ("cls_head", "model", "state_dict"):
        inner = state.get(wrapper_key) if isinstance(state, dict) else None
        if isinstance(inner, dict) and any(
            isinstance(k, str) and "." in k for k in inner.keys()
        ):
            state = inner
            break
    head = ROIHeadMLP().to(DEVICE).eval()
    head.load_state_dict(state, strict=True)
    return head


@torch.no_grad()
def run_inference(
    image_1024: Image.Image,
    sam,
    point_decoder: PointDecoder,
    cls_head: ROIHeadMLP,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (surviving_boxes_in_1024_space, scores)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    new_image = transform(image_1024).unsqueeze(0).to(DEVICE)
    features = sam.image_encoder(new_image)

    point_decoder.max_points = MAX_POINTS
    point_decoder.point_threshold = POINT_THRESHOLD
    point_decoder.nms_kernel_size = 3
    outputs = point_decoder(features)
    pred_points = outputs["pred_points"].squeeze().reshape(-1, 2)
    if pred_points.numel() == 0:
        return torch.empty(0, 4), torch.empty(0)

    all_pred_boxes = []
    cls_outs = []
    for indices in torch.arange(len(pred_points)).split(128):
        outputs_points = sam.forward_sam_with_embeddings(
            features, points=pred_points[indices]
        )
        pred_boxes = outputs_points["pred_boxes"]
        pred_logits = outputs_points["pred_ious"]

        anchor_size = 8
        anchor = torch.Tensor(
            [[-anchor_size, -anchor_size, anchor_size, anchor_size]]
        ).to(DEVICE)
        anchor_boxes = pred_points[indices].repeat(1, 2) + anchor
        anchor_boxes = anchor_boxes.clamp(0., 1024.)
        outputs_boxes = sam.forward_sam_with_embeddings(
            features, points=pred_points[indices], boxes=anchor_boxes
        )
        pred_logits = torch.cat(
            [pred_logits, outputs_boxes["pred_ious"][:, 1].unsqueeze(1)], dim=1
        )
        pred_boxes = torch.cat(
            [pred_boxes, outputs_boxes["pred_boxes"][:, 1].unsqueeze(1)], dim=1
        )

        all_pred_boxes.append(pred_boxes)
        cls_outs_ = cls_head(
            features, [pred_boxes], [text_features] * len(indices)
        )
        cls_outs_ = cls_outs_.sigmoid().view(-1, 1, 5).mean(1)
        pred_logits = cls_outs_ * pred_logits
        cls_outs.append(pred_logits)

    pred_boxes = torch.cat(all_pred_boxes)
    cls_outs = torch.cat(cls_outs)
    pred_boxes = pred_boxes[torch.arange(len(pred_boxes)),
                            torch.argmax(cls_outs, dim=1)]
    scores = cls_outs.max(1).values
    keep = vision_ops.nms(pred_boxes, scores, NMS_THRESHOLD)
    pred_boxes = pred_boxes[keep].cpu()
    scores = scores[keep].cpu()
    above = scores > SCORE_THRESHOLD
    return pred_boxes[above], scores[above]


def save_overlay(
    image_1024: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    out_path: Path,
    title: str,
) -> None:
    """Draw boxes (in 1024 coord space) on image_1024, save as PNG."""
    img = image_1024.copy()
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist()):
        color = (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    draw.text((10, 10), title, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def get_clip(prompt: str, cache: dict) -> torch.Tensor:
    if prompt in cache:
        return cache[prompt].view(1, 512).to(DEVICE)
    print(f"[clip] encoding new prompt: {prompt!r}")
    new = encode_class_names([prompt], device="cpu")
    cache[prompt] = new[prompt]
    return new[prompt].view(1, 512).to(DEVICE)


def main() -> None:
    print(f"[setup] device={DEVICE}")
    sam = build_sam_vit_h(checkpoint=str(MODELS / "sam_vit_h.pth"))
    sam.to(DEVICE).eval()
    pd_state = torch.load(
        MODELS / "point_decoder_vith.pth", map_location="cpu", weights_only=False
    )
    point_decoder = PointDecoder(sam).to(DEVICE).eval()
    point_decoder.load_state_dict(pd_state, strict=False)

    upstream_head = load_cls_head(MODELS / "MLP_small_box_w1_zeroshot.tar")
    ours_head = load_cls_head(
        REPO_ROOT / "runs" / "plan3_k32" / "checkpoints" / "best.ckpt"
    )
    print("[setup] SAM + PointDecoder + both MLPs ready")

    clip_cache = load_text_features(MODELS / "clip_text_features.pt")
    clip_cache_cpu: dict = dict(clip_cache)

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'image':>38s}  {'prompt':>26s}  "
          f"{'upstream':>9s}  {'ours':>5s}")
    print("-" * 90)
    for fname, prompts in CASES:
        path = SAMPLES / fname
        if not path.exists():
            print(f"  skip (missing): {fname}")
            continue
        image_1024, _, _ = read_image(path)

        for prompt in prompts:
            tf = get_clip(prompt, clip_cache_cpu)

            up_boxes, up_scores = run_inference(
                image_1024, sam, point_decoder, upstream_head, tf
            )
            ours_boxes, ours_scores = run_inference(
                image_1024, sam, point_decoder, ours_head, tf
            )
            up_count = up_boxes.size(0)
            ours_count = ours_boxes.size(0)

            print(f"{fname:>38s}  {prompt:>26s}  "
                  f"{up_count:>9d}  {ours_count:>5d}")

            base = fname.replace("/", "_").rsplit(".", 1)[0]
            slug = prompt.replace(" ", "_")
            save_overlay(
                image_1024, up_boxes, up_scores,
                OUT / f"{base}__{slug}__upstream.png",
                f"upstream {prompt} → {up_count}",
            )
            save_overlay(
                image_1024, ours_boxes, ours_scores,
                OUT / f"{base}__{slug}__ours.png",
                f"ours {prompt} → {ours_count}",
            )

    print(f"\n[done] overlays saved to {OUT}")


if __name__ == "__main__":
    main()

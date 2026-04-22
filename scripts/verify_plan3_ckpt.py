"""Minimal verification script for Plan 3's fine-tuned ROIHeadMLP checkpoint.

Runs upstream PseCo's zero-shot inference flow (SAM + PointDecoder +
ROIHeadMLP + CLIP text features + NMS) on a handful of FSC-147 val
images, using two different MLP checkpoints:

  1. upstream `MLP_small_box_w1_zeroshot.tar` (baseline)
  2. our Plan 3 `best.ckpt` (fine-tuned)

For each image we print (gt_count, upstream_count, ours_count) so we can
judge whether the fine-tuned weights load AND whether they produce
meaningful numbers on real inference (not just the training-time
simplified loop).

Run: python scripts/verify_plan3_ckpt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision.ops as vision_ops
import torchvision.transforms as transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT = REPO_ROOT / "external" / "PseCo"
sys.path.insert(0, str(EXT))

from models import PointDecoder, ROIHeadMLP
from ops.foundation_models.segment_anything import build_sam_vit_h

# We re-use our own CLIP cache (built with ViT-B-32-quickgelu) instead of
# upstream's `ops.dump_clip_features.dump_clip_text_features`, which has a
# `from pkg_resources import packaging` that breaks on modern setuptools.
sys.path.insert(0, str(REPO_ROOT / "src"))
from counting.models.pseco.clip_features import load_text_features

DATA_ROOT = REPO_ROOT / "datasets" / "fsc147"
MODELS = REPO_ROOT / "models" / "PseCo"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 1024

# How many val images to probe
N_SAMPLES = 5

# Inference thresholds (match upstream demo)
POINT_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.10
MAX_POINTS = 1000


def read_image(path: Path) -> Image.Image:
    """LongestMaxSize(1024) + PadIfNeeded(1024) — matches upstream demo."""
    img = Image.open(path).convert("RGB")
    transform = A.Compose([
        A.LongestMaxSize(IMAGE_SIZE),
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0,
                      position="top_left"),
    ])
    arr = transform(image=np.array(img))["image"]
    return Image.fromarray(arr)


def load_cls_head(path: Path) -> ROIHeadMLP:
    """Load ROIHeadMLP with a checkpoint that may be wrapped under
    "cls_head" (upstream format) or "model" (our training format)."""
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
    image: Image.Image,
    class_name: str,
    sam,
    point_decoder: PointDecoder,
    cls_head: ROIHeadMLP,
    text_features: torch.Tensor,
) -> int:
    """Return final proposal count after NMS + threshold."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    new_image = transform(image).unsqueeze(0).to(DEVICE)
    features = sam.image_encoder(new_image)

    point_decoder.max_points = MAX_POINTS
    point_decoder.point_threshold = POINT_THRESHOLD
    point_decoder.nms_kernel_size = 3
    outputs = point_decoder(features)
    pred_points = outputs["pred_points"].squeeze().reshape(-1, 2)

    if pred_points.numel() == 0:
        return 0

    all_pred_boxes = []
    all_pred_ious = []
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
        all_pred_ious.append(pred_logits)
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
    scores = scores[keep]
    return int((scores > SCORE_THRESHOLD).sum().item())


def main() -> None:
    # Load GT + class mapping
    splits = json.loads(
        (DATA_ROOT / "Train_Test_Val_FSC_147.json").read_text()
    )
    annotations = json.loads(
        (DATA_ROOT / "annotation_FSC147_384.json").read_text()
    )
    image_classes: dict[str, str] = {}
    for line in (DATA_ROOT / "ImageClasses_FSC147.txt").read_text().splitlines():
        if "\t" in line:
            fn, cn = line.split("\t", 1)
            image_classes[fn] = cn

    # Pick N_SAMPLES val images with varying GT counts
    val_list = splits["val"][:50]
    val_list.sort(key=lambda n: len(annotations[n]["points"]))
    stride = max(1, len(val_list) // N_SAMPLES)
    picks = [val_list[i * stride] for i in range(N_SAMPLES)]

    print(f"[setup] device={DEVICE}, samples={picks}")

    # Build shared SAM + PointDecoder (used by both MLPs)
    sam = build_sam_vit_h(checkpoint=str(MODELS / "sam_vit_h.pth"))
    sam.to(DEVICE).eval()
    pd_state = torch.load(
        MODELS / "point_decoder_vith.pth", map_location="cpu", weights_only=False
    )
    point_decoder = PointDecoder(sam).to(DEVICE).eval()
    point_decoder.load_state_dict(pd_state, strict=False)

    # Load both MLPs
    upstream_head = load_cls_head(MODELS / "MLP_small_box_w1_zeroshot.tar")
    print("[setup] upstream MLP loaded")
    ours_head = load_cls_head(
        REPO_ROOT / "runs" / "plan3_k32" / "checkpoints" / "best.ckpt"
    )
    print("[setup] Plan 3 best.ckpt loaded")

    # Load CLIP text features from our cache (built with ViT-B-32-quickgelu
    # to match the upstream MLP's training distribution). All FSC-147 class
    # names should already be in it.
    needed = {image_classes[n] for n in picks}
    print(f"[setup] loading CLIP text features for {len(needed)} classes: "
          f"{sorted(needed)}")
    clip_feats = load_text_features(MODELS / "clip_text_features.pt")
    missing = needed - set(clip_feats.keys())
    if missing:
        raise KeyError(f"CLIP cache missing classes: {sorted(missing)}")

    print(f"\n{'image':>10s}  {'class':>22s}  {'gt':>5s}  {'up':>5s}  {'ours':>5s}")
    print("-" * 60)
    for name in picks:
        class_name = image_classes[name]
        gt_count = len(annotations[name]["points"])
        img = read_image(DATA_ROOT / "images_384_VarV2" / name)
        tf = clip_feats[class_name].unsqueeze(0).to(DEVICE)

        up_count = run_inference(img, class_name, sam, point_decoder,
                                 upstream_head, tf)
        ours_count = run_inference(img, class_name, sam, point_decoder,
                                   ours_head, tf)
        print(f"{name:>10s}  {class_name:>22s}  {gt_count:>5d}  "
              f"{up_count:>5d}  {ours_count:>5d}")


if __name__ == "__main__":
    main()

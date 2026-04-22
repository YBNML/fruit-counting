"""Diagnose why apple images are undercounted.

Sweep (anchor_size, point_decoder_nms_kernel_size) for apple_1 and apple_2
and report per-config counts at two score thresholds. Tells us whether:
  - PointDecoder's own NMS is merging nearby apples (try kernel 1 vs 3)
  - SAM anchor_size is mismatched to apple scale (try 4..32)

Run from repo root: python scripts/diag_apple_undercount.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision.ops as vision_ops
import torchvision.transforms as T
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT = REPO_ROOT / "external" / "PseCo"
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models import PointDecoder, ROIHeadMLP
from ops.foundation_models.segment_anything import build_sam_vit_h

from counting.models.pseco.clip_features import encode_class_names, load_text_features

MODELS = REPO_ROOT / "models" / "PseCo"
SAMPLES = REPO_ROOT / "orchard_samples"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 1024
POINT_THRESHOLD = 0.05
MAX_POINTS = 1000
NMS_THRESHOLD = 0.3

ANCHOR_SIZES = [4, 8, 12, 16, 24, 32]
PD_NMS_KERNELS = [1, 3, 5]

GT = {
    "apple_1.jpeg": 48,
    "apple_2.jpeg": 46,
    "pear_image_1751534814_2000.0.jpg": 27,  # for comparison
}

PROMPTS = {
    "apple_1.jpeg": "apple in paper bag",
    "apple_2.jpeg": "apple in paper bag",
    "pear_image_1751534814_2000.0.jpg": "pear in paper bag",
}


def read_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    tf = A.Compose([
        A.LongestMaxSize(IMAGE_SIZE),
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0, position="top_left"),
    ])
    return Image.fromarray(tf(image=np.array(img))["image"])


def load_cls_head(path: Path) -> ROIHeadMLP:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for wk in ("cls_head", "model", "state_dict"):
        inner = state.get(wk) if isinstance(state, dict) else None
        if isinstance(inner, dict) and any(
            isinstance(k, str) and "." in k for k in inner.keys()
        ):
            state = inner
            break
    head = ROIHeadMLP().to(DEVICE).eval()
    head.load_state_dict(state, strict=True)
    return head


@torch.no_grad()
def count_for_config(
    features, pd_nms_kernel, anchor_size, cls_head, tf, point_decoder,
    score_thresholds=(0.05, 0.10, 0.15),
) -> tuple[int, dict[float, int]]:
    """Return (n_points, {score_th: count}) for this configuration."""
    point_decoder.max_points = MAX_POINTS
    point_decoder.point_threshold = POINT_THRESHOLD
    point_decoder.nms_kernel_size = pd_nms_kernel
    outputs = point_decoder(features)
    pred_points = outputs["pred_points"].squeeze().reshape(-1, 2)
    n_points = int(pred_points.size(0))
    if n_points == 0:
        return 0, {s: 0 for s in score_thresholds}

    all_boxes, all_cls = [], []
    for chunk in torch.arange(len(pred_points)).split(128):
        op = point_decoder.sam.forward_sam_with_embeddings(
            features, points=pred_points[chunk]
        )
        pred_boxes = op["pred_boxes"]
        pred_ious = op["pred_ious"]

        anchor = torch.tensor(
            [[-anchor_size, -anchor_size, anchor_size, anchor_size]],
            device=DEVICE, dtype=pred_boxes.dtype,
        )
        ab = (pred_points[chunk].repeat(1, 2) + anchor).clamp(0, IMAGE_SIZE)
        ob = point_decoder.sam.forward_sam_with_embeddings(
            features, points=pred_points[chunk], boxes=ab
        )
        pred_ious = torch.cat(
            [pred_ious, ob["pred_ious"][:, 1].unsqueeze(1)], dim=1
        )
        pred_boxes = torch.cat(
            [pred_boxes, ob["pred_boxes"][:, 1].unsqueeze(1)], dim=1
        )

        all_boxes.append(pred_boxes)
        cls = cls_head(features, [pred_boxes], [tf] * len(chunk))
        cls = cls.sigmoid().view(-1, 1, 5).mean(1)
        all_cls.append(cls * pred_ious)

    all_boxes_t = torch.cat(all_boxes)
    all_cls_t = torch.cat(all_cls)
    best = torch.argmax(all_cls_t, dim=1)
    boxes_best = all_boxes_t[torch.arange(len(all_boxes_t)), best]
    scores_best = all_cls_t.max(1).values

    keep = vision_ops.nms(boxes_best, scores_best, NMS_THRESHOLD)
    surviving = scores_best[keep]

    counts = {
        s: int((surviving > s).sum().item()) for s in score_thresholds
    }
    return n_points, counts


@torch.no_grad()
def main() -> None:
    print(f"[setup] device={DEVICE}")
    sam = build_sam_vit_h(checkpoint=str(MODELS / "sam_vit_h.pth"))
    sam.to(DEVICE).eval()
    pd_state = torch.load(
        MODELS / "point_decoder_vith.pth", map_location="cpu", weights_only=False
    )
    point_decoder = PointDecoder(sam).to(DEVICE).eval()
    point_decoder.load_state_dict(pd_state, strict=False)
    point_decoder.sam = sam  # so inner forward_sam_with_embeddings works

    cls_head = load_cls_head(MODELS / "MLP_small_box_w1_zeroshot.tar")
    clip_cache = load_text_features(MODELS / "clip_text_features.pt")
    print("[setup] ready")

    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for fname, gt in GT.items():
        path = SAMPLES / fname
        if not path.exists():
            continue
        prompt = PROMPTS[fname]
        if prompt not in clip_cache:
            clip_cache = dict(clip_cache)
            encoded = encode_class_names([prompt], device="cpu")
            clip_cache[prompt] = encoded[prompt]
        tf = clip_cache[prompt].view(1, 512).to(DEVICE)
        img = read_image(path)
        features = sam.image_encoder(to_tensor(img).unsqueeze(0).to(DEVICE))

        print(f"\n=== {fname} (GT={gt}, prompt={prompt!r}) ===")
        print(f"{'pd_nms':>7s} {'anchor':>7s} {'n_pts':>6s}  "
              f"{'c@0.05':>7s} {'c@0.10':>7s} {'c@0.15':>7s}")
        for kernel in PD_NMS_KERNELS:
            for asize in ANCHOR_SIZES:
                n_pts, counts = count_for_config(
                    features, kernel, asize, cls_head, tf, point_decoder,
                )
                print(f"{kernel:>7d} {asize:>7d} {n_pts:>6d}  "
                      f"{counts[0.05]:>7d} {counts[0.10]:>7d} {counts[0.15]:>7d}")


if __name__ == "__main__":
    main()

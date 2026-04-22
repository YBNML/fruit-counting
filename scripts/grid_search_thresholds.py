"""Grid-search (score_threshold, nms_threshold) on the 4 orchard images with GT.

Computes heavy pipeline (SAM encoder + PointDecoder + SAM box predictions +
ROIHeadMLP scoring) once per (image, prompt), then sweeps thresholds in a
lightweight inner loop.

Prints per-image best (score_th, nms_th) and the averaged MAE per config.

Run from repo root: python scripts/grid_search_thresholds.py
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

from counting.models.pseco.clip_features import (
    encode_class_names,
    load_text_features,
)

MODELS = REPO_ROOT / "models" / "PseCo"
SAMPLES = REPO_ROOT / "orchard_samples"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 1024
ANCHOR_SIZE = 8  # keep fixed per P4.2 default; swept in a later sub-pass if needed
POINT_THRESHOLD = 0.05
MAX_POINTS = 1000

GT = {
    "apple_1.jpeg": 48,
    "apple_2.jpeg": 46,
    "pear_image_1751534814_2000.0.jpg": 27,
    "pear_image_1751535399_2000.0.jpg": 54,
}

# (image, list of prompts to evaluate) — apple/pear dedicated prompts + generic
CASES: list[tuple[str, list[str]]] = [
    ("apple_1.jpeg", ["apples", "apple in paper bag", "protective fruit bag"]),
    ("apple_2.jpeg", ["apples", "apple in paper bag", "protective fruit bag"]),
    ("pear_image_1751534814_2000.0.jpg",
        ["pears", "pear in paper bag", "protective fruit bag"]),
    ("pear_image_1751535399_2000.0.jpg",
        ["pears", "pear in paper bag", "protective fruit bag"]),
]

SCORE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
NMS_THRESHOLDS = [0.3, 0.5, 0.7]


def read_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    transform = A.Compose([
        A.LongestMaxSize(IMAGE_SIZE),
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0, position="top_left"),
    ])
    return Image.fromarray(transform(image=np.array(img))["image"])


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


def get_clip(prompt: str, cache: dict) -> torch.Tensor:
    if prompt in cache:
        return cache[prompt].view(1, 512).to(DEVICE)
    new = encode_class_names([prompt], device="cpu")
    cache[prompt] = new[prompt]
    return new[prompt].view(1, 512).to(DEVICE)


@torch.no_grad()
def compute_boxes_and_scores(
    image_1024: Image.Image,
    sam,
    point_decoder: PointDecoder,
    cls_head: ROIHeadMLP,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (boxes (N,4), scores (N,)) on CPU — heavy path runs once."""
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = sam.image_encoder(to_tensor(image_1024).unsqueeze(0).to(DEVICE))
    point_decoder.max_points = MAX_POINTS
    point_decoder.point_threshold = POINT_THRESHOLD
    point_decoder.nms_kernel_size = 3
    pred_points = point_decoder(features)["pred_points"].squeeze().reshape(-1, 2)
    if pred_points.numel() == 0:
        return torch.empty(0, 4), torch.empty(0)

    all_boxes, all_cls = [], []
    for chunk in torch.arange(len(pred_points)).split(128):
        op = sam.forward_sam_with_embeddings(features, points=pred_points[chunk])
        pred_boxes = op["pred_boxes"]
        pred_ious = op["pred_ious"]

        anchor = torch.tensor(
            [[-ANCHOR_SIZE, -ANCHOR_SIZE, ANCHOR_SIZE, ANCHOR_SIZE]],
            device=DEVICE, dtype=pred_boxes.dtype,
        )
        ab = (pred_points[chunk].repeat(1, 2) + anchor).clamp(0, IMAGE_SIZE)
        ob = sam.forward_sam_with_embeddings(
            features, points=pred_points[chunk], boxes=ab
        )
        pred_ious = torch.cat(
            [pred_ious, ob["pred_ious"][:, 1].unsqueeze(1)], dim=1
        )
        pred_boxes = torch.cat(
            [pred_boxes, ob["pred_boxes"][:, 1].unsqueeze(1)], dim=1
        )

        all_boxes.append(pred_boxes)
        cls = cls_head(features, [pred_boxes], [text_features] * len(chunk))
        cls = cls.sigmoid().view(-1, 1, 5).mean(1)
        all_cls.append(cls * pred_ious)

    all_boxes_t = torch.cat(all_boxes)  # (N, 5, 4)
    all_cls_t = torch.cat(all_cls)      # (N, 5)
    best = torch.argmax(all_cls_t, dim=1)
    boxes_best = all_boxes_t[torch.arange(len(all_boxes_t)), best]
    scores_best = all_cls_t.max(1).values
    return boxes_best.cpu(), scores_best.cpu()


def count_with_thresholds(
    boxes: torch.Tensor, scores: torch.Tensor, score_th: float, nms_th: float
) -> int:
    if boxes.numel() == 0:
        return 0
    keep = vision_ops.nms(boxes, scores, nms_th)
    return int((scores[keep] > score_th).sum().item())


def main() -> None:
    print(f"[setup] device={DEVICE}")
    sam = build_sam_vit_h(checkpoint=str(MODELS / "sam_vit_h.pth"))
    sam.to(DEVICE).eval()
    pd_state = torch.load(
        MODELS / "point_decoder_vith.pth", map_location="cpu", weights_only=False
    )
    point_decoder = PointDecoder(sam).to(DEVICE).eval()
    point_decoder.load_state_dict(pd_state, strict=False)
    cls_head = load_cls_head(MODELS / "MLP_small_box_w1_zeroshot.tar")
    clip_cache = dict(load_text_features(MODELS / "clip_text_features.pt"))
    print("[setup] ready")

    # Heavy pass: for each (image, prompt), compute and stash scores/boxes
    heavy: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for fname, prompts in CASES:
        path = SAMPLES / fname
        if not path.exists():
            print(f"[skip] {fname}")
            continue
        image_1024 = read_image(path)
        for prompt in prompts:
            print(f"[heavy] {fname} × {prompt}")
            tf = get_clip(prompt, clip_cache)
            heavy[(fname, prompt)] = compute_boxes_and_scores(
                image_1024, sam, point_decoder, cls_head, tf
            )

    # Lightweight sweep across score/nms thresholds
    # For each (score_th, nms_th), average |pred - gt| over all (image, prompt)
    print(f"\n{'score_th':>9s} {'nms_th':>7s} | "
          f"{'apple_1(48)':>13s} {'apple_2(46)':>13s} {'pear_1(27)':>12s} {'pear_2(54)':>12s} | {'mean_MAE':>9s}")
    print("-" * 110)

    # Use the best prompt per image from earlier findings
    BEST_PROMPT = {
        "apple_1.jpeg": "apple in paper bag",
        "apple_2.jpeg": "apple in paper bag",
        "pear_image_1751534814_2000.0.jpg": "pear in paper bag",
        "pear_image_1751535399_2000.0.jpg": "pear in paper bag",
    }

    best_global: tuple[float, float, float] = (10**9, 0.0, 0.0)
    results_table = []

    for score_th in SCORE_THRESHOLDS:
        for nms_th in NMS_THRESHOLDS:
            counts = {}
            errs = []
            for fname, gt in GT.items():
                prompt = BEST_PROMPT[fname]
                if (fname, prompt) not in heavy:
                    continue
                b, s = heavy[(fname, prompt)]
                c = count_with_thresholds(b, s, score_th, nms_th)
                counts[fname] = c
                errs.append(abs(c - gt))
            mean_mae = sum(errs) / max(1, len(errs))
            results_table.append((score_th, nms_th, counts, mean_mae))
            a1 = counts.get("apple_1.jpeg", -1)
            a2 = counts.get("apple_2.jpeg", -1)
            p1 = counts.get("pear_image_1751534814_2000.0.jpg", -1)
            p2 = counts.get("pear_image_1751535399_2000.0.jpg", -1)
            print(f"{score_th:>9.2f} {nms_th:>7.2f} | "
                  f"{a1:>13d} {a2:>13d} {p1:>12d} {p2:>12d} | {mean_mae:>9.2f}")
            if mean_mae < best_global[0]:
                best_global = (mean_mae, score_th, nms_th)

    print("-" * 110)
    print(f"[best] mean_MAE={best_global[0]:.2f} at "
          f"score_threshold={best_global[1]}, nms_threshold={best_global[2]}")

    # Also show per-prompt stats for reference (not swept in this table)
    print("\n[per-prompt heavy counts at score_th=0.10, nms_th=0.5]")
    for (fname, prompt), (b, s) in heavy.items():
        c = count_with_thresholds(b, s, 0.10, 0.5)
        print(f"  {fname:>36s}  {prompt:>24s}  count={c:>3d}  gt={GT[fname]}")


if __name__ == "__main__":
    main()

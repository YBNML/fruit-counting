"""Zero-shot inference on user's orchard images using Plan 3 MLP vs upstream MLP.

No ground-truth counts; the point is to see whether the counts are in a
plausible range and whether our fine-tuned MLP behaves reasonably outside
the FSC-147 training distribution.

Run from repo root: python scripts/verify_orchard.py
"""

from __future__ import annotations

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

POINT_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.10
MAX_POINTS = 1000

# (image_filename, prompt) pairs. Each image tested with one natural prompt.
CASES = [
    ("apple_1.jpeg", "apples"),
    ("apple_2.jpeg", "apples"),
    ("pear_image_1751534814_2000.0.jpg", "pears"),
    ("pear_image_1751535399_2000.0.jpg", "pears"),
    ("deploy_1.jpg", "protective fruit bag"),
    ("deploy_2.jpg", "protective fruit bag"),
]


def read_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    transform = A.Compose([
        A.LongestMaxSize(IMAGE_SIZE),
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0,
                      position="top_left"),
    ])
    arr = transform(image=np.array(img))["image"]
    return Image.fromarray(arr)


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
    image: Image.Image,
    sam,
    point_decoder: PointDecoder,
    cls_head: ROIHeadMLP,
    text_features: torch.Tensor,
) -> tuple[int, int]:
    """Return (num_proposals_above_threshold, raw_num_candidate_points)."""
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
    n_points = int(pred_points.size(0))

    if n_points == 0:
        return 0, 0

    all_pred_boxes = []
    cls_outs = []
    for indices in torch.arange(n_points).split(128):
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
    scores = scores[keep]
    return int((scores > SCORE_THRESHOLD).sum().item()), n_points


def get_clip_feature(prompt: str, cache: dict) -> torch.Tensor:
    """Return cached CLIP feature reshaped to (1, 512) on the target device.

    Upstream ROIHeadMLP.forward concatenates the per-image prompt list and
    expects each element to be 2D (N_class, 512); we treat it as one class
    per image here.
    """
    if prompt in cache:
        return cache[prompt].view(1, 512).to(DEVICE)
    # Not in cache — extract now. open_clip download is cached locally.
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
    clip_cache_cpu: dict = dict(clip_cache)  # mutable copy

    print(f"\n{'image':>38s}  {'prompt':>22s}  {'points':>7s}  "
          f"{'upstream':>9s}  {'ours':>5s}")
    print("-" * 100)
    for fname, prompt in CASES:
        path = SAMPLES / fname
        if not path.exists():
            print(f"  skip (missing): {fname}")
            continue
        img = read_image(path)
        tf = get_clip_feature(prompt, clip_cache_cpu)

        up_count, n_pts = run_inference(img, sam, point_decoder,
                                        upstream_head, tf)
        ours_count, _ = run_inference(img, sam, point_decoder,
                                      ours_head, tf)
        print(f"{fname:>38s}  {prompt:>22s}  {n_pts:>7d}  "
              f"{up_count:>9d}  {ours_count:>5d}")


if __name__ == "__main__":
    main()

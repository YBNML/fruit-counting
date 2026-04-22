"""Compute MAE table for orchard images with known GT counts.

GT (user-provided):
  apple_1=48, apple_2=46, pear_1=27, pear_2=54
  (deploy images are blank — excluded)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Re-use the exact inference pipeline from verify_orchard_viz.py
import verify_orchard_viz as vov  # noqa: E402

GT = {
    "apple_1.jpeg": 48,
    "apple_2.jpeg": 46,
    "pear_image_1751534814_2000.0.jpg": 27,
    "pear_image_1751535399_2000.0.jpg": 54,
}

CASES: list[tuple[str, list[str]]] = [
    ("apple_1.jpeg", ["apples", "apple", "apple in paper bag", "protective fruit bag"]),
    ("apple_2.jpeg", ["apples", "apple", "apple in paper bag"]),
    ("pear_image_1751534814_2000.0.jpg", ["pears", "pear", "pear in paper bag"]),
    ("pear_image_1751535399_2000.0.jpg", ["pears", "pear", "pear in paper bag"]),
]


def main() -> None:
    import torch
    from counting.models.pseco.clip_features import load_text_features

    print(f"[setup] device={vov.DEVICE}")
    sam = vov.build_sam_vit_h(checkpoint=str(vov.MODELS / "sam_vit_h.pth"))
    sam.to(vov.DEVICE).eval()
    pd_state = torch.load(
        vov.MODELS / "point_decoder_vith.pth", map_location="cpu", weights_only=False
    )
    point_decoder = vov.PointDecoder(sam).to(vov.DEVICE).eval()
    point_decoder.load_state_dict(pd_state, strict=False)

    upstream_head = vov.load_cls_head(vov.MODELS / "MLP_small_box_w1_zeroshot.tar")
    ours_head = vov.load_cls_head(
        REPO_ROOT / "runs" / "plan3_k32" / "checkpoints" / "best.ckpt"
    )
    print("[setup] ready")

    clip_cache_cpu: dict = dict(load_text_features(vov.MODELS / "clip_text_features.pt"))

    results: dict[tuple[str, str, str], int] = {}

    print(f"\n{'image':>38s}  {'prompt':>22s}  {'GT':>3s}  "
          f"{'up':>4s}  {'|err|':>5s}  {'ours':>5s}  {'|err|':>6s}")
    print("-" * 100)
    for fname, prompts in CASES:
        path = vov.SAMPLES / fname
        if not path.exists():
            continue
        image_1024, _, _ = vov.read_image(path)
        gt = GT[fname]
        for prompt in prompts:
            tf = vov.get_clip(prompt, clip_cache_cpu) if hasattr(vov, "get_clip") \
                else _get_clip_fallback(prompt, clip_cache_cpu)
            up_boxes, _ = vov.run_inference(
                image_1024, sam, point_decoder, upstream_head, tf
            )
            ours_boxes, _ = vov.run_inference(
                image_1024, sam, point_decoder, ours_head, tf
            )
            up = up_boxes.size(0)
            ours = ours_boxes.size(0)
            up_err = abs(gt - up)
            ours_err = abs(gt - ours)
            results[(fname, prompt, "up")] = up
            results[(fname, prompt, "ours")] = ours
            print(f"{fname:>38s}  {prompt:>22s}  {gt:>3d}  "
                  f"{up:>4d}  {up_err:>5d}  {ours:>5d}  {ours_err:>6d}")

    # Per-prompt average MAE
    print("\n--- per-prompt average MAE (across covered images) ---")
    all_prompts = set(p for (_, p, _) in results.keys())
    for prompt in sorted(all_prompts):
        up_errs = []
        ours_errs = []
        for fname, gt in GT.items():
            if (fname, prompt, "up") in results:
                up_errs.append(abs(gt - results[(fname, prompt, "up")]))
                ours_errs.append(abs(gt - results[(fname, prompt, "ours")]))
        if up_errs:
            print(f"  {prompt:>26s}: n={len(up_errs)}  "
                  f"upstream MAE={sum(up_errs) / len(up_errs):6.2f}   "
                  f"ours MAE={sum(ours_errs) / len(ours_errs):7.2f}")


def _get_clip_fallback(prompt, cache):
    """In case vov.get_clip isn't exposed, fall back to direct access."""
    import torch
    from counting.models.pseco.clip_features import encode_class_names
    if prompt in cache:
        return cache[prompt].view(1, 512).to(vov.DEVICE)
    new = encode_class_names([prompt], device="cpu")
    cache[prompt] = new[prompt]
    return new[prompt].view(1, 512).to(vov.DEVICE)


if __name__ == "__main__":
    main()

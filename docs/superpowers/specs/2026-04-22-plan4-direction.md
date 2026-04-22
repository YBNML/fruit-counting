# Plan 4 Direction — Pivot to Upstream-First Inference

- **Date**: 2026-04-22
- **Triggered by**: Plan 3 post-mortem on 4 orchard images with user-provided ground truth

## What changed

Plan 3 shipped a fine-tuned `best.ckpt` that beat the "predict-zero" baseline on FSC-147 val (val/mae = 47.62 < 50.58), which at the time looked like a successful training. Orchard validation with 4 ground-truth images flipped the picture:

| prompt | upstream MAE | Plan 3 MAE |
|---|---|---|
| `pear in paper bag` | **6.00** ⭐ | 400 |
| `apple in paper bag` | **22.50** | 452 |
| `apples` / `pears` | 35–42 | 163–470 |
| `protective fruit bag` | 40 | 443 |

The Plan 3 fine-tuned MLP **no longer respects the text prompt** — "apples", "apple in paper bag", and "protective fruit bag" all return ~490–540 on the same image. The fine-tuning dropped CLIP alignment in favor of a generic "FSC-147-like object" detector.

The upstream pretrained MLP, in contrast, tracks prompts closely and lands within 6–22 of the actual orchard count when fed a prompt of the form `"X in paper bag"`. Upstream + prompt engineering is already production-useful on phone-taken orchard fruit images.

## Decision

**Do not ship Plan 3's fine-tuned MLP.** It is kept for traceability only (checkpoint at `~/fruit-counting/runs/plan3_k32/checkpoints/best.ckpt` on the remote host, not copied into `models/` or used by any default config).

**Pivot Plan 4 from "finish the training pipeline" to "promote the working inference path."** Concretely:

1. **P4.1** — document this direction change (this doc + README).
2. **P4.2** — rewrite `PseCoStage` to use the inference flow that actually works (the one `scripts/verify_orchard_viz.py` already exercises): SAM → PointDecoder → SAM box predictions → ROIHeadMLP → NMS → threshold. The current `PseCoStage.prepare()` imports a non-existent `PseCo.core.PseCo_inference` module; this has been broken since Plan 1 but never surfaced because only the `slow` integration test exercised it. Extend `PseCoStageConfig` with `clip_features_cache` and the tuned hyperparameters (`score_threshold`, `nms_threshold`, `anchor_size`, `point_threshold`, `max_points`).
3. **P4.3** — grid search over `(prompt, score_threshold, nms_threshold, anchor_size)` on the 4 orchard images with GT counts, document the best-performing config for each fruit type.

The heavy C3 path (port upstream's two-loss training with pseudo-box and CLIP-regions pipelines) is parked. It becomes relevant only if (a) we acquire a real labeled bagged-fruit dataset and (b) upstream + prompt engineering proves inadequate on that dataset.

## Data gap noted

The user's `original_code/test_data/00_input/20250809_*.jpg` files we copied as "deploy_*" samples are pure white (`mean=255, min=255, max=255`) — captures that failed before this session. We have no usable images of protective-bagged fruit in the current tree, so the stated production target ("봉지 과일 카운팅") cannot be evaluated yet. This is a prerequisite for any future C3 work.

## What remains out of scope

- Classifier head (ResNet18 for bag verification) — still a Plan 5+ item if it ever becomes needed.
- SR integration — deferred.
- C3 two-loss training reimplementation — conditional on labeled bag data.
- New FSC-147 fine-tuning — the upstream pretrained MLP is what we ship.

## Success criteria for Plan 4

- `counting infer` and `counting batch` run end-to-end against the real inference flow (no `PseCo.core.PseCo_inference` import).
- On the 4 orchard images with GT, the default config reproduces MAE ≤ 25 (match or beat `pear in paper bag` / `apple in paper bag` numbers above).
- Grid search findings are checked in as a CSV and/or short README section.

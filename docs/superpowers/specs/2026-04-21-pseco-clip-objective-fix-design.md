# Plan 3 Spec — PseCo Training Objective Fix (CLIP Text Features)

- **Date**: 2026-04-21
- **Scope**: Replace Plan 2's placeholder training objective with real CLIP text features + proper positive/negative labels, validated on FSC-147
- **Out of scope**: Classifier head training, SR integration, production fruit-orchard training (each → separate follow-up plan)
- **Baseline**: Plan 2's 5-epoch smoke produced `val/mae = 50.58` constant across all epochs (degenerate — model stuck predicting 0)

## 1. Background

Plan 2 built the full PseCo fine-tuning infrastructure (SAM feature cache, PointDecoder proposer, ROIHeadMLP, training loop, TensorBoard, checkpoints) and validated that it runs end-to-end on RTX 5070 in ~3 minutes for 5 epochs on FSC-147. However, the training objective was explicitly placeholder:

- Text prompts were unit vectors `1/√512` — no semantic content
- Per-proposal targets were all `1` — trivially satisfiable by "predict positive always"

Result: `train/cls` collapsed to 0, `val/mae` stuck at the dataset-mean-count value (50.58) for every epoch.

Plan 3 replaces these two placeholders with real signals: CLIP text features for the image's class name, and proposal-level pos/neg labels derived from ground-truth point annotations via box-point containment.

### Success criterion

5-epoch training on FSC-147 shows:
- `val/mae < 50.58` (degenerate baseline)
- Clear decreasing trend across epochs
- `train/cls` remains non-zero (model is actually discriminating pos/neg, not collapsing)

Upstream-level numbers (`MAE ≈ 16-20`) are **not** a target — that requires NMS, augmentation, and longer training, deferred to a later plan.

## 2. Decisions (already agreed)

| Question | Decision |
|---|---|
| Scope | Training objective fix only. Classifier and SR deferred. |
| Positive/negative labeling | Box-point containment: proposal positive iff it contains ≥1 GT point. |
| CLIP implementation | `open_clip_torch` ViT-B/32 pretrained on OpenAI weights. 512-dim text features. |
| CLIP usage | One-time extraction → `.pt` cache. Not a runtime dependency. |
| Success metric | `val/mae < 50.58` with clear decrease over 5 epochs. |
| Inference path | Unchanged. `PseCoStage` forwards the prompt string to upstream `CountingInference`; fine-tuned MLP weights are loaded via existing `init_mlp` path. |

## 3. Changes — files

### New files
| Path | Responsibility |
|---|---|
| `src/counting/models/pseco/clip_features.py` | Load `open_clip` ViT-B/32, extract 512-d text features for a list of class names, save/load dict `{class_name: tensor(512)}` as `.pt`. |
| `src/counting/models/pseco/labeling.py` | Two pure functions: `points_in_box(points, box) -> bool` and `assign_targets_from_points(boxes, points) -> LongTensor` for per-proposal pos/neg labels. |
| `tests/unit/test_pseco_labeling.py` | Label assignment — inside/outside box, multiple points, empty point list. |
| `tests/unit/test_pseco_clip_features.py` | Cache write/read roundtrip. The `open_clip`-loading test is `slow`-marked and skipped when the model download fails or environment lacks weights. |

### Modified files
| Path | What changes |
|---|---|
| `src/counting/data/formats/fsc147.py` | Add `class_name: str` field to `FSC147Record`. Parse `ImageClasses_FSC147.txt` at `FSC147Dataset.__init__`. |
| `tests/unit/test_data_fsc147.py` | Add cases covering `class_name` population and missing-file behavior. |
| `src/counting/config/train_schema.py` | `TrainModelConfig.clip_features_cache: str` (required when objective uses CLIP). |
| `tests/unit/test_config_train_schema.py` | Add validation case for the new required field. |
| `configs/train/pseco_head.yaml` | Set `clip_features_cache: models/PseCo/clip_text_features.pt`. |
| `src/counting/models/pseco/trainer.py` | Remove `_unit_prompts`, replace with CLIP cache lookup per image. Remove `targets = torch.ones(...)`, replace with `assign_targets_from_points`. Thread `class_name` and `points` through the `_CachedFSC147` dataset. |
| `src/counting/cli.py` | Add `extract-clip-features` subcommand (single purpose: build the CLIP cache file). |
| `environment-cuda.yml` / `environment.yml` | Add `open_clip_torch` to pip section. |

### Unchanged
- SAM feature cache layer (reused as-is)
- `Runner`, `CosineWithWarmup`, `EarlyStopping`, checkpoint manager
- Inference pipeline (`PseCoStage`, `build_pipeline`)
- Plan 2's 8.7 GB SAM embedding cache (re-use; no regeneration needed)
- FSC-147 image files on remote (re-use)

## 4. Data flow

### 4.1 One-time CLIP text cache generation

```
ImageClasses_FSC147.txt  ──►  extract unique class names (~147 entries)
        │
        ▼
open_clip.ViT-B-32 (CPU or GPU; seconds)
model.encode_text(tokenizer([name for name in classes]))
        │
        ▼
{ "apple": tensor(512,), "ball": tensor(512,), ..., "vehicle": tensor(512,) }
        │
        ▼
models/PseCo/clip_text_features.pt   (~300 KB total)
```

CLI: `counting extract-clip-features --dataset fsc147 --out models/PseCo/clip_text_features.pt`

(A `--classes-file` variant will be trivially addable later for custom class lists; out of Plan 3 scope.)

### 4.2 Training loop (per batch)

```
FSC147Record:
  relpath      = "7.jpg"
  class_name   = "apple"                    # NEW
  points       = [(33.7, 296.7), (23.8, 256.6), ...]
  count        = len(points)

_CachedFSC147.__getitem__:
  embedding = reader.read(relpath)          # (256, 64, 64) fp16 — Plan 2 cache
  proposals = proposer.propose(embedding)   # PointDecoder — frozen
      pred_points : (1, K, 2) in image coords
  → returns dict with class_name, points, embedding, proposals

step_fn(batch):
  embs          = stacked (B, 256, 64, 64) on device
  boxes         = _topk_points_as_boxes(proposals, k=16) on device       # Plan 2 reused
  prompts       = [clip_cache[cn].view(1, 1, 512) for cn in class_names] # NEW (one per image)
  raw_scores    = ROIHeadMLP(embs, boxes, prompts)       # (B, 16)
  targets       = assign_targets_from_points(boxes, batch["points_per_image"])  # NEW
                                                         # (B*16,) long in {0, 1}
  logits        = stack([-raw_scores, raw_scores], -1).view(B*16, 2)
  pred_counts   = (raw_scores > 0).sum(dim=1).float()    # (B,)
  loss          = pseco_head_loss(logits, targets, pred_counts, gt_counts, ...)
```

### 4.3 Key deltas vs Plan 2

| Element | Plan 2 placeholder | Plan 3 |
|---|---|---|
| text prompt | `1/√512 * ones(1, 1, 512)` | `clip_text_features[class_name]` |
| per-proposal target | all `1` | `1` if box contains any GT point, else `0` |
| class info flowing through | ignored | `FSC147Record.class_name` → prompt lookup |
| GT points used | only the scalar `count` | each `(x, y)` point — via `points_in_box` |

## 5. Labeling specification

### `points_in_box(points, box)`

```python
def points_in_box(points: Sequence[tuple[float, float]],
                  box: tuple[float, float, float, float]) -> bool:
    """True iff at least one (x, y) in `points` lies inside the half-open
    rectangle [x1, x2) × [y1, y2)."""
```

- Inclusive on `x1`/`y1`, exclusive on `x2`/`y2`. Empty `points` → `False`.

### `assign_targets_from_points(boxes_per_image, points_per_image) -> LongTensor`

```python
def assign_targets_from_points(
    boxes_per_image: list[torch.Tensor],    # each (1, K, 4) on any device
    points_per_image: list[list[tuple[float, float]]],
) -> torch.Tensor:                          # shape (B*K,), long, {0, 1}
    """For each proposal, target=1 iff the proposal box contains ≥1 point
    from the SAME image. Flattens in (image, proposal) row-major order."""
```

- Runs on CPU; the resulting tensor is moved to device by the caller.
- Works for K=16 (default) and arbitrary batch sizes.

## 6. Configuration schema delta

`TrainModelConfig` gains one field:

```python
class TrainModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sam_checkpoint: str
    init_decoder: str = ""
    init_mlp: str = ""
    clip_features_cache: str          # NEW — required (non-empty string)
```

YAML update in `configs/train/pseco_head.yaml`:

```yaml
model:
  sam_checkpoint: models/PseCo/sam_vit_h.pth
  init_decoder: models/PseCo/point_decoder_vith.pth
  init_mlp: models/PseCo/MLP_small_box_w1_zeroshot.tar
  clip_features_cache: models/PseCo/clip_text_features.pt   # NEW
```

## 7. CLI delta

`counting extract-clip-features` — new subcommand.

```
Usage: counting extract-clip-features [OPTIONS]

  Extract CLIP ViT-B/32 text features for a set of class names and save
  them to a .pt cache.

Options:
  --dataset TEXT      Either 'fsc147' (auto-derives from ImageClasses_FSC147.txt)
                      or path to a newline-delimited text file of class names.
                      [default: fsc147]
  --dataset-root PATH FSC-147 root (only used when --dataset=fsc147).
                      [default: ./datasets/fsc147]
  --out PATH          Output .pt file. [required]
  --device TEXT       cpu | mps | cuda | auto. [default: cpu]
  --help              Show this message and exit.
```

Other existing CLI commands unchanged. The trainer CLI (`counting train pseco-head`) reads the cache path from `TrainModelConfig.clip_features_cache` — no user-visible CLI flag for it.

## 8. Testing strategy

### New unit tests

| Test | Behavior verified |
|---|---|
| `test_points_in_box_basic` | Point inside, outside, on edge, empty list. |
| `test_assign_targets_from_points_shape` | `(B*K,)` shape, dtype `long`, values in `{0, 1}`. |
| `test_assign_targets_from_points_correctness` | Hand-constructed example: 2 images, 4 boxes each, known points → verify exact label vector. |
| `test_fsc147_class_name_populated` | `FSC147Record.class_name` matches `ImageClasses_FSC147.txt`. |
| `test_fsc147_missing_classes_file_raises` | Missing file → `FileNotFoundError` with path in message. |
| `test_train_schema_clip_cache_required` | Empty `clip_features_cache` → `ValidationError`. |
| `test_clip_features_cache_roundtrip` | Stub text features (non-CLIP, just tensors) write/read. |
| `test_clip_features_load_slow` (slow + skipif) | Actual `open_clip` encode for 3 class names. Skipped when network or weights missing. |

### Existing tests
- `test_config_train_schema.py` — minimal config updated to include `clip_features_cache`.
- `test_data_fsc147.py` — `_make_annotations` helper updated to include `ImageClasses_FSC147.txt`.
- All pass after edits. No change expected to non-schema / non-FSC147 tests.

### Integration smoke
- `tests/integration/test_training_smoke.py` gets two additions:
  - A fake CLIP cache is injected (`{synthetic_class: random tensor(512)}`) so the test doesn't require `open_clip`.
  - A fake `ImageClasses_FSC147.txt` is written for the tiny synthetic dataset.
- Existing `skipif(no_weights)` gate stays.

### Real validation
Not a pytest — executed manually after implementation on the remote GPU host:
1. `counting extract-clip-features --dataset fsc147 --out models/PseCo/clip_text_features.pt` (Mac, ~30s)
2. `scp models/PseCo/clip_text_features.pt remote:~/fruit-counting/models/PseCo/`
3. `counting train pseco-head --config configs/train/pseco_head.yaml --set train.epochs=5 --set run_name=plan3_e5` on remote
4. Compare TensorBoard: `val/mae` should drop below 50.58 with clear trend.
5. Record exact final MAE — used as Plan 3 success artifact (include in README).

## 9. Error handling

| Situation | Behavior |
|---|---|
| `clip_features_cache` path missing on disk | Raise `FileNotFoundError` at trainer startup, before any epoch. |
| Class name in a batch not present in CLIP cache | Raise `KeyError` naming the missing class + hint to re-run `extract-clip-features`. |
| FSC-147 root missing `ImageClasses_FSC147.txt` | Raise `FileNotFoundError` at `FSC147Dataset.__init__`. |
| Image has zero GT points (count = 0) | All proposals get target=0; training runs normally. No special case. |
| `open_clip` model download failure (no network, etc.) | `counting extract-clip-features` surfaces the underlying error; user retries once network is restored. |
| `assign_targets_from_points` receives mismatched B lengths | Raise `ValueError` with both lengths in message. |

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `open_clip` download fails on Mac | Low | Widely deployed; retry. If persistent, document manual weight download URL in README. |
| `ImageClasses_FSC147.txt` format differs from assumption | Medium | First implementation step: inspect actual file (we already have it on remote) and encode exact parser. |
| All proposals land on negative (box too small / points sparse) | Medium | Start with `half_side=16` (current default). If training shows 0 pos rate, expose `proposal_box_half_side` in config. |
| pos/neg imbalance slows convergence | Medium | Observed rate → if very skewed, add `pos_weight` to cross-entropy. Plan 3 success criterion tolerates this since baseline is so bad. |
| Upstream `ROIHeadMLP` expects prompts in a different shape than we pass | Low | Plan 2 trainer already calls `ROIHeadMLP(embs, boxes, prompts)` with shape `(1, 1, 512)` per image — verified by `training complete` in Plan 2. We only change the *contents* of those tensors. |
| Inference-time MLP loading mismatch after fine-tune | Low | Fine-tuned checkpoint uses the same `cls_head` wrapper key format (Runner's `save_checkpoint` writes raw `state_dict`, so we need to verify compatibility). See §11. |

## 11. Checkpoint compatibility note

Plan 2's `save_checkpoint` writes:
```python
{
    "model": model.state_dict(),
    "optimizer": ..., "scheduler": ..., "epoch": ..., "best_metric": ..., "config_snapshot": ...,
}
```

Plan 2's trainer loads upstream `MLP_small_box_w1_zeroshot.tar` via the `cls_head` wrapper detection logic (Plan 2 patch f999f8d). A Plan 3-trained `best.ckpt` will use the `"model"` key instead, which the same detection logic handles. So a Plan 3 checkpoint is loadable by the trainer for resume, and by `PseCoStage` for inference (inference loads via upstream `CountingInference.__init__`, which expects the `cls_head` structure — **we need to verify this path works with our `"model"`-wrapped checkpoint**).

**Action**: during Plan 3 validation step, explicitly load the Plan 3 `best.ckpt` into the inference pipeline and confirm it runs (`counting infer` on a test image). If it fails, Plan 3 adds a small checkpoint-format helper to re-wrap into `{"cls_head": state_dict}` on save. Noted as a follow-up item rather than a pre-implementation risk since inference integration tests are part of real validation.

## 12. Development order (incremental vertical slices)

1. **`clip_features.py`** — text extraction + cache write/read utilities (+ unit tests)
2. **`counting extract-clip-features` CLI** (+ smoke)
3. **`labeling.py`** — pos/neg assignment (+ unit tests)
4. **`FSC147Dataset` extension** — parse `ImageClasses_FSC147.txt`, add `class_name` to records (+ updated tests)
5. **`TrainAppConfig.model.clip_features_cache`** — schema + YAML update (+ updated tests)
6. **`trainer.py` rewrite** — use CLIP cache + real targets (remove placeholders). Update integration smoke.
7. **Full test suite pass** locally (Mac) → push → pull on remote.
8. **CLIP cache extraction** (Mac, ~30s) → scp to remote.
9. **5-epoch training** on remote GPU → TensorBoard analysis.
10. **Success judgment** → record MAE in README; close Plan 3. Or iterate (adjust proposal box size, pos_weight, etc.) if criteria not met.

## 13. Non-goals (deferred)

- Classifier learning (ResNet18) — next plan
- SR integration — next plan or later
- Custom orchard dataset training — needs our own labeling pipeline
- Multi-class softmax over all FSC-147 classes simultaneously — binary is sufficient for the success criterion
- NMS-based proposal filtering — may improve MAE but not required for "degenerate escape"
- `pos_weight` / class rebalancing — only if baseline shows severe skew
- `--classes-file` for the extract-clip-features CLI — trivially addable later

## 14. Success artifacts (end of Plan 3)

1. A Plan 3 `best.ckpt` with `val/mae < 50.58` (recorded number in README)
2. A `clip_text_features.pt` cache (checked into git via LFS if small enough, or included in README instructions; likely ~300 KB so fine to commit directly — **decision**: commit directly since small)
3. TensorBoard event file showing the decreasing curve (for qualitative inspection; not committed)
4. Updated README with "Plan 3 결과" line showing the MAE number
5. Informed decision on whether Classifier training (original Plan 3 content) is needed, based on Plan 3 results

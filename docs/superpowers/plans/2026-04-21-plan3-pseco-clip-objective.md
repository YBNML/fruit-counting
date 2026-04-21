# Plan 3 — PseCo Training Objective Fix (CLIP + Pos/Neg Labels)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Plan 2's placeholder training objective (unit-vector prompts + all-positive targets) with real CLIP ViT-B/32 text features + proposal-level pos/neg labels from GT point containment, then validate that 5-epoch training on FSC-147 produces `val/mae < 50.58` with a clear decreasing trend.

**Architecture:** Add two new small modules (`clip_features.py` for CLIP extraction/caching, `labeling.py` for pos/neg assignment), extend `FSC147Dataset` with per-image class names, extend `TrainAppConfig` with a CLIP cache path, rewrite `trainer.py`'s `step_fn`/`eval_fn` to use real prompts and targets. CLIP is a one-time cache; not a runtime dependency.

**Tech Stack:** Python 3.11, PyTorch 2.5+ (cu128 on remote for Blackwell), `open_clip_torch` (CLIP ViT-B/32), Pydantic v2, Typer, pytest. External: `external/PseCo` (submodule, unchanged).

**Spec:** `docs/superpowers/specs/2026-04-21-pseco-clip-objective-fix-design.md`

**Prerequisites:**
- Plan 1 (Foundation) complete (commit `5e2e12d` and following).
- Plan 2 (PseCo training infra) complete (commit `680fe7e` plus the two fixes `f999f8d`, `3b23f2e`, `8eac20f`).
- Remote has `~/fruit-counting/` synced, 8.7 GB SAM feature cache at `~/fruit-counting/feature_cache/fsc147_vit_h/`, FSC-147 at `~/fruit-counting/datasets/fsc147/` (including `ImageClasses_FSC147.txt`).
- `ImageClasses_FSC147.txt` format confirmed: tab-separated `<image_filename>\t<class_name>` per line. 6146 rows, 147 unique classes (e.g. `"apples"`, `"sea shells"`, `"hot air balloons"`).

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `src/counting/models/pseco/clip_features.py` | Load open_clip ViT-B/32, encode a list of class name strings to 512-d text features, save/load as `.pt` dict `{class_name: torch.Tensor(512)}`. |
| `src/counting/models/pseco/labeling.py` | Two pure functions: `points_in_box(points, box) -> bool` and `assign_targets_from_points(boxes_per_image, points_per_image) -> LongTensor` for per-proposal pos/neg labels. |
| `tests/unit/test_pseco_clip_features.py` | Cache write/read roundtrip with a synthetic `{name: tensor}` dict. The `open_clip`-loading test is `slow`-marked. |
| `tests/unit/test_pseco_labeling.py` | Box-point containment + batch-level label assignment, hand-constructed correctness case. |

### Modified files

| Path | What changes |
|---|---|
| `src/counting/data/formats/fsc147.py` | Add `class_name: str` to `FSC147Record`. Parse `ImageClasses_FSC147.txt` at `FSC147Dataset.__init__`. |
| `tests/unit/test_data_fsc147.py` | Add two tests: class_name populated, missing-classes-file raises. Update `_make_annotations` helper to write a tiny `ImageClasses_FSC147.txt`. |
| `src/counting/config/train_schema.py` | Add required `clip_features_cache: str` to `TrainModelConfig`. |
| `tests/unit/test_config_train_schema.py` | Update helper to include `clip_features_cache`; add test for empty-string rejection. |
| `configs/train/pseco_head.yaml` | Add `clip_features_cache: models/PseCo/clip_text_features.pt` under `model:`. |
| `src/counting/models/pseco/trainer.py` | Remove `_unit_prompts`, `targets = torch.ones(...)`, replace with CLIP cache lookup + `assign_targets_from_points`. Thread `class_name` and GT `points` through `_CachedFSC147`. |
| `src/counting/cli.py` | Add `extract-clip-features` subcommand. |
| `tests/integration/test_training_smoke.py` | Inject a fake CLIP cache (random tensors) and a tiny `ImageClasses_FSC147.txt`. |
| `environment.yml`, `environment-cuda.yml` | Add `open_clip_torch` to pip section. |

### Unchanged

- `original_code/**`
- `external/PseCo/` submodule
- Plan 2 SAM embedding cache on remote
- FSC-147 image files on remote
- Inference pipeline (`PseCoStage`, `build_pipeline`, CLI `infer`/`batch`)
- `Runner`, `CosineWithWarmup`, `EarlyStopping`, checkpoint manager

---

## Task 1: CLIP text feature cache utilities + unit tests

**Files:**
- Create: `src/counting/models/pseco/clip_features.py`
- Create: `tests/unit/test_pseco_clip_features.py`

---

- [ ] **Step 1: Write failing tests (roundtrip only; open_clip test is slow-marked)**

Create `tests/unit/test_pseco_clip_features.py`:

```python
from pathlib import Path

import pytest
import torch

from counting.models.pseco.clip_features import (
    load_text_features,
    save_text_features,
)


def test_save_and_load_roundtrip(tmp_path):
    features = {
        "apples": torch.randn(512),
        "sea shells": torch.randn(512),
        "hot air balloons": torch.randn(512),
    }
    out = tmp_path / "clip_text_features.pt"
    save_text_features(features, out)
    assert out.exists()

    loaded = load_text_features(out)
    assert set(loaded.keys()) == set(features.keys())
    for name, tensor in features.items():
        torch.testing.assert_close(loaded[name], tensor)


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_features(tmp_path / "missing.pt")


def test_load_rejects_nondict(tmp_path):
    out = tmp_path / "bogus.pt"
    torch.save(torch.randn(3), out)
    with pytest.raises(ValueError, match="mapping"):
        load_text_features(out)


def test_load_rejects_wrong_dim(tmp_path):
    out = tmp_path / "wrong.pt"
    torch.save({"apples": torch.randn(256)}, out)
    with pytest.raises(ValueError, match="512"):
        load_text_features(out)


@pytest.mark.slow
def test_encode_with_open_clip_real():
    """Actually load open_clip ViT-B/32 and encode three class names.

    Skipped in quick test runs; run explicitly via `pytest -m slow` after the
    weights are cached locally.
    """
    open_clip = pytest.importorskip("open_clip")  # noqa: F841
    from counting.models.pseco.clip_features import encode_class_names

    features = encode_class_names(
        ["apples", "sea shells", "hot air balloons"],
        device="cpu",
    )
    assert set(features.keys()) == {"apples", "sea shells", "hot air balloons"}
    for tensor in features.values():
        assert tensor.shape == (512,)
        assert tensor.dtype == torch.float32
```

- [ ] **Step 2: Run tests — verify they fail with ImportError**

Run: `conda run -n counting-env pytest tests/unit/test_pseco_clip_features.py -m "not slow" -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement `clip_features.py`**

Create `src/counting/models/pseco/clip_features.py`:

```python
"""CLIP ViT-B/32 text feature extraction and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

_CLIP_DIM = 512
_CLIP_MODEL_NAME = "ViT-B-32"
_CLIP_PRETRAINED = "openai"


def encode_class_names(
    class_names: Iterable[str],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Return {class_name: tensor(512,)} by forwarding class names through
    open_clip's ViT-B/32 text tower. Requires the `open_clip_torch` package."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(_CLIP_MODEL_NAME)
    model.to(device).eval()

    names = list(class_names)
    tokens = tokenizer(names).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens).float().cpu()

    return {name: feats[i] for i, name in enumerate(names)}


def save_text_features(
    features: dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    """Save `{name: tensor(512,)}` dict to a `.pt` file, creating parents."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, out)


def load_text_features(path: str | Path) -> dict[str, torch.Tensor]:
    """Load `.pt` cache, validating structure: dict[str, Tensor(512,)]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CLIP feature cache not found: {p}")
    data = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"CLIP cache at {p} is not a mapping (got {type(data).__name__})")
    for name, tensor in data.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (_CLIP_DIM,):
            raise ValueError(
                f"CLIP cache entry {name!r} has shape {tuple(tensor.shape)}; expected (512,)"
            )
    return data
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `conda run -n counting-env pytest tests/unit/test_pseco_clip_features.py -m "not slow" -v`
Expected: 4 passed, 1 deselected (slow).

- [ ] **Step 5: Commit**

```bash
git add src/counting/models/pseco/clip_features.py tests/unit/test_pseco_clip_features.py
git commit -m "feat(models): CLIP ViT-B/32 text feature extraction and caching"
```

---

## Task 2: Pos/neg labeling utilities + unit tests

**Files:**
- Create: `src/counting/models/pseco/labeling.py`
- Create: `tests/unit/test_pseco_labeling.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_pseco_labeling.py`:

```python
import pytest
import torch

from counting.models.pseco.labeling import (
    assign_targets_from_points,
    points_in_box,
)


def test_point_inside_box():
    assert points_in_box([(5.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is True


def test_point_outside_box():
    assert points_in_box([(11.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is False


def test_point_on_left_edge_included():
    assert points_in_box([(0.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is True


def test_point_on_right_edge_excluded():
    assert points_in_box([(10.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is False


def test_empty_points_is_false():
    assert points_in_box([], (0.0, 0.0, 10.0, 10.0)) is False


def test_multiple_points_any_inside():
    pts = [(50.0, 50.0), (5.0, 5.0)]
    assert points_in_box(pts, (0.0, 0.0, 10.0, 10.0)) is True


def test_assign_targets_shape_and_dtype():
    boxes = [torch.tensor([[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]])]
    points = [[(5.0, 5.0)]]
    targets = assign_targets_from_points(boxes, points)
    assert targets.shape == (2,)
    assert targets.dtype == torch.long
    assert targets.tolist() == [1, 0]


def test_assign_targets_two_images_flattened_row_major():
    # Image 0: 2 boxes; first contains a point, second does not.
    # Image 1: 2 boxes; neither contains a point.
    boxes = [
        torch.tensor([[[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]]),
        torch.tensor([[[100.0, 100.0, 110.0, 110.0], [200.0, 200.0, 210.0, 210.0]]]),
    ]
    points = [
        [(5.0, 5.0)],
        [(0.0, 0.0)],  # not inside either image-1 box
    ]
    targets = assign_targets_from_points(boxes, points)
    # Row-major: image0_box0, image0_box1, image1_box0, image1_box1
    assert targets.tolist() == [1, 0, 0, 0]


def test_assign_targets_mismatched_lengths_raises():
    boxes = [torch.zeros((1, 2, 4))]
    points: list[list[tuple[float, float]]] = [[], []]  # 2 images worth
    with pytest.raises(ValueError, match="length"):
        assign_targets_from_points(boxes, points)
```

- [ ] **Step 2: Run tests — verify they fail with ImportError**

Run: `conda run -n counting-env pytest tests/unit/test_pseco_labeling.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement `labeling.py`**

Create `src/counting/models/pseco/labeling.py`:

```python
"""Positive/negative target assignment for PseCo proposals.

Convention: a box `(x1, y1, x2, y2)` covers the half-open rectangle
`[x1, x2) × [y1, y2)`. A proposal is POSITIVE (target=1) iff at least one
ground-truth point lies within that rectangle.
"""

from __future__ import annotations

from typing import Sequence

import torch


def points_in_box(
    points: Sequence[tuple[float, float]],
    box: tuple[float, float, float, float],
) -> bool:
    """True iff any point lies in the half-open rectangle [x1,x2) × [y1,y2)."""
    x1, y1, x2, y2 = box
    for x, y in points:
        if x1 <= x < x2 and y1 <= y < y2:
            return True
    return False


def assign_targets_from_points(
    boxes_per_image: list[torch.Tensor],
    points_per_image: list[list[tuple[float, float]]],
) -> torch.Tensor:
    """Return a flat (B*K,) long tensor of pos/neg labels.

    Each `boxes_per_image[i]` has shape `(1, K, 4)` (matching PseCo's
    `ROIHeadMLP.forward`). `points_per_image[i]` is the list of GT points for
    image `i`. Output is flattened in row-major (image, proposal) order.
    """
    if len(boxes_per_image) != len(points_per_image):
        raise ValueError(
            f"length mismatch: boxes={len(boxes_per_image)} points={len(points_per_image)}"
        )

    labels: list[int] = []
    for boxes_tensor, gt_points in zip(boxes_per_image, points_per_image):
        boxes = boxes_tensor.reshape(-1, 4).tolist()
        for box in boxes:
            labels.append(1 if points_in_box(gt_points, tuple(box)) else 0)
    return torch.tensor(labels, dtype=torch.long)
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `conda run -n counting-env pytest tests/unit/test_pseco_labeling.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/counting/models/pseco/labeling.py tests/unit/test_pseco_labeling.py
git commit -m "feat(models): pos/neg target assignment via box-point containment"
```

---

## Task 3: FSC147Dataset class_name extension

**Files:**
- Modify: `src/counting/data/formats/fsc147.py`
- Modify: `tests/unit/test_data_fsc147.py`

---

- [ ] **Step 1: Update the test `_make_annotations` helper to write ImageClasses_FSC147.txt, and add two new tests**

Edit `tests/unit/test_data_fsc147.py` — replace the `_make_annotations` function and append two tests. Full replacement content for the file:

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from counting.data.formats.fsc147 import FSC147Dataset


def _make_annotations(root: Path):
    imgs = root / "images_384_VarV2"
    imgs.mkdir(parents=True)
    for name in ("1.jpg", "2.jpg", "3.jpg"):
        arr = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(imgs / name)

    (root / "annotation_FSC147_384.json").write_text(json.dumps({
        "1.jpg": {
            "points": [[10, 10], [20, 20]],
            "box_examples_coordinates": [[[0, 0], [0, 30], [30, 30], [30, 0]]],
        },
        "2.jpg": {
            "points": [[5, 5]],
            "box_examples_coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0]]],
        },
        "3.jpg": {"points": [], "box_examples_coordinates": []},
    }))
    (root / "Train_Test_Val_FSC_147.json").write_text(json.dumps({
        "train": ["1.jpg", "2.jpg"],
        "val": ["3.jpg"],
        "test": [],
    }))
    # Tab-separated filename<TAB>class_name, one per line
    (root / "ImageClasses_FSC147.txt").write_text(
        "1.jpg\tapples\n2.jpg\tapples\n3.jpg\tsea shells\n"
    )


def test_fsc147_loads_split(tmp_path):
    _make_annotations(tmp_path)
    ds = FSC147Dataset(tmp_path, split="train")

    assert len(ds) == 2
    rec = ds[0]
    assert rec.relpath in {"1.jpg", "2.jpg"}
    img = rec.read_rgb()
    assert img.shape[:2] == (48, 64)
    assert len(rec.points) >= 0
    assert isinstance(rec.count, int)


def test_fsc147_points_match_annotations(tmp_path):
    _make_annotations(tmp_path)
    ds = FSC147Dataset(tmp_path, split="train")
    by_name = {r.relpath: r for r in ds}
    assert by_name["1.jpg"].count == 2
    assert by_name["2.jpg"].count == 1
    assert by_name["1.jpg"].points == [(10.0, 10.0), (20.0, 20.0)]


def test_fsc147_bad_split_raises(tmp_path):
    import pytest

    _make_annotations(tmp_path)
    with pytest.raises(ValueError, match="split"):
        FSC147Dataset(tmp_path, split="bogus")


def test_fsc147_missing_files_raise(tmp_path):
    import pytest

    (tmp_path / "images_384_VarV2").mkdir()
    with pytest.raises(FileNotFoundError):
        FSC147Dataset(tmp_path, split="train")


def test_fsc147_malformed_json_raises_with_path(tmp_path):
    import pytest

    _make_annotations(tmp_path)
    (tmp_path / "annotation_FSC147_384.json").write_text("{ not valid json")

    with pytest.raises(ValueError, match="annotation_FSC147_384.json"):
        FSC147Dataset(tmp_path, split="train")


def test_fsc147_class_name_populated(tmp_path):
    _make_annotations(tmp_path)
    ds = FSC147Dataset(tmp_path, split="train")
    by_name = {r.relpath: r for r in ds}
    assert by_name["1.jpg"].class_name == "apples"
    assert by_name["2.jpg"].class_name == "apples"


def test_fsc147_missing_classes_file_raises(tmp_path):
    import pytest

    _make_annotations(tmp_path)
    (tmp_path / "ImageClasses_FSC147.txt").unlink()
    with pytest.raises(FileNotFoundError, match="ImageClasses_FSC147.txt"):
        FSC147Dataset(tmp_path, split="train")
```

- [ ] **Step 2: Run tests — expect failures on the two new tests (class_name missing) and on existing tests (helper writes new file, but FSC147Dataset doesn't read it yet)**

Run: `conda run -n counting-env pytest tests/unit/test_data_fsc147.py -v`
Expected: mix of failures; specifically `test_fsc147_class_name_populated` and `test_fsc147_missing_classes_file_raises` fail. The other existing tests may still pass because the current dataset doesn't require the classes file.

- [ ] **Step 3: Modify `FSC147Record` and `FSC147Dataset` to parse and expose class_name**

Replace the contents of `src/counting/data/formats/fsc147.py`:

```python
"""FSC-147 dataset adapter.

Expected layout (matches upstream PseCo):
    root/
      images_384_VarV2/<image_id>.jpg
      annotation_FSC147_384.json     # per-image points + box examples
      Train_Test_Val_FSC_147.json    # split mapping
      ImageClasses_FSC147.txt        # tab-separated <image>\t<class_name>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from counting.utils.image import read_image_rgb

_VALID_SPLITS = {"train", "val", "test"}


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def _load_image_classes(path: Path) -> dict[str, str]:
    """Parse tab-separated `<image_filename>\t<class_name>` file."""
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected tab-separated <image>\\t<class>, got {line!r}"
                )
            mapping[parts[0]] = parts[1]
    return mapping


@dataclass(frozen=True)
class FSC147Record:
    path: Path
    relpath: str
    points: list[tuple[float, float]]
    box_examples: list[list[tuple[float, float]]]
    count: int
    class_name: str

    def read_rgb(self) -> np.ndarray:
        return read_image_rgb(self.path)


class FSC147Dataset:
    def __init__(self, root: str | Path, *, split: str = "train") -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Valid: {sorted(_VALID_SPLITS)}")
        self.root = Path(root)
        self.split = split

        img_dir = self.root / "images_384_VarV2"
        ann_path = self.root / "annotation_FSC147_384.json"
        split_path = self.root / "Train_Test_Val_FSC_147.json"
        classes_path = self.root / "ImageClasses_FSC147.txt"
        for p in (img_dir, ann_path, split_path, classes_path):
            if not p.exists():
                raise FileNotFoundError(f"FSC-147 artifact missing: {p}")

        splits = _load_json(split_path)
        annotations = _load_json(ann_path)
        image_classes = _load_image_classes(classes_path)

        self._records: list[FSC147Record] = []
        for name in splits.get(split, []):
            ann = annotations.get(name, {})
            pts = [(float(x), float(y)) for x, y in ann.get("points", [])]
            boxes = [
                [(float(x), float(y)) for x, y in box]
                for box in ann.get("box_examples_coordinates", [])
            ]
            self._records.append(FSC147Record(
                path=img_dir / name,
                relpath=name,
                points=pts,
                box_examples=boxes,
                count=len(pts),
                class_name=image_classes.get(name, ""),
            ))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> FSC147Record:
        return self._records[idx]

    def __iter__(self) -> Iterator[FSC147Record]:
        return iter(self._records)
```

- [ ] **Step 4: Run tests — verify all pass**

Run: `conda run -n counting-env pytest tests/unit/test_data_fsc147.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/counting/data/formats/fsc147.py tests/unit/test_data_fsc147.py
git commit -m "feat(data): FSC147Dataset parses ImageClasses_FSC147.txt into class_name"
```

---

## Task 4: TrainAppConfig `clip_features_cache` field

**Files:**
- Modify: `src/counting/config/train_schema.py`
- Modify: `tests/unit/test_config_train_schema.py`
- Modify: `configs/train/pseco_head.yaml`

---

- [ ] **Step 1: Update the test helper and add a rejection test**

Replace `tests/unit/test_config_train_schema.py` entirely:

```python
import pytest
from pydantic import ValidationError

from counting.config.train_schema import TrainAppConfig


def _minimal():
    return {
        "run_name": "pseco_head_v1",
        "device": "cuda",
        "seed": 42,
        "output_dir": "./runs",
        "data": {
            "format": "fsc147",
            "root": "./datasets/fsc147",
            "train_split": "train",
            "val_split": "val",
            "image_size": 1024,
        },
        "model": {
            "sam_checkpoint": "models/PseCo/sam_vit_h.pth",
            "init_decoder": "models/PseCo/point_decoder_vith.pth",
            "init_mlp": "models/PseCo/MLP_small_box_w1_zeroshot.tar",
            "clip_features_cache": "models/PseCo/clip_text_features.pt",
        },
        "cache": {
            "enabled": True,
            "dir": "./feature_cache/fsc147_vit_h",
            "dtype": "float16",
            "augment_variants": 1,
        },
        "train": {
            "batch_size": 8,
            "epochs": 30,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_steps": 500,
            "scheduler": "cosine",
            "loss_weights": {"cls": 1.0, "count": 0.1},
            "early_stopping": {"patience": 5, "metric": "val_mae", "mode": "min"},
        },
        "logging": {
            "tensorboard": True,
            "log_every_n_steps": 20,
            "save_every_n_epochs": 1,
        },
    }


def test_valid_parses():
    cfg = TrainAppConfig.model_validate(_minimal())
    assert cfg.run_name == "pseco_head_v1"
    assert cfg.data.format == "fsc147"
    assert cfg.train.scheduler == "cosine"
    assert cfg.model.clip_features_cache.endswith("clip_text_features.pt")


def test_scheduler_must_be_known():
    d = _minimal()
    d["train"]["scheduler"] = "tangent"
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_dtype_must_be_float16_or_float32():
    d = _minimal()
    d["cache"]["dtype"] = "int8"
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_augment_variants_positive():
    d = _minimal()
    d["cache"]["augment_variants"] = 0
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_early_stopping_mode_literal():
    d = _minimal()
    d["train"]["early_stopping"]["mode"] = "sideways"
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_clip_features_cache_required_non_empty():
    d = _minimal()
    d["model"]["clip_features_cache"] = ""
    with pytest.raises(ValidationError, match="clip_features_cache"):
        TrainAppConfig.model_validate(d)
```

- [ ] **Step 2: Run tests — expect failure on `test_clip_features_cache_required_non_empty` (field not defined)**

Run: `conda run -n counting-env pytest tests/unit/test_config_train_schema.py -v`
Expected: `test_valid_parses` fails (missing field not tolerated by `extra="forbid"` → wait, this is ADDING a field to the schema, so existing without it should fail. But the test dict HAS the field, so `test_valid_parses` should pass AFTER schema change. Before schema change, the extra field in the dict fails validation.)

Actually, you'll see: multiple failures because `_minimal()` now includes `clip_features_cache` but the schema doesn't allow it (extra="forbid"). The new rejection test also fails.

- [ ] **Step 3: Add `clip_features_cache` field with validator to `TrainModelConfig`**

Edit `src/counting/config/train_schema.py` — replace the `TrainModelConfig` class block only (keep everything else in the file intact):

Find:
```python
class TrainModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sam_checkpoint: str
    init_decoder: str = ""
    init_mlp: str = ""
```

Replace with:
```python
class TrainModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sam_checkpoint: str
    init_decoder: str = ""
    init_mlp: str = ""
    clip_features_cache: str

    @field_validator("clip_features_cache")
    @classmethod
    def _non_empty_cache_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("clip_features_cache must be a non-empty path")
        return v
```

Also ensure `field_validator` is imported at the top of the file. Find the pydantic import line:
```python
from pydantic import BaseModel, ConfigDict, Field
```
Replace with:
```python
from pydantic import BaseModel, ConfigDict, Field, field_validator
```

- [ ] **Step 4: Update YAML config**

Edit `configs/train/pseco_head.yaml` — find the `model:` block and add the cache path. Replace:

```yaml
model:
  sam_checkpoint: models/PseCo/sam_vit_h.pth
  init_decoder: models/PseCo/point_decoder_vith.pth
  init_mlp: models/PseCo/MLP_small_box_w1_zeroshot.tar
```

With:

```yaml
model:
  sam_checkpoint: models/PseCo/sam_vit_h.pth
  init_decoder: models/PseCo/point_decoder_vith.pth
  init_mlp: models/PseCo/MLP_small_box_w1_zeroshot.tar
  clip_features_cache: models/PseCo/clip_text_features.pt
```

- [ ] **Step 5: Run tests — verify all pass**

Run: `conda run -n counting-env pytest tests/unit/test_config_train_schema.py -v`
Expected: 6 passed.

Also:
```bash
conda run -n counting-env counting validate-config configs/train/pseco_head.yaml 2>&1 | head -3
```

Actually `validate-config` expects an `AppConfig` (inference) not `TrainAppConfig`. The pipeline YAML and the train YAML are different schemas; don't run `validate-config` on the train YAML.

Instead, a direct check:
```bash
conda run -n counting-env python -c "
import yaml
from counting.config.train_schema import TrainAppConfig
cfg = TrainAppConfig.model_validate(yaml.safe_load(open('configs/train/pseco_head.yaml')))
print('OK', cfg.run_name, cfg.model.clip_features_cache)
"
```
Expected: `OK pseco_head_v1 models/PseCo/clip_text_features.pt`.

- [ ] **Step 6: Commit**

```bash
git add src/counting/config/train_schema.py configs/train/pseco_head.yaml tests/unit/test_config_train_schema.py
git commit -m "feat(config): TrainModelConfig.clip_features_cache (required)"
```

---

## Task 5: `extract-clip-features` CLI command

**Files:**
- Modify: `src/counting/cli.py`

No unit test (small glue; exercised in Task 9 real validation run).

---

- [ ] **Step 1: Append the new subcommand to `cli.py`**

Edit `src/counting/cli.py` — append at end of file (after all existing commands, before `if __name__ == "__main__":`):

```python


@app.command("extract-clip-features")
def extract_clip_features(
    dataset: str = typer.Option(
        "fsc147",
        "--dataset",
        help="Either 'fsc147' (auto-derive from ImageClasses_FSC147.txt) or a path to a newline-delimited text file of class names.",
    ),
    dataset_root: str = typer.Option(
        "./datasets/fsc147",
        "--dataset-root",
        help="FSC-147 root (used when --dataset=fsc147).",
    ),
    out: str = typer.Option(..., "--out", "-o", help="Output .pt file"),
    device: str = typer.Option(
        "cpu", "--device", help="cpu | mps | cuda | auto"
    ),
) -> None:
    """Extract CLIP ViT-B/32 text features for a set of class names."""
    from pathlib import Path

    from counting.models.pseco.clip_features import (
        encode_class_names,
        save_text_features,
    )
    from counting.utils.device import resolve_device

    if dataset == "fsc147":
        classes_file = Path(dataset_root) / "ImageClasses_FSC147.txt"
        if not classes_file.exists():
            raise typer.BadParameter(f"Classes file not found: {classes_file}")
        names: list[str] = []
        seen: set[str] = set()
        with classes_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                cname = parts[1]
                if cname not in seen:
                    seen.add(cname)
                    names.append(cname)
    else:
        p = Path(dataset)
        if not p.exists():
            raise typer.BadParameter(f"Class list file not found: {p}")
        names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    resolved_device = resolve_device(device)
    console.print(f"[loading] open_clip ViT-B/32 on {resolved_device}")
    features = encode_class_names(names, device=resolved_device)
    save_text_features(features, out)
    console.print(f"[green]done[/green] {len(features)} classes → {out}")
```

- [ ] **Step 2: Verify help renders**

Run: `conda run -n counting-env counting extract-clip-features --help 2>&1 | head -15`
Expected: help text listing `--dataset`, `--dataset-root`, `--out`, `--device`.

- [ ] **Step 3: Run non-slow tests — verify no regressions**

Run: `conda run -n counting-env pytest -m "not slow" -q 2>&1 | tail -3`
Expected: tests pass with new counts reflecting Tasks 1–4.

- [ ] **Step 4: Commit**

```bash
git add src/counting/cli.py
git commit -m "feat(cli): extract-clip-features command"
```

---

## Task 6: Trainer rewrite — real CLIP prompts + real pos/neg targets

**Files:**
- Modify: `src/counting/models/pseco/trainer.py`

---

- [ ] **Step 1: Rewrite `trainer.py`**

The full file replacement is below. Key changes vs current state:
- `_CachedFSC147.__getitem__` now includes `class_name` and `points` per sample.
- `_collate_batch` gathers `class_names` and `points_per_image`.
- `train_pseco_head` loads the CLIP cache once at startup (fail fast on missing file or missing class), moves cached features to device.
- `step_fn` and `eval_fn` build `prompts` per image from the CLIP cache (not unit vectors).
- `step_fn` builds targets via `assign_targets_from_points`.
- `eval_fn` computes MAE using `pred_count = (raw_scores > 0).sum(dim=1)` — unchanged.
- `_unit_prompts` is removed.

Replace the entire file content with:

```python
"""PseCo ROIHeadMLP fine-tuning: orchestrates cache reader, proposer, and trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from counting.config.train_schema import TrainAppConfig
from counting.data.cache import FeatureCacheReader
from counting.data.formats.fsc147 import FSC147Dataset
from counting.models.pseco.clip_features import load_text_features
from counting.models.pseco.labeling import assign_targets_from_points
from counting.models.pseco.losses import pseco_head_loss
from counting.models.pseco.proposals import PointDecoderProposer
from counting.training.callbacks import CosineWithWarmup, EarlyStopping
from counting.training.runner import Runner
from counting.utils.device import resolve_device


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external" / "PseCo"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class _CachedFSC147(Dataset):
    """Yields cached embeddings + proposals + class_name + GT points."""

    def __init__(
        self,
        fsc: FSC147Dataset,
        reader: FeatureCacheReader,
        proposer: PointDecoderProposer,
    ):
        self._recs = list(fsc)
        self._reader = reader
        self._proposer = proposer

    def __len__(self) -> int:
        return len(self._recs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self._recs[idx]
        emb = self._reader.read(rec.relpath)
        proposals = self._proposer.propose(emb)
        return {
            "image_id": rec.relpath,
            "class_name": rec.class_name,
            "points": rec.points,
            "embedding": torch.from_numpy(emb).to(torch.float32),
            "proposals": proposals,
            "gt_count": rec.count,
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_ids": [b["image_id"] for b in batch],
        "class_names": [b["class_name"] for b in batch],
        "points_per_image": [b["points"] for b in batch],
        "embeddings": torch.stack([b["embedding"] for b in batch]),
        "proposals": [b["proposals"] for b in batch],
        "gt_counts": torch.tensor([b["gt_count"] for b in batch], dtype=torch.float32),
    }


def _require_classes_in_cache(
    records: list[Any], clip_cache: dict[str, torch.Tensor], split: str
) -> None:
    """Fail fast if any record's class_name is missing from the CLIP cache."""
    unknown: set[str] = set()
    for rec in records:
        if rec.class_name and rec.class_name not in clip_cache:
            unknown.add(rec.class_name)
    if unknown:
        missing = ", ".join(sorted(unknown)[:5])
        raise KeyError(
            f"{split}: {len(unknown)} class(es) missing from CLIP feature cache: "
            f"{missing}{' ...' if len(unknown) > 5 else ''}. "
            "Re-run `counting extract-clip-features`."
        )


def train_pseco_head(cfg: TrainAppConfig) -> None:
    _ensure_external_on_path()
    from models import ROIHeadMLP

    device = resolve_device(cfg.device)

    fsc_train = FSC147Dataset(cfg.data.root, split=cfg.data.train_split)
    fsc_val = FSC147Dataset(cfg.data.root, split=cfg.data.val_split)

    # --- CLIP text features: load and move to device once ---
    clip_cache_cpu = load_text_features(cfg.model.clip_features_cache)
    clip_cache = {name: tensor.to(device) for name, tensor in clip_cache_cpu.items()}

    _require_classes_in_cache(list(fsc_train), clip_cache, cfg.data.train_split)
    _require_classes_in_cache(list(fsc_val), clip_cache, cfg.data.val_split)

    reader = FeatureCacheReader(cfg.cache.dir)

    proposer = PointDecoderProposer(
        sam_checkpoint=cfg.model.sam_checkpoint,
        point_decoder_checkpoint=cfg.model.init_decoder,
        device=device,
    )
    proposer.prepare()

    train_ds = _CachedFSC147(fsc_train, reader, proposer)
    val_ds = _CachedFSC147(fsc_val, reader, proposer)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_batch,
    )

    head = ROIHeadMLP().to(device)
    if cfg.model.init_mlp:
        state = torch.load(cfg.model.init_mlp, map_location="cpu", weights_only=False)
        # PseCo MLP checkpoints store the state_dict under "cls_head"; other
        # generic conventions use "model" or "state_dict". Walk whichever
        # wrapper key is present before passing to load_state_dict.
        if isinstance(state, dict):
            for wrapper_key in ("cls_head", "model", "state_dict"):
                inner = state.get(wrapper_key)
                if isinstance(inner, dict) and any(
                    isinstance(k, str) and "." in k for k in inner.keys()
                ):
                    state = inner
                    break
        head.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    total_steps = cfg.train.epochs * max(1, len(train_loader))
    scheduler = CosineWithWarmup(
        base_lr=cfg.train.lr,
        warmup_steps=cfg.train.warmup_steps,
        total_steps=total_steps,
        min_lr=max(1e-6, cfg.train.lr * 0.01),
    )
    es = EarlyStopping(
        patience=cfg.train.early_stopping.patience,
        mode=cfg.train.early_stopping.mode,
    )

    run_dir = Path(cfg.output_dir) / cfg.run_name
    runner = Runner(
        model=head,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=es,
        device=device,
        run_dir=run_dir,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        save_every_n_epochs=cfg.logging.save_every_n_epochs,
        config_snapshot=cfg.model_dump(mode="json"),
        tensorboard=cfg.logging.tensorboard,
    )

    def _build_prompts(class_names: list[str]) -> list[torch.Tensor]:
        """Return one (1, 1, 512) prompt tensor per image based on its class."""
        return [clip_cache[cn].view(1, 1, 512) for cn in class_names]

    def step_fn(batch, model):
        embs = batch["embeddings"].to(device)
        gt_counts = batch["gt_counts"].to(device)
        B = embs.size(0)

        bboxes_per_image = _topk_points_as_boxes(
            batch["proposals"], k=16, image_size=1024, device=device,
        )
        prompts = _build_prompts(batch["class_names"])

        raw_scores = model(embs, bboxes_per_image, prompts)  # (B, 16)
        raw_scores = raw_scores.reshape(B * 16)

        # Convert single-class scores into binary logits [-s, s]
        logits = torch.stack([-raw_scores, raw_scores], dim=1)  # (B*16, 2)

        # Real pos/neg targets from GT points
        targets_cpu = assign_targets_from_points(
            bboxes_per_image, batch["points_per_image"]
        )
        targets = targets_cpu.to(device)

        pred_counts = (raw_scores.detach().reshape(B, 16) > 0).sum(dim=1).float()

        return pseco_head_loss(
            logits=logits,
            targets=targets,
            pred_counts=pred_counts,
            gt_counts=gt_counts,
            cls_weight=cfg.train.loss_weights.get("cls", 1.0),
            count_weight=cfg.train.loss_weights.get("count", 0.1),
        )

    def eval_fn(batch, model) -> dict[str, float]:
        embs = batch["embeddings"].to(device)
        gt = batch["gt_counts"]
        B = embs.size(0)
        bboxes = _topk_points_as_boxes(
            batch["proposals"], k=16, image_size=1024, device=device,
        )
        prompts = _build_prompts(batch["class_names"])
        raw_scores = model(embs, bboxes, prompts).reshape(B, 16)  # (B, 16)
        pred_counts = (raw_scores > 0).sum(dim=1).float().cpu()
        mae = F.l1_loss(pred_counts, gt).item()
        return {"mae": mae}

    for epoch in range(cfg.train.epochs):
        runner.state.epoch = epoch
        runner.train_epoch(train_loader, step_fn)
        val_metrics = runner.validate(val_loader, eval_fn)
        runner.maybe_save(val_metrics, metric_key="mae")
        if runner.early_stopping.update(val_metrics.get("mae", float("inf"))):
            break

    runner.close()
    proposer.cleanup()


def _topk_points_as_boxes(
    proposals: list[dict[str, torch.Tensor]],
    *,
    k: int,
    image_size: int,
    half_side: float = 16.0,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Return per-image (1, k, 4) boxes centered on the top-k predicted points."""
    boxes_per_image: list[torch.Tensor] = []
    for p in proposals:
        pts = p["pred_points"].squeeze(0)[:k]
        if pts.size(0) < k:
            pad = torch.zeros((k - pts.size(0), 2))
            pts = torch.cat([pts, pad], dim=0)
        x = pts[:, 0].clamp(0, image_size)
        y = pts[:, 1].clamp(0, image_size)
        boxes = torch.stack([
            (x - half_side).clamp(0),
            (y - half_side).clamp(0),
            (x + half_side).clamp(max=image_size),
            (y + half_side).clamp(max=image_size),
        ], dim=1).unsqueeze(0)
        boxes_per_image.append(boxes.to(device))
    return boxes_per_image
```

- [ ] **Step 2: Run unit tests — verify nothing regresses (trainer has no direct unit tests)**

Run: `conda run -n counting-env pytest -m "not slow" -q 2>&1 | tail -3`
Expected: all unit tests pass. Integration smoke will fail in Task 7 until its fixture is updated.

- [ ] **Step 3: Commit**

```bash
git add src/counting/models/pseco/trainer.py
git commit -m "feat(models): real CLIP prompts and pos/neg targets in trainer"
```

---

## Task 7: Integration smoke test update

**Files:**
- Modify: `tests/integration/test_training_smoke.py`

---

- [ ] **Step 1: Replace the integration smoke test to include a fake CLIP cache and ImageClasses_FSC147.txt**

Replace the contents of `tests/integration/test_training_smoke.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from counting.data.cache import FeatureCacheWriter

ROOT = Path(__file__).resolve().parents[2]


def _weights_available() -> bool:
    import yaml

    cfg_path = ROOT / "configs" / "train" / "pseco_head.yaml"
    if not cfg_path.exists():
        return False
    cfg = yaml.safe_load(cfg_path.read_text())
    need = [cfg["model"]["sam_checkpoint"], cfg["model"]["init_decoder"]]
    return all((ROOT / p).exists() for p in need)


def _external_available() -> bool:
    return (ROOT / "external" / "PseCo" / "models.py").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not (_weights_available() and _external_available()),
    reason="PseCo weights or external/ not present",
)
def test_train_pseco_head_smoke(tmp_path):
    """Run 2 epochs on a tiny synthetic cached dataset with a fake CLIP cache."""
    import json

    import torch
    from PIL import Image

    from counting.config.train_schema import TrainAppConfig
    from counting.models.pseco.clip_features import save_text_features
    from counting.models.pseco.trainer import train_pseco_head

    dataset_root = tmp_path / "fsc"
    (dataset_root / "images_384_VarV2").mkdir(parents=True)

    names = ["1.jpg", "2.jpg"]
    for n in names:
        Image.fromarray(
            (np.random.rand(64, 64, 3) * 255).astype("uint8")
        ).save(dataset_root / "images_384_VarV2" / n)
    (dataset_root / "annotation_FSC147_384.json").write_text(json.dumps({
        "1.jpg": {"points": [[5, 5], [10, 10]], "box_examples_coordinates": []},
        "2.jpg": {"points": [[20, 20]], "box_examples_coordinates": []},
    }))
    (dataset_root / "Train_Test_Val_FSC_147.json").write_text(json.dumps({
        "train": ["1.jpg"], "val": ["2.jpg"], "test": [],
    }))
    (dataset_root / "ImageClasses_FSC147.txt").write_text(
        "1.jpg\ttestfruit\n2.jpg\ttestfruit\n"
    )

    cache_dir = tmp_path / "cache"
    writer = FeatureCacheWriter(
        cache_dir=cache_dir,
        meta={"source": "smoke", "hash": "smoke0000smoke00"},
        shard_size=4,
    )
    writer.open()
    for n in names:
        writer.write(n, np.random.randn(256, 64, 64).astype(np.float16))
    writer.close()

    # Fake CLIP features — random 512-d tensor for the test class
    clip_cache_path = tmp_path / "clip.pt"
    save_text_features({"testfruit": torch.randn(512)}, clip_cache_path)

    cfg = TrainAppConfig.model_validate({
        "run_name": "smoke",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "format": "fsc147", "root": str(dataset_root),
            "train_split": "train", "val_split": "val", "image_size": 1024,
        },
        "model": {
            "sam_checkpoint": str(ROOT / "models" / "PseCo" / "sam_vit_h.pth"),
            "init_decoder": str(ROOT / "models" / "PseCo" / "point_decoder_vith.pth"),
            "init_mlp": "",
            "clip_features_cache": str(clip_cache_path),
        },
        "cache": {
            "enabled": True, "dir": str(cache_dir),
            "dtype": "float16", "augment_variants": 1,
        },
        "train": {
            "batch_size": 1, "epochs": 2, "lr": 1e-4,
            "weight_decay": 1e-4, "warmup_steps": 0,
            "scheduler": "cosine",
            "loss_weights": {"cls": 1.0, "count": 0.1},
            "early_stopping": {"patience": 5, "metric": "val_mae", "mode": "min"},
        },
        "logging": {
            "tensorboard": False, "log_every_n_steps": 1, "save_every_n_epochs": 1,
        },
    })

    train_pseco_head(cfg)

    ckpt_dir = tmp_path / "runs" / "smoke" / "checkpoints"
    assert (ckpt_dir / "best.ckpt").exists()
    assert (ckpt_dir / "last.ckpt").exists()
```

- [ ] **Step 2: Run full unit test suite — should all still pass**

Run: `conda run -n counting-env pytest -m "not slow" -q 2>&1 | tail -3`
Expected: all pass. Exact new count: Plan 2 final was 67 unit tests; Plan 3 adds `clip_features` (4 non-slow) + `labeling` (9) + `fsc147 class_name` (2 new) + `train_schema cache` (1 new) = 67 + 16 = **83 unit tests**.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_training_smoke.py
git commit -m "test(integration): update training smoke for CLIP cache + classes file"
```

---

## Task 8: Add open_clip_torch to environment files

**Files:**
- Modify: `environment.yml`
- Modify: `environment-cuda.yml`

---

- [ ] **Step 1: Add open_clip_torch to pip section in `environment.yml`**

Find the pip block in `environment.yml` and add `open_clip_torch` to the list. After the change the pip section should read:

```yaml
  - pip:
      - typer>=0.12
      - pydantic>=2.0
      - albumentations
      - timm
      - rich
      - open_clip_torch
```

- [ ] **Step 2: Add open_clip_torch to pip section in `environment-cuda.yml`**

Same change for the CUDA env file. After the change:

```yaml
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu128
      - torch>=2.5
      - torchvision>=0.20
      - typer>=0.12
      - pydantic>=2.0
      - albumentations
      - timm
      - rich
      # PseCo upstream dependencies (ops/ops.py, CLIP text extraction helpers)
      - matplotlib
      - ftfy
      - gpustat
      - open_clip_torch
```

- [ ] **Step 3: Install into the local Mac env so we can run Task 9 Step 1 immediately**

```bash
conda run -n counting-env pip install open_clip_torch 2>&1 | tail -3
```
Expected: Successfully installed open_clip_torch-<version> (and its torch/torchvision deps that may already be satisfied).

- [ ] **Step 4: Commit**

```bash
git add environment.yml environment-cuda.yml
git commit -m "build(env): add open_clip_torch for CLIP text feature extraction"
```

- [ ] **Step 5: Push all Plan 3 code changes**

```bash
git push
```
Expected: all Task 1–8 commits reach `origin/main`.

- [ ] **Step 6: Remote sync and install open_clip_torch on remote**

```bash
ssh kimhj@205.200.0.113 'cd ~/fruit-counting && git pull --ff-only 2>&1 | tail -3 && ~/miniforge3/bin/conda run -n counting-env pip install open_clip_torch 2>&1 | tail -3'
```
Expected: pull succeeds, open_clip installation completes. (No re-creation of the conda env.)

- [ ] **Step 7: Remote sanity — all unit tests green**

```bash
ssh kimhj@205.200.0.113 'cd ~/fruit-counting && ~/miniforge3/bin/conda run -n counting-env pytest -m "not slow" -q 2>&1 | tail -3'
```
Expected: 83 passed.

---

## Task 9: Real validation — generate CLIP cache, deploy, train 5 epochs, judge success

No unit tests. Manual validation with concrete commands and expected outcomes.

---

- [ ] **Step 1: Extract CLIP cache on Mac (CPU is fine for text-only)**

```bash
conda run -n counting-env counting extract-clip-features \
  --dataset fsc147 \
  --dataset-root /Users/khj/YBNML_macmini/counting/FSC147_384_V2 \
  --out /Users/khj/YBNML_macmini/counting/models/PseCo/clip_text_features.pt \
  --device cpu
```

Wait — the Mac workspace has FSC-147 at `FSC147_384_V2/` but the remote has it at `datasets/fsc147/`. Our command uses the Mac path to find the class list. The written cache `.pt` file is the same regardless of dataset root, so we deploy it to the remote.

If the Mac no longer has `ImageClasses_FSC147.txt` (it might have only been on remote), use the remote's copy instead:

```bash
# Fallback: fetch ImageClasses_FSC147.txt from remote first
scp kimhj@205.200.0.113:~/fruit-counting/datasets/fsc147/ImageClasses_FSC147.txt \
  /tmp/ImageClasses_FSC147.txt
mkdir -p /tmp/fsc-stub/
mv /tmp/ImageClasses_FSC147.txt /tmp/fsc-stub/

mkdir -p /Users/khj/YBNML_macmini/counting/models/PseCo/
conda run -n counting-env counting extract-clip-features \
  --dataset fsc147 \
  --dataset-root /tmp/fsc-stub \
  --out /Users/khj/YBNML_macmini/counting/models/PseCo/clip_text_features.pt \
  --device cpu
```

Expected: `done 147 classes → /Users/khj/YBNML_macmini/counting/models/PseCo/clip_text_features.pt`. First run downloads open_clip ViT-B/32 weights (~600 MB) to the local cache `~/.cache/torch/hub/`.

- [ ] **Step 2: Deploy CLIP cache to remote**

```bash
scp /Users/khj/YBNML_macmini/counting/models/PseCo/clip_text_features.pt \
  kimhj@205.200.0.113:~/fruit-counting/models/PseCo/clip_text_features.pt
```
Expected: transfer completes (file is ~300 KB).

- [ ] **Step 3: Kick off 5-epoch training on remote**

```bash
ssh kimhj@205.200.0.113 "bash -lc 'rm -rf ~/fruit-counting/runs/plan3_e5 && > ~/train_plan3.log && cd ~/fruit-counting && nohup ~/miniforge3/bin/conda run -n counting-env counting train pseco-head --config configs/train/pseco_head.yaml --set train.epochs=5 --set run_name=plan3_e5 >> ~/train_plan3.log 2>&1 &' && sleep 3 && pgrep -af 'counting train pseco-head' | head -3"
```
Expected: 3 processes listed (bash wrapper, conda run wrapper, counting CLI python). Training runs in background.

- [ ] **Step 4: Wait for training completion (~3–5 minutes at batch_size=8 on RTX 5070)**

Poll with a short loop until the process ends and checkpoints exist. One-liner:

```bash
until ssh kimhj@205.200.0.113 'test -f ~/fruit-counting/runs/plan3_e5/checkpoints/last.ckpt && ! pgrep -f "counting train pseco-head" > /dev/null'; do sleep 30; done && echo "DONE"
```

Expected: `DONE` prints once training is finished and `last.ckpt` is on disk.

- [ ] **Step 5: Extract TensorBoard metrics**

```bash
ssh kimhj@205.200.0.113 '~/miniforge3/bin/conda run -n counting-env python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(\"/home/kimhj/fruit-counting/runs/plan3_e5/tensorboard\")
ea.Reload()
for tag in ea.Tags()[\"scalars\"]:
    scalars = ea.Scalars(tag)
    vals = [round(s.value, 4) for s in scalars]
    if len(vals) <= 10:
        print(f\"{tag}: {vals}\")
    else:
        print(f\"{tag}: n={len(vals)} first3={vals[:3]} last3={vals[-3:]} min={min(vals):.4f} max={max(vals):.4f}\")
"'
```
Expected output pattern:

- `val/mae`: list of 5 values. **First value should be < 50.58 already** (initial CLIP prompts already break the degenerate symmetry); subsequent values should trend lower or at least not collapse back up.
- `train/cls`: non-zero throughout; monotone decrease from start would be ideal, oscillation is OK, collapse to 0.0 immediately would signal another degenerate situation.
- `train/count`: decreasing trend.

- [ ] **Step 6: Judge success**

Record the final `val/mae` from Step 5. Plan 3 is **successful** if:
1. Any `val/mae` epoch < 50.58 (strictly below the degenerate baseline).
2. `train/cls` is not flat-at-zero throughout.
3. Directional improvement (either `val/mae` decreases over epochs, or is meaningfully below baseline from the first epoch).

If all three hold → proceed to Step 7. If any fails → debugging: most common causes are (a) positive rate too low (`half_side=16` vs image_size=1024 means boxes are tiny), (b) CLIP cache path mismatch. Diagnose by running a shorter (`--limit`-style) smoke and printing pos/neg ratios. Do not commit until objective is fixed.

- [ ] **Step 7: Update README with Plan 3 results**

On Mac, edit `README.md` — find the `## 진행 상황` block and replace with:

```markdown
## 진행 상황

- [x] Foundation (Plan 1): 스캐폴드, 설정, 데이터 진단, 추론 파이프라인
- [x] PseCo 학습 인프라 (Plan 2): SAM 피처 캐시 + ROIHeadMLP 파인튜닝 루프 (placeholder objective)
- [x] PseCo 학습 목표 수정 (Plan 3): CLIP 텍스트 피처 + pos/neg 레이블 — **val/mae = <FILL>** (5 epoch, FSC-147 val)
- [ ] Classifier 학습 · SR 통합 · 정돈 (Plan 4)
```

Replace `<FILL>` with the best `val/mae` value from Step 5. Commit:

```bash
git add README.md
git commit -m "docs: Plan 3 result — val/mae=<FILL> on FSC-147 after 5 epochs"
git push
```

- [ ] **Step 8: Commit CLIP text feature cache to repo**

Since the file is small (~300 KB) and reproducible-but-expensive, commit it so others can skip Step 1:

```bash
git add models/PseCo/clip_text_features.pt
git commit -m "feat(data): CLIP text features cache for FSC-147 (147 classes, ViT-B/32)"
git push
```

Note: `.gitignore` has `/models/` anchored at top-level — this path IS top-level `models/`, so the file WILL be ignored by default. Force-add:

```bash
git add -f models/PseCo/clip_text_features.pt
git commit -m "feat(data): CLIP text features cache for FSC-147 (147 classes, ViT-B/32)"
git push
```

Alternatively, uncomment or tighten the `.gitignore` rule to allow `clip_text_features.pt` specifically:

```
# After the existing "pretrained weights" block, add:
!models/PseCo/clip_text_features.pt
```

Either approach is fine; prefer the negation for clarity.

---

## Self-Review

**Spec coverage** — each section of the spec maps to a task:
- §1 Background: covered by the plan goal / "Success criterion" recap in Task 9 Step 6.
- §2 Decisions: reflected in Tasks 1 (CLIP source, 512-d), 2 (containment labeling), 4 (CLIP cache field), and trainer (Task 6).
- §3 File changes: covered by File Structure and Tasks 1–8.
- §4 Data flow (4.1 extract, 4.2 training loop, 4.3 deltas): covered by Tasks 1, 5, 6, 9.
- §5 Labeling spec (signatures + half-open convention + row-major): covered in Task 2 test vectors and implementation.
- §6 Config schema delta: Task 4.
- §7 CLI delta: Task 5.
- §8 Testing strategy (eight tests + integration update): Tasks 1, 2, 3, 4, 7.
- §9 Error handling (6 cases): trainer `_require_classes_in_cache` (Task 6), `_load_image_classes` (Task 3), `load_text_features` (Task 1), `assign_targets_from_points` ValueError (Task 2).
- §10 Risks: validation in Task 9 Step 6 explicitly names two of the most likely failure modes.
- §11 Checkpoint compatibility note: not a separate task; after Task 9 succeeds, the "action" is a manual `counting infer` check which can be added as follow-up. For this plan the trainer's own state_dict load path exercises the format round-trip.
- §12 Development order: matches Tasks 1–9 order.
- §13 Non-goals: respected (no Classifier, SR, NMS, pos_weight, `--classes-file`).
- §14 Success artifacts: Task 9 Steps 7–8 commit the README update and the cache file.

**Placeholder scan** — all code blocks are complete; no "TODO" / "implement later" / vague instructions. Two `<FILL>` placeholders in Task 9 Step 7 are explicit variable substitutions for the measured `val/mae` value (they become a concrete number when the step runs), not plan-level placeholders.

**Type consistency** — function signatures match across tasks:
- `encode_class_names(class_names, device=...) → dict[str, Tensor]` (Task 1) used in Task 5 CLI and Task 9 Step 1.
- `save_text_features(dict, path)` / `load_text_features(path)` (Task 1) used in Task 5, 6, 7.
- `assign_targets_from_points(boxes_per_image, points_per_image) → LongTensor(B*K,)` (Task 2) used in Task 6 `step_fn`.
- `FSC147Record.class_name: str` (Task 3) used in Task 6 `_CachedFSC147.__getitem__` and `_require_classes_in_cache`.
- `TrainModelConfig.clip_features_cache: str` (Task 4) read by Task 6 `train_pseco_head`.
- `_topk_points_as_boxes(... device=...)` unchanged from Plan 2 (accepted as-is).

One outstanding check: in Task 6 the trainer uses `batch["class_names"]` and `batch["points_per_image"]`, both of which are produced by `_collate_batch` (Task 6). Consistent.

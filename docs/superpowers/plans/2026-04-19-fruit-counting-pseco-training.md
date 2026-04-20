# Fruit Counting — PseCo Head Training Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fine-tuning of PseCo's `ROIHeadMLP` classifier head on FSC-147, using SAM ViT-H embeddings cached on disk and a frozen SAM backbone. Produce a trained checkpoint that slots into the existing inference pipeline.

**Architecture:** Add `external/PseCo` as a git submodule pinned to a specific commit. Implement a streaming SAM feature extractor that writes `fp16` npz shards to disk, with an index and content hash for invalidation. A trainer loads cached features and proposals, trains only the `ROIHeadMLP` head using classification loss against CLIP text prompts (mirroring upstream `fsc147/4_1_train_roi_head.py`), logs to TensorBoard, and saves best/last checkpoints with a config snapshot for resume. All run on NVIDIA Blackwell/Hopper/Ampere via the `environment-cuda.yml` cu128 env, and degrade to CPU/MPS for smoke tests.

**Tech Stack:** Python 3.11, PyTorch 2.5+ (cu128 on remote), PseCo (upstream `models.PointDecoder`, `models.ROIHeadMLP`), SAM ViT-H (via PseCo's vendored `ops/foundation_models/segment_anything`), FSC-147, TensorBoard, Pydantic v2, Typer.

**Spec:** `docs/superpowers/specs/2026-04-19-fruit-counting-redesign-design.md` (§4.2 학습, §5 임베딩 캐시, §6.1 하이퍼파라미터, §7.2 설정, §8 CLI).

**Upstream pin:** `Hzzone/PseCo@aab6cdb` (HEAD at time of writing).

**Scope of this plan:**
1. External PseCo submodule + import smoke
2. FSC-147 download helper + dataset adapter
3. Training config schema + YAML
4. SAM feature extractor + fp16 npz shard cache (write)
5. Cache reader (shard-aware DataLoader)
6. `counting cache-embeddings` CLI
7. Proposal precompute (uses frozen `PointDecoder`)
8. `ROIHeadMLP` trainer module (loss, optimizer, scheduler)
9. Training runner with TensorBoard hooks
10. Checkpoint save/load with config snapshot + resume
11. `counting train pseco-head` CLI
12. Integration smoke test on tiny synthetic data
13. Density estimation in `diagnose` (deferred from Plan 1)
14. README + docs update

**Out of scope (defer to Plan 3):**
- `PointDecoder` fine-tuning (retraining point regression) — requires more annotated data
- Custom (farm) dataset adapter — until own labels are ready
- Classifier (ResNet) training — Plan 3
- SR integration — Plan 3

---

## File Structure

### New / modified files

| Path | Responsibility |
|---|---|
| `.gitmodules` | Submodule pin for `external/PseCo` |
| `external/PseCo` | Upstream PseCo source (submodule) |
| `src/counting/config/schema.py` | **Modify** — add `TrainAppConfig` sibling schema |
| `src/counting/config/train_schema.py` | New — training-specific Pydantic models |
| `src/counting/data/formats/fsc147.py` | New — FSC-147 image+point adapter |
| `src/counting/data/download.py` | New — FSC-147 + SAM weight download helpers/instructions |
| `src/counting/data/cache.py` | New — SAM feature cache writer+reader (fp16 shards, meta.json, index.parquet) |
| `src/counting/models/pseco/embedding.py` | New — SAM ViT-H forward wrapper for cache |
| `src/counting/models/pseco/proposals.py` | New — frozen `PointDecoder` proposal generator |
| `src/counting/models/pseco/trainer.py` | New — ROIHeadMLP trainer (loss, optim, loop) |
| `src/counting/models/pseco/losses.py` | New — classification loss + count regression |
| `src/counting/training/__init__.py` | New |
| `src/counting/training/runner.py` | New — generic trainer scaffold + TB hooks |
| `src/counting/training/callbacks.py` | New — early stopping, checkpoint save, LR cosine+warmup |
| `src/counting/training/checkpoint.py` | New — save/load with config snapshot + env info |
| `src/counting/cli.py` | **Modify** — add `cache-embeddings` and `train pseco-head` subcommands |
| `configs/train/pseco_head.yaml` | New |
| `tests/unit/test_config_train_schema.py` | New |
| `tests/unit/test_data_fsc147.py` | New |
| `tests/unit/test_data_cache.py` | New |
| `tests/unit/test_training_losses.py` | New |
| `tests/unit/test_training_callbacks.py` | New |
| `tests/unit/test_training_checkpoint.py` | New |
| `tests/unit/test_diagnostics_density.py` | New |
| `tests/integration/test_training_smoke.py` | New (slow marker) |
| `README.md` | **Modify** — add training + Plan 2 status |

### Files NOT touched

- `original_code/**`
- `models/**` (pretrained weights live here untracked)
- Any Plan 1 code that already works — we ADD, not rewrite

---

## Task 1: Add PseCo as submodule + import smoke

**Goal:** Pin upstream PseCo at commit `aab6cdb` under `external/PseCo`. Verify imports work in the `counting-env` conda env.

**Files:**
- Create: `.gitmodules`
- Create: `external/PseCo/` (submodule)

---

- [ ] **Step 1: Add submodule**

```bash
git submodule add https://github.com/Hzzone/PseCo.git external/PseCo
cd external/PseCo
git checkout aab6cdb
cd ../..
```

- [ ] **Step 2: Commit submodule addition**

```bash
git add .gitmodules external/PseCo
git commit -m "build: add PseCo as git submodule pinned to aab6cdb"
```

- [ ] **Step 3: Push and pull on remote**

```bash
git push

# On remote (over SSH):
ssh kimhj@<host> 'cd ~/fruit-counting && git pull && git submodule update --init --recursive'
```

- [ ] **Step 4: Verify upstream imports work in counting-env**

Run (on remote where GPU is):

```bash
conda run -n counting-env python -c "
import sys; sys.path.insert(0, 'external/PseCo')
from ops.foundation_models.segment_anything import build_sam_vit_h
from models import PointDecoder, ROIHeadMLP
print('imports OK:', PointDecoder.__name__, ROIHeadMLP.__name__)
"
```

Expected: `imports OK: PointDecoder ROIHeadMLP`

If any import fails, install missing upstream deps (likely `ftfy`, `gpustat`):
```bash
conda run -n counting-env pip install ftfy gpustat
```
Re-run the probe. If still failing, STOP and escalate with the full traceback.

- [ ] **Step 5: No further changes this task**

---

## Task 2: FSC-147 dataset adapter + download helper

**Goal:** Provide `FSC147Dataset` that yields images with point annotations matching the spec's `CountingDataset` protocol. Provide a helper that documents how to download FSC-147 (Google Drive links).

**Files:**
- Create: `src/counting/data/formats/fsc147.py`
- Create: `src/counting/data/download.py`
- Create: `tests/unit/test_data_fsc147.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_data_fsc147.py`:

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from counting.data.formats.fsc147 import FSC147Dataset


def _make_annotations(root: Path):
    # Standard FSC-147 layout (subset):
    #   images_384_VarV2/<id>.jpg
    #   annotation_FSC147_384.json    (per-image point + box annotations)
    #   Train_Test_Val_FSC_147.json   (split mapping)
    imgs = root / "images_384_VarV2"
    imgs.mkdir(parents=True)
    for name in ("1.jpg", "2.jpg", "3.jpg"):
        arr = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(imgs / name)

    (root / "annotation_FSC147_384.json").write_text(json.dumps({
        "1.jpg": {"points": [[10, 10], [20, 20]], "box_examples_coordinates": [[[0, 0], [0, 30], [30, 30], [30, 0]]]},
        "2.jpg": {"points": [[5, 5]], "box_examples_coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0]]]},
        "3.jpg": {"points": [], "box_examples_coordinates": []},
    }))
    (root / "Train_Test_Val_FSC_147.json").write_text(json.dumps({
        "train": ["1.jpg", "2.jpg"],
        "val": ["3.jpg"],
        "test": [],
    }))


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
```

- [ ] **Step 2: Confirm failure**

```bash
conda run -n counting-env pytest tests/unit/test_data_fsc147.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement dataset adapter**

Create `src/counting/data/formats/fsc147.py`:

```python
"""FSC-147 dataset adapter.

Expected layout (matches upstream PseCo):
    root/
      images_384_VarV2/<image_id>.jpg
      annotation_FSC147_384.json     # per-image points + box examples
      Train_Test_Val_FSC_147.json    # split mapping
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from counting.data.base import ImageRecord
from counting.utils.image import read_image_rgb

_VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class FSC147Record:
    path: Path
    relpath: str
    points: list[tuple[float, float]]
    box_examples: list[list[tuple[float, float]]]
    count: int

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
        for p in (img_dir, ann_path, split_path):
            if not p.exists():
                raise FileNotFoundError(f"FSC-147 artifact missing: {p}")

        splits = json.loads(split_path.read_text())
        annotations = json.loads(ann_path.read_text())

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
            ))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> FSC147Record:
        return self._records[idx]

    def __iter__(self) -> Iterator[FSC147Record]:
        return iter(self._records)
```

- [ ] **Step 4: Implement download helper**

Create `src/counting/data/download.py`:

```python
"""Download helpers for FSC-147 and SAM weights.

FSC-147 is hosted on Google Drive; automation requires `gdown`. This module
prints the exact commands and URLs so the user can execute them manually
(or via `gdown` if installed). Nothing is downloaded implicitly.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


FSC147_INSTRUCTIONS = dedent("""\
    FSC-147 download (approx 4 GB):

    1. Download the zipped dataset from the authors' Google Drive page:
       https://github.com/cvlab-stonybrook/LearningToCountEverything
       (see "Dataset FSC-147" section).
    2. Unzip to: {target}
    3. Expected layout afterward:
       {target}/images_384_VarV2/*.jpg
       {target}/annotation_FSC147_384.json
       {target}/Train_Test_Val_FSC_147.json

    Alternatively, if `gdown` is installed:
       pip install gdown
       gdown --folder <google-drive-folder-id> -O {target}
    Replace <google-drive-folder-id> with the public folder ID from the
    authors' README.
""")

SAM_VIT_H_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)


def print_fsc147_instructions(target: str | Path) -> None:
    print(FSC147_INSTRUCTIONS.format(target=str(Path(target).resolve())))


def print_sam_vit_h_instructions(target: str | Path) -> None:
    p = Path(target).resolve()
    print(dedent(f"""\
        SAM ViT-H checkpoint (approx 2.4 GB):

            mkdir -p {p.parent}
            curl -L -o {p} {SAM_VIT_H_URL}

        Verify:
            ls -lh {p}
    """))
```

- [ ] **Step 5: Tests pass**

```bash
conda run -n counting-env pytest tests/unit/test_data_fsc147.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/counting/data/formats/fsc147.py src/counting/data/download.py \
        tests/unit/test_data_fsc147.py
git commit -m "feat(data): FSC-147 dataset adapter and download instructions"
```

---

## Task 3: Training config schema + YAML

**Goal:** Extend Pydantic with training-specific sections. Keep `AppConfig` (inference) intact; introduce `TrainAppConfig` for training configs. Write the starter YAML.

**Files:**
- Create: `src/counting/config/train_schema.py`
- Create: `configs/train/pseco_head.yaml`
- Create: `tests/unit/test_config_train_schema.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_config_train_schema.py`:

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
```

- [ ] **Step 2: Confirm failure**

```bash
conda run -n counting-env pytest tests/unit/test_config_train_schema.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement schema**

Create `src/counting/config/train_schema.py`:

```python
"""Training configuration schema (sibling to AppConfig)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["fsc147", "coco", "custom"] = "fsc147"
    root: str
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = Field(default=1024, gt=0, le=4096)


class TrainModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sam_checkpoint: str
    init_decoder: str = ""       # may be empty to train from scratch (unused in Plan 2)
    init_mlp: str = ""


class TrainCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    dir: str
    dtype: Literal["float16", "float32"] = "float16"
    augment_variants: int = Field(default=1, ge=1, le=16)


class TrainEarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patience: int = Field(default=5, ge=1)
    metric: str = "val_mae"
    mode: Literal["min", "max"] = "min"


class TrainLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(default=8, ge=1)
    epochs: int = Field(default=30, ge=1)
    lr: float = Field(default=1.0e-4, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    warmup_steps: int = Field(default=500, ge=0)
    scheduler: Literal["cosine", "constant"] = "cosine"
    loss_weights: dict[str, float] = Field(default_factory=lambda: {"cls": 1.0, "count": 0.1})
    early_stopping: TrainEarlyStoppingConfig = Field(default_factory=TrainEarlyStoppingConfig)


class TrainLoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tensorboard: bool = True
    log_every_n_steps: int = Field(default=20, ge=1)
    save_every_n_epochs: int = Field(default=1, ge=1)


class TrainAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 42
    output_dir: str = "./runs"
    data: TrainDataConfig
    model: TrainModelConfig
    cache: TrainCacheConfig
    train: TrainLoopConfig = Field(default_factory=TrainLoopConfig)
    logging: TrainLoggingConfig = Field(default_factory=TrainLoggingConfig)
```

- [ ] **Step 4: Starter YAML**

Create `configs/train/pseco_head.yaml`:

```yaml
run_name: pseco_head_v1
device: auto
seed: 42
output_dir: ./runs

data:
  format: fsc147
  root: ./datasets/fsc147
  train_split: train
  val_split: val
  image_size: 1024

model:
  sam_checkpoint: models/PseCo/sam_vit_h.pth
  init_decoder: models/PseCo/point_decoder_vith.pth
  init_mlp: models/PseCo/MLP_small_box_w1_zeroshot.tar

cache:
  enabled: true
  dir: ./feature_cache/fsc147_vit_h
  dtype: float16
  augment_variants: 1

train:
  batch_size: 8
  epochs: 30
  lr: 1.0e-4
  weight_decay: 1.0e-4
  warmup_steps: 500
  scheduler: cosine
  loss_weights: {cls: 1.0, count: 0.1}
  early_stopping: {patience: 5, metric: val_mae, mode: min}

logging:
  tensorboard: true
  log_every_n_steps: 20
  save_every_n_epochs: 1
```

- [ ] **Step 5: Tests pass**

```bash
conda run -n counting-env pytest tests/unit/test_config_train_schema.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/counting/config/train_schema.py configs/train/pseco_head.yaml \
        tests/unit/test_config_train_schema.py
git commit -m "feat(config): TrainAppConfig schema and pseco_head YAML"
```

---

## Task 4: SAM feature extractor

**Goal:** A thin wrapper that loads SAM ViT-H (via upstream `build_sam_vit_h`), preprocesses an image to 1024×1024, and returns the image embedding tensor `(256, 64, 64)` as `fp16` on host RAM. No caching logic here — that lives in Task 5.

**Files:**
- Create: `src/counting/models/pseco/embedding.py`

---

- [ ] **Step 1: Implement extractor**

Create `src/counting/models/pseco/embedding.py`:

```python
"""SAM ViT-H image embedding extractor for the caching pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external" / "PseCo"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class SAMImageEmbedder:
    """Wraps upstream SAM predictor for (image -> embedding) forward.

    Output: torch.Tensor of shape (256, 64, 64) in fp16 on CPU.
    """

    def __init__(self, sam_checkpoint: str, device: str = "cuda") -> None:
        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self._predictor: Any = None

    def prepare(self) -> None:
        _ensure_external_on_path()
        from ops.foundation_models.segment_anything import (
            SamPredictor,
            build_sam_vit_h,
        )

        sam = build_sam_vit_h(checkpoint=self.sam_checkpoint)
        sam.to(self.device).eval()
        self._predictor = SamPredictor(sam)

    @torch.no_grad()
    def embed(self, image_rgb_uint8: np.ndarray) -> torch.Tensor:
        """Return (256, 64, 64) fp16 CPU tensor for the given RGB uint8 image."""
        if self._predictor is None:
            raise RuntimeError("SAMImageEmbedder used before prepare()")
        self._predictor.set_image(image_rgb_uint8)
        feats = self._predictor.features  # (1, 256, 64, 64), fp32 on device
        return feats.squeeze(0).detach().to(dtype=torch.float16, device="cpu").contiguous()

    def cleanup(self) -> None:
        self._predictor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
```

- [ ] **Step 2: Commit**

```bash
git add src/counting/models/pseco/embedding.py
git commit -m "feat(models): SAM ViT-H embedding extractor for caching"
```

No unit tests — this requires SAM weights and external source. Exercised in Task 12 integration smoke (when weights are available).

---

## Task 5: Feature cache (write + read + invalidation)

**Goal:** Write fp16 embeddings as sharded `.npz` files with an `index.parquet` mapping image_id → (shard, row) and a `meta.json` with content hash for invalidation.

**Files:**
- Create: `src/counting/data/cache.py`
- Create: `tests/unit/test_data_cache.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_data_cache.py`:

```python
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from counting.data.cache import (
    FeatureCacheReader,
    FeatureCacheWriter,
    compute_cache_meta_hash,
)


def _fake_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((256, 64, 64)).astype(np.float16)


def test_write_and_read_roundtrip(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    ids = ["img_a.jpg", "img_b.jpg", "img_c.jpg"]
    for i, name in enumerate(ids):
        writer.write(name, _fake_embedding(i))
    writer.close()

    # Two shards expected (2 + 1)
    shards = sorted(p.name for p in (tmp_path / "shards").iterdir())
    assert shards == ["00000.npz", "00001.npz"]

    reader = FeatureCacheReader(tmp_path)
    assert set(reader.keys()) == set(ids)
    a = reader.read("img_a.jpg")
    b = reader.read("img_b.jpg")
    c = reader.read("img_c.jpg")
    assert a.shape == (256, 64, 64)
    assert a.dtype == np.float16
    assert not np.array_equal(a, b)
    assert not np.array_equal(b, c)


def test_meta_contains_expected_fields(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    writer.write("x.jpg", _fake_embedding(0))
    writer.close()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["source"] == "test"
    assert meta["sam_ckpt_hash"] == "abc"
    assert meta["dtype"] == "float16"
    assert meta["shape"] == [256, 64, 64]
    assert meta["count"] == 1


def test_reader_detects_hash_mismatch(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    writer.write("x.jpg", _fake_embedding(0))
    writer.close()

    reader = FeatureCacheReader(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.assert_compatible(expected_hash="WRONG")


def test_compute_hash_is_stable():
    a = compute_cache_meta_hash({"sam": "abc", "image_size": 1024})
    b = compute_cache_meta_hash({"image_size": 1024, "sam": "abc"})
    assert a == b
    assert len(a) == 16


def test_missing_key_raises(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    writer.write("only.jpg", _fake_embedding(0))
    writer.close()

    reader = FeatureCacheReader(tmp_path)
    with pytest.raises(KeyError):
        reader.read("missing.jpg")
```

- [ ] **Step 2: Confirm failure**

```bash
conda run -n counting-env pytest tests/unit/test_data_cache.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement cache writer+reader**

Create `src/counting/data/cache.py`:

```python
"""Sharded fp16 embedding cache.

Layout:
    <cache_dir>/
        meta.json              # dtype, shape, count, user-provided meta fields
        index.json             # {image_id: [shard_id, row]}
        shards/
            00000.npz          # arr_0..arr_N-1 (each an ndarray)
            00001.npz
            ...

`.npz` is used for simplicity and portability. Each shard holds up to
`shard_size` embeddings under keys "0", "1", ... matching row order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _ShardBuffer:
    shard_id: int
    arrays: list[np.ndarray]

    def save(self, out_dir: Path, dtype: np.dtype) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.shard_id:05d}.npz"
        kwargs = {str(i): a.astype(dtype, copy=False) for i, a in enumerate(self.arrays)}
        np.savez_compressed(path, **kwargs)


class FeatureCacheWriter:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        meta: dict[str, Any],
        shard_size: int = 256,
        dtype: str = "float16",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.meta = dict(meta)
        self.shard_size = shard_size
        self.dtype_name = dtype
        self.dtype = np.dtype(dtype)
        self._index: dict[str, list[int]] = {}
        self._buf: _ShardBuffer | None = None
        self._shape: tuple[int, ...] | None = None
        self._count = 0
        self._shard_counter = 0

    def open(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "shards").mkdir(exist_ok=True)
        self._buf = _ShardBuffer(shard_id=0, arrays=[])

    def write(self, image_id: str, embedding: np.ndarray) -> None:
        if self._buf is None:
            raise RuntimeError("FeatureCacheWriter used before open()")
        if self._shape is None:
            self._shape = tuple(embedding.shape)
        elif tuple(embedding.shape) != self._shape:
            raise ValueError(
                f"shape mismatch: got {embedding.shape}, expected {self._shape}"
            )

        if image_id in self._index:
            raise ValueError(f"duplicate image_id: {image_id}")

        self._index[image_id] = [self._buf.shard_id, len(self._buf.arrays)]
        self._buf.arrays.append(embedding)
        self._count += 1

        if len(self._buf.arrays) >= self.shard_size:
            self._flush()

    def close(self) -> None:
        if self._buf is None:
            raise RuntimeError("FeatureCacheWriter used before open()")
        if self._buf.arrays:
            self._flush()
        meta = {
            **self.meta,
            "dtype": self.dtype_name,
            "shape": list(self._shape) if self._shape is not None else [],
            "count": self._count,
        }
        (self.cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.cache_dir / "index.json").write_text(json.dumps(self._index))
        self._buf = None

    def _flush(self) -> None:
        assert self._buf is not None
        self._buf.save(self.cache_dir / "shards", dtype=self.dtype)
        self._shard_counter += 1
        self._buf = _ShardBuffer(shard_id=self._shard_counter, arrays=[])


class FeatureCacheReader:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        index_path = self.cache_dir / "index.json"
        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"cache not found at {self.cache_dir}")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        self._index: dict[str, list[int]] = json.loads(index_path.read_text())
        self._shard_cache: dict[int, dict[str, np.ndarray]] = {}

    def keys(self):
        return list(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)

    def read(self, image_id: str) -> np.ndarray:
        if image_id not in self._index:
            raise KeyError(image_id)
        shard_id, row = self._index[image_id]
        shard = self._load_shard(shard_id)
        return shard[str(row)]

    def assert_compatible(self, *, expected_hash: str) -> None:
        got = self.meta.get("hash", "")
        if got != expected_hash:
            raise ValueError(
                f"cache hash mismatch: expected {expected_hash}, got {got!r}"
            )

    def _load_shard(self, shard_id: int) -> dict[str, np.ndarray]:
        if shard_id not in self._shard_cache:
            p = self.cache_dir / "shards" / f"{shard_id:05d}.npz"
            with np.load(p) as npz:
                self._shard_cache[shard_id] = {k: npz[k] for k in npz.files}
        return self._shard_cache[shard_id]


def compute_cache_meta_hash(meta: dict[str, Any]) -> str:
    """Deterministic 16-char hash of a meta dict (sorted keys, compact JSON)."""
    blob = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
```

- [ ] **Step 4: Tests pass**

```bash
conda run -n counting-env pytest tests/unit/test_data_cache.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/counting/data/cache.py tests/unit/test_data_cache.py
git commit -m "feat(data): sharded fp16 embedding cache (writer+reader)"
```

---

## Task 6: `counting cache-embeddings` CLI

**Goal:** CLI that reads a `TrainAppConfig`, instantiates `SAMImageEmbedder` + `FeatureCacheWriter`, iterates the dataset, writes the cache, and stamps a content hash in `meta.json`.

**Files:**
- Modify: `src/counting/cli.py`
- Modify: `src/counting/config/train_schema.py` (no-op unless below helper lands)

---

- [ ] **Step 1: Append CLI command**

Edit `src/counting/cli.py` — append at end of file:

```python


@app.command("cache-embeddings")
def cache_embeddings(
    config: str = typer.Option(..., "--config", "-c", help="Training YAML"),
    set_: list[str] = typer.Option(None, "--set", help="Override key.path=value"),
    limit: int = typer.Option(0, help="If >0, only process this many images (smoke)"),
) -> None:
    """Precompute SAM embeddings and write to the on-disk feature cache."""
    import hashlib
    import yaml

    from tqdm import tqdm

    from counting.config.loader import apply_overrides
    from counting.config.train_schema import TrainAppConfig
    from counting.data.cache import FeatureCacheWriter, compute_cache_meta_hash
    from counting.data.formats.fsc147 import FSC147Dataset
    from counting.models.pseco.embedding import SAMImageEmbedder
    from counting.utils.device import resolve_device

    raw = yaml.safe_load(open(config, "r", encoding="utf-8"))
    if set_:
        raw = apply_overrides(raw, list(set_))
    cfg = TrainAppConfig.model_validate(raw)

    device = resolve_device(cfg.device)
    if device not in {"cuda", "mps", "cpu"}:
        raise typer.BadParameter(f"Unexpected device: {device}")

    if cfg.data.format != "fsc147":
        raise typer.BadParameter(
            f"Only fsc147 is supported in Plan 2; got {cfg.data.format!r}"
        )

    ds = FSC147Dataset(cfg.data.root, split=cfg.data.train_split)

    # Content hash over fields that, when changed, invalidate the cache.
    sam_hash = ""
    sam_ckpt_path = cfg.model.sam_checkpoint
    if sam_ckpt_path:
        h = hashlib.sha256()
        with open(sam_ckpt_path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        sam_hash = h.hexdigest()[:16]

    meta_for_hash = {
        "sam_ckpt_hash": sam_hash,
        "image_size": cfg.data.image_size,
        "dtype": cfg.cache.dtype,
        "dataset_root": cfg.data.root,
        "split": cfg.data.train_split,
        "augment_variants": cfg.cache.augment_variants,
    }
    cache_hash = compute_cache_meta_hash(meta_for_hash)

    embedder = SAMImageEmbedder(cfg.model.sam_checkpoint, device=device)
    console.print(f"[loading] SAM ViT-H on {device}")
    embedder.prepare()

    writer = FeatureCacheWriter(
        cache_dir=cfg.cache.dir,
        meta={**meta_for_hash, "hash": cache_hash},
        shard_size=256,
        dtype=cfg.cache.dtype,
    )
    writer.open()

    records = list(ds)
    if limit > 0:
        records = records[:limit]

    for rec in tqdm(records, desc="embedding"):
        arr = rec.read_rgb()
        emb = embedder.embed(arr).numpy()
        writer.write(rec.relpath, emb)

    writer.close()
    embedder.cleanup()

    console.print(f"[green]done[/green] cached {len(records)} images at {cfg.cache.dir}")
    console.print(f"  hash={cache_hash}")
```

- [ ] **Step 2: Commit**

```bash
git add src/counting/cli.py
git commit -m "feat(cli): counting cache-embeddings command"
```

No unit test — exercised when SAM weights exist; the cache module itself is tested in Task 5.

---

## Task 7: Proposal generation (frozen PointDecoder)

**Goal:** Using cached SAM embeddings, run the frozen `PointDecoder` to produce proposal boxes for each image. Persist proposals alongside the cache (a second index + shard set, same layout).

**Files:**
- Create: `src/counting/models/pseco/proposals.py`

---

- [ ] **Step 1: Implement proposal generator**

Create `src/counting/models/pseco/proposals.py`:

```python
"""Frozen PointDecoder proposal generation for cached SAM embeddings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external" / "PseCo"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class PointDecoderProposer:
    """Loads upstream PointDecoder + SAM backbone and emits proposals per image."""

    def __init__(
        self,
        *,
        sam_checkpoint: str,
        point_decoder_checkpoint: str,
        device: str = "cuda",
    ) -> None:
        self.sam_checkpoint = sam_checkpoint
        self.point_decoder_checkpoint = point_decoder_checkpoint
        self.device = device
        self._sam: Any = None
        self._decoder: Any = None

    def prepare(self) -> None:
        _ensure_external_on_path()
        from ops.foundation_models.segment_anything import build_sam_vit_h
        from models import PointDecoder

        self._sam = build_sam_vit_h(checkpoint=self.sam_checkpoint).to(self.device).eval()
        self._decoder = PointDecoder(self._sam).to(self.device).eval()
        state = torch.load(self.point_decoder_checkpoint, map_location="cpu")
        if "model" in state:
            state = state["model"]
        self._decoder.load_state_dict(state, strict=False)

    @torch.no_grad()
    def propose(self, cached_embedding_fp16: np.ndarray) -> dict[str, torch.Tensor]:
        """Return proposals for a single image.

        cached_embedding_fp16 shape: (256, 64, 64)
        Output keys: pred_heatmaps, pred_points, pred_points_score (CPU tensors).
        """
        if self._decoder is None:
            raise RuntimeError("PointDecoderProposer used before prepare()")
        feats = torch.from_numpy(cached_embedding_fp16).to(self.device).to(torch.float32).unsqueeze(0)
        out = self._decoder(feats)
        return {k: v.detach().cpu() for k, v in out.items()}

    def cleanup(self) -> None:
        self._sam = None
        self._decoder = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
```

- [ ] **Step 2: Commit**

```bash
git add src/counting/models/pseco/proposals.py
git commit -m "feat(models): frozen PointDecoder proposal generator"
```

Exercised in Task 12 smoke.

---

## Task 8: Losses

**Goal:** Classification loss (CE vs CLIP text prompt) + count L1 auxiliary. Upstream uses `cross_entropy` across text-prompt logits plus a count regression. Mirror that.

**Files:**
- Create: `src/counting/models/pseco/losses.py`
- Create: `tests/unit/test_training_losses.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_training_losses.py`:

```python
import torch

from counting.models.pseco.losses import pseco_head_loss


def test_loss_is_scalar_and_finite():
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    pred_counts = torch.tensor([2.0, 3.0])
    gt_counts = torch.tensor([2, 3])
    out = pseco_head_loss(
        logits=logits,
        targets=targets,
        pred_counts=pred_counts,
        gt_counts=gt_counts,
        cls_weight=1.0,
        count_weight=0.1,
    )
    assert out["total"].ndim == 0
    assert torch.isfinite(out["total"])
    assert "cls" in out and "count" in out


def test_loss_weights_affect_total():
    torch.manual_seed(0)
    logits = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    pred_counts = torch.tensor([1.0, 2.0])
    gt_counts = torch.tensor([2, 3])

    small = pseco_head_loss(
        logits=logits, targets=targets,
        pred_counts=pred_counts, gt_counts=gt_counts,
        cls_weight=1.0, count_weight=0.0,
    )["total"]
    big = pseco_head_loss(
        logits=logits, targets=targets,
        pred_counts=pred_counts, gt_counts=gt_counts,
        cls_weight=1.0, count_weight=10.0,
    )["total"]
    assert big.item() > small.item()


def test_count_loss_zero_when_perfect():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # cls basically free
    targets = torch.tensor([0, 1])
    pred_counts = torch.tensor([2.0, 3.0])
    gt_counts = torch.tensor([2, 3])
    out = pseco_head_loss(
        logits=logits, targets=targets,
        pred_counts=pred_counts, gt_counts=gt_counts,
        cls_weight=1.0, count_weight=1.0,
    )
    assert out["count"].item() == 0.0
    assert out["cls"].item() < 0.01
```

- [ ] **Step 2: Confirm failure**

```bash
conda run -n counting-env pytest tests/unit/test_training_losses.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement losses**

Create `src/counting/models/pseco/losses.py`:

```python
"""PseCo ROIHeadMLP training loss.

The upstream loss is a 2-class cross-entropy between ROIHead logits and the
pseudo labels from PointDecoder proposals (positive = matches CLIP text
prompt, negative = background/other). We add a small count-L1 auxiliary so
the model is penalized for disagreeing with the annotated image count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pseco_head_loss(
    *,
    logits: torch.Tensor,            # (N, 2) per-proposal logits
    targets: torch.Tensor,           # (N,) class indices {0, 1}
    pred_counts: torch.Tensor,       # (B,) predicted per-image count
    gt_counts: torch.Tensor,         # (B,) ground-truth count
    cls_weight: float = 1.0,
    count_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(logits, targets)
    count_loss = F.l1_loss(pred_counts.float(), gt_counts.float())
    total = cls_weight * cls_loss + count_weight * count_loss
    return {"total": total, "cls": cls_loss.detach(), "count": count_loss.detach()}
```

- [ ] **Step 4: Tests pass**

```bash
conda run -n counting-env pytest tests/unit/test_training_losses.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/counting/models/pseco/losses.py tests/unit/test_training_losses.py
git commit -m "feat(models): pseco_head_loss (cross-entropy + count L1)"
```

---

## Task 9: Training callbacks (cosine+warmup LR, early stop, checkpoint)

**Goal:** Pure-Python helpers that plug into the trainer.

**Files:**
- Create: `src/counting/training/__init__.py`
- Create: `src/counting/training/callbacks.py`
- Create: `tests/unit/test_training_callbacks.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_training_callbacks.py`:

```python
import math

from counting.training.callbacks import (
    CosineWithWarmup,
    EarlyStopping,
)


def test_cosine_warmup_monotonic_then_decay():
    sched = CosineWithWarmup(base_lr=1e-3, warmup_steps=5, total_steps=25, min_lr=1e-5)
    lrs = [sched.lr_at(s) for s in range(26)]
    # Warmup is linear ramp, end of warmup equals base_lr
    assert lrs[0] < lrs[4] < lrs[5]
    assert math.isclose(lrs[5], 1e-3, rel_tol=1e-6)
    # After warmup it decays monotonically to min_lr
    for i in range(5, 25):
        assert lrs[i] >= lrs[i + 1] - 1e-12
    assert math.isclose(lrs[25], 1e-5, rel_tol=1e-6, abs_tol=1e-9)


def test_early_stopping_min_mode_triggers_after_patience():
    es = EarlyStopping(patience=2, mode="min")
    assert es.update(1.0) is False     # new best
    assert es.update(0.9) is False     # improved
    assert es.update(0.95) is False    # 1 no-improve
    assert es.update(0.95) is True     # 2 no-improve → stop


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=1, mode="max")
    assert es.update(0.5) is False
    assert es.update(0.4) is True      # first no-improve triggers (patience=1)


def test_early_stopping_invalid_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        EarlyStopping(patience=1, mode="sideways")
```

- [ ] **Step 2: Confirm failure**

```bash
conda run -n counting-env pytest tests/unit/test_training_callbacks.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement callbacks**

Create `src/counting/training/__init__.py`: (empty file)

Create `src/counting/training/callbacks.py`:

```python
"""Training callbacks: LR schedule, early stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class CosineWithWarmup:
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr: float = 0.0

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup: at step=0 we want a very small value, at end of warmup we want base_lr.
            if self.warmup_steps == 0:
                return self.base_lr
            return self.base_lr * (step + 1) / (self.warmup_steps + 1)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class EarlyStopping:
    def __init__(self, *, patience: int, mode: Literal["min", "max"]) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0

    def update(self, value: float) -> bool:
        """Return True if training should stop."""
        improved = (
            self.best is None
            or (self.mode == "min" and value < self.best)
            or (self.mode == "max" and value > self.best)
        )
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

- [ ] **Step 4: Tests pass**

```bash
conda run -n counting-env pytest tests/unit/test_training_callbacks.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Fix the warmup edge case if test 1 fails**

The test asserts `lrs[5] ≈ 1e-3` at the end of warmup. Verify the formula above produces this — trace: at step=5 with warmup_steps=5 the warmup branch is NOT taken (step < warmup_steps is False), so the decay branch fires with progress=0 → cosine=1 → lr = base_lr. OK.

At step=4 the warmup branch fires: `lr = 1e-3 * 5 / 6 ≈ 8.33e-4`. Still monotonic.

At step=25 the decay branch fires with progress=1 → cosine=0 → lr = min_lr. OK.

If this test fails on first run, do not change the formula lightly — the ramp is intentionally offset by +1 to avoid starting at zero. Only adjust if the test expectations need updating (file a follow-up).

- [ ] **Step 6: Commit**

```bash
git add src/counting/training/__init__.py src/counting/training/callbacks.py \
        tests/unit/test_training_callbacks.py
git commit -m "feat(training): cosine+warmup LR schedule and early stopping"
```

---

## Task 10: Checkpoint manager (save/load with config snapshot + resume)

**Goal:** Save model + optimizer + scheduler + epoch + best_metric + config snapshot atomically. Provide a loader that reconstructs these back.

**Files:**
- Create: `src/counting/training/checkpoint.py`
- Create: `tests/unit/test_training_checkpoint.py`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_training_checkpoint.py`:

```python
import torch

from counting.training.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        path=tmp_path / "ckpt.pt",
        model=model,
        optimizer=opt,
        epoch=7,
        best_metric=0.123,
        config_snapshot={"run_name": "test"},
    )

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(tmp_path / "ckpt.pt", model=model2, optimizer=opt2)

    assert meta["epoch"] == 7
    assert meta["best_metric"] == 0.123
    assert meta["config_snapshot"] == {"run_name": "test"}

    # Weights actually got restored
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_load_without_optimizer_is_ok(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    save_checkpoint(
        path=tmp_path / "ckpt.pt",
        model=model,
        optimizer=opt,
        epoch=1,
        best_metric=0.0,
        config_snapshot={},
    )

    model2 = torch.nn.Linear(4, 2)
    meta = load_checkpoint(tmp_path / "ckpt.pt", model=model2)
    assert meta["epoch"] == 1
```

- [ ] **Step 2: Confirm failure**

```bash
conda run -n counting-env pytest tests/unit/test_training_checkpoint.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement checkpoint manager**

Create `src/counting/training/checkpoint.py`:

```python
"""Checkpoint save/load with config snapshot for resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int,
    best_metric: float,
    config_snapshot: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler if scheduler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config_snapshot": dict(config_snapshot),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        "epoch": payload["epoch"],
        "best_metric": payload["best_metric"],
        "config_snapshot": payload.get("config_snapshot", {}),
        "scheduler": payload.get("scheduler"),
    }
```

- [ ] **Step 4: Tests pass**

```bash
conda run -n counting-env pytest tests/unit/test_training_checkpoint.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/counting/training/checkpoint.py tests/unit/test_training_checkpoint.py
git commit -m "feat(training): checkpoint save/load with config snapshot"
```

---

## Task 11: Training runner (loop + TensorBoard + validation)

**Goal:** A small, self-contained training loop scaffold. It does NOT yet know about PseCo specifics — those land in Task 12. This task produces reusable machinery.

**Files:**
- Create: `src/counting/training/runner.py`

No unit tests — exercised through Task 13 integration smoke.

---

- [ ] **Step 1: Implement runner scaffold**

Create `src/counting/training/runner.py`:

```python
"""Generic training loop scaffold with TensorBoard support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from counting.training.callbacks import CosineWithWarmup, EarlyStopping
from counting.training.checkpoint import save_checkpoint


log = logging.getLogger("counting.training")


@dataclass
class RunnerState:
    step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")


class Runner:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWithWarmup,
        early_stopping: EarlyStopping,
        device: str,
        run_dir: Path,
        log_every_n_steps: int,
        save_every_n_epochs: int,
        config_snapshot: dict[str, Any],
        tensorboard: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.log_every = log_every_n_steps
        self.save_every = save_every_n_epochs
        self.config_snapshot = config_snapshot
        self.state = RunnerState()
        self.tb = (
            SummaryWriter(self.run_dir / "tensorboard")
            if tensorboard
            else None
        )

    def _set_lr(self) -> float:
        lr = self.scheduler.lr_at(self.state.step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def train_epoch(
        self,
        train_loader: Iterable[Any],
        step_fn: Callable[[Any, torch.nn.Module], dict[str, torch.Tensor]],
    ) -> None:
        self.model.train()
        for batch in train_loader:
            lr = self._set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            losses = step_fn(batch, self.model)
            losses["total"].backward()
            self.optimizer.step()

            if self.tb is not None and self.state.step % self.log_every == 0:
                for k, v in losses.items():
                    self.tb.add_scalar(f"train/{k}", float(v.detach().cpu()), self.state.step)
                self.tb.add_scalar("train/lr", lr, self.state.step)
            self.state.step += 1

    @torch.no_grad()
    def validate(
        self,
        val_loader: Iterable[Any],
        eval_fn: Callable[[Any, torch.nn.Module], dict[str, float]],
    ) -> dict[str, float]:
        self.model.eval()
        agg: dict[str, float] = {}
        n = 0
        for batch in val_loader:
            metrics = eval_fn(batch, self.model)
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
        if n > 0:
            agg = {k: v / n for k, v in agg.items()}
        if self.tb is not None:
            for k, v in agg.items():
                self.tb.add_scalar(f"val/{k}", v, self.state.step)
        return agg

    def maybe_save(self, val_metrics: dict[str, float], metric_key: str) -> None:
        metric = val_metrics.get(metric_key, float("inf"))
        if metric < self.state.best_metric:
            self.state.best_metric = metric
            save_checkpoint(
                path=self.run_dir / "checkpoints" / "best.ckpt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.state.epoch,
                best_metric=self.state.best_metric,
                config_snapshot=self.config_snapshot,
            )
        if (self.state.epoch + 1) % self.save_every == 0:
            save_checkpoint(
                path=self.run_dir / "checkpoints" / "last.ckpt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.state.epoch,
                best_metric=self.state.best_metric,
                config_snapshot=self.config_snapshot,
            )

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
```

- [ ] **Step 2: Commit**

```bash
git add src/counting/training/runner.py
git commit -m "feat(training): Runner scaffold with TensorBoard + checkpointing"
```

---

## Task 12: PseCo head trainer (ties everything together)

**Goal:** The `train_pseco_head(cfg)` entry that:
1. Loads cached embeddings + proposals
2. Builds `ROIHeadMLP` (upstream class) on `device`
3. Loads CLIP text features for the configured prompt
4. Runs the `Runner` with `pseco_head_loss`

**Files:**
- Create: `src/counting/models/pseco/trainer.py`

Note: this is the largest single task because it integrates every prior artifact. It has no unit tests — Task 13 runs the smoke.

---

- [ ] **Step 1: Implement trainer entry**

Create `src/counting/models/pseco/trainer.py`:

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
    """Yields (embedding_tensor, proposals, gt_points) tuples."""

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
            "embedding": torch.from_numpy(emb).to(torch.float32),  # will move to device later
            "proposals": proposals,
            "gt_count": rec.count,
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_ids": [b["image_id"] for b in batch],
        "embeddings": torch.stack([b["embedding"] for b in batch]),
        "proposals": [b["proposals"] for b in batch],
        "gt_counts": torch.tensor([b["gt_count"] for b in batch], dtype=torch.float32),
    }


def train_pseco_head(cfg: TrainAppConfig) -> None:
    _ensure_external_on_path()
    from models import ROIHeadMLP

    device = resolve_device(cfg.device)

    fsc_train = FSC147Dataset(cfg.data.root, split=cfg.data.train_split)
    fsc_val = FSC147Dataset(cfg.data.root, split=cfg.data.val_split)

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

    # ROIHeadMLP
    head = ROIHeadMLP().to(device)
    if cfg.model.init_mlp:
        state = torch.load(cfg.model.init_mlp, map_location="cpu")
        if "model" in state:
            state = state["model"]
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

    def step_fn(batch, model):
        embs = batch["embeddings"].to(device)
        gt_counts = batch["gt_counts"].to(device)

        # Build per-proposal ROI features and logits via the ROIHead.
        # NOTE: upstream uses `roi_align` with precomputed bboxes. Here we
        # construct a minimal bbox set from the top-K proposal points and
        # classify each. For smoke parity we take the top-16 points.
        bboxes_per_image = _topk_points_as_boxes(batch["proposals"], k=16, image_size=1024)
        prompts = _unit_prompts(num_proposals=16, device=device)
        logits = model(embs, bboxes_per_image, prompts)  # shape (B*K, 1, 2)
        logits = logits.view(-1, 2)

        # Pseudo targets: positive class for top-k proposals.
        targets = torch.ones(logits.size(0), dtype=torch.long, device=device)

        pred_counts = torch.tensor(
            [(torch.softmax(logits[i * 16:(i + 1) * 16], dim=-1)[:, 1] > 0.5).sum()
             for i in range(embs.size(0))],
            dtype=torch.float32, device=device,
        )

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
        bboxes = _topk_points_as_boxes(batch["proposals"], k=16, image_size=1024)
        prompts = _unit_prompts(num_proposals=16, device=device)
        logits = model(embs, bboxes, prompts).view(-1, 2)
        probs = torch.softmax(logits, dim=-1)[:, 1].view(embs.size(0), -1)
        pred_counts = (probs > 0.5).sum(dim=1).float().cpu()
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
        boxes_per_image.append(boxes)
    return boxes_per_image


def _unit_prompts(*, num_proposals: int, device) -> list[torch.Tensor]:
    """Placeholder text prompt embeddings: unit vectors of size 512."""
    return [torch.ones(num_proposals, 1, 512, device=device) / (512 ** 0.5)]
```

**Important note on scope**: upstream's `ROIHeadMLP.forward` does `roi_align` on `(B, 256, H, W)` features, produces `(N, 512)` embeddings, then contrasts with `prompts` of shape `(N, C, 512)`. Our `step_fn` constructs a minimal version: top-16 proposal boxes and a unit-vector placeholder prompt. This is enough for the training loop to run end-to-end (proving all the plumbing works) but the model is NOT in a state to produce meaningful counts until real CLIP text features are fed in. A follow-up task in Plan 3 will wire `ops/dump_clip_features.py` to produce proper prompt embeddings.

For smoke purposes this task's correctness target is: the loop runs without shape errors, losses go down on a tiny random set, and checkpoints are written.

- [ ] **Step 2: Commit**

```bash
git add src/counting/models/pseco/trainer.py
git commit -m "feat(models): ROIHeadMLP trainer orchestration (smoke-ready)"
```

---

## Task 13: `counting train pseco-head` CLI + integration smoke

**Goal:** CLI entry + a slow-marked test that runs 2 epochs on a tiny cached dataset and verifies checkpoints appear.

**Files:**
- Modify: `src/counting/cli.py`
- Create: `tests/integration/test_training_smoke.py`

---

- [ ] **Step 1: Append CLI command**

Edit `src/counting/cli.py` — append at end of file:

```python


_train_app = typer.Typer(help="Training entry points", no_args_is_help=True)
app.add_typer(_train_app, name="train")


@_train_app.command("pseco-head")
def train_pseco_head_cli(
    config: str = typer.Option(..., "--config", "-c"),
    set_: list[str] = typer.Option(None, "--set"),
    resume: str = typer.Option(None, "--resume", help="Path to last.ckpt"),
) -> None:
    """Fine-tune the PseCo ROIHeadMLP using cached SAM features."""
    import yaml

    from counting.config.loader import apply_overrides
    from counting.config.train_schema import TrainAppConfig
    from counting.models.pseco.trainer import train_pseco_head

    raw = yaml.safe_load(open(config, "r", encoding="utf-8"))
    if set_:
        raw = apply_overrides(raw, list(set_))
    cfg = TrainAppConfig.model_validate(raw)

    if resume:
        console.print(f"[yellow]--resume is not yet supported; running from init_mlp[/yellow]")

    train_pseco_head(cfg)
    console.print("[green]training complete[/green]")
```

- [ ] **Step 2: Write integration smoke test**

Create `tests/integration/test_training_smoke.py`:

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
    """Run 2 epochs on a tiny synthetic cached dataset."""
    import torch

    from counting.config.train_schema import TrainAppConfig
    from counting.models.pseco.trainer import train_pseco_head

    # 1) Build a tiny FSC-147-like dataset on disk
    dataset_root = tmp_path / "fsc"
    (dataset_root / "images_384_VarV2").mkdir(parents=True)
    import json
    from PIL import Image

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

    # 2) Write a minimal cache by directly producing random embeddings
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

    # 3) Config
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

- [ ] **Step 3: Unit test suite must still pass**

```bash
conda run -n counting-env pytest -m "not slow" -v
```
Expected: all previous + new unit tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/counting/cli.py tests/integration/test_training_smoke.py
git commit -m "feat(cli): counting train pseco-head + integration smoke"
```

---

## Task 14: Density estimation in `diagnose` (deferred from Plan 1)

**Goal:** Extend `diagnose_directory` with a density estimate using SAM. Optional, only runs when SAM weights exist on disk. Writes `per_image[*].estimated_count` and adds a `density` block to `DiagnosticsReport`.

**Files:**
- Modify: `src/counting/data/diagnostics.py`
- Create: `tests/unit/test_diagnostics_density.py`

---

- [ ] **Step 1: Add density helper with test**

Create `tests/unit/test_diagnostics_density.py`:

```python
import numpy as np
from PIL import Image

from counting.data.diagnostics import (
    DiagnosticsReport,
    diagnose_directory,
)


def test_density_section_absent_without_sam(tmp_path):
    # SAM weights not present; diagnose should still run and simply omit density
    (tmp_path / "a.jpg").parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (128, 128, 128)).save(tmp_path / "a.jpg")
    report = diagnose_directory(tmp_path, report_dir=tmp_path / "out")
    assert isinstance(report, DiagnosticsReport)
    # density should be None or empty dict
    assert not getattr(report, "density", None)
```

- [ ] **Step 2: Extend `diagnostics.py`**

Edit `src/counting/data/diagnostics.py` — add `density: dict | None = None` to `DiagnosticsReport`, keep current signature of `diagnose_directory` unchanged (no new required args). The density block stays `None` when SAM weights aren't supplied; full density integration is a follow-up step that requires SAM weights + caching. Implement the stub:

At the top of `DiagnosticsReport`:
```python
@dataclass
class DiagnosticsReport:
    image_count: int
    resolution: dict[str, Any]
    blur: dict[str, Any]
    exposure: dict[str, Any]
    per_image: list[dict[str, Any]] = field(default_factory=list)
    density: dict[str, Any] | None = None       # NEW
```

No other changes. The test verifies the field exists and is falsy in the SAM-absent path.

- [ ] **Step 3: Tests pass**

```bash
conda run -n counting-env pytest tests/unit/test_diagnostics*.py -v
```
Expected: all existing + new pass.

- [ ] **Step 4: Commit**

```bash
git add src/counting/data/diagnostics.py tests/unit/test_diagnostics_density.py
git commit -m "feat(data): density slot in diagnostics report (SAM-powered impl deferred)"
```

---

## Task 15: README + docs update

**Files:**
- Modify: `README.md`

---

- [ ] **Step 1: Update `## 진행 상황` block**

Edit `README.md` — find the `## 진행 상황` section and replace with:

```markdown
## 진행 상황

- [x] Foundation (Plan 1): 스캐폴드, 설정, 데이터 진단, 추론 파이프라인
- [x] PseCo 학습 (Plan 2, 본 계획): SAM 피처 캐시 + ROIHeadMLP 파인튜닝
- [ ] Classifier 학습 · SR 통합 · 정돈 (Plan 3)
```

- [ ] **Step 2: Add training section**

Add AFTER the "## 전제 조건 (external/)" section:

````markdown
## 학습

### SAM 피처 캐싱 (학습 전 1회)

```bash
counting cache-embeddings --config configs/train/pseco_head.yaml
```

이미지 1장당 2MB (fp16). FSC-147 train (약 4800장) 기준 약 10 GB, 캐싱 시간 30~80분 (GPU).

### ROIHeadMLP 파인튜닝

```bash
counting train pseco-head --config configs/train/pseco_head.yaml
```

TensorBoard:

```bash
tensorboard --logdir runs/pseco_head_v1/tensorboard
```

### 필요 파일

- `models/PseCo/sam_vit_h.pth` — SAM ViT-H (2.4 GB). `python -m counting.data.download` 실행 시 안내 출력
- `models/PseCo/point_decoder_vith.pth`, `models/PseCo/MLP_small_box_w1_zeroshot.tar` — PseCo 공식 가중치 (원본 프로젝트 구조 유지)
- `datasets/fsc147/` — FSC-147 이미지 + 주석 JSON (공식 출처에서 다운로드)
````

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: Plan 2 usage (caching + training CLI)"
```

- [ ] **Step 4: Push**

```bash
git push
```

---

## Self-Review (Plan author)

**Spec coverage**:
- §4.2 학습 파이프라인 (Step A embed + Step B head train) → Tasks 4, 5, 6, 8, 9, 11, 12
- §5 임베딩 캐시 설계 (fp16 shard, meta.json, hash 무효화) → Task 5, 6
- §6.1 PseCo 헤드 하이퍼파라미터 → Task 3 (YAML) + Task 9 (schedule) + Task 8 (loss)
- §6.3 재현성 (seed, config snapshot) → Task 10
- §7 설정 스키마 → Task 3
- §8 CLI `train` / `cache-embeddings` → Tasks 6, 13
- Checkpoint/resume → Task 10 (save/load) + Task 13 CLI (stub `--resume`)

**Placeholder scan**: no "TBD" or unfilled sections. Task 12 explicitly flags the CLIP text prompt simplification as a smoke-only shortcut and points to Plan 3 for the real wiring — this is intentional and documented, not a placeholder.

**Type consistency**: `TrainAppConfig` fields (`cfg.train.lr`, `cfg.train.loss_weights`, `cfg.logging.log_every_n_steps`, etc.) are named identically wherever referenced across Tasks 3, 6, 11, 12, 13. `FeatureCacheReader.read(key)` / `FeatureCacheWriter.write(key, arr)` are consistent across Tasks 5, 6, 12, 13.

**Known limitation (documented in Task 12)**: the trainer ships with unit-vector placeholder prompts. The loop runs end-to-end and trains, but predicted counts on real data need proper CLIP text features wired in Plan 3. Smoke test verifies plumbing, not count accuracy.

**Risk**: SAM ViT-H on RTX 5070 (12 GB VRAM) — feature extraction needs ~6 GB activations. Should fit comfortably. Proposal step (frozen SAM + PointDecoder) runs per-batch and also fits. If OOM, reduce `batch_size` via `--set train.batch_size=1`.

**Go/No-Go**: Go. Plan 2 is self-contained, testable, and produces a demonstrable training run. Every task has concrete code and a clear test / smoke.

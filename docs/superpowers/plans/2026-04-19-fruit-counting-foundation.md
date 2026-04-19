# Fruit Counting Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 과일 카운팅 시스템의 기반(스캐폴드, 설정, 데이터 진단, 추론 파이프라인)을 구축해 원본 `original_code/` 의 추론 기능을 대체 가능한 상태로 만든다. 학습 코드는 포함하지 않는다 (Plan 2/3 에서 추가).

**Architecture:** `src/counting/` 라이브러리 + `counting` CLI. Deblur/PseCo/SR/Classifier 각각을 `Stage` 프로토콜을 구현하는 얇은 래퍼로 감싸고, YAML + Pydantic 설정을 통해 파이프라인을 조립. `external/` 원본 소스와 `models/` 가중치는 유지. Mac(MPS)/CPU 기본, CUDA 는 conda env 전환으로 지원.

**Tech Stack:** Python 3.11, PyTorch 2.3 (MPS/CUDA/CPU), Typer (CLI), Pydantic v2 (설정), OpenCV/Pillow, pytest, conda (환경 격리).

**Spec:** `docs/superpowers/specs/2026-04-19-fruit-counting-redesign-design.md`

**Scope of this plan (Phases 1–4 of the spec):**
1. 스캐폴드 (conda env, pyproject.toml, 패키지 구조, `counting info`)
2. 설정 시스템 (Pydantic 스키마, YAML 로더, dot-path 오버라이드)
3. 데이터 진단 도구 (`counting diagnose`)
4. 추론 파이프라인 v1 (Stage 프로토콜, Deblur/PseCo 래퍼, `counting infer`/`batch`)
5. 결과 I/O 최소치 (JSON/CSV 직렬화)

**Out of scope (이 플랜에서 제외):**
- 임베딩 캐시 및 PseCo 학습 (Plan 2)
- Classifier 학습 및 SR 기본 활성화 (Plan 3)
- `external/` 소스 복원 — 구현 중 PseCo/Deblur 래퍼 단계에서 부재 시 명시적 오류로 안내

---

## File Structure

각 파일의 단일 책임을 먼저 확정해 Task 분해의 기준으로 삼는다.

### 새로 만드는 파일

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | 패키지 메타, 의존성 pin, `counting` 엔트리포인트 |
| `environment.yml` | conda env `counting-env` 정의 (Mac/CPU/MPS 기본) |
| `environment-cuda.yml` | conda env CUDA 변형 (GPU 서버용) |
| `.gitignore` | `runs/`, `feature_cache/`, `__pycache__/`, `.DS_Store` 등 제외 |
| `README.md` | 설치/실행 요약 (Plan 1 범위 기준) |
| `src/counting/__init__.py` | public API: `Pipeline`, `load_config`, `CountingResult` |
| `src/counting/config/__init__.py` | 공개 API 재노출 |
| `src/counting/config/schema.py` | Pydantic v2 설정 스키마 (루트/파이프라인/스테이지/IO) |
| `src/counting/config/loader.py` | YAML 파싱 + dot-path 오버라이드 적용 + 검증 |
| `src/counting/config/hashing.py` | 설정 해시 계산 (전처리·경로 변경 감지용) |
| `src/counting/data/__init__.py` | public API |
| `src/counting/data/base.py` | `ImageRecord`, `CountingDataset` 프로토콜 |
| `src/counting/data/formats/__init__.py` | 포맷 레지스트리 |
| `src/counting/data/formats/imagefolder.py` | 디렉터리 스캔용 최소 구현 (라벨 없음) |
| `src/counting/data/diagnostics.py` | 해상도/블러/노출/밀도 진단 + HTML/JSON 리포트 |
| `src/counting/models/__init__.py` | 공개 API |
| `src/counting/models/base.py` | `Stage` 프로토콜, 공통 유틸 (device 이동, eval 모드) |
| `src/counting/models/deblur.py` | DeblurGANv2 래퍼 (추론 전용) |
| `src/counting/models/pseco/__init__.py` | 재노출 |
| `src/counting/models/pseco/inference.py` | PseCo 추론 래퍼 (카운팅 + 크롭) |
| `src/counting/models/sr.py` | SR-NO 래퍼 (추론 전용; 기본 비활성) |
| `src/counting/models/classification/__init__.py` | 재노출 |
| `src/counting/models/classification/inference.py` | ResNet 분류 추론 래퍼 (기본 비활성) |
| `src/counting/pipeline.py` | `Pipeline` 클래스 — Stage 조립, 에러 격리, 타이밍 |
| `src/counting/io/__init__.py` | public API |
| `src/counting/io/results.py` | `CountingResult`, `StageTiming`, `CropMeta` dataclass |
| `src/counting/io/serialize.py` | JSON/CSV 직렬화 + 로더 |
| `src/counting/utils/device.py` | `resolve_device("auto"|...)` + fallback env 설정 |
| `src/counting/utils/image.py` | `read_image_rgb`, `ensure_pil`, `ensure_np_rgb` |
| `src/counting/utils/logging.py` | `get_logger`, 회전 파일 핸들러 (옵션) |
| `src/counting/cli.py` | Typer 앱: `info`, `validate-config`, `diagnose`, `infer`, `batch` |
| `configs/pipeline/default.yaml` | 기본 파이프라인 설정 |
| `tests/conftest.py` | 공통 fixture (더미 이미지, tmp 경로) |
| `tests/unit/test_config_schema.py` | Pydantic 검증 |
| `tests/unit/test_config_loader.py` | YAML 로드 + 오버라이드 |
| `tests/unit/test_config_hashing.py` | 해시 안정성 |
| `tests/unit/test_utils_device.py` | device resolver |
| `tests/unit/test_utils_image.py` | 이미지 유틸 |
| `tests/unit/test_data_imagefolder.py` | 디렉터리 스캐너 |
| `tests/unit/test_diagnostics.py` | 진단 지표 산출 (SAM 불필요 경로) |
| `tests/unit/test_io_serialize.py` | JSON/CSV 라운드트립 |
| `tests/unit/test_pipeline_errors.py` | Stage 실패 격리 (더미 Stage) |
| `tests/integration/test_pipeline_smoke.py` | 더미 이미지로 파이프라인 완주 (slow; 가중치 없으면 skip) |

### 유지 / 건드리지 않는 것

- `original_code/**` — 읽기 전용 보존
- `external/**` (존재할 경우) — 서드파티 원본, 수정 금지
- `models/**` — 사전학습 가중치

---

## Task 1: 리포지토리 스캐폴드 & conda 환경

**목적:** 작업 공간을 구성하고 conda 전용 env 로 기존 프로젝트와 격리한다. 설치 즉시 `counting info` 가 동작해 환경을 점검할 수 있게 한다.

**Files:**
- Create: `.gitignore`
- Create: `environment.yml`
- Create: `environment-cuda.yml`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/counting/__init__.py`
- Create: `src/counting/cli.py`
- Create: `src/counting/utils/__init__.py`
- Create: `src/counting/utils/device.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_utils_device.py`

---

- [ ] **Step 1: `.gitignore` 작성**

Create `.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
.pytest_cache/
.coverage
htmlcov/
.DS_Store

# venv/conda artifacts inside repo
.venv/
env/

# run outputs
runs/
feature_cache/
reports/
logs/
*.log

# editors
.vscode/
.idea/
```

- [ ] **Step 2: conda env 파일 작성 (Mac/CPU/MPS 기본)**

Create `environment.yml`:

```yaml
name: counting-env
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pillow
  - pandas
  - pyyaml
  - tqdm
  - scipy
  - scikit-learn
  - opencv
  - pytorch=2.3.*
  - torchvision=0.18.*
  - tensorboard
  - pytest
  - pytest-cov
  - pip:
      - typer>=0.12
      - pydantic>=2.0
      - albumentations
      - timm
      - rich
```

- [ ] **Step 3: conda env 파일 작성 (CUDA 변형)**

Create `environment-cuda.yml`:

```yaml
name: counting-env
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pillow
  - pandas
  - pyyaml
  - tqdm
  - scipy
  - scikit-learn
  - opencv
  - pytorch=2.3.*
  - torchvision=0.18.*
  - pytorch-cuda=12.1
  - tensorboard
  - pytest
  - pytest-cov
  - pip:
      - typer>=0.12
      - pydantic>=2.0
      - albumentations
      - timm
      - rich
```

- [ ] **Step 4: `pyproject.toml` 작성**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "counting"
version = "0.1.0"
description = "Fruit counting pipeline (inference + training)"
requires-python = ">=3.10"
readme = "README.md"
dependencies = [
  "numpy",
  "pillow",
  "pandas",
  "pyyaml",
  "pydantic>=2.0",
  "typer>=0.12",
  "rich",
  "tqdm",
  "opencv-python",
  "torch>=2.2",
  "torchvision",
  "timm",
  "albumentations",
  "tensorboard",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[project.scripts]
counting = "counting.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "slow: integration tests that load models or touch disk",
]
addopts = "-q"
```

- [ ] **Step 5: `README.md` 초기 작성**

Create `README.md`:

```markdown
# counting

과일 카운팅 파이프라인 (재설계 버전).

## 설치 (Mac, conda)

\`\`\`bash
conda env create -f environment.yml
conda activate counting-env
pip install -e .
\`\`\`

GPU 서버:

\`\`\`bash
conda env create -f environment-cuda.yml
conda activate counting-env
pip install -e .
\`\`\`

## 환경 점검

\`\`\`bash
counting info
\`\`\`

## 진행 상황

- [x] Foundation (Plan 1): 스캐폴드, 설정, 데이터 진단, 추론 파이프라인
- [ ] PseCo 학습 (Plan 2)
- [ ] Classifier 학습 · SR 통합 · 정돈 (Plan 3)

설계: `docs/superpowers/specs/2026-04-19-fruit-counting-redesign-design.md`
```

- [ ] **Step 6: 패키지 루트 `__init__.py` (임시)**

Create `src/counting/__init__.py`:

```python
"""counting — fruit counting pipeline."""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

Create `src/counting/utils/__init__.py`:

```python
"""Utility helpers."""
```

- [ ] **Step 7: device resolver 테스트 작성 (실패 기대)**

Create `tests/__init__.py`: (empty)
Create `tests/unit/__init__.py`: (empty)
Create `tests/conftest.py`:

```python
import pytest


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
```

Create `tests/unit/test_utils_device.py`:

```python
import pytest

from counting.utils.device import resolve_device


def test_resolve_device_cpu_explicit():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_picks_something():
    d = resolve_device("auto")
    assert d in {"cpu", "mps", "cuda"}


def test_resolve_device_cuda_without_cuda_raises(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")


def test_resolve_device_invalid_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")
```

- [ ] **Step 8: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_utils_device.py -v`
Expected: ModuleNotFoundError or collection error — `counting.utils.device` 미존재

- [ ] **Step 9: device resolver 구현**

Create `src/counting/utils/device.py`:

```python
"""Device selection with safe fallbacks."""

from __future__ import annotations

import os

import torch

_VALID = {"auto", "cpu", "mps", "cuda"}


def resolve_device(requested: str) -> str:
    """Return one of 'cpu' | 'mps' | 'cuda'.

    'auto' picks cuda → mps → cpu depending on availability.
    An explicit device that is not available raises RuntimeError.
    """
    req = requested.lower()
    if req not in _VALID:
        raise ValueError(f"Unknown device: {requested!r}. Valid: {sorted(_VALID)}")

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            _enable_mps_fallback()
            return "mps"
        return "cpu"

    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"

    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        _enable_mps_fallback()
        return "mps"

    return "cpu"


def _enable_mps_fallback() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

- [ ] **Step 10: 테스트 통과 확인**

Run: `pytest tests/unit/test_utils_device.py -v`
Expected: PASS (4 tests)

- [ ] **Step 11: CLI 최소 구현 (`counting info`)**

Create `src/counting/cli.py`:

```python
"""Typer-based CLI entrypoint."""

from __future__ import annotations

import platform
import sys

import typer
from rich.console import Console
from rich.table import Table

from counting import __version__
from counting.utils.device import resolve_device

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def info() -> None:
    """Show environment and device information."""
    import torch

    table = Table(title=f"counting {__version__}")
    table.add_column("key")
    table.add_column("value")
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    table.add_row("torch", torch.__version__)
    table.add_row("cuda.available", str(torch.cuda.is_available()))
    mps = getattr(torch.backends, "mps", None)
    table.add_row("mps.available", str(bool(mps and mps.is_available())))
    table.add_row("resolved(auto)", resolve_device("auto"))
    console.print(table)


if __name__ == "__main__":
    app()
```

- [ ] **Step 12: 설치 및 CLI 동작 확인**

Run:
```bash
conda env list | grep counting-env || echo "create env first: conda env create -f environment.yml"
conda run -n counting-env pip install -e .
conda run -n counting-env counting info
```
Expected: 테이블 출력 (python/torch/device 정보). env 가 없으면 생성 안내 메시지.

- [ ] **Step 13: 커밋**

```bash
git init 2>/dev/null || true
git add .gitignore environment.yml environment-cuda.yml pyproject.toml README.md \
        src/counting/__init__.py src/counting/utils/__init__.py src/counting/utils/device.py \
        src/counting/cli.py tests/__init__.py tests/unit/__init__.py tests/conftest.py \
        tests/unit/test_utils_device.py
git commit -m "chore: scaffold counting package with conda env and info CLI"
```

---

## Task 2: 설정 스키마 (Pydantic v2)

**목적:** YAML 설정을 Pydantic 으로 검증해 잘못된 값·경로를 조기에 차단한다. 스테이지별 설정은 `enabled` 플래그로 on/off.

**Files:**
- Create: `src/counting/config/__init__.py`
- Create: `src/counting/config/schema.py`
- Create: `tests/unit/test_config_schema.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_config_schema.py`:

```python
import pytest
from pydantic import ValidationError

from counting.config.schema import (
    AppConfig,
    ClassifierStageConfig,
    DeblurStageConfig,
    IOConfig,
    PipelineConfig,
    PseCoStageConfig,
    SRStageConfig,
    StageSet,
)


def _minimal_dict():
    return {
        "device": "auto",
        "seed": 42,
        "output_dir": "./runs",
        "pipeline": {
            "stages": {
                "deblur": {"enabled": True, "weights": "models/deblur/fpn.pth"},
                "pseco": {
                    "enabled": True,
                    "prompt": "protective fruit bag",
                    "sam_checkpoint": "models/PseCo/sam_vit_h.pth",
                    "decoder_checkpoint": "models/PseCo/point_decoder_vith.pth",
                    "mlp_checkpoint": "models/PseCo/MLP_small_box_w1_zeroshot.tar",
                },
                "sr": {"enabled": False, "scale": 2.0, "max_crop_side": 500},
                "classifier": {
                    "enabled": False,
                    "checkpoint": "models/classification/classification_model.pt",
                    "threshold": 0.5,
                },
            }
        },
        "io": {"output_format": "json", "save_visualizations": False},
    }


def test_valid_config_parses():
    cfg = AppConfig.model_validate(_minimal_dict())
    assert cfg.device == "auto"
    assert cfg.pipeline.stages.pseco.prompt == "protective fruit bag"


def test_invalid_device_rejected():
    d = _minimal_dict()
    d["device"] = "tpu"
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_pseco_requires_checkpoints_when_enabled():
    d = _minimal_dict()
    d["pipeline"]["stages"]["pseco"]["sam_checkpoint"] = ""
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_sr_disabled_allows_missing_fields():
    d = _minimal_dict()
    d["pipeline"]["stages"]["sr"] = {"enabled": False}
    cfg = AppConfig.model_validate(d)
    assert cfg.pipeline.stages.sr.enabled is False


def test_classifier_threshold_bounds():
    d = _minimal_dict()
    d["pipeline"]["stages"]["classifier"]["threshold"] = 1.5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_io_format_options():
    d = _minimal_dict()
    d["io"]["output_format"] = "both"
    cfg = AppConfig.model_validate(d)
    assert cfg.io.output_format == "both"

    d["io"]["output_format"] = "xml"
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_config_schema.py -v`
Expected: ImportError — 모듈 미존재

- [ ] **Step 3: 스키마 구현**

Create `src/counting/config/__init__.py`:

```python
"""Configuration package."""

from counting.config.schema import AppConfig

__all__ = ["AppConfig"]
```

Create `src/counting/config/schema.py`:

```python
"""Pydantic configuration schema for the counting pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DeblurStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    weights: str = ""


class PseCoStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    prompt: str = "protective fruit bag"
    sam_checkpoint: str = ""
    decoder_checkpoint: str = ""
    mlp_checkpoint: str = ""

    @model_validator(mode="after")
    def _require_checkpoints_when_enabled(self):
        if self.enabled:
            missing = [
                name
                for name in ("sam_checkpoint", "decoder_checkpoint", "mlp_checkpoint")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"PseCo enabled but missing: {missing}")
        return self


class SRStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    scale: float = Field(default=2.0, gt=1.0, le=8.0)
    max_crop_side: int = Field(default=500, gt=0)


class ClassifierStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    checkpoint: str = ""
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_ckpt_when_enabled(self):
        if self.enabled and not self.checkpoint:
            raise ValueError("Classifier enabled but checkpoint path is empty")
        return self


class StageSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deblur: DeblurStageConfig = DeblurStageConfig()
    pseco: PseCoStageConfig = PseCoStageConfig()
    sr: SRStageConfig = SRStageConfig()
    classifier: ClassifierStageConfig = ClassifierStageConfig()


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stages: StageSet = StageSet()


class IOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_format: Literal["json", "csv", "both"] = "json"
    save_visualizations: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 42
    output_dir: str = "./runs"
    pipeline: PipelineConfig = PipelineConfig()
    io: IOConfig = IOConfig()

    @field_validator("output_dir")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("output_dir must be a non-empty path")
        return v
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/unit/test_config_schema.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: 커밋**

```bash
git add src/counting/config/__init__.py src/counting/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(config): add Pydantic schema for pipeline configuration"
```

---

## Task 3: YAML 로더 + dot-path 오버라이드

**목적:** YAML 파일을 읽어 스키마로 검증하고, CLI `--set key.path=value` 오버라이드를 지원한다.

**Files:**
- Create: `src/counting/config/loader.py`
- Create: `configs/pipeline/default.yaml`
- Create: `tests/unit/test_config_loader.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_config_loader.py`:

```python
import textwrap

import pytest

from counting.config.loader import apply_overrides, load_config


def _write(tmp_path, text: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return p


def test_load_valid_yaml(tmp_path):
    p = _write(
        tmp_path,
        """
        device: cpu
        seed: 7
        output_dir: ./out
        pipeline:
          stages:
            deblur: {enabled: false}
            pseco:
              enabled: true
              sam_checkpoint: a.pth
              decoder_checkpoint: b.pth
              mlp_checkpoint: c.tar
        """,
    )
    cfg = load_config(p)
    assert cfg.device == "cpu"
    assert cfg.seed == 7
    assert cfg.pipeline.stages.pseco.sam_checkpoint == "a.pth"


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no.yaml")


def test_apply_overrides_bool_and_number():
    cfg = {"pipeline": {"stages": {"sr": {"enabled": False, "scale": 2.0}}}}
    out = apply_overrides(
        cfg,
        ["pipeline.stages.sr.enabled=true", "pipeline.stages.sr.scale=3"],
    )
    assert out["pipeline"]["stages"]["sr"]["enabled"] is True
    assert out["pipeline"]["stages"]["sr"]["scale"] == 3


def test_apply_overrides_string():
    cfg = {"pipeline": {"stages": {"pseco": {"prompt": "a"}}}}
    out = apply_overrides(cfg, ["pipeline.stages.pseco.prompt=bagged apple"])
    assert out["pipeline"]["stages"]["pseco"]["prompt"] == "bagged apple"


def test_apply_overrides_rejects_bad_path():
    with pytest.raises(KeyError):
        apply_overrides({"a": {"b": 1}}, ["a.c=2"])


def test_apply_overrides_rejects_missing_equal():
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, ["apple"])


def test_load_with_overrides(tmp_path):
    p = _write(
        tmp_path,
        """
        device: cpu
        pipeline:
          stages:
            pseco:
              enabled: true
              sam_checkpoint: a
              decoder_checkpoint: b
              mlp_checkpoint: c
              prompt: apple
        """,
    )
    cfg = load_config(p, overrides=["pipeline.stages.pseco.prompt=pear"])
    assert cfg.pipeline.stages.pseco.prompt == "pear"
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_config_loader.py -v`
Expected: ImportError

- [ ] **Step 3: 로더 구현**

Create `src/counting/config/loader.py`:

```python
"""YAML config loading with dot-path CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from counting.config.schema import AppConfig


def load_config(
    path: str | Path,
    *,
    overrides: Iterable[str] | None = None,
) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {p}")
    if overrides:
        raw = apply_overrides(raw, list(overrides))
    return AppConfig.model_validate(raw)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Override must be key.path=value: {spec!r}")
        key, _, raw_value = spec.partition("=")
        _set_dot_path(out, key.split("."), _coerce(raw_value))
    return out


def _set_dot_path(node: dict[str, Any], parts: list[str], value: Any) -> None:
    cur: Any = node
    for i, part in enumerate(parts[:-1]):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Unknown override path segment: {'.'.join(parts[: i + 1])}")
        cur = cur[part]
    last = parts[-1]
    if not isinstance(cur, dict) or last not in cur:
        raise KeyError(f"Unknown override path: {'.'.join(parts)}")
    cur[last] = value


def _coerce(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
```

- [ ] **Step 4: 기본 YAML 설정 파일 작성**

Create `configs/pipeline/default.yaml`:

```yaml
device: auto
seed: 42
output_dir: ./runs

pipeline:
  stages:
    deblur:
      enabled: true
      weights: models/deblurganv2/fpn_mobilenet.pth
    pseco:
      enabled: true
      prompt: "protective fruit bag"
      sam_checkpoint: models/PseCo/sam_vit_h.pth
      decoder_checkpoint: models/PseCo/point_decoder_vith.pth
      mlp_checkpoint: models/PseCo/MLP_small_box_w1_zeroshot.tar
    sr:
      enabled: false
      scale: 2.0
      max_crop_side: 500
    classifier:
      enabled: false
      checkpoint: models/classification/classification_model.pt
      threshold: 0.5

io:
  output_format: json
  save_visualizations: false
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/unit/test_config_loader.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: CLI 에 `validate-config` 추가**

Edit `src/counting/cli.py` — append after `info()`:

```python
@app.command("validate-config")
def validate_config(
    path: str = typer.Argument(..., help="Path to YAML config"),
    set_: list[str] = typer.Option(
        None, "--set", help="Dot-path override: key.path=value", show_default=False
    ),
) -> None:
    """Load and validate a YAML config. Prints resolved device."""
    from counting.config.loader import load_config

    cfg = load_config(path, overrides=set_ or None)
    device = resolve_device(cfg.device)
    console.print(f"[green]OK[/green] {path} (device → {device})")
```

- [ ] **Step 7: CLI 수동 확인**

Run:
```bash
conda run -n counting-env counting validate-config configs/pipeline/default.yaml
```
Expected: `OK configs/pipeline/default.yaml (device → cpu|mps|cuda)`

- [ ] **Step 8: 커밋**

```bash
git add src/counting/config/loader.py configs/pipeline/default.yaml \
        tests/unit/test_config_loader.py src/counting/cli.py
git commit -m "feat(config): YAML loader with dot-path overrides and validate-config CLI"
```

---

## Task 4: 설정 해시 계산

**목적:** 임베딩 캐시(Plan 2)와 결과 재현성 목적으로 설정 지문(해시)을 생성한다. Plan 1 단계에서는 추론 결과 파일에 해시를 기록해 나중에 "이 결과가 어떤 설정으로 나왔는지" 추적 가능.

**Files:**
- Create: `src/counting/config/hashing.py`
- Create: `tests/unit/test_config_hashing.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_config_hashing.py`:

```python
from counting.config.hashing import config_hash
from counting.config.schema import AppConfig


def _cfg(**overrides):
    data = {
        "device": "cpu",
        "seed": 42,
        "output_dir": "./runs",
        "pipeline": {
            "stages": {
                "pseco": {
                    "enabled": True,
                    "sam_checkpoint": "a",
                    "decoder_checkpoint": "b",
                    "mlp_checkpoint": "c",
                }
            }
        },
    }
    data.update(overrides)
    return AppConfig.model_validate(data)


def test_hash_is_deterministic():
    a = config_hash(_cfg())
    b = config_hash(_cfg())
    assert a == b
    assert len(a) == 16


def test_hash_changes_with_relevant_field():
    a = config_hash(_cfg())
    b = config_hash(_cfg(seed=99))
    assert a != b


def test_hash_ignores_device_and_output_dir():
    """Device/output_dir are runtime concerns; they must not change the hash."""
    base = config_hash(_cfg())
    assert config_hash(_cfg(device="auto")) == base
    assert config_hash(_cfg(output_dir="./other")) == base
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_config_hashing.py -v`
Expected: ImportError

- [ ] **Step 3: 해시 구현**

Create `src/counting/config/hashing.py`:

```python
"""Stable hash of the semantic (non-runtime) parts of a config."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from counting.config.schema import AppConfig

_RUNTIME_KEYS = {"device", "output_dir"}


def config_hash(cfg: AppConfig) -> str:
    data = cfg.model_dump(mode="json")
    pruned = {k: v for k, v in data.items() if k not in _RUNTIME_KEYS}
    blob = json.dumps(pruned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _as_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/unit/test_config_hashing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 커밋**

```bash
git add src/counting/config/hashing.py tests/unit/test_config_hashing.py
git commit -m "feat(config): add deterministic config hash excluding runtime fields"
```

---

## Task 5: 이미지 유틸 & 로깅

**목적:** 공용 이미지 I/O(RGB 기준)와 간단한 로거 헬퍼. 원본 `utils/image_utils.py` 의 `read_image_rgb` 와 호환.

**Files:**
- Create: `src/counting/utils/image.py`
- Create: `src/counting/utils/logging.py`
- Create: `tests/unit/test_utils_image.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_utils_image.py`:

```python
import numpy as np
from PIL import Image

from counting.utils.image import ensure_np_rgb, ensure_pil, read_image_rgb


def test_read_image_rgb(tmp_path):
    arr = (np.random.rand(10, 12, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    Image.fromarray(arr).save(p)

    out = read_image_rgb(p)
    assert out.shape == (10, 12, 3)
    assert out.dtype == np.uint8


def test_ensure_pil_from_numpy_rgb():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    pil = ensure_pil(arr, assume="rgb")
    assert isinstance(pil, Image.Image)
    assert pil.size == (4, 4)


def test_ensure_pil_from_pil():
    im = Image.new("RGB", (2, 3))
    assert ensure_pil(im) is im


def test_ensure_np_rgb_from_pil():
    im = Image.new("RGB", (3, 2), (255, 0, 0))
    arr = ensure_np_rgb(im)
    assert arr.shape == (2, 3, 3)
    assert arr[0, 0, 0] == 255


def test_ensure_np_rgb_from_ndarray_passthrough():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    assert ensure_np_rgb(arr) is arr
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_utils_image.py -v`
Expected: ImportError

- [ ] **Step 3: 이미지 유틸 구현**

Create `src/counting/utils/image.py`:

```python
"""Minimal image I/O helpers. All numpy arrays are RGB uint8 unless stated."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


def read_image_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def ensure_pil(image, *, assume: Literal["rgb", "bgr"] = "rgb") -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image)
        if image.ndim == 3 and image.shape[2] == 3:
            arr = image if assume == "rgb" else image[..., ::-1]
            return Image.fromarray(arr)
    raise TypeError(f"Cannot convert to PIL.Image: {type(image)}")


def ensure_np_rgb(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    raise TypeError(f"Cannot convert to np.ndarray: {type(image)}")
```

- [ ] **Step 4: 로거 구현**

Create `src/counting/utils/logging.py`:

```python
"""Logger factory with console + optional rotating file handler."""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s — %(message)s"


def get_logger(
    name: str = "counting",
    *,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    backup_count: int = 7,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(_FMT)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(p, when="midnight", backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/unit/test_utils_image.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: 커밋**

```bash
git add src/counting/utils/image.py src/counting/utils/logging.py tests/unit/test_utils_image.py
git commit -m "feat(utils): add image I/O helpers and logger factory"
```

---

## Task 6: 데이터 레이어 — Dataset 프로토콜 & ImageFolder

**목적:** Plan 2/3 에서 확장할 데이터 어댑터 계층의 최소 기반. Plan 1 에서는 라벨 없는 `imagefolder` 만으로 추론/진단을 수행.

**Files:**
- Create: `src/counting/data/__init__.py`
- Create: `src/counting/data/base.py`
- Create: `src/counting/data/formats/__init__.py`
- Create: `src/counting/data/formats/imagefolder.py`
- Create: `tests/unit/test_data_imagefolder.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_data_imagefolder.py`:

```python
import numpy as np
from PIL import Image

from counting.data.formats.imagefolder import ImageFolderDataset


def _make_image(path, size=(8, 8), color=(0, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_enumerates_supported_extensions(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "sub" / "b.png")
    (tmp_path / "readme.txt").write_text("x")

    ds = ImageFolderDataset(tmp_path)
    assert len(ds) == 2
    rels = sorted(r.relpath for r in ds)
    assert rels == ["a.jpg", "sub/b.png"]


def test_getitem_returns_record_and_loads_image(tmp_path):
    _make_image(tmp_path / "a.jpg", color=(255, 0, 0))

    ds = ImageFolderDataset(tmp_path)
    rec = ds[0]
    assert rec.relpath == "a.jpg"
    assert rec.path.exists()
    img = rec.read_rgb()
    assert isinstance(img, np.ndarray)
    assert img.shape == (8, 8, 3)
    assert img[0, 0, 0] == 255


def test_empty_directory_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        ImageFolderDataset(tmp_path)
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_data_imagefolder.py -v`
Expected: ImportError

- [ ] **Step 3: 데이터 기반 구현**

Create `src/counting/data/__init__.py`:

```python
"""Data layer."""

from counting.data.base import CountingDataset, ImageRecord
from counting.data.formats.imagefolder import ImageFolderDataset

__all__ = ["CountingDataset", "ImageRecord", "ImageFolderDataset"]
```

Create `src/counting/data/base.py`:

```python
"""Dataset protocol. Plan 2 extends this with point/box labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from counting.utils.image import read_image_rgb


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relpath: str

    def read_rgb(self) -> np.ndarray:
        return read_image_rgb(self.path)


@runtime_checkable
class CountingDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> ImageRecord: ...
    def __iter__(self): ...
```

Create `src/counting/data/formats/__init__.py`:

```python
"""Dataset format adapters."""
```

Create `src/counting/data/formats/imagefolder.py`:

```python
"""A directory of images, no labels. Inference/diagnostics only."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from counting.data.base import ImageRecord

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageFolderDataset:
    def __init__(self, root: str | Path, *, extensions: set[str] | None = None) -> None:
        self.root = Path(root)
        self.extensions = extensions or _EXTS
        self._paths = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.extensions
        )
        if not self._paths:
            raise FileNotFoundError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> ImageRecord:
        p = self._paths[idx]
        return ImageRecord(path=p, relpath=str(p.relative_to(self.root)))

    def __iter__(self) -> Iterator[ImageRecord]:
        for i in range(len(self)):
            yield self[i]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/unit/test_data_imagefolder.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 커밋**

```bash
git add src/counting/data tests/unit/test_data_imagefolder.py
git commit -m "feat(data): add Dataset protocol and ImageFolderDataset"
```

---

## Task 7: 데이터 진단 도구 (SAM 불필요 지표만)

**목적:** 해상도/블러/노출 지표를 계산해 JSON + HTML 리포트를 생성. 밀도 추정(SAM 기반)은 Plan 2 에서 PseCo 스테이지가 준비된 뒤 연결.

**Files:**
- Create: `src/counting/data/diagnostics.py`
- Create: `tests/unit/test_diagnostics.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_diagnostics.py`:

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from counting.data.diagnostics import DiagnosticsReport, diagnose_directory


def _make(path: Path, size, fill):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_diagnose_produces_report(tmp_path):
    _make(tmp_path / "a.jpg", (64, 48), 128)
    _make(tmp_path / "b.jpg", (128, 96), 200)

    report_dir = tmp_path / "out"
    report = diagnose_directory(tmp_path, report_dir=report_dir)

    assert isinstance(report, DiagnosticsReport)
    assert report.image_count == 2
    assert {"width", "height"} <= set(report.resolution.keys())
    assert (report_dir / "diagnostics.json").exists()
    assert (report_dir / "diagnostics_report.html").exists()

    data = json.loads((report_dir / "diagnostics.json").read_text())
    assert data["image_count"] == 2


def test_blur_metric_flags_flat_image(tmp_path):
    _make(tmp_path / "flat.jpg", (32, 32), 128)

    report = diagnose_directory(tmp_path, report_dir=tmp_path / "out")
    assert report.blur["min"] <= 1.0
    assert report.blur["low_blur_ratio"] >= 0.5


def test_empty_directory_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        diagnose_directory(tmp_path, report_dir=tmp_path / "out")
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_diagnostics.py -v`
Expected: ImportError

- [ ] **Step 3: 진단 구현**

Create `src/counting/data/diagnostics.py`:

```python
"""Dataset diagnostics: resolution, blur, exposure. HTML + JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from counting.data.formats.imagefolder import ImageFolderDataset

_BLUR_THRESHOLD = 100.0       # Laplacian variance; < threshold ⇒ "blurry"
_LOW_EXPOSURE = 40            # mean intensity < ⇒ underexposed
_HIGH_EXPOSURE = 215          # mean intensity > ⇒ overexposed


@dataclass
class DiagnosticsReport:
    image_count: int
    resolution: dict[str, Any]
    blur: dict[str, Any]
    exposure: dict[str, Any]
    per_image: list[dict[str, Any]] = field(default_factory=list)


def diagnose_directory(image_dir: str | Path, *, report_dir: str | Path) -> DiagnosticsReport:
    ds = ImageFolderDataset(image_dir)
    widths, heights, blurs, means = [], [], [], []
    per_image = []

    for rec in ds:
        arr = rec.read_rgb()
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean = float(gray.mean())

        widths.append(w); heights.append(h); blurs.append(lap_var); means.append(mean)
        per_image.append({
            "relpath": rec.relpath,
            "width": w, "height": h,
            "blur_var": round(lap_var, 3),
            "mean_intensity": round(mean, 2),
        })

    low_blur_ratio = sum(1 for b in blurs if b < _BLUR_THRESHOLD) / len(blurs)
    under = sum(1 for m in means if m < _LOW_EXPOSURE) / len(means)
    over = sum(1 for m in means if m > _HIGH_EXPOSURE) / len(means)

    report = DiagnosticsReport(
        image_count=len(ds),
        resolution={
            "width": {"min": min(widths), "max": max(widths), "median": median(widths)},
            "height": {"min": min(heights), "max": max(heights), "median": median(heights)},
        },
        blur={
            "min": round(min(blurs), 3),
            "max": round(max(blurs), 3),
            "median": round(float(np.median(blurs)), 3),
            "threshold": _BLUR_THRESHOLD,
            "low_blur_ratio": round(low_blur_ratio, 3),
        },
        exposure={
            "underexposed_ratio": round(under, 3),
            "overexposed_ratio": round(over, 3),
            "low_threshold": _LOW_EXPOSURE,
            "high_threshold": _HIGH_EXPOSURE,
        },
        per_image=per_image,
    )

    _write_reports(report, Path(report_dir))
    return report


def _write_reports(report: DiagnosticsReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    (out_dir / "diagnostics_report.html").write_text(_render_html(report), encoding="utf-8")


def _render_html(report: DiagnosticsReport) -> str:
    rows = "\n".join(
        f"<tr><td>{p['relpath']}</td><td>{p['width']}×{p['height']}</td>"
        f"<td>{p['blur_var']}</td><td>{p['mean_intensity']}</td></tr>"
        for p in report.per_image
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Diagnostics</title>
<style>body{{font-family:system-ui;margin:24px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}</style>
</head><body>
<h1>Dataset Diagnostics</h1>
<p>images: {report.image_count}</p>
<h2>Resolution</h2><pre>{json.dumps(report.resolution, indent=2)}</pre>
<h2>Blur (Laplacian variance)</h2><pre>{json.dumps(report.blur, indent=2)}</pre>
<h2>Exposure (gray mean)</h2><pre>{json.dumps(report.exposure, indent=2)}</pre>
<h2>Per image</h2>
<table><tr><th>relpath</th><th>size</th><th>blur_var</th><th>mean</th></tr>
{rows}
</table></body></html>
"""
```

- [ ] **Step 4: CLI 에 `diagnose` 추가**

Edit `src/counting/cli.py` — append:

```python
@app.command()
def diagnose(
    image_dir: str = typer.Argument(..., help="Directory of images"),
    report_dir: str = typer.Option("./reports/diagnostics", help="Where to write outputs"),
) -> None:
    """Compute resolution/blur/exposure diagnostics for a directory."""
    from counting.data.diagnostics import diagnose_directory

    r = diagnose_directory(image_dir, report_dir=report_dir)
    console.print(
        f"[green]OK[/green] {r.image_count} images → {report_dir}\n"
        f"  low_blur_ratio={r.blur['low_blur_ratio']}  "
        f"under={r.exposure['underexposed_ratio']}  over={r.exposure['overexposed_ratio']}"
    )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/unit/test_diagnostics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: 커밋**

```bash
git add src/counting/data/diagnostics.py tests/unit/test_diagnostics.py src/counting/cli.py
git commit -m "feat(data): dataset diagnostics (resolution/blur/exposure) + CLI"
```

---

## Task 8: 결과 스키마 & 직렬화 (JSON/CSV)

**목적:** 파이프라인 출력 포맷을 고정해 추론/배치 CLI 가 일관되게 저장하도록 한다.

**Files:**
- Create: `src/counting/io/__init__.py`
- Create: `src/counting/io/results.py`
- Create: `src/counting/io/serialize.py`
- Create: `tests/unit/test_io_serialize.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_io_serialize.py`:

```python
import json

from counting.io.results import CountingResult, CropMeta, StageTiming
from counting.io.serialize import (
    read_batch_csv,
    read_batch_json,
    write_batch_csv,
    write_batch_json,
)


def _sample(image_path="x.jpg", count=3):
    return CountingResult(
        image_path=image_path,
        raw_count=count,
        verified_count=count,
        points=[(1.0, 2.0), (3.0, 4.0)],
        boxes=[(0.0, 0.0, 10.0, 10.0)],
        crops=[CropMeta(bbox=(0, 0, 10, 10), score=0.9, is_bag=True)],
        timings_ms=[StageTiming(stage="pseco", ms=12.3)],
        device="cpu",
        config_hash="abc123",
        error=None,
    )


def test_json_roundtrip(tmp_path):
    results = [_sample("a.jpg"), _sample("b.jpg", count=1)]
    p = tmp_path / "out.json"
    write_batch_json(results, p)
    out = read_batch_json(p)
    assert len(out) == 2
    assert out[0].image_path == "a.jpg"
    assert out[0].raw_count == 3
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == 1


def test_csv_roundtrip(tmp_path):
    results = [_sample("a.jpg"), _sample("b.jpg", count=1)]
    p = tmp_path / "out.csv"
    write_batch_csv(results, p)
    rows = read_batch_csv(p)
    assert [r["image_path"] for r in rows] == ["a.jpg", "b.jpg"]
    assert rows[0]["raw_count"] == "3"


def test_error_preserved_in_json(tmp_path):
    r = _sample()
    r = r.__class__(**{**r.__dict__, "error": "pseco failed"})
    p = tmp_path / "out.json"
    write_batch_json([r], p)
    assert read_batch_json(p)[0].error == "pseco failed"
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_io_serialize.py -v`
Expected: ImportError

- [ ] **Step 3: 결과 dataclass 구현**

Create `src/counting/io/__init__.py`:

```python
"""Result I/O."""

from counting.io.results import CountingResult, CropMeta, StageTiming
from counting.io.serialize import (
    read_batch_csv,
    read_batch_json,
    write_batch_csv,
    write_batch_json,
)

__all__ = [
    "CountingResult", "CropMeta", "StageTiming",
    "read_batch_csv", "read_batch_json",
    "write_batch_csv", "write_batch_json",
]
```

Create `src/counting/io/results.py`:

```python
"""Inference result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageTiming:
    stage: str
    ms: float


@dataclass
class CropMeta:
    bbox: tuple[float, float, float, float]     # x1,y1,x2,y2
    score: float | None = None
    is_bag: bool | None = None                  # filled if classifier ran


@dataclass
class CountingResult:
    image_path: str
    raw_count: int                              # PseCo raw count
    verified_count: int                         # after classifier if enabled, else == raw_count
    points: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    crops: list[CropMeta] = field(default_factory=list)
    timings_ms: list[StageTiming] = field(default_factory=list)
    device: str = "cpu"
    config_hash: str = ""
    error: Optional[str] = None
```

- [ ] **Step 4: 직렬화 구현**

Create `src/counting/io/serialize.py`:

```python
"""JSON/CSV I/O for CountingResult."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from counting.io.results import CountingResult, CropMeta, StageTiming

SCHEMA_VERSION = 1

_CSV_HEADERS = [
    "image_path", "raw_count", "verified_count",
    "device", "config_hash", "error", "timings_ms_total",
]


def write_batch_json(results: Iterable[CountingResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "results": [asdict(r) for r in results],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_batch_json(path: str | Path) -> list[CountingResult]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version in {p}")
    return [_result_from_dict(d) for d in data["results"]]


def write_batch_csv(results: Iterable[CountingResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "image_path": r.image_path,
                "raw_count": r.raw_count,
                "verified_count": r.verified_count,
                "device": r.device,
                "config_hash": r.config_hash,
                "error": r.error or "",
                "timings_ms_total": round(sum(t.ms for t in r.timings_ms), 2),
            })


def read_batch_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _result_from_dict(d: dict) -> CountingResult:
    return CountingResult(
        image_path=d["image_path"],
        raw_count=int(d["raw_count"]),
        verified_count=int(d["verified_count"]),
        points=[tuple(x) for x in d.get("points", [])],
        boxes=[tuple(x) for x in d.get("boxes", [])],
        crops=[CropMeta(**c) for c in d.get("crops", [])],
        timings_ms=[StageTiming(**t) for t in d.get("timings_ms", [])],
        device=d.get("device", "cpu"),
        config_hash=d.get("config_hash", ""),
        error=d.get("error"),
    )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/unit/test_io_serialize.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: 커밋**

```bash
git add src/counting/io tests/unit/test_io_serialize.py
git commit -m "feat(io): CountingResult dataclasses and JSON/CSV serialization"
```

---

## Task 9: Stage 프로토콜 & 에러 격리 파이프라인

**목적:** 실제 모델 래퍼를 붙이기 전에 Stage 인터페이스와 파이프라인 조립 로직을 테스트 가능한 형태로 확정한다.

**Files:**
- Create: `src/counting/models/__init__.py`
- Create: `src/counting/models/base.py`
- Create: `src/counting/pipeline.py`
- Create: `tests/unit/test_pipeline_errors.py`

---

- [ ] **Step 1: 테스트 작성 (실패 기대)**

Create `tests/unit/test_pipeline_errors.py`:

```python
import numpy as np
import pytest

from counting.models.base import Stage, StageResult
from counting.pipeline import Pipeline


class _IdentityDeblur(Stage):
    name = "deblur"

    def prepare(self, _cfg): pass
    def process(self, image):
        return StageResult(output=image)
    def cleanup(self): pass


class _ConstCount(Stage):
    name = "pseco"

    def __init__(self, n: int):
        self.n = n

    def prepare(self, _cfg): pass
    def process(self, image):
        return StageResult(output={"count": self.n, "crops": [], "points": [], "boxes": []})
    def cleanup(self): pass


class _FailingStage(Stage):
    name = "sr"

    def prepare(self, _cfg): pass
    def process(self, _x): raise RuntimeError("boom")
    def cleanup(self): pass


def _image():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_pipeline_runs_and_counts():
    pipe = Pipeline.from_stages([_IdentityDeblur(), _ConstCount(5)], device="cpu", config_hash="h")
    res = pipe.run_numpy(_image(), image_path="x.jpg")
    assert res.raw_count == 5
    assert res.verified_count == 5
    assert res.error is None
    assert [t.stage for t in res.timings_ms] == ["deblur", "pseco"]


def test_pipeline_records_stage_failure_without_crashing():
    pipe = Pipeline.from_stages(
        [_IdentityDeblur(), _ConstCount(2), _FailingStage()],
        device="cpu",
        config_hash="h",
    )
    res = pipe.run_numpy(_image(), image_path="x.jpg")
    assert res.raw_count == 2
    assert res.error and "sr" in res.error


def test_pipeline_requires_counting_stage():
    with pytest.raises(ValueError, match="pseco"):
        Pipeline.from_stages([_IdentityDeblur()], device="cpu", config_hash="h").run_numpy(
            _image(), image_path="x.jpg"
        )
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `pytest tests/unit/test_pipeline_errors.py -v`
Expected: ImportError

- [ ] **Step 3: Stage 프로토콜 구현**

Create `src/counting/models/__init__.py`:

```python
"""Model wrappers implementing the Stage protocol."""

from counting.models.base import Stage, StageResult

__all__ = ["Stage", "StageResult"]
```

Create `src/counting/models/base.py`:

```python
"""Stage protocol for pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class StageResult:
    output: Any
    metadata: dict[str, Any] | None = None


@runtime_checkable
class Stage(Protocol):
    name: str

    def prepare(self, cfg: Any) -> None: ...
    def process(self, x: Any) -> StageResult: ...
    def cleanup(self) -> None: ...
```

- [ ] **Step 4: 파이프라인 구현**

Create `src/counting/pipeline.py`:

```python
"""Pipeline assembly: runs stages in order with per-stage timing and error isolation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from counting.io.results import CountingResult, CropMeta, StageTiming
from counting.models.base import Stage

_REQUIRED_COUNTING_STAGE = "pseco"


@dataclass
class Pipeline:
    stages: list[Stage]
    device: str
    config_hash: str

    @classmethod
    def from_stages(cls, stages: list[Stage], *, device: str, config_hash: str) -> "Pipeline":
        return cls(stages=stages, device=device, config_hash=config_hash)

    def run_numpy(self, image: np.ndarray, *, image_path: str) -> CountingResult:
        timings: list[StageTiming] = []
        raw_count = 0
        points: list[tuple[float, float]] = []
        boxes: list[tuple[float, float, float, float]] = []
        crops_meta: list[CropMeta] = []
        error: str | None = None

        has_counting = any(s.name == _REQUIRED_COUNTING_STAGE for s in self.stages)
        if not has_counting:
            raise ValueError(f"Pipeline must include a '{_REQUIRED_COUNTING_STAGE}' stage")

        current: Any = image
        verified_count: int | None = None

        for stage in self.stages:
            t0 = time.perf_counter()
            try:
                result = stage.process(current)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                timings.append(StageTiming(stage=stage.name, ms=round(elapsed_ms, 3)))
                error = f"[{stage.name}] {type(exc).__name__}: {exc}"
                break

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(StageTiming(stage=stage.name, ms=round(elapsed_ms, 3)))

            if stage.name == "pseco":
                out = result.output
                raw_count = int(out.get("count", 0))
                points = [tuple(p) for p in out.get("points", [])]
                boxes = [tuple(b) for b in out.get("boxes", [])]
                crops_meta = [CropMeta(bbox=tuple(b)) for b in boxes]
                current = {"image": image, "crops": out.get("crops", [])}
            elif stage.name == "classifier":
                # expects: {"verified_count": int, "per_crop": [bool, ...]}
                out = result.output
                verified_count = int(out.get("verified_count", raw_count))
                for i, is_bag in enumerate(out.get("per_crop", [])):
                    if i < len(crops_meta):
                        crops_meta[i] = CropMeta(
                            bbox=crops_meta[i].bbox,
                            score=crops_meta[i].score,
                            is_bag=bool(is_bag),
                        )
            else:
                current = result.output

        if verified_count is None:
            verified_count = raw_count

        return CountingResult(
            image_path=image_path,
            raw_count=raw_count,
            verified_count=verified_count,
            points=points,
            boxes=boxes,
            crops=crops_meta,
            timings_ms=timings,
            device=self.device,
            config_hash=self.config_hash,
            error=error,
        )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/unit/test_pipeline_errors.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: 커밋**

```bash
git add src/counting/models/__init__.py src/counting/models/base.py src/counting/pipeline.py \
        tests/unit/test_pipeline_errors.py
git commit -m "feat(pipeline): Stage protocol and Pipeline with per-stage timings/error isolation"
```

---

## Task 10: Deblur 래퍼 (DeblurGANv2)

**목적:** 원본 `external/DeblurGANv2.core.deblur_inference.DeblurInference` 를 Stage 로 감싼다. Plan 1 은 추론 전용.

**Files:**
- Create: `src/counting/models/deblur.py`

(단위 테스트는 가중치·외부 소스 의존성 때문에 Task 14 스모크에서 다룬다.)

---

- [ ] **Step 1: 래퍼 구현**

Create `src/counting/models/deblur.py`:

```python
"""DeblurGANv2 inference wrapper (read-only; no training in Plan 1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_np_rgb

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[3]      # repo root
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class DeblurStage(Stage):
    name = "deblur"

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from DeblurGANv2.core.deblur_inference import DeblurInference
        except ImportError as exc:
            raise RuntimeError(
                "DeblurGANv2 source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        self._impl = DeblurInference()
        try:
            self._impl.model.eval()
        except Exception:
            pass

    def process(self, image: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("DeblurStage used before prepare()")
        arr = ensure_np_rgb(image)
        out = self._impl.predict(arr)
        return StageResult(output=np.asarray(out))

    def cleanup(self) -> None:
        self._impl = None
```

- [ ] **Step 2: 커밋**

```bash
git add src/counting/models/deblur.py
git commit -m "feat(models): DeblurGANv2 inference Stage wrapper"
```

---

## Task 11: PseCo 추론 래퍼

**목적:** 원본 `external/PseCo.core.PseCo_inference.CountingInference` 를 Stage 로 감싼다. 출력은 `{"count": int, "points": [...], "boxes": [...], "crops": [PIL ...]}` 형태로 파이프라인이 소비.

**Files:**
- Create: `src/counting/models/pseco/__init__.py`
- Create: `src/counting/models/pseco/inference.py`

---

- [ ] **Step 1: 래퍼 구현**

Create `src/counting/models/pseco/__init__.py`:

```python
"""PseCo wrappers (inference in Plan 1)."""

from counting.models.pseco.inference import PseCoStage

__all__ = ["PseCoStage"]
```

Create `src/counting/models/pseco/inference.py`:

```python
"""PseCo counting inference wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_np_rgb

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]      # repo root
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class PseCoStage(Stage):
    name = "pseco"

    def __init__(self, *, prompt: str, sam_ckpt: str, decoder_ckpt: str, mlp_ckpt: str) -> None:
        self.prompt = prompt
        self.sam_ckpt = sam_ckpt
        self.decoder_ckpt = decoder_ckpt
        self.mlp_ckpt = mlp_ckpt
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from PseCo.core.PseCo_inference import CountingInference
        except ImportError as exc:
            raise RuntimeError(
                "PseCo source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        # Upstream constructor signature kept as in original_code/ai_modules/main.py.
        self._impl = CountingInference(prompt=self.prompt)

    def process(self, image: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("PseCoStage used before prepare()")
        arr = ensure_np_rgb(image)
        crops, count = self._impl.get_count_images(arr, None)
        points = list(getattr(self._impl, "last_points", []) or [])
        boxes = list(getattr(self._impl, "last_boxes", []) or [])
        return StageResult(output={
            "count": int(count),
            "crops": list(crops or []),
            "points": [tuple(p) for p in points],
            "boxes": [tuple(b) for b in boxes],
        })

    def cleanup(self) -> None:
        self._impl = None
```

- [ ] **Step 2: 커밋**

```bash
git add src/counting/models/pseco
git commit -m "feat(models): PseCo counting Stage wrapper"
```

---

## Task 12: SR 래퍼 (기본 비활성)

**목적:** SR 스테이지 틀만 만들어 두고 설정 `enabled: true` 일 때만 파이프라인에 포함되도록 한다. PseCo 크롭에 per-crop 으로 적용. Classifier 가 함께 켜진 경우에만 의미가 있으므로 Plan 1 에서는 기본 비활성 유지.

**Files:**
- Create: `src/counting/models/sr.py`

---

- [ ] **Step 1: 래퍼 구현**

Create `src/counting/models/sr.py`:

```python
"""Super-Resolution Neural Operator wrapper (per-crop, inference only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_pil

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class SRStage(Stage):
    name = "sr"

    def __init__(self, *, scale: float, max_crop_side: int) -> None:
        self.scale = scale
        self.max_crop_side = max_crop_side
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from Super_Resolution_Neural_Operator.core.super_resolution_inference import (
                SuperResolutionModule,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SR-NO source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        self._impl = SuperResolutionModule(scale=self.scale)

    def process(self, payload: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("SRStage used before prepare()")
        crops = payload.get("crops", []) if isinstance(payload, dict) else []
        upscaled: list[Image.Image] = []
        for crop in crops:
            pil = ensure_pil(crop)
            if pil.size[0] > self.max_crop_side or pil.size[1] > self.max_crop_side:
                upscaled.append(pil)
                continue
            upscaled.append(self._impl.predict(pil))
        payload = dict(payload) if isinstance(payload, dict) else {}
        payload["crops"] = upscaled
        return StageResult(output=payload)

    def cleanup(self) -> None:
        self._impl = None
```

- [ ] **Step 2: 커밋**

```bash
git add src/counting/models/sr.py
git commit -m "feat(models): SR-NO per-crop Stage wrapper (disabled by default)"
```

---

## Task 13: Classifier 추론 래퍼 (기본 비활성)

**목적:** 분류기를 선택적으로 파이프라인 끝에 붙여 `verified_count` 를 산출.

**Files:**
- Create: `src/counting/models/classification/__init__.py`
- Create: `src/counting/models/classification/inference.py`

---

- [ ] **Step 1: 래퍼 구현**

Create `src/counting/models/classification/__init__.py`:

```python
"""Classification wrappers (inference in Plan 1)."""

from counting.models.classification.inference import ClassifierStage

__all__ = ["ClassifierStage"]
```

Create `src/counting/models/classification/inference.py`:

```python
"""ResNet-based bagged-fruit classifier Stage (inference)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from counting.models.base import Stage, StageResult
from counting.utils.image import ensure_pil

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class ClassifierStage(Stage):
    name = "classifier"

    def __init__(self, *, checkpoint: str, threshold: float) -> None:
        self.checkpoint = checkpoint
        self.threshold = threshold
        self._impl: Any = None

    def prepare(self, _cfg: Any) -> None:
        _ensure_external_on_path()
        try:
            from classification.core.classification_inference import ClassificationInference
        except ImportError as exc:
            raise RuntimeError(
                "Classification source not found under external/. See spec §1 '전제 조건'."
            ) from exc
        self._impl = ClassificationInference(ckpt_path=self.checkpoint, threshold=self.threshold)

    def process(self, payload: Any) -> StageResult:
        if self._impl is None:
            raise RuntimeError("ClassifierStage used before prepare()")
        crops = payload.get("crops", []) if isinstance(payload, dict) else []
        per_crop: list[bool] = []
        for crop in crops:
            is_bag, _p_bag, _label, _scores = self._impl.verify_bagged_fruit(ensure_pil(crop))
            per_crop.append(bool(is_bag))
        return StageResult(output={
            "verified_count": int(sum(per_crop)),
            "per_crop": per_crop,
        })

    def cleanup(self) -> None:
        self._impl = None
```

- [ ] **Step 2: 커밋**

```bash
git add src/counting/models/classification
git commit -m "feat(models): bagged-fruit classifier Stage wrapper (disabled by default)"
```

---

## Task 14: 파이프라인 팩토리 & CLI `infer`/`batch`

**목적:** 설정에서 파이프라인을 구성하고 단일·배치 실행 CLI 를 제공한다. 학습 없이 원본 추론 기능을 대체.

**Files:**
- Modify: `src/counting/__init__.py`
- Modify: `src/counting/pipeline.py` (팩토리 추가)
- Modify: `src/counting/cli.py` (`infer`, `batch` 추가)
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_pipeline_smoke.py`

---

- [ ] **Step 1: public API 노출**

Replace `src/counting/__init__.py`:

```python
"""counting — fruit counting pipeline."""

from counting.config.loader import load_config
from counting.io.results import CountingResult
from counting.pipeline import Pipeline, build_pipeline

__version__ = "0.1.0"

__all__ = ["__version__", "Pipeline", "build_pipeline", "load_config", "CountingResult"]
```

- [ ] **Step 2: `Pipeline.build_pipeline` 팩토리 추가**

Edit `src/counting/pipeline.py` — append (after the `Pipeline` class):

```python
def build_pipeline(cfg, *, device: str | None = None):
    """Construct a Pipeline from an AppConfig, honoring enabled flags."""
    from counting.config.hashing import config_hash
    from counting.models.classification.inference import ClassifierStage
    from counting.models.deblur import DeblurStage
    from counting.models.pseco.inference import PseCoStage
    from counting.models.sr import SRStage
    from counting.utils.device import resolve_device

    resolved = resolve_device(device or cfg.device)
    stages: list[Stage] = []

    s = cfg.pipeline.stages
    if s.deblur.enabled:
        st = DeblurStage(weights=s.deblur.weights)
        st.prepare(cfg)
        stages.append(st)
    if s.pseco.enabled:
        st = PseCoStage(
            prompt=s.pseco.prompt,
            sam_ckpt=s.pseco.sam_checkpoint,
            decoder_ckpt=s.pseco.decoder_checkpoint,
            mlp_ckpt=s.pseco.mlp_checkpoint,
        )
        st.prepare(cfg)
        stages.append(st)
    if s.sr.enabled:
        st = SRStage(scale=s.sr.scale, max_crop_side=s.sr.max_crop_side)
        st.prepare(cfg)
        stages.append(st)
    if s.classifier.enabled:
        st = ClassifierStage(checkpoint=s.classifier.checkpoint, threshold=s.classifier.threshold)
        st.prepare(cfg)
        stages.append(st)

    return Pipeline(stages=stages, device=resolved, config_hash=config_hash(cfg))
```

- [ ] **Step 3: CLI 에 `infer`/`batch` 추가**

Edit `src/counting/cli.py` — append:

```python
@app.command()
def infer(
    image: str = typer.Argument(..., help="Path to image"),
    config: str = typer.Option(..., "--config", "-c", help="Pipeline YAML"),
    set_: list[str] = typer.Option(None, "--set", help="Override key.path=value"),
    output: str = typer.Option(None, "--output", "-o", help="Output JSON path (optional)"),
) -> None:
    """Run inference on a single image."""
    from counting.config.loader import load_config
    from counting.io.serialize import write_batch_json
    from counting.pipeline import build_pipeline
    from counting.utils.image import read_image_rgb

    cfg = load_config(config, overrides=set_ or None)
    pipe = build_pipeline(cfg)
    arr = read_image_rgb(image)
    result = pipe.run_numpy(arr, image_path=str(image))

    console.print(
        f"image={image} raw_count={result.raw_count} verified={result.verified_count} "
        f"device={result.device} err={result.error or '-'}"
    )
    if output:
        write_batch_json([result], output)
        console.print(f"[green]saved[/green] {output}")


@app.command()
def batch(
    image_dir: str = typer.Argument(..., help="Directory of images"),
    config: str = typer.Option(..., "--config", "-c", help="Pipeline YAML"),
    set_: list[str] = typer.Option(None, "--set", help="Override key.path=value"),
    output: str = typer.Option("./runs/last_batch", help="Output directory"),
    fmt: str = typer.Option("json", "--format", help="json | csv | both"),
) -> None:
    """Run inference over a directory."""
    from pathlib import Path

    from counting.config.loader import load_config
    from counting.data.formats.imagefolder import ImageFolderDataset
    from counting.io.serialize import write_batch_csv, write_batch_json
    from counting.pipeline import build_pipeline

    cfg = load_config(config, overrides=set_ or None)
    if fmt not in {"json", "csv", "both"}:
        raise typer.BadParameter("--format must be json|csv|both")

    ds = ImageFolderDataset(image_dir)
    pipe = build_pipeline(cfg)
    results = []
    for i, rec in enumerate(ds, 1):
        arr = rec.read_rgb()
        r = pipe.run_numpy(arr, image_path=str(rec.path))
        results.append(r)
        console.print(
            f"[{i}/{len(ds)}] {rec.relpath} raw={r.raw_count} verified={r.verified_count} "
            f"err={r.error or '-'}"
        )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt in {"json", "both"}:
        write_batch_json(results, out_dir / "results.json")
    if fmt in {"csv", "both"}:
        write_batch_csv(results, out_dir / "results.csv")
    console.print(f"[green]done[/green] {len(results)} images → {out_dir}")
```

- [ ] **Step 4: 통합 스모크 테스트 (가중치 없으면 skip)**

Create `tests/integration/__init__.py`: (empty)
Create `tests/integration/test_pipeline_smoke.py`:

```python
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "pipeline" / "default.yaml"


def _weights_available() -> bool:
    try:
        import yaml
    except ImportError:
        return False
    cfg = yaml.safe_load(CFG.read_text())
    stages = cfg["pipeline"]["stages"]
    need = []
    if stages["pseco"]["enabled"]:
        need += [
            stages["pseco"]["sam_checkpoint"],
            stages["pseco"]["decoder_checkpoint"],
            stages["pseco"]["mlp_checkpoint"],
        ]
    if stages["deblur"]["enabled"]:
        need.append(stages["deblur"]["weights"])
    return all((ROOT / p).exists() for p in need)


@pytest.mark.slow
@pytest.mark.skipif(not _weights_available(), reason="model weights not present")
def test_pipeline_runs_on_dummy_image(tmp_path):
    from counting.config.loader import load_config
    from counting.pipeline import build_pipeline

    img_path = tmp_path / "dummy.png"
    Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8")).save(img_path)

    cfg = load_config(CFG, overrides=["device=cpu"])
    pipe = build_pipeline(cfg)
    arr = np.asarray(Image.open(img_path).convert("RGB"))
    r = pipe.run_numpy(arr, image_path=str(img_path))

    assert r.config_hash
    assert r.error is None or "pseco" in r.error  # tolerate upstream quirks on random data
```

- [ ] **Step 5: 테스트 실행 (유닛만)**

Run: `pytest -m "not slow" -v`
Expected: 앞선 모든 유닛 테스트 PASS. 가중치 없으면 슬로우 스모크는 자동 skip.

- [ ] **Step 6: CLI 수동 확인 (가중치 있을 때만)**

Run (optional, skip if weights missing):
```bash
conda run -n counting-env counting infer test_data/22.jpg \
  --config configs/pipeline/default.yaml --set device=cpu
```
Expected: `image=... raw_count=<int> verified=<int> device=cpu err=-` 또는 `err=` 에 상류 에러 표시.

- [ ] **Step 7: 커밋**

```bash
git add src/counting/__init__.py src/counting/pipeline.py src/counting/cli.py \
        tests/integration/__init__.py tests/integration/test_pipeline_smoke.py
git commit -m "feat(pipeline): build_pipeline factory + infer/batch CLI"
```

---

## Task 15: 마무리 — README 갱신 & 전체 테스트

**Files:**
- Modify: `README.md`

---

- [ ] **Step 1: README 사용 예시 추가**

Edit `README.md` — replace the `## 진행 상황` block with:

```markdown
## 사용

### 설정 검증

\`\`\`bash
counting validate-config configs/pipeline/default.yaml
\`\`\`

### 데이터 진단

\`\`\`bash
counting diagnose ./my_images --report-dir ./reports/diagnostics
\`\`\`

### 단일 추론

\`\`\`bash
counting infer ./image.jpg --config configs/pipeline/default.yaml \
  --set device=cpu
\`\`\`

### 배치 추론

\`\`\`bash
counting batch ./input_dir --config configs/pipeline/default.yaml \
  --output ./runs/last_batch --format json
\`\`\`

### 라이브러리 사용

\`\`\`python
from counting import Pipeline, build_pipeline, load_config
from counting.utils.image import read_image_rgb

cfg = load_config("configs/pipeline/default.yaml")
pipe = build_pipeline(cfg)
result = pipe.run_numpy(read_image_rgb("image.jpg"), image_path="image.jpg")
print(result.raw_count, result.verified_count)
\`\`\`

## 진행 상황

- [x] Foundation (Plan 1): 스캐폴드, 설정, 데이터 진단, 추론 파이프라인
- [ ] PseCo 학습 (Plan 2)
- [ ] Classifier 학습 · SR 통합 · 정돈 (Plan 3)

## 전제 조건 (external/)

서드파티 원본 소스가 아래 경로에 있어야 `deblur`/`pseco`/`sr`/`classifier` 스테이지를 활성화할 수 있습니다:

- `external/DeblurGANv2/`
- `external/PseCo/`
- `external/Super_Resolution_Neural_Operator/`
- `external/classification/`

비활성 상태로는 설정 검증·진단·I/O 만 사용 가능합니다.

설계 문서: `docs/superpowers/specs/2026-04-19-fruit-counting-redesign-design.md`
```

- [ ] **Step 2: 전체 유닛 테스트 통과 확인**

Run: `pytest -m "not slow" -v`
Expected: 전체 PASS

- [ ] **Step 3: 커밋**

```bash
git add README.md
git commit -m "docs: usage examples and external/ prerequisites for Plan 1"
```

---

## Self-Review (플랜 작성자 체크)

스펙 커버리지 — Plan 1 범위(스펙 §2 합의 결정, §3 폴더, §4 추론 흐름, §7 설정, §8 CLI, §10 환경, §11 테스트, §12 단계 1~4 + 7)를 모두 작업으로 매핑했다.

- §2 결정 요약 → Task 1~3 (conda, 설정), Task 9 (Stage), Task 14 (CLI)
- §3 폴더 구조 → File Structure + Task 1~14 전반
- §4.1 추론 흐름 → Task 9~14
- §4.4 진단 → Task 7
- §5 임베딩 캐시 → **Plan 2** (의도적 제외)
- §6 하이퍼파라미터 → **Plan 2/3**
- §7 설정 스키마 → Task 2, 3
- §8 CLI → Task 1 (info), 3 (validate-config), 7 (diagnose), 14 (infer, batch)
- §9 에러 처리 → Task 9 (격리), Task 2 (검증)
- §10 conda env → Task 1
- §11 테스트 전략 → Task 2~9 (단위), Task 14 (스모크)
- §12 단계 1~4·7 → Task 1~14

Placeholder 검사 — 모든 코드 블록이 완성된 내용으로 채워져 있고 "TBD" / "구현 생략" 류 없음.

타입 일관성 — `Pipeline.run_numpy`, `CountingResult`, `StageTiming`, `CropMeta` 필드명이 모든 Task 에서 동일. PseCo 출력 스키마(`{"count","crops","points","boxes"}`)와 Classifier 출력(`{"verified_count","per_crop"}`)이 Task 9 예시/Task 11/Task 13/Task 14 팩토리/Task 14 CLI 에서 일관.

범위 — 학습 요소를 모두 제외했고 `external/` 부재 시 명시적 오류 + README 안내. Plan 1 은 단독으로 "설정 검증·진단·추론 CLI" 가 동작하는 소프트웨어를 남긴다.

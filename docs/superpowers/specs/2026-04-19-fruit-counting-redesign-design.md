# Fruit Counting — 전면 재설계 Design Spec

- **Date**: 2026-04-19
- **Scope**: 과일 카운팅 파이프라인 전면 재설계 (학습 + 추론)
- **Out of scope**: 병충해(화상병) 검출 — 별도 폴더에서 진행
- **Legacy**: `original_code/` 는 보존, 새 패키지는 `src/counting/` 로 분리

## 1. 배경 및 목표

### 배경
기존 `original_code/` 는 추론 전용 파이프라인이며, 다음과 같은 한계가 있다.

- 경로가 하드코딩되어 환경 이식성이 낮음
- 전역 변수 기반 모델 싱글톤, 단일 `main.py` 에 모든 책임 집중
- 예외 처리 누락 (`def_total` 미정의 가능 경로 존재)
- 학습 코드 부재 — 가중치 파일만 존재하고 재현·개선 경로가 없음
- 모듈/테스트 분리가 불명확

### 목표
- 라이브러리 + 얇은 CLI 구조로 재설계해 **노트북/서버/Mac** 모두에서 동일하게 동작
- **학습 코드 포함**: PseCo 파인튜닝(백본 동결 + 헤드만) + Classifier 학습 (옵션)
- **데이터 진단 도구**로 기존 농장 이미지의 학습 적합성을 수치로 판단
- 설정은 YAML + Pydantic 으로 검증, 실험은 TensorBoard 로 추적
- 기존 다른 프로젝트와 **conda 전용 env** 로 격리

### 비목표 (YAGNI)
- HTTP API 서버화 (필요해지면 CLI 위에 얇게 추가)
- 폴링 루프 서비스 (운영 자동화는 cron/systemd 에 위임)
- Deblur / SR 재학습 (사전학습 그대로 사용)
- 병충해 검출 (다른 폴더에서 진행)

### 전제 조건
- `external/` 서드파티 원본 소스가 존재해야 한다:
  - `external/DeblurGANv2/`
  - `external/PseCo/` (SAM 포함)
  - `external/Super_Resolution_Neural_Operator/`
  - `external/classification/`
  원본 `original_code/ai_modules/main.py` 가 `sys.path` 에 추가하는 경로와 동일. 현재 스냅샷에는 포함되지 않았으므로 구현 1단계에서 복원 여부 확인.
- 사전학습 가중치는 `models/` 아래 기존 구조 유지:
  - `models/PseCo/point_decoder_vith.pth`, `models/PseCo/MLP_small_box_w1_zeroshot.tar`, `models/PseCo/sam_vit_h.pth` (추가 필요)
  - `models/classification/classification_model.pt`
  - Deblur / SR 가중치는 기존 위치
- **데이터 진단 도구**의 밀도 추정은 SAM 가중치를 사용하므로 PseCo 의존성과 공유.

## 2. 합의된 설계 결정

| 항목 | 결정 |
|---|---|
| 실행 형태 | 라이브러리 + 얇은 CLI (`counting train/infer/batch/diagnose`) |
| 학습 범위 | PseCo 헤드 파인튜닝 + Classifier. Deblur/SR 은 사전학습 사용 |
| 학습 전략 | **백본 동결 + SAM 임베딩 캐싱** 기본. 풀 파인튜닝은 플래그로 지원하되 Mac 경고 |
| 데이터 | `Dataset` 인터페이스 (FSC-147/COCO/custom), 데이터 진단 도구 포함 |
| 실행 환경 | Mac (MPS) 1차, GPU 서버 2차. `device=auto|cpu|mps|cuda` |
| 설정 관리 | YAML + Pydantic (v2). CLI dot-path 오버라이드 (`--set train.lr=5e-5`) |
| 실험 추적 | TensorBoard (로컬, 외부 계정 불필요) |
| 결과 출력 | JSON 기본 + CSV 옵션. 라이브러리는 dataclass 반환 |
| 패키징 | `pyproject.toml` + `pip install -e .`. 엔트리포인트 `counting` |
| 테스트 | pytest. 단위 테스트 중심 + 스모크 통합 (`slow` 마커) |
| 환경 격리 | **conda 전용 env (`counting-env`)** — 기존 프로젝트와 충돌 방지 |

## 3. 폴더 구조

```
counting/
├── original_code/                  # 기존 (보존)
├── src/counting/                   # 새 패키지 (import: counting)
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py               # Pydantic v2 스키마
│   │   └── loader.py               # YAML 로딩 + dot-path 오버라이드
│   ├── data/
│   │   ├── base.py                 # Dataset 인터페이스
│   │   ├── formats/                # fsc147.py / coco.py / custom.py / imagefolder.py
│   │   ├── diagnostics.py          # 해상도/블러/노출/밀도 진단
│   │   └── cache.py                # SAM 임베딩 캐시 (fp16 npz shard)
│   ├── models/
│   │   ├── base.py                 # Stage 프로토콜
│   │   ├── deblur.py               # DeblurGANv2 래퍼 (추론)
│   │   ├── pseco/
│   │   │   ├── inference.py
│   │   │   ├── trainer.py          # 헤드 파인튜닝
│   │   │   └── feature_cache.py    # 캐시 생성·로딩 헬퍼
│   │   ├── sr.py                   # SR-NO 래퍼 (추론)
│   │   └── classification/
│   │       ├── inference.py
│   │       └── trainer.py
│   ├── pipeline.py                 # Stage 조합, device/에러 격리
│   ├── training/
│   │   ├── runner.py               # 공통 학습 루프 + TB 훅
│   │   └── callbacks.py            # 체크포인트/조기종료
│   ├── io/
│   │   ├── results.py              # CountingResult dataclass
│   │   └── serialize.py            # JSON/CSV 직렬화
│   └── cli.py                      # Typer 엔트리포인트
├── configs/
│   ├── pipeline/default.yaml
│   ├── train/pseco_head.yaml
│   └── train/classifier.yaml
├── external/                       # 서드파티 소스 (원본과 동일 위치)
├── scripts/                        # 1회성 유틸
├── tests/
│   ├── unit/
│   └── integration/                # slow 마커
├── docs/superpowers/specs/
├── environment.yml                 # conda env (Mac/CPU/MPS 기본)
├── environment-cuda.yml            # GPU 서버용 (선택)
├── pyproject.toml
└── README.md
```

### 핵심 구조 원칙
- **`external/` 유지**: 서드파티 모델 원본 소스(DeblurGANv2/PseCo/SR-NO/classification)는 수정 금지. `src/counting/models/*` 는 얇은 래퍼 (`Stage` 구현) 역할만.
- **Stage 프로토콜**: `prepare(cfg) → process(input) → cleanup()` 통일. 새 단계 추가·순서 변경 쉬움.
- **디바이스 추상화**: 설정 `device` 하나로 모든 Stage 제어. Stage 는 `to(device)` 만 호출.

## 4. 파이프라인 및 데이터 흐름

### 4.1 추론 파이프라인

```
  Image (path/PIL/np)
         │
         ▼
  ┌──────────────┐
  │  Deblur      │  DeblurGANv2 (사전학습, 추론 전용)
  └──────┬───────┘
         │ RGB np.ndarray
         ▼
  ┌──────────────┐
  │  PseCo       │  SAM(ViT-H) + point decoder + MLP
  │  - embed     │
  │  - decode    │  → points + bboxes + count
  └──────┬───────┘
         │ count(int), crops[PIL]
         ▼
  [optional SR per crop]  ──►  [optional Classifier]
                                (bag / not-bag)
         │
         ▼
  CountingResult
  { raw_count, verified_count,
    points[], boxes[], crops_meta[],
    timings_ms, device, config_hash, error? }
```

- **옵션 단계는 `enabled: bool`** 로 on/off. 개발 중 PseCo 단독 성능을 본 뒤 SR/Classifier 활성 여부 판단.
- **실패 격리**: 한 단계가 실패해도 앞 단계 결과는 살리고 `error` 필드 기록 → 배치 처리 중단 없음.
- **타이밍 계측**: 각 Stage 경과 시간을 `timings_ms` 에 기록해 병목 진단.

### 4.2 PseCo 헤드 학습 파이프라인

```
  Dataset (images + point labels)
         │
         ▼
  [Step A] 임베딩 사전계산 (1회성, 증분 가능)
  ├─ SAM(ViT-H) forward, no_grad
  └─ fp16 npz shard → feature_cache/
         │
         ▼
  [Step B] 헤드 학습 (반복)
  ├─ DataLoader: (cached_embedding, targets)
  ├─ point_decoder + MLP 업데이트 (백본 freeze)
  ├─ Loss: point_loss + 0.1 · count_L1 (PseCo 원 가중치)
  ├─ TensorBoard: loss / val MAE / 샘플 시각화
  └─ ckpt: best(val MAE) + last + epoch_N
```

### 4.3 Classifier 학습 파이프라인 (옵션)

- ResNet18 (timm) ImageNet 사전학습 → head 교체 (512 → 1, sigmoid)
- 입력: PseCo 크롭 (혹은 SR 결과) → 224×224
- Augmentation: flip / colorjitter / random resized crop (0.8~1.0)
- Loss: BCEWithLogits (`pos_weight` 로 클래스 불균형 보정)
- Eval: accuracy / precision / recall / F1 / AUROC

### 4.4 데이터 진단 도구

별도 CLI `counting diagnose <dir>` — 학습 전 데이터 적합성 판단.

- **해상도 분포**: min/max/median, 이상치 강조
- **블러 지표**: Laplacian variance, 임계 이하 이미지 비율
- **노출/대비**: 히스토그램, 과노출/저노출 비율
- **객체 밀도 추정**: SAM 프롬프트 기반 대략 카운트 → 이미지당 대상 수 분포
- 출력: `diagnostics_report.html` (썸네일 + 플롯) + `diagnostics.json`

## 5. 임베딩 캐시 설계

### 5.1 동기
SAM ViT-H forward 는 Mac MPS 에서 2~5 s/이미지. 매 에폭 반복 시 수십 시간 소요. 백본이 동결되면 임베딩은 불변이므로 디스크에 1회 저장.

### 5.2 레이아웃

```
feature_cache/<dataset_id>/
├── meta.json        # preprocess_hash, sam_ckpt_hash, sam_arch, dtype, image_size
├── index.parquet    # image_id → shard_id, row_offset, label_ref
└── shards/
    ├── 00000.npz    # (B, 256, 64, 64) float16
    ├── 00001.npz
    └── ...
```

### 5.3 규칙
- **fp16 저장**: 정밀도 손실 무시 가능, 용량 절반 (1장 ≈ 2 MB → 1만 장 ≈ 20 GB)
- **Shard 단위 ~256 장**: 랜덤 접근 효율 + 파일 수 억제
- **해시 기반 무효화**: `meta.json` 의 `preprocess_hash + sam_ckpt_hash` 가 현재와 다르면 오류 + 재생성 명령 안내
- **증분 재생성**: `rebuild=missing` 모드로 신규 이미지만 추가
- **용량 사전 점검**: 20 GB 초과 예상 시 사용자 확인 프롬프트
- **증강 대응**: 캐시 후 증강 불가 → 두 옵션
  - (a) **증강 없음** (기본, 빠름, 데이터 많을 때)
  - (b) **`augment_variants=N`**: 각 이미지 N 변형본을 모두 캐시 (용량 N 배)

소규모(≤ 500 장) 데이터는 (b) 권장을 진단/리포트에 명시.

## 6. 하이퍼파라미터 기본값

### 6.1 PseCo 헤드

| 항목 | 기본값 |
|---|---|
| optimizer | AdamW (wd = 1e-4) |
| lr (head) | 1e-4, linear warmup 500 steps |
| scheduler | cosine, min_lr = 1e-6 |
| batch_size | 8 |
| epochs | 30 |
| loss | `point_loss + 0.1 · count_L1` |
| early stopping | patience = 5, metric = val_mae, mode = min |
| seed | 42 |

### 6.2 Classifier

| 항목 | 기본값 |
|---|---|
| backbone | ResNet18 (ImageNet pretrained) |
| head | Linear(512 → 1) + sigmoid |
| optimizer | AdamW, lr = 3e-4, wd = 1e-4 |
| scheduler | cosine, 20 epochs |
| batch_size | 32 |
| augmentation | RandomHorizontalFlip, ColorJitter(0.2), RandomResizedCrop(224, scale = 0.8~1.0) |
| loss | BCEWithLogitsLoss (`pos_weight` 자동 계산) |

### 6.3 재현성
- `seed` 하나로 torch/numpy/random/DataLoader worker 전부 고정
- 각 run 에 `config.yaml` 스냅샷 + `env.txt` (pip freeze / conda list) 저장
- TensorBoard `hparams/` 탭에 하이퍼파라미터 전체 기록

## 7. 설정 스키마 (YAML + Pydantic)

### 7.1 파이프라인 설정 (`configs/pipeline/default.yaml`)

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
      max_crop_side: 500         # 이 이상 크롭은 스킵
    classifier:
      enabled: false
      checkpoint: models/classification/classification_model.pt
      threshold: 0.5

io:
  output_format: json            # json | csv | both
  save_visualizations: false
```

### 7.2 학습 설정 — PseCo 헤드 (`configs/train/pseco_head.yaml`)

```yaml
run_name: pseco_head_v1
device: auto
seed: 42
output_dir: ./runs

data:
  format: fsc147                 # fsc147 | coco | custom
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
  loss_weights: {point: 1.0, count_l1: 0.1}
  early_stopping: {patience: 5, metric: val_mae, mode: min}

logging:
  tensorboard: true
  log_every_n_steps: 20
  save_every_n_epochs: 1
```

### 7.3 학습 설정 — Classifier (`configs/train/classifier.yaml`)

```yaml
run_name: clf_bag_v1
device: auto
seed: 42

data:
  format: imagefolder
  root: ./datasets/classification
  train_split: train
  val_split: val
  image_size: 224

model:
  backbone: resnet18
  pretrained: true

train:
  batch_size: 32
  epochs: 20
  lr: 3.0e-4
  weight_decay: 1.0e-4
  scheduler: cosine
  augmentation:
    horizontal_flip: true
    color_jitter: 0.2
    random_resized_crop: [0.8, 1.0]

logging:
  tensorboard: true
```

### 7.4 오버라이드 규칙
- CLI 에서 `--set train.lr=5e-5` 형태의 dot-path 오버라이드 지원 (간단 파서, Hydra 없이 구현).
- 오버라이드 결과도 Pydantic 검증을 통과해야 함.

## 8. CLI 인터페이스 (Typer)

```bash
# 데이터 진단
counting diagnose ./my_images --report-dir ./reports/my_images

# 임베딩 사전계산 (선택, train 이 자동 호출도 가능)
counting cache-embeddings --config configs/train/pseco_head.yaml

# 학습
counting train pseco-head --config configs/train/pseco_head.yaml
counting train classifier --config configs/train/classifier.yaml

# 재개
counting train pseco-head --resume runs/pseco_head_v1/checkpoints/last.ckpt

# 추론 (단일)
counting infer ./image.jpg \
  --config configs/pipeline/default.yaml \
  --set pipeline.stages.classifier.enabled=true

# 배치 처리 (기존 main.py 대체)
counting batch ./input_dir --config configs/pipeline/default.yaml \
  --output ./results --format json

# 환경/설정 확인
counting info
counting validate-config <path>
```

### 라이브러리 사용

```python
from counting import Pipeline, load_config

cfg = load_config("configs/pipeline/default.yaml")
pipeline = Pipeline(cfg)
result = pipeline.run("image.jpg")
print(result.verified_count, result.raw_count)
```

### run 레이아웃

```
runs/<run_name>/
├── config.yaml          # 스냅샷
├── env.txt              # pip freeze + conda list
├── tensorboard/         # TB 이벤트
├── checkpoints/
│   ├── best.ckpt
│   └── last.ckpt
└── logs/train.log
```

### 설정 검증 시점
- **시작 즉시**: 타입/필수 필드, 경로/체크포인트 존재
- **캐시 사용 시**: 해시 비교 → 불일치면 명확한 에러 + 재생성 안내
- **디바이스**: `device=cuda` 인데 CUDA 없으면 즉시 실패. `auto` 면 `cuda → mps → cpu` 순

## 9. 에러 처리 방침

- **학습**: 예외 발생 시 `last.ckpt` 저장 후 종료. `--resume` 으로 이어서 학습 가능.
- **추론**: 이미지 단위 실패는 결과에 `error` 기록하고 건너뜀 (기존 동작 유지). 모델 로딩 실패 등 치명 오류는 즉시 중단.
- **설정 검증 실패**: Pydantic 에러 메시지를 사람이 읽을 수 있도록 래핑해 출력.
- **Mac MPS op 미지원**: `PYTORCH_ENABLE_MPS_FALLBACK=1` 기본 설정, fallback 발생 시 경고 로그.

## 10. 환경 (conda)

전용 env `counting-env` 로 기존 다른 프로젝트와 격리.

### `environment.yml` (Mac / CPU / MPS 기본)

```yaml
name: counting-env
channels: [pytorch, conda-forge]
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
  - jupyterlab
  - pytest
  - pytest-cov
  - pip:
    - typer
    - pydantic>=2.0
    - albumentations
    - timm
    - rich
```

### `environment-cuda.yml` (GPU 서버)

```yaml
name: counting-env
channels: [pytorch, nvidia, conda-forge]
dependencies:
  - python=3.11
  - pytorch=2.3.*
  - torchvision=0.18.*
  - pytorch-cuda=12.1
  # ... 나머지는 environment.yml 과 동일
```

설치:

```bash
conda env create -f environment.yml      # 또는 environment-cuda.yml
conda activate counting-env
pip install -e .
```

## 11. 테스트 전략

| 대상 | 형태 | 목적 |
|---|---|---|
| 설정 로더 | 단위 | 유효/무효 YAML, 오버라이드, 타입 검증, 경로 체크 |
| 데이터 로더 | 단위 | 포맷(fsc147/coco/custom/imagefolder) 어댑터가 동일 Dataset 인터페이스 반환 |
| 임베딩 캐시 | 단위 | 생성/로드/해시 불일치 감지/증분 재생성 |
| Stage 프로토콜 | 단위 | 더미 Stage 로 파이프라인 조립, 에러 격리 동작 |
| Pipeline 스모크 | 통합 (slow) | 더미 이미지 1장으로 전체 추론 완주 |
| Trainer 스모크 | 통합 (slow) | 2 epoch, 이미지 4장으로 학습 루프 완주 + ckpt 저장 |
| 진단 도구 | 단위 | 더미 이미지로 해상도/블러 지표 산출 |
| I/O | 단위 | JSON/CSV 직렬화 라운드트립 |

- 기본: `pytest -m "not slow"` (수 초)
- 전체: `pytest` (수 분)
- 가중치 필요 테스트는 fixture 로 자동 skip (가중치 없으면 통과)
- 커버리지 목표: 핵심 모듈(config, data, cache, pipeline) ≥ 80%. 모델 래퍼는 스모크만.

## 12. 개발 순서 (점진적 수직 슬라이스)

한 번에 모두 만들지 않고 작동하는 얇은 슬라이스부터 쌓는다.

1. **스캐폴드** — conda env, `pyproject.toml`, 패키지 구조, `counting info`
2. **설정** — Pydantic 스키마 + YAML 로더 + 오버라이드 + 테스트
3. **데이터 진단** — `counting diagnose`. 기존 농장 이미지 적합성부터 판단
4. **추론 파이프라인 v1** — Stage 인터페이스 + Deblur/PseCo 래퍼 + `infer`/`batch`
5. **임베딩 캐시** — 생성/로드/해시 검증 + `counting cache-embeddings`
6. **PseCo 헤드 학습** — trainer + TensorBoard + 체크포인트/재개
7. **결과 I/O** — JSON/CSV 출력, 결과 스키마 확정
8. **Classifier** — 학습 + 추론 통합 (PseCo 단독으로 충분하면 중단)
9. **SR 통합** — 필요 시 활성화
10. **정돈** — README, 사용 예시, 리포트 샘플

1~4 까지 오면 원본의 추론 부분과 기능적으로 동등해지고, 5~6 에서 학습이 붙는다.

## 13. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| PseCo 원 구현 구조에 깊게 의존 | `external/PseCo` 는 수정 금지, 얇은 래퍼만. 필요 시 원 구현 함수를 import 해 재활용 |
| Mac MPS 연산 호환성 이슈 | `PYTORCH_ENABLE_MPS_FALLBACK=1` 기본, 경고 로그, 느려지면 안내 |
| 임베딩 캐시 용량 폭주 | 용량 사전 계산, 20 GB 초과 예상 시 확인 프롬프트 |
| 기존 가중치와 신 코드 버전 불일치 | 체크포인트에 설정 스냅샷 포함, 로드 시 구조 검증 |
| 학습 데이터 부적합 | 진단 도구를 3 단계에 배치해 본격 학습 전 발견 |
| Mac 에서 풀 파인튜닝 불가 | 기본값을 백본 동결로, 풀 파인튜닝 플래그 시 Mac 경고. GPU 서버 이관 지침을 README 에 명시 |

## 14. 향후 확장 가능성 (현재는 비목표)

현 설계는 다음 확장을 막지 않는다.

- **HTTP API** — `counting.Pipeline` 을 FastAPI 로 감싸면 끝
- **폴링 서비스** — `counting batch` 를 cron/systemd 로 주기 실행
- **웹 대시보드** — TensorBoard + 진단 리포트 HTML 로 충분, 추가 UI 는 필요 시
- **병충해 연동** — 별도 폴더의 결과와 합치는 집계 스크립트를 `scripts/` 에 추가

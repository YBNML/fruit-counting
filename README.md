# counting

과일 카운팅 파이프라인 (재설계 버전).

## 설치 (Mac, conda)

```bash
conda env create -f environment.yml
conda activate counting-env
pip install -e .
```

GPU 서버:

```bash
conda env create -f environment-cuda.yml
conda activate counting-env
pip install -e .
```

## 환경 점검

```bash
counting info
```

## 사용

### 설정 검증

```bash
counting validate-config configs/pipeline/default.yaml
```

### 데이터 진단

```bash
counting diagnose ./my_images --report-dir ./reports/diagnostics
```

### 단일 추론

```bash
counting infer ./image.jpg --config configs/pipeline/default.yaml \
  --set device=cpu
```

### 배치 추론

```bash
counting batch ./input_dir --config configs/pipeline/default.yaml \
  --output ./runs/last_batch --format json
```

### 라이브러리 사용

```python
from counting import Pipeline, build_pipeline, load_config
from counting.utils.image import read_image_rgb

cfg = load_config("configs/pipeline/default.yaml")
pipe = build_pipeline(cfg)
result = pipe.run_numpy(read_image_rgb("image.jpg"), image_path="image.jpg")
print(result.raw_count, result.verified_count)
```

### 합성 렌더 채점 (orchard_synth)

합성 렌더(orchard_synth 출력)의 정답 자세·깊이·박스를 기준으로 과수원 파이프라인(오라클 검출 → 추적 → 3D 라프팅 → 다중 시야 중복 제거 → 나무 배정)을 채점하는 별도 도구다. 아래는 명령과 규칙만 적는다.

```bash
counting eval truth <bundle...> [--pitches 30,50] [--sides A,B] [--variant count1]
counting eval ladder --config configs/eval/apple_val.yaml [--ladder configs/eval/sweeps/ladder.yaml] [--rungs L1,L3] [--with-2x2] [--subsets]
counting eval sweep --config configs/eval/apple_val.yaml --grid configs/eval/sweeps/val/<grid>.yaml [--ladder configs/eval/sweeps/ladder.yaml]
```

- `eval truth`는 하나 이상의 묶음에서 패스/나무 정답 표를 찍고 요약·기하·분할 확인을 돈다.
- `eval ladder`는 오라클 사다리(와, 학습 검출기의 캐시 검출 칸을 추가한 `--ladder configs/eval/sweeps/ladder_l5.yaml`)를 돌려 실험 폴더를 쓴다.
- `eval sweep`은 격자 파일의 점들을(사다리 칸과 곱해서도) 돌려 `sweep.csv`가 들어간 같은 모양의 실험 폴더를 쓴다.

학습 검출기는 다음 두 명령으로 만들고 캐시한다. 캐시는 가중치마다 한 번만 쓴다(묶음의 캐시 파일이 이미 있으면 `detect`는 아무것도 쓰지 않고 멈춘다).

```bash
counting train detector --config configs/train/<config>.yaml
counting detect --config configs/detect/<config>.yaml --weights <weights path> --run <bundle> [--run <bundle> ...]
```

- `detect`가 쓰는 캐시 폴더는 `<output_dir>/<가중치 sha256 앞 16자>/<run_id>.npz`다.

```bash
counting eval detmetrics --config configs/eval/apple_val.yaml --cache runs/detect/<가중치 sha256 앞 16자> [--nms-iou 0.5] [--edge-snap-px 2]
```

- 캐시를 읽을 때의 후처리(클래스 무관 NMS, 가장자리 맞춤, 최소 상자 크기)는 설정의 `detector.post`를 따른다. NMS와 가장자리 맞춤은 위 두 옵션으로 그때만 바꿀 수 있고, 최소 상자 크기는 `--set detector.post.min_short_px=<px>`로 바꾼다.

학습 검출기의 캐시 검출을 사다리에 얹어 돌리려면:

```bash
counting eval ladder --config configs/eval/apple_val.yaml --ladder configs/eval/sweeps/ladder_l5.yaml --set detector.cache=runs/detect/<가중치 sha256 앞 16자>
```

test 분할은 `configs/eval/frozen/`에 커밋된 설정으로만 돈다. `--set`·`--run`을 받지 않고, 작업 트리가 깨끗해야 하며, 캐시를 쓰는 칸은 고정 설정에 캐시 digest가 박혀 있어야 하고 그 digest가 `counting detect`의 기록과 맞아야 한다(캐시 칸의 검출기·추적기 설정도 고정 설정 그대로다). 한 번 돌 때마다 `runs/eval/test_log.csv`에 한 줄이 남는다.

결과는 `<output_dir>/<run_name>/<experiment_id>/` 폴더에 `results.json`, `config.yaml`, `record.json`, `report.md`, `per_pass.csv`, `per_tree.csv`(사다리나 단일 실행)·`sweep.csv`(격자), 그리고 학습 검출기 칸을 채점한 사다리라면 `budget.csv`가 함께 생긴다. 같은 실험 id를 다시 돌리면 `results.json`이 바이트 단위로 같아야 하고, 다르면 거부되며 아무 파일도 덮어쓰지 않는다.

합성 렌더·정답 자세·보정 전 값이다. 실제 과원 정확도가 아니다. 검출기는 COCO 사전학습 가중치에서 시작해 연구·비상업 용도로만 쓴다.

## 진행 상황

- [x] Foundation (Plan 1): 스캐폴드, 설정, 데이터 진단, 추론 파이프라인
- [x] PseCo 학습 인프라 (Plan 2): SAM 피처 캐시 + ROIHeadMLP 파인튜닝 루프 (placeholder objective)
- [x] PseCo 학습 목표 수정 (Plan 3): CLIP 텍스트 피처 + pos/neg 레이블 + 좌표 스케일 수정. FSC-147 val/mae 47.62로 placeholder baseline은 돌파했으나 **오소차드 4장 GT 검증에서 MAE 400+, prompt-agnostic으로 수렴해 사용 불가**로 판정. 체크포인트 `runs/plan3_k32/checkpoints/best.ckpt` (원격) 보존만, 기본 설정에는 미사용.
- [x] **Plan 4**: upstream MLP 기반 실사용 가능 추론 파이프라인 정비 (`docs/superpowers/specs/2026-04-22-plan4-direction.md`)
  - P4.1 방향 문서화 ✅
  - P4.2 `PseCoStage` 재작성 (SAM → PointDecoder → SAM 박스 예측 → ROIHeadMLP → NMS → threshold, `counting infer` 실제 동작) ✅
  - P4.3 프롬프트 + 하이퍼파라미터 grid search ✅
- 오소차드 4장 GT 검증 요약 (최적 설정 `anchor_size=24 score=0.05 nms=0.3`):

  | image | GT | pred | \|err\| |
  |---|---|---|---|
  | apple_1 | 48 | 28 | 20 |
  | apple_2 | 46 | 30 | 16 |
  | pear_1  | 27 | 30 | 3 |
  | pear_2  | 54 | 62 | 8 |

  **Mean MAE 11.75** — 파인튜닝 없이 upstream MLP + 프롬프트 + SAM anchor 튜닝으로 phone-taken 과수원 이미지에서 실용 수준. 사과가 여전히 -35% 언더카운트인데 SAM+PointDecoder의 검출 한계. 남은 개선은 자체 도메인 라벨링 후 파인튜닝이 필요.

## 전제 조건 (external/)

서드파티 원본 소스가 아래 경로에 있어야 `deblur`/`pseco`/`sr`/`classifier` 스테이지를 활성화할 수 있습니다:

- `external/DeblurGANv2/`
- `external/PseCo/`
- `external/Super_Resolution_Neural_Operator/`
- `external/classification/`

비활성 상태로는 설정 검증·진단·I/O 만 사용 가능합니다.

설계 문서: `docs/superpowers/specs/2026-04-19-fruit-counting-redesign-design.md`
플랜 문서: `docs/superpowers/plans/2026-04-19-fruit-counting-foundation.md`

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

- `models/PseCo/sam_vit_h.pth` — SAM ViT-H (2.4 GB)
- `models/PseCo/point_decoder_vith.pth`, `models/PseCo/MLP_small_box_w1_zeroshot.tar` — PseCo 공식 가중치
- `datasets/fsc147/` — FSC-147 이미지 + 주석 JSON

`python -c "from counting.data.download import print_sam_vit_h_instructions; print_sam_vit_h_instructions('models/PseCo/sam_vit_h.pth')"` 로 SAM 다운로드 명령을 확인할 수 있습니다.

## 라이선스

이 저장소의 코드는 Apache License 2.0을 따릅니다(`LICENSE`). `external/PseCo` 서브모듈과 따로 내려받는 모델 가중치(SAM, PseCo, COCO 사전 학습 가중치 등)는 각자의 라이선스와 이용 조건을 따릅니다.

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

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
플랜 문서: `docs/superpowers/plans/2026-04-19-fruit-counting-foundation.md`

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

## 진행 상황

- [x] Foundation (Plan 1): 스캐폴드, 설정, 데이터 진단, 추론 파이프라인
- [ ] PseCo 학습 (Plan 2)
- [ ] Classifier 학습 · SR 통합 · 정돈 (Plan 3)

설계: `docs/superpowers/specs/2026-04-19-fruit-counting-redesign-design.md`

# wafer-defect-suite

반도체 웨이퍼 결함을 **팹 검사 흐름(공정 → 웨이퍼 패턴 → 결함 위치)** 3단계로 분석하는 end-to-end 포트폴리오.
세 가지 모달리티(테이블 · 이미지맵 · 검출)에 각각 다른 ML 기법을 적용하고, **정직한 평가(leak-free · per-class · 불균형)** 로 검증한다.

> **핵심 서사**: 합성 데이터만으로 학습하면 실제에서 무너진다(0.99→0.36). 이를 정량·인과로 규명하고,
> **실데이터 학습 + 구조적 개선(SE-ResNet 앙상블)으로 0.935까지** 끌어올린다. 단순 튜닝·자기지도·전이는 효과 없음을 정직하게 보고한다.

---

## 3-스테이지 결과 요약

| 스테이지 | 질문 | 데이터 | 기법 | 핵심 결과 |
|---|---|---|---|---|
| **1. 공정** | 왜 생기나 | Meruva (실, 결함 7/5000) | ML vs DL 이상탐지 | **LOF(ML) PR-AUC 0.341 > AutoEncoder(DL) 0.211**, recall@100=1.0 |
| **2. 웨이퍼 패턴** ★ | 어떤 패턴 | MixedWM38(합성)+WM-811K(실) | 멀티라벨 분류·전이·SE-ResNet | 합성 0.985 → **전이 0.364** → SE-ResNet 6앙상블 **0.935**(macro-F1) |
| **3. 결함 위치** | 어디 | WM-811K(실) / ELLIMAC(합성) | Grad-CAM(메인) + YOLO(부록) | 실데이터 위치탐지 / ELLIMAC bestV2 mAP@0.5 **0.739** |

---

## Stage 1 — 공정 센서 이상탐지 (ML vs DL)
결함 7/5000(0.14%) → 지도분류 불가 → 정상만 학습하는 이상탐지. leak-free(정상 80% 학습).
양성 7건이라 단일 split은 노이즈가 커서 **30회 반복 CV(mean±std)** 로 평가.

| 모델 | PR-AUC (30-seed) | ROC-AUC | recall@100 |
|---|---|---|---|
| **Mahalanobis (ML)** | **0.295 ± 0.028** | 0.954 | 0.81 |
| OneClassSVM (ML) | 0.294 ± 0.030 | 0.958 | 0.89 |
| LOF (ML) | 0.294 ± 0.039 | 0.962 | **0.94** |
| AutoEncoder (DL) | 0.215 ± **0.095** | 0.887 | 0.67 |
| IsolationForest (ML) | 0.119 ± 0.055 | 0.950 | 0.84 |

→ **고전 ML 3종 통계적 동률(~0.295), DL은 더 낮고 분산 큼(±0.095, 불안정).** 소표본·저차원 테이블엔 DL 불필요.

**★ 피처 엔지니어링 (데이터셋 메커니즘 = "센서 비정상 조합"에 정합):**

| 피처셋 | 최고 모델 | PR-AUC |
|---|---|---|
| base (6) | kNN | 0.308 |
| **+선별 교호작용·전력·step (16)** | **Mahalanobis** | **0.446** |
| 전수 쌍조합 (36~41) | (붕괴) | 0.07~0.22 |

→ 결함=다변량 조합이라 **공분산 모델(Mahalanobis/GMM)이 최적, IForest(축정렬) 최악**. **선별 교호작용**으로 0.30→**0.45**, 단 **무차별 전수조합은 차원의 저주로 역효과**(도메인 선별>무차별). rank-앙상블 0.304.
위험신호: pressure↓(z −2.31)·etch_rate↑·temp↑.
`run.py`(단일) · `rigor.py`(30-seed CV·앙상블) · `sweep.py`(피처×모델 스윕)

## Stage 2 — 웨이퍼 패턴 (플래그십)
| 실험 | macro-F1 | mAP | 메시지 |
|---|---|---|---|
| 합성(MixedWM38) | 0.985 | 0.999 | 합성은 쉬움 |
| 합성→실제 전이 | **0.364** | 0.417 | **도메인 갭 규명**(정상 오탐 0.957) |
| A 증강 진단 | 0.311 | 0.413 | "노이즈 원인설" 기각(증강 무효) |
| 실데이터 lot-split (baseline) | 0.859 | 0.930 | 단일 |
| 실데이터 + 자기지도 | 0.869 | 0.928 | SSL +0.01 (효과 미미) |
| CNN 강화(증강+width64) +보정 | 0.902 ± 0.008 | 0.955 | 3-seed |
| SE-ResNet 단일 (깊이+SE attention) | 0.912 → 보정 0.928 | 0.974 | ★★ 돌파 |
| **SE-ResNet 6-앙상블(w48×3+w64×3)+보정** | **0.935** | — | ★ 최종 best |

- 핵심: **구조적 변화(깊이+SE attention)+앙상블** 이 결정적(0.86→**0.935**). Loc 0.72→0.86, Scratch 0.78→0.87, Near-full→1.0
- per-class(6앙상블+보정): Edge-Ring 0.994 / NF 1.0 / Center 0.98 / Edge-Loc·Donut 0.93 / Random 0.92 / Scratch 0.87 / Loc 0.86
- (negative, 정직) 입력 패딩·해상도↑·class-balanced·**SimCLR·MAE·DINOv2·TTA** 모두 효과 없음 → "구조 > 단순튜닝" 입증
- 평가: **lot 그룹 분할**(누수 차단), 다중 seed, 임계 val-only
- `train_real --augment --width 64 --loss asl` · `rigor`(임계보정·혼동) · `transfer_eval` · `ssl_pretrain`

## Stage 3 — 결함 위치
- **B(메인, 실데이터)**: Stage2 실모델 **Grad-CAM** → WM-811K 실제 맵의 결함 위치 히트맵 (합성 무관)
  `python -m src.stage3_localization.gradcam`
- **A(부록, 합성)**: ELLIMAC 폴리곤→bbox 정제 + cls6 제거 → bestV2 **mAP@0.5 0.739**
  `python -m src.stage3_detection.benchmark`

---

## 데모 웹 (`web/`)
React(Vite) 단일앱. 5화면 모두 실제 기능:
Dashboard · Stage1(테이블·관리도·파라미터 추천·ML vs DL) · Stage2(갤러리·판정·Grad-CAM 히트맵·합성vs실데이터 토글) · Stage3(Grad-CAM 위치탐지 + ELLIMAC 검출) · Experiments(추이·per-class·혼동행렬).
```bash
cd web && npm install && npm run dev   # http://localhost:5173
```
**LIVE 추론(선택)** — FastAPI 백엔드 연결 시 실제 모델 추론:
```bash
python -m backend.prep_samples            # 최초 1회
uvicorn backend.main:app --port 8000      # 백엔드
```
웹이 백엔드를 감지하면 상단 배지가 **LIVE**로 바뀌고, Stage1 슬라이더→실시간 LOF 점수,
Stage2 "⚡LIVE 추론"→실제 WaferCNN 예측(실모델 vs 합성모델 전이 실패를 실시간으로). 백엔드 없으면 정적 데모로 폴백.

## 구조
```
src/stage1_process/   공정 이상탐지(ML vs DL)
src/stage2_wafermap/  데이터셋·모델·학습(합성/실데이터)·전이·자기지도·Grad-CAM은 stage3로
src/stage3_localization/  Grad-CAM 위치탐지(실)   src/stage3_detection/  ELLIMAC YOLO(합성)
src/common/           metrics(멀티라벨)·seed
web/                  React 데모 콘솔
docs/                 PRD · EDA · 웹 디자인 프롬프트
data/ , experiments/  데이터·산출물 (git 제외)
```

## 데이터 & 라이선스
데이터는 레포에 미포함(`.gitignore`). 출처에서 받아 `data/` 에 둔다.
- WM-811K (MIR Lab / Wu et al. 2015) · MixedWM38 (Kaggle/Wang et al.) · ELLIMAC (Kaggle/Roboflow) · Meruva (Kaggle). 연구·포트폴리오 목적.

## 평가 원칙
leak-free(lot 그룹 분할, seed 고정, 임계 val-only) · per-class·불균형(macro-F1·mAP·PR-AUC) · **정직성**(전이 실패·negative result 그대로 보고).

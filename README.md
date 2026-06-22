# wafer-defect-suite

반도체 웨이퍼 결함을 **팹 검사 흐름(공정 → 웨이퍼 패턴 → 결함 위치)** 3단계로 분석하는 end-to-end 포트폴리오.
세 가지 모달리티(테이블 · 이미지맵 · 검출)에 각각 다른 ML 기법을 적용하고, **정직한 평가(leak-free · per-class · 불균형)** 로 검증한다.

> **핵심 서사**: 합성 데이터만으로 학습하면 실제에서 무너진다(0.99→0.36). 이를 정량·인과로 규명하고,
> **실데이터 lot-split 학습(0.86)** 으로 회복한다. 무지성으로 증강·자기지도를 붙여도 안 좋아진다는 것도 데이터로 보였다.

---

## 3-스테이지 결과 요약

| 스테이지 | 질문 | 데이터 | 기법 | 핵심 결과 |
|---|---|---|---|---|
| **1. 공정** | 왜 생기나 | Meruva (실, 결함 7/5000) | ML vs DL 이상탐지 | **LOF(ML) PR-AUC 0.341 > AutoEncoder(DL) 0.211**, recall@100=1.0 |
| **2. 웨이퍼 패턴** ★ | 어떤 패턴 | MixedWM38(합성)+WM-811K(실) | 멀티라벨 분류·전이·자기지도 | 합성 0.985 → **전이 0.364** → 실데이터 **0.859**(macro-F1) |
| **3. 결함 위치** | 어디 | WM-811K(실) / ELLIMAC(합성) | Grad-CAM(메인) + YOLO(부록) | 실데이터 위치탐지 / ELLIMAC bestV2 mAP@0.5 **0.739** |

---

## Stage 1 — 공정 센서 이상탐지 (ML vs DL)
결함 7/5000(0.14%) → 지도분류 불가 → 정상만 학습하는 이상탐지. leak-free(정상 80% 학습).

| 모델 | PR-AUC | ROC-AUC | recall@100 |
|---|---|---|---|
| **LOF (ML)** | **0.341** | 0.977 | **1.000** |
| Mahalanobis (ML) | 0.274 | 0.956 | 0.714 |
| OneClassSVM (ML) | 0.266 | 0.959 | 0.857 |
| AutoEncoder (DL) | 0.211 | 0.930 | 0.857 |
| IsolationForest (ML) | 0.055 | 0.902 | 0.714 |

→ **소표본·저차원 테이블엔 고전 ML이 DL보다 우세.** 위험신호: pressure↓(z −2.31)·etch_rate↑·temp↑.
`python -m src.stage1_process.run`

## Stage 2 — 웨이퍼 패턴 (플래그십)
| 실험 | macro-F1 | mAP | 메시지 |
|---|---|---|---|
| 합성(MixedWM38) | 0.985 | 0.999 | 합성은 쉬움 |
| 합성→실제 전이 | **0.364** | 0.417 | **도메인 갭 규명**(정상 오탐 0.957) |
| A 증강 진단 | 0.311 | 0.413 | "노이즈 원인설" 기각(증강 무효) |
| **실데이터 lot-split** | **0.859** | 0.930 | ★ 메인 벤치마크 |
| 실데이터 + 자기지도 | 0.869 | 0.928 | SSL +0.01 (효과 미미) |

- per-class: Edge-Ring 0.99 / Center 0.93 / … / **Scratch 0.78 · Loc 0.72(최난)**
- 평가: **lot 단위 그룹 분할**(같은 lot 누수 차단), seed 고정, 임계 val-only
- `python -m src.stage2_wafermap.train_real --loss asl` · `transfer_eval` · `ssl_pretrain`

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

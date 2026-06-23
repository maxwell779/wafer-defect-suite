# PRD — Semiconductor Defect Analysis Suite

> 반도체 웨이퍼 결함을 **팹 검사 흐름(공정 → 웨이퍼 패턴 → 국소 위치)** 으로 분석하는
> end-to-end 포트폴리오. 코드(`wafer-defect-suite`) + 데모 웹(React).
> *문서 기준일: 2026-06-22 (진행 현황은 §4의 스냅샷).*

## 1. 비전 / 대상
- **한 줄**: "공정값에서 **왜** 생기고, 웨이퍼에서 **어떤** 패턴이며, 다이에서 **어디** 있는지"를 세 모달리티(테이블·이미지맵·검출)로 잇는 반도체 결함 분석 콘솔.
- **대상 독자**: 반도체/제조 비전·ML 직군 채용 담당. "여러 데이터 유형에 맞는 기법 선택 + 정직한 평가"를 보여주는 게 목적.

## 2. 목표 / 비목표 / 성공 기준
**목표**
- 3-스테이지 각각에서 **다른 ML 역량**을 정직한 평가(leak-free·per-class·불균형)로 증명
- busbar 역량 재사용: 이상탐지·대조학습·leak-free·검출
- 결과를 **React 웹 데모**로 시각화

**비목표(지금 안 함)**
- ❌ **합성 결함 생성**: 합성→실제 전이 실패를 이미 입증 → 가치 낮음. *보류(옵션)*
- ❌ 실시간 팹 연동, 대규모 분산학습, SOTA 경쟁

**성공 기준 (Definition of Done) — 달성 현황**
- Stage1: ML 5종 vs DL + 변수기여 + 30-seed CV 결론 ✅ (고전ML>DL, DL 불안정)
- Stage2: 실데이터 lot-split macro-F1 ≥ 0.85 ✅ → **강화로 0.890±0.007 / 보정 ~0.90** ✅. (자기지도는 **효과 없음**으로 정직 보고 — 당초 가정과 다름)
- Stage3: ELLIMAC bestV2 mAP@0.5 0.739 ✅ + **실데이터 Grad-CAM 위치탐지(B) 추가** ✅. (YOLO11 재학습은 미실시)
- 웹: 5화면 React + FastAPI LIVE 추론 ✅
- 전체: README 재현법 + 정직한 결과표(negative 포함) ✅

## 3. 아키텍처 — 3 스테이지
| 스테이지 | 질문 | 데이터 | 핵심 기법 | 평가 |
|---|---|---|---|---|
| **1. 공정** | 왜 생기나 | Meruva CSV (5k, 결함 7) | **ML vs DL 비교** + 변수기여 | PR-AUC·recall@k |
| **2. 웨이퍼 패턴** ★ | 어떤 패턴 | MixedWM38 + WM-811K | 멀티라벨 분해·합성→실제 전이·**자기지도** | macro-F1·mAP·per-class |
| **3. 국소 위치** | 어디 | ELLIMAC (YOLO) | 라벨정제 + 검출 | mAP@0.5 |

## 4. 스테이지별 상세 (✅ 전부 완료)

### Stage 1 — 공정 센서 (Meruva) ※ML vs DL ✅
- **ML vs DL** 5종 + **30회 반복 CV(mean±std)** — 양성 7건 단일 split 노이즈 제거
- 결과: **Mahalanobis 0.295±0.028 / OCSVM 0.294 / LOF 0.294 (통계 동률)** > **AutoEncoder(DL) 0.215±0.095(불안정)** > IForest 0.119
- 결론: **소표본·저차원엔 고전 ML 우세, DL 불필요·불안정.** 변수기여 pressure↓·etch↑·temp↑
- `src/stage1_process/run.py`(단일) · `rigor.py`(반복 CV)

### Stage 2 — 웨이퍼 패턴 ★플래그십 ✅
| 실험 | macro-F1 | 비고 |
|---|---|---|
| 합성(MixedWM38) | 0.985 | 합성 쉬움 |
| 합성→실제 전이 | **0.364** (정상오탐 0.957) | 도메인갭 규명 |
| A 증강 진단 | 0.311 | "노이즈 원인설" 기각(negative) |
| 실데이터 lot-split(baseline) | 0.859 | 단일 |
| C 자기지도(SimCLR) | 0.869 / 저라벨 −0.02 | **효과 없음(negative)** |
| CNN 강화(증강+width64)+보정 | 0.902 ± 0.008 (3-seed) | |
| **SE-ResNet(깊이+SE attention)** | 0.912 → 보정 **0.928** | ★★ 돌파 |
| **SE-ResNet 앙상블(3)+보정** | **0.929** | ★ best (Loc 0.85·Scratch 0.86·NF 1.0) |
| (negative) 패딩·해상도·balanced·SimCLR·MAE·DINO | 0.85~0.87 / 0.33 | 모두 효과 없음(정직) |
- 평가: lot 그룹 분할·다중 seed·임계 val-only. 혼동행렬 산출 완료.

### Stage 3 — 결함 위치 ✅ (A+B)
- **B(메인, 실데이터)**: Stage2 실모델 **Grad-CAM** → WM-811K 실제 맵 결함 위치 히트맵(합성 무관). `stage3_localization/gradcam.py`
- **A(부록, 실제 칩표면 사진)**: ELLIMAC 폴리곤→bbox 정제+cls6 제거 → YOLO11m **mAP@0.5 0.753**(11l 동률). `stage3_detection/benchmark.py`
  - (YOLO11 재학습은 미실시 — bestV2 벤치마크로 대체, 합성이라 일반화 한계 명시)

## 5. 데이터 & 라이선스
> 데이터는 **레포에 포함하지 않음**(대용량·라이선스). `.gitignore`의 `data/`. 사용자는 출처에서 직접 받아 `data/`에 둠.

| 데이터 | 출처 | 비고 |
|---|---|---|
| WM-811K (LSWMD.pkl) | MIR Lab / Wu et al. 2015 (Kaggle 재배포) | 실데이터 811k, 연구용 |
| MixedWM38 (npz) | Kaggle(co1d7era) / Wang et al. | 합성 멀티라벨 38k |
| ELLIMAC (YOLO) | Kaggle(ellimaaac) / Roboflow | 실제 칩표면 사진+검출 라벨(증강), 다른 도메인 |
| Meruva (csv) | Kaggle(Meruva Kodanda Suraj) | 공정센서 테이블(합성 가능성) |
- 각 데이터 **출처/라이선스 링크를 README에 명시**, 포트폴리오/연구 목적 사용.

## 6. 데모 웹 (React) — "Defect Analysis Console"
상세 사양: [`web_design_prompt.md`](web_design_prompt.md). 5화면 = Dashboard / Stage1 Process / **Stage2 WaferMap(합성vs실데이터 토글=전이 스토리)** / Stage3 Detection / Experiments(정직한 수치·혼동행렬). 백엔드 FastAPI(추론) 또는 정적 데모 JSON 폴백.

## 7. 기술 스택 / 구조
- 학습: PyTorch (`src/stage2_wafermap/` 구축됨)
- 백엔드: FastAPI + 저장 `*.pt` 추론 / 프론트: React+Vite, 차트 Recharts, 웨이퍼맵 canvas
- 레포: `src/`(stage1/2/3) · `web/` · `experiments/`(git제외) · `docs/` · `data/`(git제외)
- 재현성: seed 고정, `requirements.txt`, lot-split

## 8. 산출물 (Deliverables)
- 학습된 모델: 합성·실데이터·SSL encoder (`experiments/*/best.pt`, git제외)
- 결과 리포트: 스테이지별 `test_report.txt`/`test_metrics.json` + 종합 비교표
- 웹 데모: `web/`(프론트+백엔드) 또는 claude.ai 단일파일 데모
- 문서: README(재현법·결과·데이터 출처), 본 PRD, EDA 노트, HANDOFF

## 9. 로드맵
1. **Stage2 마무리**: SSL 파인튜닝 비교 + 임계보정 ← 지금
2. **Stage1**: ML vs DL + 변수기여
3. **Stage3**: 라벨정제 + YOLO11 + 벤치마크
4. **웹 데모**: 백엔드 → 프론트 → 디자인
5. 종합 README/리포트 + GitHub 정리
> 병렬화: 독립 실험은 백그라운드 동시 실행.

## 10. 평가 원칙 (busbar 계승)
- **leak-free**: split 고정, lot 그룹(WM-811K), 임계는 val에서만
- **per-class·불균형**: macro-F1·mAP·PR-AUC, 희귀클래스 별도, 정확도 단일지표 금지
- **정직성**: 합성 고득점도 전이 실패를 그대로 보고

## 11. 리스크 & 대응
| 리스크 | 영향 | 대응 |
|---|---|---|
| 도메인 갭(합성→실제) | 합성모델 실전 무용 | 실데이터 학습(B) 완료, 한계 그대로 보고 |
| **Meruva 양성 7건** | 통계적 불안정, 결론 과신 위험 | "데모/탐색 수준" 명시, recall@k로만, 일반화 자제 |
| ELLIMAC 다른 도메인(칩표면) | 우리 웨이퍼 과제 일반화 한계 | 라벨정제 + 도메인 차이 명시(검출 스킬 데모) |
| 누수/과적합 | 수치 부풀림 | lot-split·seed·임계 val-only |
| 웹 라이브 추론 복잡도 | 일정 지연 | **정적 데모(JSON) 폴백** 우선 |

## 12. 열린 질문
- 웹: 정적 데모로 충분한가, FastAPI 라이브까지 갈까?
- 자기지도가 **full-label에서도** 이득 있나, 저라벨에서만인가? (실험으로 확인 중)
- 진행 순서: 모델 스테이지 먼저 vs 웹 먼저?
- Stage1 결론을 어디까지 일반화해 말할 수 있나(7건 한계)?

## 13. 의사결정 로그
- 2026-06-22: 합성 결함 생성 **보류**. Stage1 **ML vs DL 비교 추가**. 실험 병렬화. PRD에 성공기준·데이터/라이선스·리스크·산출물·열린질문 보강.

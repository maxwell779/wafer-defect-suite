# PRD — Semiconductor Defect Analysis Suite

> 반도체 웨이퍼 결함을 **팹 검사 흐름(공정 → 웨이퍼 패턴 → 국소 위치)** 으로 분석하는
> end-to-end 포트폴리오. 코드(`wafer-defect-suite`) + 데모 웹(React).

## 1. 비전 / 한 줄
"공정값에서 **왜** 생기고, 웨이퍼에서 **어떤** 패턴이며, 다이에서 **어디** 있는지"를
세 모달리티(테이블·이미지맵·검출)로 잇는 반도체 결함 분석 콘솔.

## 2. 목표 / 비목표
**목표**
- 3-스테이지 각각에서 **다른 ML 역량**을 정직한 평가(leak-free·per-class·불균형)로 증명
- busbar 프로젝트 역량 재사용: 이상탐지·대조학습·leak-free·검출
- 결과를 **React 웹 데모**로 시각화(채용 어필)

**비목표(지금 안 함)**
- ❌ **합성 결함 생성**(synthetic defect gen): 합성→실제 전이 실패를 이미 입증 → 합성 추가 가치 낮음. *보류(옵션)*
- ❌ 실시간 팹 연동, 대규모 분산학습

## 3. 아키텍처 — 3 스테이지

| 스테이지 | 질문 | 데이터 | 핵심 기법 | 평가 |
|---|---|---|---|---|
| **1. 공정** | 왜 생기나 | Meruva CSV (5k, 결함 7) | **ML vs DL 비교** + 변수기여 | PR-AUC·recall@k |
| **2. 웨이퍼 패턴** ★ | 어떤 패턴 | MixedWM38 + WM-811K | 멀티라벨 분해·합성→실제 전이·**자기지도** | macro-F1·mAP·per-class |
| **3. 국소 위치** | 어디 | ELLIMAC (YOLO) | 라벨정제 + 검출 | mAP@0.5 |

## 4. 스테이지별 상세

### Stage 1 — 공정 센서 (Meruva) ※ML vs DL 비교
- 데이터: 6 수치(temp/pressure/gas/etch_rate/voltage/current) + step, **결함 7/5000(0.14%)**
- ★ **ML vs DL 정면 비교** (작은·불균형 테이블에서 무엇이 맞나):
  - 고전 ML: **IsolationForest · One-Class SVM · LOF · Mahalanobis(가우시안)**
  - DL: **AutoEncoder 재구성오차**
  - 가설/결론 후보: "양성 7건뿐 → 고전 ML(특히 Mahalanobis/IForest)이 DL보다 안정적·해석가능"
- 변수기여: permutation importance / 정상 대비 z-gap (이미 EDA: pressure↓·temp↑·etch↑)
- 평가: **PR-AUC, recall@k**(정확도 금지), 비지도라 라벨은 평가에만 사용

### Stage 2 — 웨이퍼 패턴 ★플래그십 (진행 현황)
| 실험 | 결과 | 상태 |
|---|---|---|
| 합성(MixedWM38) 멀티라벨 | macro-F1 0.985 / mAP 0.999 | ✅ |
| 합성→실제 전이 | macro-F1 **0.364**, 정상오탐 0.957 → 도메인갭 규명 | ✅ |
| A 진단(증강) | 전이 0.311(↓) → "노이즈가 원인" 기각, 갭은 구조적 | ✅ |
| 실데이터(WM-811K) lot-split | macro-F1 **0.859** / mAP 0.930 ← 메인 벤치마크 | ✅ |
| **C 자기지도(SimCLR 150k)** | 사전학습 → 파인튜닝(전체+저라벨) vs scratch | 🔄 진행 |
- 남은 것: SSL 파인튜닝 비교(특히 **저라벨 10%** 에서 SSL 이득), 클래스별 **임계 val 보정**, 혼동분석

### Stage 3 — 국소 위치 (ELLIMAC)
- 라벨 정제(cls6 폴리곤 108파일 제거/매핑) → 깨끗한 YOLO 데이터
- YOLO11 학습 + 제공 `bestV2.pt` 벤치마크(mAP@0.5)
- Stage2와 연계: 패턴분류 → 위치검출

## 5. 데모 웹 (React) — "Defect Analysis Console"
busbar 콘솔처럼. 상세 사양은 [`web_design_prompt.md`](web_design_prompt.md).
- **Dashboard**: 3-스테이지 파이프라인 다이어그램 + 스테이지별 핵심 지표
- **Stage1 Process**: 공정값 입력/선택 → 이상점수 + 변수기여 막대 + ML/DL 비교표
- **Stage2 WaferMap** ★: 웨이퍼맵 선택 → 멀티라벨 예측 막대 + 맵 시각화 + **합성모델 vs 실데이터모델 동일맵 비교**(전이 스토리)
- **Stage3 Detection**: 이미지 → YOLO 박스 오버레이
- **Experiments**: 비교표·혼동행렬(정직한 수치 그대로)
- 백엔드: **FastAPI**(학습모델 추론 서빙) / 또는 정적 데모(사전계산 결과 JSON)

## 6. 기술 스택 / 구조
- 학습: PyTorch (이미 구축, `src/stage2_wafermap/`)
- 백엔드: FastAPI + 저장된 `*.pt` 추론
- 프론트: React + (차트: Recharts/Plotly), 웨이퍼맵 캔버스 렌더
- 레포: `src/`(stage1/2/3) · `web/`(frontend+backend) · `experiments/`(산출물, git제외) · `docs/`

## 7. 로드맵 (단계)
1. **Stage2 마무리**: SSL 파인튜닝 비교 + 임계보정 (진행) ← 지금
2. **Stage1 구현**: ML vs DL 비교 + 변수기여
3. **Stage3 구현**: 라벨정제 + YOLO11 + 벤치마크
4. **웹 데모**: 백엔드(추론 API) → 프론트(React) → 디자인
5. 종합 README/리포트 + GitHub 정리

> 병렬화: 독립 실험(SSL 파인튜닝 변형, Stage1 모델들)은 백그라운드 동시 실행 가능.

## 8. 평가 원칙 (busbar 계승)
- **leak-free**: split 고정, lot 단위 그룹(WM-811K), 임계는 val에서만
- **per-class·불균형**: macro-F1·mAP·PR-AUC, 희귀클래스 별도 추적, 정확도 단일지표 금지
- **정직성**: 합성 고득점도 전이 실패를 그대로 보고(스토리의 핵심)

## 9. 의사결정 로그
- 2026-06-22: 합성 결함 생성 **보류**(전이 실패 입증으로 가치↓). Stage1 **ML vs DL 비교 추가**. 실험 병렬화.

# Claude 디자인 프롬프트 — "Semiconductor Defect Analysis Console" (React)

> 반도체 웨이퍼 결함 분석 포트폴리오의 **데모 웹앱**을 설계/구현해 주세요.
> 톤·완성도는 실무 검사 콘솔 수준. 깔끔·기술적, 데이터 중심. (이전 busbar 검사 콘솔의 후속)

---

## 0. 무엇을 만드나
3-스테이지(공정 → 웨이퍼 패턴 → 국소 위치) 결함 분석 결과를 보여주는 **단일 페이지 React 앱**(탭/라우트 5개).
백엔드는 **FastAPI**(저장된 PyTorch 모델 추론) 또는 **정적 데모**(사전계산 JSON) 둘 다 지원하게.

## 1. 화면 구성 (5)

### ① Dashboard
- 상단: **3-스테이지 파이프라인 다이어그램** (공정→웨이퍼맵→다이, 화살표로 연결)
- 스테이지별 **핵심 지표 카드**:
  - Stage1: 결함 7/5000(0.14%), best AD model PR-AUC
  - Stage2: 실데이터 macro-F1 **0.859** / mAP 0.930
  - Stage3: 검출 mAP@0.5
- 한 줄 서사: "왜(공정) → 무엇(패턴) → 어디(위치)"

### ② Stage 1 — Process Monitor
- 좌: 공정값 입력/슬라이더(temperature, pressure, gas_flow, etch_rate, voltage, current) 또는 샘플 선택
- 우: **이상 점수 게이지** + **변수 기여 막대**(pressure↓·temp↑·etch_rate↑가 위험신호)
- 하단: **ML vs DL 비교표**(IsolationForest / One-Class SVM / LOF / Mahalanobis / AutoEncoder의 PR-AUC·recall@k)

### ③ Stage 2 — Wafer Map Classifier ★메인
- 좌: 웨이퍼맵 갤러리/선택(52×52, 값 0=다이없음/1=정상/2=불량를 색으로). 업로드도.
- 중: 선택맵 **확대 렌더**(캔버스). 색: 0=연회색, 1=옅은청, 2=빨강.
- 우: **멀티라벨 예측 막대**(8클래스 확률, 임계선 표시) + 예측 라벨 칩
- ★ **모델 토글: "합성 학습" vs "실데이터 학습"** → 같은 맵에 두 모델 예측을 나란히 → **전이 실패 스토리**가 보이게
  (합성모델은 실제맵을 Random으로 오인하는 등)

### ④ Stage 3 — Defect Localization
- 이미지 선택 → **YOLO 박스 오버레이**(클래스별 색), confidence 표시
- 제공 모델(bestV2) vs 재학습 모델 비교 토글(있으면)

### ⑤ Experiments (정직한 수치 그대로)
- **비교표**: 합성 0.985 → 전이 0.364 → 실데이터 0.859 (+ A 진단 0.311, SSL 결과)
- **혼동행렬** 히트맵(실데이터 8×8)
- per-class F1/mAP 막대 (Loc·Scratch 가장 낮음을 그대로 노출)

## 2. 실제 데이터/수치 (목업에 사용)
- Stage2 실데이터 per-class F1: Edge-Ring 0.99, Center 0.93, Donut 0.92, Random 0.88, Near-full 0.83, Edge-Loc 0.83, Scratch 0.78, **Loc 0.72**
- 전이 실패: 합성모델 정상 오탐율 0.957, Center→Random 오인 다수
- 클래스 8: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random
- 웨이퍼맵 샘플: 백엔드가 `/api/wafermap/{id}` 로 52×52 정수배열 제공(또는 정적 JSON)

## 3. 디자인 가이드
- **깔끔·기술적**, 반도체/팹 느낌. 다크 또는 라이트 1택(일관). 강조색 1~2개.
- 색 절제: 웨이퍼맵의 빨강(불량)·차트 강조만. 장식 과다 금지.
- 반응형, 카드 기반 레이아웃, 모노스페이스는 수치/코드에만.
- 한국어 UI(영문 병기 가능).

## 4. 기술
- React(+Vite), 차트 Recharts 또는 Plotly, 웨이퍼맵은 `<canvas>` 직접 렌더
- 백엔드 FastAPI: `/api/stage1/score`, `/api/stage2/predict?model=real|synth`, `/api/stage3/detect`, `/api/metrics`
- 모델 없을 때를 위한 **정적 데모 모드**(사전계산 JSON) 폴백 포함

## 5. 산출물
- `web/frontend/`(React) + `web/backend/`(FastAPI) 구조
- 더미 데이터로도 동작하는 데모(모델 연결 전 확인용)
- README에 실행법

## 6. 정확성 가드
- **반도체 웨이퍼** 결함분석(자동차 busbar 아님). 카메라/이미지맵 기반.
- 비지도/멀티라벨/검출이 스테이지별로 다름 — 혼동 금지.
- 수치는 위 실제값 사용, 과장 금지(Loc/Scratch가 낮은 것도 그대로).

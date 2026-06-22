# Claude 디자인 프롬프트 v2 — "Wafer Defect Analysis Console" (기능 풍부 React 앱)

> ❗ v1 데모는 **겉모습만 있고 기능이 0** 이었음. 이번엔 **busbar 검사 콘솔처럼 실제로 동작하는 기능**을
> 많이 넣어야 함. 정적 그림 금지 — **인터랙션·필터·차트·추천·판정**이 mock 데이터로 실제 작동해야 함.
> 첨부 zip의 **실제 웨이퍼/검출 이미지**와 **데이터 JSON**을 활용해 현실감 있게.

---

## 0. 산출물
- React(+Vite, Tailwind) **단일 또는 멀티 파일** 앱. claude.ai Artifact에서 **미리보기 동작** 필수.
- 첨부 `data/*.json` 을 import 해서 구동(또는 동등한 mock 내장). 첨부 `assets/*.png|jpg` 를 이미지로 사용.
- 5개 탭, 각 탭이 **실제 기능**을 가져야 함(아래 "기능" 항목 전부 동작).

## 1. 공통 셸
- 상단바: 로고 "Wafer Defect Console", 탭 5개, 우측 DEMO 뱃지.
- 톤: 깔끔·기술적(라이트), 강조색 1~2개, 웨이퍼맵 빨강(불량)만 색 강조. 한국어 UI.

---

## 2. 탭별 기능 (★실제 동작 필수)

### ① Dashboard
- 3-스테이지 파이프라인 다이어그램(공정→웨이퍼맵→다이) + 스테이지 지표 카드.
- **실시간 느낌 요약**: 최근 처리 lot 수, 결함율, 경보 수(아래 데이터에서 집계).
- 각 카드 클릭 → 해당 스테이지 탭으로 이동.

### ② Stage 1 — Process Monitor ★기능 많이
데이터: `data/process_runs.json` (공정 run 수백 개, 6파라미터 + step + 결함라벨 + 이상점수).
**필수 기능:**
1. **Run 테이블**: 정렬/필터(공정단계, 상태 OK/WARN/FAIL), 검색. 행 클릭 → 상세.
2. **관리도(Control Chart)**: 파라미터별 라인차트 + 관리한계선(UCL/LCL, mean±3σ). 한계 벗어난 run 빨강 표시.
3. **파라미터 분포**: 정상 vs 결함 히스토그램/박스플롯 토글.
4. **수율-파라미터 관계**: 산점도(파라미터 x vs 이상점수/결함) + **변수 중요도 막대**
   (pressure↓ −2.31, etch_rate↑ +1.39, temp↑ +1.37 …). "어떤 파라미터가 수율에 영향".
5. **파라미터 조정 추천 패널**: run 선택 → 정상 평균 대비 편차 큰 파라미터 표시 +
   "pressure 690 < 정상 760 → **+70 상향 권장**" 식 추천. 슬라이더로 값 바꾸면 이상점수 재계산(클라이언트 휴리스틱).
6. **ML vs DL 비교표**: `data/stage1_results.json` (LOF PR-AUC 0.341 1위 … AutoEncoder 0.211).
   "소표본엔 고전 ML 우세" 결론 배지.

### ③ Stage 2 — Wafer Map Analyzer ★메인, 판정 기능
데이터: `data/wafermaps.json` (맵 메타 + 라벨), 이미지 `assets/wm_*.png`.
**필수 기능:**
1. **맵 갤러리**: 클래스/소스(합성·실제) 필터, 썸네일 그리드. 클릭 → 분석.
2. **맵 뷰어**: 선택맵 확대(canvas, 0=연회색/1=옅은청/2=빨강).
3. **결함 판정**: **멀티라벨 예측 막대**(8클래스 확률 + 임계선) → 예측 결함 칩 + 판정(OK/NG).
4. **위치 판정(localization)**: 맵 위에 **결함 영역 하이라이트/히트맵 오버레이**
   (어느 부분이 결함인지). 토글로 원본/히트맵.
5. **합성 vs 실데이터 모델 토글**: 같은 맵 두 예측 나란히 →
   합성모델이 실제맵을 Random으로 오인·정상 오탐 0.957 = **전이 실패 스토리** 시각화.
6. **업로드**: 사용자 맵 업로드 → (mock) 예측.

### ④ Stage 3 — Defect Localization (검출)
이미지 `assets/ellimac_*.jpg`.
**필수 기능:**
1. 이미지 선택/업로드 → **YOLO 박스 오버레이**(클래스별 색 + confidence).
2. confidence 슬라이더로 박스 필터.
3. 클래스 범례 + 검출 개수 집계.

### ⑤ Experiments
데이터: `data/stage2_metrics.json`.
1. **비교표**: 합성 0.985 → 전이 0.364 → A증강 0.311 → 실데이터 0.859 → +SSL 0.869.
2. **per-class F1 막대**(Edge-Ring 0.99 … Loc 0.72 최저 — 그대로 노출).
3. **혼동행렬 히트맵**(8×8, 인터랙티브 툴팁).
4. 정직성 노트: "A·C(증강·자기지도) negative result도 그대로 보고".

---

## 3. 데이터/이미지 (첨부 zip)
- `data/process_runs.json` — 공정 run 배열(Stage1 테이블·관리도·추천용)
- `data/stage1_results.json` — ML vs DL 비교 + 변수기여
- `data/wafermaps.json` — 웨이퍼맵 메타(클래스/소스/예측)
- `data/stage2_metrics.json` — 비교표·per-class·혼동행렬
- `assets/wm_*.png` — 실제 렌더된 웨이퍼맵(합성·실제, 클래스별)
- `assets/ellimac_*.jpg` — 검출용 실제 웨이퍼 이미지
> 데이터 없으면 동등 mock을 내장하되, **첨부 수치/이미지를 우선** 사용.

## 4. 품질 기준 (busbar 수준)
- 모든 탭에 **동작하는 인터랙션**(필터·정렬·선택·슬라이더·토글). 빈 껍데기 금지.
- 차트는 Recharts(또는 동등), 웨이퍼맵은 canvas.
- 반응형, 카드 레이아웃, 로딩/빈상태 처리.

## 5. 정확성 가드
- **반도체 웨이퍼**(자동차 busbar 아님). 8클래스: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random.
- 수치는 첨부값 그대로, 과장 금지(Loc/Scratch 낮음, A·C negative 그대로).
- Stage1=비지도 이상탐지, Stage2=멀티라벨 분류+위치, Stage3=검출 — 혼동 금지.

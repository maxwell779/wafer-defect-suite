# 실험 종합 분석 & 성능 향상 전략

> 현재까지 실험을 **버그/문제 점검 → 데이터 재분석 → stage·data별 향상 전략**으로 정리.

---

## 1. 실험 결과 버그/문제 점검 (audit)

| 항목 | 점검 결과 | 조치 |
|---|---|---|
| MixedWM38 컬럼→클래스 매핑 | **공간 시그니처로 검증 OK** (col0 중앙=Center, col3 가장자리=Edge-Ring, col5 전면=NF, col7 산포=Random) | 버그 아님 ✓ |
| Stage1 30-seed CV | 결함 **7건은 매 split 고정**(변동=normal split·모델 랜덤만) → 분산이 과소평가될 수 있음 | repeated **stratified k-fold(결함도 분할)** 로 보강 권장 |
| Stage2 "~0.900" | seed42 보정(+0.0098)을 평균에 외삽한 값 | 전 seed에 보정 적용해 **실측 mean±std**로 교체 필요 |
| 증강 유효성 | 회전/플립은 웨이퍼 패턴에 **라벨 불변**(Center/Ring/Loc/Scratch 모두) → 유효 ✓. `aug_noise=0.05`는 실데이터엔 과할 수 있음 | aug_noise 0/0.02도 비교 |
| **resize 52×52 nearest** | 632가지 가변크기·비정사각(예 53×58)을 **정사각 강제 → 종횡비 왜곡** | aspect 보존 pad→resize 실험 ★ |
| Stage3 polygon→bbox | 폴리곤 외접 bbox 근사(표준), YOLO11 **재학습 미실시** | 정식 학습+정제 시 보강 |
| ELLIMAC 의미 | 클래스가 wafer-map 패턴(Center 등)인데 "이미지 검출"이라 도메인 불명확 + 합성 | 부록 유지, 한계 명시 |
| 정상 오탐(전이) | 합성모델 정상 오탐 0.957 — 정상('none')에 배경노이즈 ~16% | 실데이터 학습으로 해결됨 ✓ |

**결론**: 치명 버그는 없음. 개선 여지 = ① k-fold CV로 분산 정확화, ② 보정 실측화, ③ resize 왜곡 제거.

---

## 2. 데이터 재분석 — 무엇이 성능을 제한하나

### WM-811K (실, Stage2 메인)
- **632가지 가변크기** → 52 정사각 resize가 정보 손실/왜곡. 큰 맵((53,58))은 다운샘플로 미세패턴 소실.
- **극불균형**: Edge-Ring 9680 ↔ **Near-full 149·Scratch 1193·Donut 555·Random 866**. 희귀클래스가 macro-F1 끌어내림.
- **저밀도 클래스 = 난이도**: Scratch(밀도0.10)·Loc(0.13)이 최난(현 F1 0.80/0.81). 미세·국소라 저해상도에 취약.
- **63.8만 미라벨** = 미활용 자원(자기지도 실패했지만 방법 개선 여지).
- **lot 구조**(46k lot×1–25) → 평가는 lot-split 필수(완료). 학습엔 lot 내 상관 활용 여지.

### MixedWM38 (합성, 전이/멀티라벨)
- 배경노이즈 ~8%(정상도) → 실데이터와 분포 다름(전이 실패 핵심).
- 양립불가 쌍(C↔D, EL↔ER)·NF/Random 실데이터 차용 → **혼합은 6클래스 한정**.
- 멀티라벨 분해 검증은 여기서만 가능(실데이터는 단일라벨).

### Meruva (실, Stage1)
- **양성 7건** = 통계적 천장. PR-AUC 0.295±0.03이 사실상 한계. 일반화 주장 자제.
- 분리신호 pressure/etch/temp만 유효(3변수). 나머지 노이즈.

### ELLIMAC (합성, Stage3)
- Roboflow 합성·폴리곤 라벨·cls6 노이즈. 실전 일반화 불가. 스킬 데모용.

---

## 3. 성능 향상 전략 (대대적, stage·data별)

### ★ Stage 2 — 웨이퍼 패턴 (플래그십, 최대 투자)
현재 0.890±0.007(보정 ~0.90). 병목 = Loc·Scratch·Edge-Loc + 희귀클래스.

**A. 입력/전처리**
1. **종횡비 보존 pad→resize** (정사각 패딩 후 리사이즈) — 왜곡 제거 ★최우선
2. **해상도↑** 52→64/96 (미세 Scratch/Loc 보존)
3. **극좌표 변환**(웨이퍼는 원형 → r,θ 인코딩)으로 Edge/Ring/Center 회전불변 강화
4. die-mask 채널 추가(유효 영역 명시)

**B. 모델**
1. 더 큰 백본(ResNet18/스케일업) + GAP→head
2. **CoordConv/attention**으로 위치 패턴(Center vs Loc) 구분력↑
3. TTA(회전4×) 추론 앙상블

**C. 불균형 (희귀 NF/Donut/Scratch/Loc)**
1. **class-balanced sampling**(WeightedRandomSampler) ★
2. ASL γ 튜닝 / focal / per-class loss weight
3. 희귀클래스 **표적 증강**(회전·탄성변형)

**D. 데이터 활용**
1. normal_cap 스윕(특이도 vs 균형)
2. **자기지도 재도전**: 강한 증강(cutout/elastic)·**MAE**·더 큰 백본·더 긴 학습(현 SimCLR는 약증강으로 실패)
3. WM-811K 전체 라벨(현재 일부) 활용

**E. 평가/튜닝**
1. **lot k-fold CV(5겹)** → mean±std 정확화 ★
2. per-class 임계 전 seed 실측 적용
3. 멀티라벨(MixedWM38)에서 분해 성능 별도 보고

### Stage 1 — 공정 (소표본 한계 내 최선)
1. **stratified k-fold(결함 포함)** CV로 분산 정확화 ★
2. feature engineering(비율·교호작용: pressure/etch)
3. **AD 앙상블**(LOF+Maha+OCSVM 점수 평균) → 안정성↑
4. (옵션) 합성 양성 생성(SMOTE류)로 지도학습 보조 — 단 한계 명시

### Stage 3 — 검출/위치
1. **B(Grad-CAM)**: 정량화(IoU/포인팅 게임)로 위치 정확도 수치화
2. **A(ELLIMAC)**: 라벨 전수 정제 → **YOLO11 정식 학습** vs bestV2 비교
3. 증강·해상도 스윕

---

## 4. 우선순위 실험 계획 (ROI 순)

| # | 실험 | 기대효과 | 비용 |
|---|---|---|---|
| 1 | Stage2 **종횡비 보존 + 해상도 64** | 왜곡 제거, Scratch/Loc↑ | 중 |
| 2 | Stage2 **class-balanced sampling + ASL튜닝** | 희귀클래스 macro-F1↑ | 중 |
| 3 | Stage2 **lot 5-fold CV + 전-seed 임계보정** | 신뢰구간 정확화 | 중 |
| 4 | Stage2 **자기지도 재도전(MAE/강증강)** | 미라벨 활용, 난클래스↑ | 상 |
| 5 | Stage1 **stratified k-fold + AD 앙상블** | 분산 정확·안정 | 저 |
| 6 | Stage3 **YOLO11 학습 + Grad-CAM 정량화** | 검출 비교·위치 수치 | 상 |
| 7 | Stage2 **극좌표/CoordConv** | 위치 패턴 구분 | 상 |

> 권장 착수: **1→2→3**(Stage2 핵심 ROI) → 5(빠른 보강) → 4/6(연구성).

# Stage 1 — 공정 센서 이상탐지 (Meruva CSV)

**상태: 예정 (placeholder)**

## 문제
공정 파라미터(temperature/pressure/gas_flow/etch_rate/voltage/current)로 결함 웨이퍼를 탐지.
**결함 7 / 5000 (0.14%)** → 지도 분류 불가 → **비지도 이상탐지**.

## EDA 발견 (분리 신호)
| feature | 정상 평균 | 결함 평균 | z-gap |
|---|---|---|---|
| pressure_torr | 759.8 | 690.1 | **2.31** |
| etch_rate_nm_min | 95.1 | 106.2 | 1.38 |
| temperature_c | 450.1 | 470.5 | 1.37 |
| gas/voltage/current | — | — | <0.5 (거의 무신호) |

→ "압력 저하 + 고온 + 식각률↑" 가 결함 신호.

## 계획
- IsolationForest / One-Class SVM / Mahalanobis / AutoEncoder 비교
- 평가: 결함 7건에 대한 recall@k, PR-AUC (정확도 금지 — 불균형 함정)
- 변수 기여도(SHAP/permutation) 로 "어떤 공정값이 위험한가" 설명

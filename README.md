# wafer-defect-suite

반도체 웨이퍼 결함을 **팹 검사 흐름(공정 → 웨이퍼 패턴 → 국소 위치)** 3단계로 분석하는 포트폴리오.

## 3-스테이지

| 스테이지 | 질문 | 데이터 | 문제 유형 | 상태 |
|---|---|---|---|---|
| **1. 공정 센서** | 왜 생기나 | Meruva CSV (5k, 결함 7건) | 테이블 **이상탐지**(극희귀) | 예정 |
| **2. 웨이퍼 패턴** ★ | 어떤 패턴인가 | MixedWM38(38k, 멀티라벨) + WM-811K(811k 실데이터) | **멀티라벨 분해** + 합성→실제 전이 + 자기지도 | **진행** |
| **3. 국소 위치** | 어디 있나 | ELLIMAC (YOLO, 4.5k) | **검출** + 라벨노이즈 정제 | 예정 |

## 데이터 위치
레포 내부 `data/` 에 보관 (대용량이라 git 에는 제외 — `.gitignore`):
```
data/Wafer_Map_Datasets.npz            # MixedWM38
data/wafer_datasets/LSWMD/LSWMD.pkl    # WM-811K
data/wafer_datasets/meruva_csv/...     # 공정센서
data/wafer_datasets/ELLIMAC/...        # YOLO 검출
```

## 데이터 (EDA 요약)
- **MixedWM38** (`Wafer_Map_Datasets.npz`): 38,015 × 52×52, 값{0,1,2}, 라벨 8차원 멀티핫(평균 2.3/장). 정상1k+단일7k+혼합30k. 합성(배경노이즈~8%). **NF(149)·Random(866)은 단독만** 등장(실데이터 차용).
- **WM-811K** (`LSWMD.pkl`): 811,457장, **632가지 가변크기**, **63.8만 미라벨**, lot 46,293개(웨이퍼 1–25). 극불균형(NF 149 ~ Edge-Ring 9680). **lot 단위 split 필수**.
- **Meruva** (`*.csv`): 공정센서 6변수, 결함 7/5000(0.14%). 분리신호=압력↓·온도↑·식각률↑.
- **ELLIMAC** (YOLO): 4,531장(train 3172/valid 907/test 452), 6클래스 + cls6(폴리곤·라벨노이즈 108개), 학습된 `bestV2.pt` 포함.

## Stage 2 베이스라인 실행
```bash
# 레포 루트(g:/wafer-defect-suite)에서
python -m src.stage2_wafermap.train --epochs 20            # 전체 학습
python -m src.stage2_wafermap.train --epochs 2 --subset 4000  # 스모크 테스트
python -m src.stage2_wafermap.train --loss asl              # Asymmetric Loss (불균형)
```
산출물: `experiments/stage2_*/` (best 가중치 + per-class 리포트).

## 설계 원칙 (busbar 프로젝트 교훈 계승)
- **leak-free**: split 고정 seed, 임계값은 val에서만 보정(평가셋 누수 금지).
- **per-class 분석**: 희귀클래스(NF·Random) 별도 추적. macro-F1·mAP·exact-match·hamming.
- **불균형 정면대응**: pos_weight BCE / Asymmetric Loss.

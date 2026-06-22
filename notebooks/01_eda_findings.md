# EDA 정리 — 4개 데이터셋 (WM811k_img 제외)

> 원본 데이터: `g:/pro-vision/data/wafer_datasets/` (+ `Wafer_Map_Datasets.npz`, `LSWMD.pkl`)

## 1. MixedWM38 (`Wafer_Map_Datasets.npz`) — 합성, 멀티라벨 ★Stage 2 본체
- `arr_0`: (38015, 52, 52) int, 값 {0=다이없음, 1=정상, 2=불량}
- `arr_1`: (38015, 8) 멀티핫 — `[Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random]`
- 라벨 카디널리티: 0개 1,000 / 1개 7,015 / 2개 13,000 / 3개 13,000 / 4개 4,000 (평균 2.3)
- 클래스 빈도: Scratch 19k, Loc 18k, Center/Edge-Loc 13k, Donut/Edge-Ring 12k, **Random 866, Near-full 149**
- **동시발생 규칙(데이터에 내재)**: Center↔Donut = 0, Edge-Loc↔Edge-Ring = 0 (양립불가). **NF·Random 은 단독만** 등장
- 결함밀도(die 중 불량): normal 0.078, single 0.29, mixed 0.33 (배경 노이즈 큼)

## 2. WM-811K / LSWMD (`LSWMD.pkl`) — 실제, 단일라벨, 대량
- 811,457장, columns: `waferMap, dieSize, lotName, waferIndex, trianTestLabel, failureType`
- **라벨 172,950 / 미라벨(NA) 638,507 (78%)** → 자기지도 사전학습 자원
- 제공 split: Training 54,355 / Test 118,595
- **632가지 가변 크기** (최다 (32,29) 108k, (25,27) 64k, (49,39) 39k …)
- lot 46,293개 × waferIndex 1–25 → **lot 단위 분할 필수**(웨이퍼 단위 분할 시 leak)
- failureType: none 147k, Edge-Ring 9680, Edge-Loc 5189, Center 4294, Loc 3593, Scratch 1193, Random 866, Donut 555, **Near-full 149**
- 결함밀도: Near-full 0.88·Random 0.48(전면) ↔ **Scratch 0.10·Loc 0.13(희소=난이도↑)**, none 도 0.16

## 3. Meruva (`semiconductor_wafer_defect_dataset.csv`) — 공정센서 테이블 → Stage 1
- 5,000행, 6 수치 + process_step + defect_label
- 결함 **7 / 5000 (0.14%)** — 극희귀(비지도 AD)
- 분리신호: pressure↓(z=2.31), etch_rate↑·temp↑(z≈1.4), 나머지 무신호
- process_step(Litho/Oxid/CMP/Etch/Deposition) 골고루 분포

## 4. ELLIMAC — YOLO 검출 → Stage 3
- 4,531장 (3172/907/452), 6클래스, 박스 대부분 1개
- 이미지 크기 혼재(742×576/640×480/224×224 …)
- **cls6 폴리곤(라벨노이즈) 108파일** + 학습된 `bestV2.pt`(52MB)

## 통합 서사: 팹 결함분석 3-스테이지
**공정(Meruva) → 웨이퍼 패턴(MixedWM38+WM-811K) → 국소 위치(ELLIMAC)**
= "왜 → 무엇 → 어디" 로 내려가는 end-to-end.

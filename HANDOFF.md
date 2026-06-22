# HANDOFF — wafer-defect-suite (새 채팅 인수인계)

> 새 Claude Code 세션은 **이 파일과 `README.md`, `notebooks/01_eda_findings.md` 를 먼저 읽고** 이어서 작업하면 됩니다.

## 0. 맥락
- 이전 프로젝트(버스바 카메라 비전 결함검사, ReConPatch)는 마무리 단계. **이건 새 개인 포트폴리오 프로젝트.**
- 사용자: 비전/이상탐지 실무 경험(불균형, leak-free 평가, 대조학습 ReConPatch, YOLO crop). 한국어로 소통.
- 환경: Windows, **터미널이 cmd** → 경로 이동은 `cd /d G:\wafer-defect-suite`. conda env **`mate-x_anomalib_env`**. GPU **A100 80GB**, torch 2.6+cu124.

## 1. 프로젝트 한 줄 정의
반도체 웨이퍼 결함을 **팹 검사 흐름 3-스테이지**로 분석:
**공정센서(왜) → 웨이퍼맵 패턴(무엇) → 국소 위치(어디)**

| 스테이지 | 데이터 | 문제 | 상태 |
|---|---|---|---|
| 1. 공정센서 | Meruva CSV | 테이블 이상탐지(결함 7/5000) | placeholder |
| **2. 웨이퍼 패턴** ★ | MixedWM38 + WM-811K | **멀티라벨 분해** + 합성→실제 전이 + 자기지도 | **베이스라인 동작** |
| 3. 국소 위치 | ELLIMAC | YOLO 검출 + 라벨노이즈 정제 | placeholder |

## 2. 데이터 (레포 `data/`, git 제외)
- `data/Wafer_Map_Datasets.npz` = **MixedWM38**: `arr_0`(38015,52,52) 값{0,1,2}, `arr_1`(38015,8) 멀티핫.
  - 클래스순서 `config.WM_CLASSES` = [Center,Donut,Edge-Loc,Edge-Ring,Loc,Near-full,Scratch,Random]
  - 평균 2.3라벨/장. 정상1k/단일7k/혼합30k. 배경노이즈~8%.
  - **NF(149)·Random(866)은 단독만** 등장(WM-811K 실데이터 차용). Center↔Donut, Edge-Loc↔Edge-Ring 양립불가(co=0).
- `data/wafer_datasets/LSWMD/LSWMD.pkl` = **WM-811K 실데이터** 811,457장. **632가지 가변크기**, **63.8만 미라벨**, lot 46,293개(웨이퍼1–25). 극불균형. ★**lot 단위 split 필수**(leak 방지).
- `data/wafer_datasets/meruva_csv/...` = 공정센서 6변수, 결함 7/5000. 신호: pressure↓·temp↑·etch_rate↑.
- `data/wafer_datasets/ELLIMAC/` = YOLO(images/labels train3172/valid907/test452), 6클래스. ★**cls6 폴리곤 라벨노이즈 108파일** + 학습된 `model/Model/bestV2.pt`(52MB).

## 3. 코드 구조 / 동작 확인됨
```
src/common/{seed,metrics}.py          # 멀티라벨 지표: per-class F1/mAP/exact-match/hamming
src/stage2_wafermap/
  dataset.py   # MixedWM38 로드 + combo-stratified leak-free split(3채널 one-hot)
  model.py     # WaferCNN (3 conv block→GAP→FC, 8 logits)
  losses.py    # BCE(pos_weight) / AsymmetricLoss
  train.py     # 학습+best저장+테스트리포트  evaluate.py # 체크포인트 재평가
```
실행 (레포 루트에서):
```
python -m src.stage2_wafermap.train --epochs 20 --loss asl
python -m src.stage2_wafermap.train --epochs 2 --subset 4000   # 스모크(통과확인됨)
```
산출물: `experiments/stage2_<loss>_w<width>/` (best.pt, test_report.txt, test_metrics.json).
스모크 결과(2ep/4k): mAP≈0.78, macroF1 낮음(에폭부족+임계0.5 미보정).

## 4. 설계 원칙 (버스바 교훈 계승)
- **leak-free**: split seed 고정, 임계값은 val에서만 보정(평가셋 누수 금지).
- **per-class 분석 필수**: 희귀 NF·Random 따로 추적. 정확도 단일지표 금지(불균형 함정).
- **불균형 정면대응**: pos_weight / ASL.
- 과대포장 금지(특히 Meruva 양성 7건은 지도학습 불가 → 비지도 AD 데모로만).

## 5. git
- 레포 `g:/wafer-defect-suite` 에 `git init` 완료, 커밋 2개(main). 데이터/experiments 는 `.gitignore`.
- 사용자가 GitHub 빈 repo 생성 후 직접:
  `git remote add origin https://github.com/maxwell779/<repo>.git && git push -u origin main`
  (이 환경은 push 인증 막힘 → 사용자가 실행)

## 6. 다음 작업 (우선순위)
1. **스테이지2 본학습**: `--loss asl --epochs 20` → per-class 수치 확보 → **클래스별 임계값 val 보정**(leak-free) 추가
2. **합성→실제 전이**: WM-811K 단일결함 맵(가변→리사이즈)으로 MixedWM38 학습모델 평가 스크립트
3. **자기지도**: WM-811K 63.8만 미라벨로 대조학습/MAE 사전학습 → 희귀클래스 boost
4. 스테이지1(Meruva: IsolationForest/AE/Mahalanobis + 변수기여), 스테이지3(ELLIMAC 라벨정제→YOLO11, bestV2.pt 벤치마크)

## 7. 직전 멈춘 지점
스테이지2 베이스라인 골격 완성·스모크 통과. **다음 행동 후보 = 스테이지2 본학습(ASL 20epoch) 실행해 첫 정식 per-class 수치 뽑기.**

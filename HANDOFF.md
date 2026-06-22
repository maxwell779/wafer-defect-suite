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

## 7. 진행 결과 (수치)
- **합성(MixedWM38) 학습** `train.py --loss asl`: test macro-F1 **0.985**, mAP 0.999 (합성이라 쉬움)
- **합성→실제 전이** `transfer_eval.py`: macro-F1 **0.364**, 정상 오탐 0.957 → **도메인 갭 규명**
  (Edge-Ring/Near-full 전이됨, Center/Loc/Scratch/Random 붕괴 → Random으로 오인)
- **실데이터(WM-811K) 학습** `train_real.py --loss asl` (lot-split leak-free): test macro-F1 **0.859**,
  mAP 0.930, exact 0.866. (Loc 0.72·Scratch 0.78 최난, Edge-Ring 0.99) ← **현재 메인 벤치마크**

## 8. 직전 멈춘 지점 / 다음 할 일
- **A (진단) 완료** `train.py --augment` + transfer_eval: 증강 합성모델 전이 macro-F1 **0.311**(기본 0.364↓).
  → "노이즈가 원인" 가설 **기각**, 갭은 구조적. 싼 증강으론 못 메움(=B/C 필요 입증).
- **C (자기지도) 완료 — negative result**: SimCLR(150k 미라벨, 30ep) NT-Xent loss 4.98 정체(약한 증강→부분붕괴).
  파인튜닝: 전체 0.869(+0.01 미미), 저라벨10% 0.665 vs scratch 0.684(오히려↓). → 나이브 SSL 효과없음.
  개선여지: 강한 증강(cutout/erase)·MAE·큰 백본·더 긴 학습. **Stage2 확정 = 실데이터 0.859(B).**
- **Stage1 완료** `src/stage1_process/run.py`: ML vs DL 이상탐지. LOF 최고(PR-AUC 0.341, recall@100=1.0) > AutoEncoder(0.211). "소표본엔 고전ML". 변수기여 pressure↓·etch↑·temp↑.
- **Stage3 완료**: B(메인) `stage3_localization/gradcam.py` 실데이터 Grad-CAM 위치탐지(합성 무관). A(부록) `stage3_detection/benchmark.py` ELLIMAC 폴리곤→bbox 정제+cls6제거, bestV2 test mAP@0.5=0.739(합성 명시).
- **3스테이지 코드 전부 완료.**
- **웹 데모 완료** `web/` 정식 React(Vite) 앱: 5화면 기능형(Dashboard/Stage1 테이블·관리도·추천·ML vs DL/Stage2 갤러리·판정·Grad-CAM히트맵·합성vs실토글/Stage3 Grad-CAM위치+ELLIMAC검출/Experiments). 실데이터JSON+이미지자산 포함, 의존성없는 SVG차트. 실행 `cd web && npm install && npm run dev`(node 필요, 서버엔 미설치→로컬/CI에서).
- **종합 README 완료**(루트), web vite build 검증 통과.
- **FastAPI 백엔드 완료** `backend/`: Stage1 LOF score / Stage2 WaferCNN predict·sample·gradcam / Stage3 YOLO detect.
  엔드포인트 curl 검증 완료(실모델 Center 0.857 vs 합성 Edge-Loc 0.882=전이실패 라이브). 웹 LIVE 모드 연동(api.js, 폴백 유지).
  실행: `python -m backend.prep_samples` → `uvicorn backend.main:app --port 8000`. samples.npz·pt·data는 git 제외.
- (다음) GitHub push, 선택: 배포(GitHub Pages/서버), 스크린샷.
- **PRD/웹 보강 완료**: docs/PRD.md, docs/web_design_prompt.md(v1), docs/web_design_prompt_v2.md(기능형), wafer_web_design.zip(실이미지+데이터)

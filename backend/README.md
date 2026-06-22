# Backend — 라이브 추론 API (FastAPI)

학습된 실제 모델로 추론을 서빙. 웹(`web/`)이 LIVE 모드로 호출.

## 실행
```bash
# 레포 루트에서
python -m backend.prep_samples            # 최초 1회: samples.npz 생성(2GB pkl 1회 로드)
uvicorn backend.main:app --port 8000      # 서버 (http://localhost:8000)
```

## 엔드포인트
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 상태·device |
| POST | `/api/stage1/score` | 공정 6파라미터 → 이상점수·변수z·조정추천 (LOF) |
| POST | `/api/stage2/predict` | 52×52 맵 → real·synth 모델 멀티라벨 예측 |
| GET | `/api/stage2/sample/{class}` | 실데이터 샘플 맵 + 예측 + Grad-CAM(52×52) |
| POST | `/api/stage3/detect` | 이미지 업로드 → YOLO 박스 |
| GET | `/api/stage3/detect_sample/{idx}` | ELLIMAC test 샘플 검출 |
| GET | `/api/metrics`·`/process_runs`·`/wafermaps` | 정적 데이터 |

## 모델 출처
- Stage1: Meruva CSV로 LOF 즉시 fit
- Stage2: `experiments/stage2_real_asl/best.pt`(실) + `stage2_asl_w32/best.pt`(합성)
- Stage3: `data/.../ELLIMAC/model/Model/bestV2.pt`

> 모델 가중치(`experiments/*.pt`)·데이터는 git 미포함 — 학습 스크립트로 재생성 후 구동.

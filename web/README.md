# Wafer Defect Console (web)

`wafer-defect-suite` 3-스테이지 결과를 보여주는 React(Vite) 데모 콘솔.

## 실행
```bash
cd web
npm install
npm run dev      # 개발 서버 (http://localhost:5173)
npm run build    # 정적 빌드 → dist/
npm run preview  # 빌드 결과 미리보기
```

## 화면
- **Dashboard** — 3-스테이지 개요 + KPI
- **Stage 1 · 공정 모니터** — Run 테이블·관리도(UCL/LCL)·파라미터 조정 추천·변수중요도·ML vs DL 표
- **Stage 2 · 웨이퍼맵** — 갤러리·멀티라벨 판정·Grad-CAM 위치 히트맵·합성 vs 실데이터 토글
- **Stage 3 · 결함 검출/위치** — Grad-CAM 위치탐지(실데이터, 메인) + ELLIMAC YOLO 검출(합성, 부록)
- **Experiments** — 학습전략 추이·per-class F1·혼동행렬

## 데이터
- `src/appdata/*.json` — 학습 파이프라인 산출(process_runs, stage1_results, wafermaps, stage2_metrics)
- `public/assets/*` — 웨이퍼맵·Grad-CAM·ELLIMAC 실제 이미지

현재는 사전계산 결과(mock) 기반 데모. 라이브 추론은 FastAPI 백엔드 연결 시 확장 가능.

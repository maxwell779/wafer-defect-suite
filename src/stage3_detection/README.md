# Stage 3 — 결함 위치 검출 (ELLIMAC, YOLO)

**상태: 예정 (placeholder)**

## 문제
웨이퍼 이미지에서 결함 패턴의 **위치(bounding box)** 를 검출.

## EDA 발견
- 4,531장 (train 3172 / valid 907 / test 452), 클래스 6 = `Center, Donut, Edge-Loc, Edge-Ring, Loc, Scratch`
- 박스/이미지: 대부분 1개, 일부 2~4개(혼합)
- 이미지 크기 제각각 (742×576, 640×480, 224×224 …) — Roboflow 합성 혼재
- ★ **라벨 노이즈**: `cls6`(yaml에 없는 7번째 id)가 **108개 파일에서 폴리곤(세그) 포맷**으로 존재 → 정제 필요
- 학습된 **`bestV2.pt` (52MB, YOLO)** 포함 → 재현/벤치마크 기준선

## 계획
- 라벨 정제(cls6 폴리곤 제거 또는 클래스 매핑) → 깨끗한 YOLO 데이터셋
- `bestV2.pt` 재현 평가(mAP@0.5) → 재학습(YOLO11) 비교
- 웨이퍼맵(Stage 2)과 연계: "패턴 분류 → 위치 검출" 흐름

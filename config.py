"""Central paths & constants for wafer-defect-suite.

데이터는 busbar 프로젝트 폴더에 있으므로 절대경로로 참조한다.
(레포는 g:/wafer-defect-suite, 데이터는 g:/pro-vision/data/wafer_datasets)
"""
from pathlib import Path

# ── 데이터 루트 (외부 참조) ───────────────────────────────────────────
DATA_ROOT = Path("g:/pro-vision/data/wafer_datasets")

# Stage 2 — 웨이퍼맵 (합성 멀티라벨 + 실제)
MIXEDWM38_NPZ = Path("g:/pro-vision/data/Wafer_Map_Datasets.npz")   # (38015,52,52)+(38015,8)
LSWMD_PKL     = DATA_ROOT / "LSWMD/LSWMD.pkl"                         # WM-811K 실데이터 811k

# Stage 1 — 공정 센서 (테이블)
MERUVA_CSV    = DATA_ROOT / "meruva_csv/semiconductor_wafer_defect_dataset.csv"

# Stage 3 — 결함 검출 (YOLO)
ELLIMAC_DIR   = DATA_ROOT / "ELLIMAC"

# ── 산출물 ────────────────────────────────────────────────────────────
EXPERIMENTS = Path(__file__).parent / "experiments"
EXPERIMENTS.mkdir(exist_ok=True)

# ── 도메인 상수 ───────────────────────────────────────────────────────
# MixedWM38 라벨 8열 순서 (희귀 NF=149 / Random=866 위치로 검증됨)
WM_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]

# EDA에서 확인된 물리적 양립불가 쌍 (동시발생=0) — 모델 사후보정/사전지식에 활용 가능
MUTUALLY_EXCLUSIVE = [("Center", "Donut"), ("Edge-Loc", "Edge-Ring")]

SEED = 42

"""Central paths & constants for wafer-defect-suite.

데이터는 레포 내부 data/ 에 둔다 (git 에는 올리지 않음 — .gitignore 의 data/).
경로는 레포 위치에 무관하게 동작하도록 __file__ 기준 상대경로.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"

# ── 데이터 ────────────────────────────────────────────────────────────
DATA_ROOT = DATA / "wafer_datasets"

# Stage 2 — 웨이퍼맵 (합성 멀티라벨 + 실제)
MIXEDWM38_NPZ = DATA / "Wafer_Map_Datasets.npz"          # (38015,52,52)+(38015,8)
LSWMD_PKL     = DATA_ROOT / "LSWMD/LSWMD.pkl"            # WM-811K 실데이터 811k

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

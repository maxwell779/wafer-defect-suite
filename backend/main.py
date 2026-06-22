"""FastAPI 라이브 추론 백엔드 — wafer-defect-suite.

실행 (레포 루트):
    python -m backend.prep_samples          # 최초 1회 (samples.npz 생성)
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations
import sys, json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from backend import models

app = FastAPI(title="Wafer Defect Suite API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

APPDATA = Path(__file__).resolve().parents[1] / "web/src/appdata"


def _json(name):
    return json.load(open(APPDATA / name, encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"ok": True, "device": ("cuda" if __import__("torch").cuda.is_available() else "cpu")}


# ── 정적 데이터 ───────────────────────────────────────────────────────
@app.get("/api/metrics")
def metrics(): return _json("stage2_metrics.json")
@app.get("/api/process_runs")
def process_runs(): return _json("process_runs.json")
@app.get("/api/wafermaps")
def wafermaps(): return _json("wafermaps.json")


# ── Stage 1 ───────────────────────────────────────────────────────────
class ProcParams(BaseModel):
    temperature_c: float; pressure_torr: float; gas_flow_sccm: float
    etch_rate_nm_min: float; voltage_v: float; current_ma: float

@app.post("/api/stage1/score")
def stage1_score(p: ProcParams):
    return models.stage1().score(p.model_dump())


# ── Stage 2 ───────────────────────────────────────────────────────────
class WaferMap(BaseModel):
    wmap: list[list[int]]            # 52x52 {0,1,2}

@app.post("/api/stage2/predict")
def stage2_predict(m: WaferMap):
    return models.stage2().predict(m.wmap)

@app.get("/api/stage2/sample/{cls_name}")
def stage2_sample(cls_name: str):
    return models.stage2().sample(cls_name)


# ── Stage 3 ───────────────────────────────────────────────────────────
@app.post("/api/stage3/detect")
async def stage3_detect(file: UploadFile = File(...), conf: float = 0.25):
    import tempfile, os
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(data); tmp.close()
    try:
        return {"boxes": models.stage3().detect(tmp.name, conf=conf)}
    finally:
        os.unlink(tmp.name)

@app.get("/api/stage3/detect_sample/{idx}")
def stage3_detect_sample(idx: int, conf: float = 0.25):
    img = config.ELLIMAC_DIR / "images/images/test"
    jpgs = sorted(img.glob("*.jpg"))
    p = jpgs[idx % len(jpgs)]
    return {"image": p.name, "boxes": models.stage3().detect(str(p), conf=conf)}

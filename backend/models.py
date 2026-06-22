"""모델 로딩/추론 (Stage1 sklearn · Stage2 WaferCNN · Stage3 YOLO).
무거운 모델은 lazy 싱글톤으로 첫 호출 시 로드.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.stage2_wafermap.model import build_model
from src.stage3_localization.gradcam import gradcam as cam_fn

CLS = config.WM_CLASSES
FEATS = ["temperature_c", "pressure_torr", "gas_flow_sccm", "etch_rate_nm_min", "voltage_v", "current_ma"]
SAMPLES = Path(__file__).resolve().parent / "samples.npz"


def _eng(X):
    """6 raw → 11 (base + 비율·전력·교호) — Stage1 best 피처(결함=센서 조합)."""
    t, p, g, e, v, c = X.T
    return np.c_[X, p / (e + 1e-6), t / (p + 1e-6), e * t, v * c, g / (p + 1e-6)]


# ── Stage 1 (sklearn — 교호작용 피처 + Mahalanobis, best) ──────────────
class Stage1:
    def __init__(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.covariance import EllipticEnvelope
        df = pd.read_csv(config.MERUVA_CSV)
        X = df[FEATS].to_numpy(float); y = df["defect_label"].to_numpy(int)
        Xn = X[y == 0]
        self.mean, self.std = Xn.mean(0), Xn.std(0)            # recommendations용(raw)
        self.sc = StandardScaler().fit(_eng(Xn))               # 교호작용 피처 스케일
        self.ee = EllipticEnvelope(contamination=0.05, random_state=config.SEED).fit(self.sc.transform(_eng(Xn)))
        s = -self.ee.score_samples(self.sc.transform(_eng(X)))
        self.smin, self.smax = float(s.min()), float(s.max())

    def score(self, params: dict):
        x = np.array([[float(params[f]) for f in FEATS]])
        s = float(-self.ee.score_samples(self.sc.transform(_eng(x)))[0])
        norm = (s - self.smin) / (self.smax - self.smin + 1e-9)
        z = {f: float((x[0, i] - self.mean[i]) / (self.std[i] + 1e-9)) for i, f in enumerate(FEATS)}
        recs = []
        for i, f in enumerate(FEATS):
            if abs(z[f]) > 1.2:
                d = float(self.mean[i] - x[0, i])
                recs.append({"feature": f, "z": round(z[f], 2), "delta": round(d, 1),
                             "msg": f"정상 평균 {self.mean[i]:.0f} 대비 {'높음' if z[f] > 0 else '낮음'} → {d:+.0f} {'하향' if z[f] > 0 else '상향'} 권장"})
        return {"anomaly_score": round(min(max(norm, 0.0), 1.0), 3),
                "z": {k: round(v, 2) for k, v in z.items()}, "recommendations": recs}


# ── Stage 2 (WaferCNN real + synth) ──────────────────────────────────
class Stage2:
    def __init__(self):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 실모델 = SE-ResNet 앙상블(best 0.935); 없으면 CNN 폴백
        ens = [("stage2_real_asl_resnet", 48), ("stage2_real_asl_resnet2", 48), ("stage2_real_asl_resnet3", 48),
               ("stage2_real_asl_r64s42", 64), ("stage2_real_asl_r64s2", 64), ("stage2_real_asl_r64s3", 64)]
        self.real_models = []
        for d, w in ens:
            p = config.EXPERIMENTS / d / "best.pt"
            if p.exists():
                self.real_models.append(self._load(p, "resnet", w))
        if not self.real_models:                               # 폴백
            self.real_models = [self._load(config.EXPERIMENTS / "stage2_real_asl_w64aug/best.pt", "cnn", 64)]
        self.real_layer = getattr(self.real_models[0], "layers", [None])[-1]  # CAM용
        self.synth = self._load(config.EXPERIMENTS / "stage2_asl_w32/best.pt", "cnn", 32)
        self.samples = np.load(SAMPLES) if SAMPLES.exists() else None

    def _load(self, p, arch, width):
        m = build_model(arch, 3, len(CLS), width).to(self.device).eval()
        m.load_state_dict(self.torch.load(p, map_location=self.device))
        return m

    def _x(self, wmap):
        m = np.asarray(wmap).astype(np.int64)
        oh = np.stack([(m == 0), (m == 1), (m == 2)], 0).astype(np.float32)
        return self.torch.from_numpy(oh)[None].to(self.device)

    def predict(self, wmap):
        x = self._x(wmap)
        with self.torch.no_grad():
            pr = np.mean([self.torch.sigmoid(m(x))[0].cpu().numpy() for m in self.real_models], axis=0)
            ps = self.torch.sigmoid(self.synth(x))[0].cpu().numpy()
        return {"classes": CLS, "pred_real": [round(float(v), 3) for v in pr],
                "pred_synth": [round(float(v), 3) for v in ps]}

    def gradcam(self, wmap, cls_idx):
        cam = cam_fn(self.real_models[0], self._x(wmap), int(cls_idx), self.real_layer)  # 52x52 [0,1]
        return cam.round(3).tolist()

    def sample(self, cls_name):
        if self.samples is None:
            raise FileNotFoundError("samples.npz 없음 — python -m backend.prep_samples 먼저 실행")
        ci = CLS.index(cls_name)
        idx = np.where(self.samples["real_lbl"] == ci)[0]
        m = self.samples["real_maps"][idx[0]] if len(idx) else self.samples["real_maps"][0]
        out = self.predict(m)
        out["wafermap"] = m.astype(int).tolist()
        out["gradcam"] = self.gradcam(m, ci)
        out["true_class"] = cls_name
        return out


# ── Stage 3 (YOLO) ───────────────────────────────────────────────────
class Stage3:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(str(config.ELLIMAC_DIR / "model/Model/bestV2.pt"))

    def detect(self, source, conf=0.25):
        r = self.model.predict(source, conf=conf, verbose=False)[0]
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxyn[0].tolist()
            boxes.append({"x": round(x1, 4), "y": round(y1, 4), "w": round(x2 - x1, 4),
                          "h": round(y2 - y1, 4), "cls": self.model.names[int(b.cls)],
                          "conf": round(float(b.conf), 3)})
        return boxes


# lazy 싱글톤
_S = {}
def stage1(): return _S.setdefault("s1", Stage1())
def stage2(): return _S.setdefault("s2", Stage2())
def stage3(): return _S.setdefault("s3", Stage3())

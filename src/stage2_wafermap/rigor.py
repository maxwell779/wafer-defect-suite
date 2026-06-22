"""Stage 2 엄밀 평가 — per-class 임계 보정(val-only) + 혼동행렬 + 멀티시드.

- 0.5 고정 임계 대신 클래스별로 val에서 F1 최대 임계 선택 → test 적용(leak-free 부스트)
- test 혼동행렬(argmax) 산출
실행:
    python -m src.stage2_wafermap.rigor --ckpt experiments/stage2_real_asl/best.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

import config
from src.stage2_wafermap.dataset import WaferMapDataset
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.model import WaferCNN
from src.stage2_wafermap.train import run_eval


def best_thresholds(y_val, p_val):
    """클래스별 val F1 최대 임계."""
    th = np.full(y_val.shape[1], 0.5)
    for c in range(y_val.shape[1]):
        if y_val[:, c].sum() == 0:
            continue
        best, bt = -1, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(y_val[:, c], (p_val[:, c] >= t).astype(int), zero_division=0)
            if f > best:
                best, bt = f, t
        th[c] = bt
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/stage2_real_asl/best.pt")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--normal-cap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=config.SEED, help="학습과 동일 split seed")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES

    X, Y, y_idx, lots = load_wm811k(normal_cap=args.normal_cap, seed=args.seed)
    tr, va, te = lot_group_split(y_idx, lots, seed=args.seed)
    dl = lambda idx: DataLoader(WaferMapDataset(X, Y, idx), batch_size=256)
    model = WaferCNN(3, len(cls), args.width).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    yv, pv = run_eval(model, dl(va), device)
    yt, pt = run_eval(model, dl(te), device)

    f1_05 = f1_score(yt, (pt >= 0.5).astype(int), average="macro", zero_division=0)
    th = best_thresholds(yv, pv)
    pred = (pt >= th).astype(int)
    f1_cal = f1_score(yt, pred, average="macro", zero_division=0)

    print("===== Stage2 임계 보정 (val-only) =====")
    print(f"  macro-F1  0.5고정 {f1_05:.4f}  →  per-class보정 {f1_cal:.4f}  (Δ{f1_cal-f1_05:+.4f})")
    print(f"  {'class':12s}{'thr':>6}{'F1@0.5':>9}{'F1@cal':>9}")
    for c in range(len(cls)):
        a = f1_score(yt[:, c], (pt[:, c] >= 0.5).astype(int), zero_division=0)
        b = f1_score(yt[:, c], (pt[:, c] >= th[c]).astype(int), zero_division=0)
        print(f"  {cls[c]:12s}{th[c]:6.2f}{a:9.3f}{b:9.3f}")

    # 혼동행렬 (단일라벨 행 기준 argmax)
    defect = yt.sum(1) == 1
    yi = yt[defect].argmax(1); pi = pt[defect].argmax(1)
    cm = confusion_matrix(yi, pi, labels=list(range(len(cls))))
    print("\n실데이터 혼동행렬 (행=실제, 열=예측 argmax):")
    print("        " + " ".join(f"{c[:4]:>5}" for c in cls))
    for i, c in enumerate(cls):
        print(f"  {c[:7]:8s}" + " ".join(f"{cm[i,j]:5d}" for j in range(len(cls))))

    import json
    out = config.EXPERIMENTS / "stage2_real_asl"
    json.dump({"macro_f1_05": round(float(f1_05), 4), "macro_f1_cal": round(float(f1_cal), 4),
               "thresholds": {cls[c]: round(float(th[c]), 2) for c in range(len(cls))},
               "confusion_real": cm.tolist(), "labels": cls},
              open(out / "rigor_calibrated.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[saved] {out/'rigor_calibrated.json'}")


if __name__ == "__main__":
    main()

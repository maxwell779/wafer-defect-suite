"""Stage 2 앙상블 평가 — 동일 split(seed42)에서 여러 모델 확률 평균 + 임계보정.

멤버: CNN 3 init + SE-ResNet. test에서 단일모델 vs 앙상블 macro-F1 비교.
실행:  python -m src.stage2_wafermap.ensemble_eval
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import config
from src.stage2_wafermap.dataset import WaferMapDataset
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.train import run_eval
from src.stage2_wafermap.rigor import best_thresholds

# (경로, arch, width) — 모두 split seed=42 에서 학습. SE-ResNet 패밀리 앙상블(최강).
MEMBERS = [
    ("experiments/stage2_real_asl_resnet/best.pt", "resnet", 48),
    ("experiments/stage2_real_asl_resnet2/best.pt", "resnet", 48),
    ("experiments/stage2_real_asl_resnet3/best.pt", "resnet", 48),
]


def probs(path, arch, width, dl, device):
    m = build_model(arch, 3, len(config.WM_CLASSES), width).to(device).eval()
    m.load_state_dict(torch.load(path, map_location=device))
    y, p = run_eval(m, dl, device)
    return y, p


def macro(y, p, th=0.5):
    return f1_score(y, (p >= th).astype(int), average="macro", zero_division=0)


def main():
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES
    X, Y, y_idx, lots = load_wm811k(normal_cap=10000, seed=config.SEED)
    tr, va, te = lot_group_split(y_idx, lots, seed=config.SEED)
    dlv = DataLoader(WaferMapDataset(X, Y, va), batch_size=256)
    dlt = DataLoader(WaferMapDataset(X, Y, te), batch_size=256)

    members = [m for m in MEMBERS if os.path.exists(m[0])]
    pv_list, pt_list = [], []
    print("===== 단일 모델 (split42 test) =====")
    for path, arch, width in members:
        yv, pv = probs(path, arch, width, dlv, device)
        yt, pt = probs(path, arch, width, dlt, device)
        pv_list.append(pv); pt_list.append(pt)
        th = best_thresholds(yv, pv)
        print(f"  {arch:6s} {os.path.basename(os.path.dirname(path)):28s} "
              f"macroF1 0.5={macro(yt, pt):.4f}  보정={f1_score(yt,(pt>=th).astype(int),average='macro',zero_division=0):.4f}")

    pv_ens = np.mean(pv_list, axis=0); pt_ens = np.mean(pt_list, axis=0)
    yt = yt  # 동일 split이라 y 동일
    f05 = macro(yt, pt_ens)
    th = best_thresholds(yv, pv_ens)
    fcal = f1_score(yt, (pt_ens >= th).astype(int), average="macro", zero_division=0)
    print(f"\n===== 앙상블({len(members)}모델) =====")
    print(f"  macro-F1  0.5={f05:.4f}  →  per-class보정={fcal:.4f}")
    for c in range(len(cls)):
        print(f"    {cls[c]:12s} thr {th[c]:.2f}  F1 {f1_score(yt[:,c],(pt_ens[:,c]>=th[c]).astype(int),zero_division=0):.3f}")


if __name__ == "__main__":
    main()

"""Stage 2 앙상블 평가 — 다모델(SE-ResNet w48+w64) + TTA(회전/플립) + 임계보정.
동일 split(seed42)에서 확률 평균. 웨이퍼 패턴은 회전불변이라 TTA 유효.
실행:  python -m src.stage2_wafermap.ensemble_eval [--tta]
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
from sklearn.metrics import f1_score

import config
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.rigor import best_thresholds

CLS = config.WM_CLASSES
MEMBERS = [
    ("experiments/stage2_real_asl_resnet/best.pt", "resnet", 48),
    ("experiments/stage2_real_asl_resnet2/best.pt", "resnet", 48),
    ("experiments/stage2_real_asl_resnet3/best.pt", "resnet", 48),
    ("experiments/stage2_real_asl_r64s42/best.pt", "resnet", 64),
    ("experiments/stage2_real_asl_r64s2/best.pt", "resnet", 64),
    ("experiments/stage2_real_asl_r64s3/best.pt", "resnet", 64),
]


def onehot(maps):
    return np.stack([(maps == 0), (maps == 1), (maps == 2)], axis=1).astype(np.float32)


def predict(model, maps, device, tta, bs=512):
    tfs = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    acc = np.zeros((len(maps), len(CLS)), dtype=np.float64)
    for k, flip in tfs:
        mm = np.rot90(maps, k, axes=(1, 2))
        if flip:
            mm = mm[:, :, ::-1]
        mm = np.ascontiguousarray(mm)
        for i in range(0, len(mm), bs):
            x = torch.from_numpy(onehot(mm[i:i + bs])).to(device)
            with torch.no_grad():
                acc[i:i + bs] += torch.sigmoid(model(x)).cpu().numpy()
    return acc / len(tfs)


def macro(y, p, th=0.5):
    return f1_score(y, (p >= th).astype(int), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", action="store_true", help="회전/플립 TTA(8배)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y, y_idx, lots = load_wm811k(normal_cap=10000, seed=config.SEED)
    tr, va, te = lot_group_split(y_idx, lots, seed=config.SEED)
    Xv, Yv, Xt, Yt = X[va], Y[va], X[te], Y[te]

    members = [m for m in MEMBERS if os.path.exists(m[0])]
    print(f"멤버 {len(members)}개 | TTA={'ON' if args.tta else 'OFF'}")
    pv, pt = [], []
    for path, arch, width in members:
        m = build_model(arch, 3, len(CLS), width).to(device).eval()
        m.load_state_dict(torch.load(path, map_location=device))
        pvi, pti = predict(m, Xv, device, args.tta), predict(m, Xt, device, args.tta)
        pv.append(pvi); pt.append(pti)
        th = best_thresholds(Yv, pvi)
        print(f"  {os.path.basename(os.path.dirname(path)):26s} w{width} "
              f"0.5={macro(Yt, pti):.4f} 보정={f1_score(Yt,(pti>=th).astype(int),average='macro',zero_division=0):.4f}")

    pv_e, pt_e = np.mean(pv, 0), np.mean(pt, 0)
    th = best_thresholds(Yv, pv_e)
    f05, fcal = macro(Yt, pt_e), f1_score(Yt, (pt_e >= th).astype(int), average="macro", zero_division=0)
    print(f"\n앙상블({len(members)}) {'+TTA' if args.tta else ''}: macro-F1 0.5={f05:.4f} → 보정={fcal:.4f}")
    for c in range(len(CLS)):
        print(f"    {CLS[c]:12s} {f1_score(Yt[:,c],(pt_e[:,c]>=th[c]).astype(int),zero_division=0):.3f}")


if __name__ == "__main__":
    main()

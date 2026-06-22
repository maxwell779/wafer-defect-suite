"""저장된 체크포인트로 테스트셋 재평가 (per-class 리포트).

    python -m src.stage2_wafermap.evaluate --ckpt experiments/stage2_bce_w32/best.pt
"""
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

import config
from src.common.metrics import multilabel_report, format_report
from src.stage2_wafermap.dataset import load_mixedwm38, make_splits, WaferMapDataset
from src.stage2_wafermap.model import WaferCNN
from src.stage2_wafermap.train import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES
    X, Y = load_mixedwm38(config.MIXEDWM38_NPZ)
    _, _, te = make_splits(Y, seed=config.SEED)
    te_dl = DataLoader(WaferMapDataset(X, Y, te), batch_size=args.batch)

    model = WaferCNN(in_ch=3, n_classes=len(cls), width=args.width).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    gts, probs = run_eval(model, te_dl, device)
    summ, per = multilabel_report(gts, probs, cls, thresh=args.thresh)
    print(format_report(summ, per))


if __name__ == "__main__":
    main()

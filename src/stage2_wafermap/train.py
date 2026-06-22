"""Stage 2 — MixedWM38 멀티라벨 분류 베이스라인 학습.

실행 (레포 루트 g:/wafer-defect-suite 에서):
    python -m src.stage2_wafermap.train --epochs 20
    python -m src.stage2_wafermap.train --epochs 2 --subset 4000   # 스모크
    python -m src.stage2_wafermap.train --loss asl                 # 불균형 손실
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.common.seed import set_seed
from src.common.metrics import multilabel_report, format_report
from src.stage2_wafermap.dataset import (
    load_mixedwm38, make_splits, WaferMapDataset, pos_weight_from,
)
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.losses import build_loss


def run_eval(model, loader, device):
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy()
            probs.append(p); gts.append(y.numpy())
    return np.concatenate(gts), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["bce", "asl"], default="bce")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--arch", choices=["cnn","resnet","resnet_cbam"], default="cnn")
    ap.add_argument("--subset", type=int, default=0, help=">0 이면 빠른 스모크용 일부만")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--augment", action="store_true", help="실제모사 증강(노이즈+회전) — A 진단실험")
    args = ap.parse_args()

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES
    out = config.EXPERIMENTS / f"stage2_{args.arch}_{args.loss}_w{args.width}{'_aug' if args.augment else ''}"
    out.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────
    X, Y = load_mixedwm38(config.MIXEDWM38_NPZ)
    tr, va, te = make_splits(Y, seed=config.SEED)
    if args.subset:
        tr, va, te = tr[: args.subset], va[: args.subset // 4], te[: args.subset // 4]
    print(f"[data] train {len(tr)} | val {len(va)} | test {len(te)} | device {device}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh,
                                   num_workers=args.workers, pin_memory=(device == "cuda"))
    tr_dl = dl(WaferMapDataset(X, Y, tr, augment=args.augment, seed=config.SEED), True)
    va_dl = dl(WaferMapDataset(X, Y, va), False)
    te_dl = dl(WaferMapDataset(X, Y, te), False)

    # ── model / loss / optim ──────────────────────────────────────────
    model = build_model(args.arch, in_ch=3, n_classes=len(cls), width=args.width).to(device)
    pw = pos_weight_from(Y, tr).to(device) if args.loss == "bce" else None
    criterion = build_loss(args.loss, pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ── train loop ────────────────────────────────────────────────────
    best_macro, best_path = -1.0, out / "best.pt"
    for ep in range(1, args.epochs + 1):
        model.train(); t = time.time(); tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        gts, probs = run_eval(model, va_dl, device)
        summ, _ = multilabel_report(gts, probs, cls)
        print(f"[ep {ep:02d}] loss {tot/len(tr):.4f} | val macroF1 {summ['macro_f1']:.4f} "
              f"mAP {summ['mAP']:.4f} exact {summ['exact_match']:.4f} ({time.time()-t:.0f}s)")
        if summ["macro_f1"] > best_macro:
            best_macro = summ["macro_f1"]
            torch.save(model.state_dict(), best_path)

    # ── final test (best 가중치) ───────────────────────────────────────
    model.load_state_dict(torch.load(best_path))
    gts, probs = run_eval(model, te_dl, device)
    summ, per = multilabel_report(gts, probs, cls)
    report = format_report(summ, per)
    print("\n===== TEST (best val macro-F1) =====\n" + report)
    (out / "test_report.txt").write_text(report, encoding="utf-8")
    json.dump({"summary": summ, "per_class": per, "args": vars(args)},
              open(out / "test_metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

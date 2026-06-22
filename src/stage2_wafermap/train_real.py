"""Stage 2-B — WM-811K(실데이터) 학습 (lot 단위 leak-free).

합성(MixedWM38) 전이 실패를 극복하는 "실제 벤치마크".
모델/손실/지표는 train.py 와 동일, 데이터만 WM-811K + lot-split.

실행:
    python -m src.stage2_wafermap.train_real --epochs 25 --loss asl
    python -m src.stage2_wafermap.train_real --normal-cap 0   # 정상 전체 포함
"""
from __future__ import annotations
import argparse, json, time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.common.seed import set_seed
from src.common.metrics import multilabel_report, format_report
from src.stage2_wafermap.dataset import WaferMapDataset, pos_weight_from
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.model import WaferCNN
from src.stage2_wafermap.losses import build_loss
from src.stage2_wafermap.train import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["bce", "asl"], default="asl")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--normal-cap", type=int, default=10000, help="정상('none') 표본 상한(0=전체)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--init", default="", help="SSL 사전학습 encoder.pt 경로(features 초기화)")
    ap.add_argument("--label-frac", type=float, default=1.0, help="train 라벨 사용 비율(저라벨 실험)")
    args = ap.parse_args()

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES
    tag = ("_ssl" if args.init else "") + (f"_f{args.label_frac:g}" if args.label_frac < 1.0 else "")
    out = config.EXPERIMENTS / f"stage2_real_{args.loss}{tag}"
    out.mkdir(parents=True, exist_ok=True)

    # ── data (lot-split leak-free) ────────────────────────────────────
    print("[load] WM-811K + 52x52 리사이즈 ...")
    t = time.time()
    X, Y, y_idx, lots = load_wm811k(normal_cap=args.normal_cap, seed=config.SEED)
    tr, va, te = lot_group_split(y_idx, lots, seed=config.SEED)
    if args.label_frac < 1.0:                       # 저라벨 실험 (SSL 이득 부각)
        rng = np.random.default_rng(config.SEED)
        tr = rng.choice(tr, int(len(tr) * args.label_frac), replace=False)
    print(f"  maps {len(X)} | train {len(tr)} val {len(va)} test {len(te)} "
          f"| lots {len(set(lots))} | ({time.time()-t:.0f}s)  [lot-leak 검증 통과]")
    print(f"  class 분포(train): " + ", ".join(
        f"{cls[i]}:{int(Y[tr][:,i].sum())}" for i in range(len(cls))) +
        f", normal:{int((y_idx[tr]<0).sum())}")

    dl = lambda idx, sh: DataLoader(WaferMapDataset(X, Y, idx), batch_size=args.batch,
                                    shuffle=sh, num_workers=args.workers,
                                    pin_memory=(device == "cuda"))
    tr_dl, va_dl, te_dl = dl(tr, True), dl(va, False), dl(te, False)

    # ── model/loss/optim ──────────────────────────────────────────────
    model = WaferCNN(in_ch=3, n_classes=len(cls), width=args.width).to(device)
    if args.init:                                   # SSL 사전학습 encoder 로드
        model.features.load_state_dict(torch.load(args.init, map_location=device))
        print(f"  [init] SSL encoder 로드: {args.init}")
    pw = pos_weight_from(Y, tr).to(device) if args.loss == "bce" else None
    criterion = build_loss(args.loss, pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_macro, best_path = -1.0, out / "best.pt"
    for ep in range(1, args.epochs + 1):
        model.train(); t = time.time(); tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        gts, probs = run_eval(model, va_dl, device)
        summ, _ = multilabel_report(gts, probs, cls)
        print(f"[ep {ep:02d}] loss {tot/len(tr):.4f} | val macroF1 {summ['macro_f1']:.4f} "
              f"mAP {summ['mAP']:.4f} ({time.time()-t:.0f}s)")
        if summ["macro_f1"] > best_macro:
            best_macro = summ["macro_f1"]; torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path))
    gts, probs = run_eval(model, te_dl, device)
    summ, per = multilabel_report(gts, probs, cls)
    report = format_report(summ, per)
    print("\n===== TEST (WM-811K, lot-split) =====\n" + report)
    (out / "test_report.txt").write_text(report, encoding="utf-8")
    json.dump({"summary": summ, "per_class": per, "args": vars(args)},
              open(out / "test_metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

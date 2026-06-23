"""Stage2 라벨노이즈 정제 (confident learning, cleanlab multilabel).

가설: WM-811K 천장(0.935)의 주원인 = Loc↔Edge-Loc/Center 라벨모호성(노이즈).
절차:
  1) train을 lot-group 3-fold로 OOF 예측확률 산출 (누수 없이)
  2) cleanlab.multilabel 로 오라벨 후보 식별
  3) 오라벨 제거(정제) 후 재학습 → 동일 test로 baseline 대비 비교
출력: 오라벨 비율/클래스별 분포 + 정제 전후 macro-F1.

실행: python -m src.stage2_wafermap.label_audit --epochs 20 --frac-clean 1.0
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch
from torch.utils.data import DataLoader

import config
from src.common.seed import set_seed
from src.common.metrics import multilabel_report, format_report
from src.stage2_wafermap.dataset import WaferMapDataset, CH
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.losses import build_loss
from src.stage2_wafermap.train import run_eval


def train_one(X, Y, tr_idx, va_idx, cls, device, epochs, arch, width, loss, seed, augment=True):
    set_seed(seed)
    mk = lambda idx, sh, aug=False: DataLoader(
        WaferMapDataset(X, Y, idx, augment=aug, seed=seed), batch_size=256, shuffle=sh,
        num_workers=0, pin_memory=(device == "cuda"))
    tr_dl = mk(tr_idx, True, augment); va_dl = mk(va_idx, False)
    model = build_model(arch, in_ch=CH["onehot"], n_classes=len(cls), width=width).to(device)
    crit = build_loss(loss)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_sd = -1, None
    for ep in range(epochs):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_v = crit(model(x), y); loss_v.backward(); opt.step()
        sch.step()
        g, p = run_eval(model, va_dl, device)
        f = multilabel_report(g, p, cls)[0]["macro_f1"]
        if f > best: best, best_sd = f, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--arch", default="resnet"); ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--loss", default="asl"); ap.add_argument("--normal-cap", type=int, default=10000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES
    print("[load] WM-811K ...", flush=True)
    X, Y, y_idx, lots = load_wm811k(normal_cap=args.normal_cap, seed=config.SEED, size=52)
    tr, va, te = lot_group_split(y_idx, lots, seed=config.SEED)
    print(f"  train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    # ── 1) train 3-fold(lot) OOF 예측확률 ──────────────────────────────
    from sklearn.model_selection import GroupKFold
    lot_tr = np.array(lots)[tr]
    oof = np.zeros((len(tr), len(cls)), dtype=np.float32)
    gkf = GroupKFold(n_splits=3)
    for k, (a, b) in enumerate(gkf.split(tr, groups=lot_tr)):
        t0 = time.time()
        m = train_one(X, Y, tr[a], tr[b], cls, device, args.epochs, args.arch, args.width, args.loss, seed=k)
        dl = DataLoader(WaferMapDataset(X, Y, tr[b]), batch_size=256)
        _, p = run_eval(m, dl, device); oof[b] = p
        print(f"  [fold {k}] OOF 산출 ({time.time()-t0:.0f}s)", flush=True)

    # ── 2) cleanlab 멀티라벨 오라벨 식별 ───────────────────────────────
    from cleanlab.multilabel_classification.filter import find_label_issues
    labels = [list(np.where(Y[i] > 0.5)[0]) for i in tr]
    issues = find_label_issues(labels=labels, pred_probs=oof)   # bool mask
    n_iss = int(issues.sum())
    print(f"\n[cleanlab] 오라벨 후보 {n_iss}/{len(tr)} ({100*n_iss/len(tr):.1f}%)", flush=True)
    per_cls = {cls[c]: int(sum(1 for i in np.where(issues)[0] if Y[tr[i], c] > 0.5)) for c in range(len(cls))}
    print("  클래스별 오라벨:", per_cls, flush=True)

    # ── 3) 정제(오라벨 제거) 후 재학습 vs baseline ─────────────────────
    te_dl = DataLoader(WaferMapDataset(X, Y, te), batch_size=256)
    print("\n[baseline] 전체 train 학습 ...", flush=True)
    mb = train_one(X, Y, tr, va, cls, device, args.epochs, args.arch, args.width, args.loss, seed=0)
    gb, pb = run_eval(mb, te_dl, device); sb = multilabel_report(gb, pb, cls)[0]
    clean_tr = tr[~issues]
    print(f"[clean] 정제 train({len(clean_tr)}) 학습 ...", flush=True)
    mc = train_one(X, Y, clean_tr, va, cls, device, args.epochs, args.arch, args.width, args.loss, seed=0)
    gc, pc = run_eval(mc, te_dl, device); sc = multilabel_report(gc, pc, cls)[0]

    print("\n===== 라벨정제 효과 (동일 test) =====", flush=True)
    print(f"  baseline macro-F1 {sb['macro_f1']:.4f} | mAP {sb['mAP']:.4f}", flush=True)
    print(f"  cleaned  macro-F1 {sc['macro_f1']:.4f} | mAP {sc['mAP']:.4f}", flush=True)
    print(f"  Δ macro-F1 {sc['macro_f1']-sb['macro_f1']:+.4f}", flush=True)
    out = config.EXPERIMENTS / "stage2_label_audit"; out.mkdir(parents=True, exist_ok=True)
    json.dump({"n_issues": n_iss, "frac": n_iss/len(tr), "per_class": per_cls,
               "baseline": sb, "cleaned": sc}, open(out/"audit.json", "w"), indent=2, default=float)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()

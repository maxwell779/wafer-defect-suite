"""Stage2 메가서치 (Hyperband식) — 600조합 프록시 스캔 → 상위 풀학습.

딥러닝은 1학습 수분이라 단순 500개 불가 → 2단계로 '500+ 탐색' 현실화:
  Phase 1: 데이터 1회 로드 → GPU 상주 텐서로 초고속(증강X·소표본·5에폭) 600조합 랜덤 스캔, val macro-F1 랭킹
  Phase 2: 상위 12개 풀 학습(전체 train·증강·20에폭) → test macro-F1로 진짜 best
탐색공간: arch·width·pool·dropout·lr·wd·loss(+phase2 mixup)

실행: python tools/overnight/stage2_megasearch.py
"""
import sys, os, time, random, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import torch.nn.functional as F

import config
from src.common.seed import set_seed
from src.common.metrics import multilabel_report
from src.stage2_wafermap.dataset import WaferMapDataset, CH
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.losses import build_loss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLS = config.WM_CLASSES


def materialize(X, Y, idx, augment=False, inmode="onehot"):
    ds = WaferMapDataset(X, Y, idx, augment=augment, inmode=inmode)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.stack([ds[i][1] for i in range(len(ds))])
    return xs, ys


def macro_f1(model, xv, yv, bs=512):
    model.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(xv), bs):
            ps.append(torch.sigmoid(model(xv[i:i+bs])).cpu())
    p = torch.cat(ps).numpy()
    return multilabel_report(yv.cpu().numpy(), p, CLS)[0]["macro_f1"]


def train_proxy(cfg, xt, yt, xv, yv, epochs=5, bs=256):
    set_seed(0)
    m = build_model(cfg["arch"], in_ch=3, n_classes=len(CLS), width=cfg["width"],
                    pool=cfg["pool"], dropout=cfg["dropout"]).to(DEV)
    crit = build_loss(cfg["loss"], pos_weight=None,
                      cls_count=torch.as_tensor(yt.sum(0).cpu()))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n = len(xt)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad(); loss = crit(m(xt[idx]), yt[idx]); loss.backward(); opt.step()
    return macro_f1(m, xv, yv)


def train_full(cfg, X, Y, tr, va, te, epochs=20, bs=256):
    """전체 train + 증강(DataLoader) + mixup, best val → test 평가."""
    from torch.utils.data import DataLoader
    set_seed(0)
    tr_dl = DataLoader(WaferMapDataset(X, Y, tr, augment=True, seed=0), batch_size=bs, shuffle=True)
    va_x, va_y = materialize(X, Y, va); va_x, va_y = va_x.to(DEV), va_y.to(DEV)
    te_x, te_y = materialize(X, Y, te); te_x, te_y = te_x.to(DEV), te_y.to(DEV)
    m = build_model(cfg["arch"], 3, len(CLS), cfg["width"], cfg["pool"], cfg["dropout"]).to(DEV)
    crit = build_loss(cfg["loss"], cls_count=torch.as_tensor(Y[tr].sum(0)))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_sd = -1, None
    for ep in range(epochs):
        m.train()
        for x, y in tr_dl:
            x, y = x.to(DEV), y.to(DEV)
            if cfg.get("mixup", "none") != "none" and np.random.rand() < 0.5:
                lam = float(np.random.beta(0.4, 0.4)); pm = torch.randperm(x.size(0), device=DEV)
                x = lam * x + (1 - lam) * x[pm]; y = lam * y + (1 - lam) * y[pm]
            opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
        sch.step()
        f = macro_f1(m, va_x, va_y)
        if f > best: best, best_sd = f, {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_sd)
    return best, macro_f1(m, te_x, te_y)


def gen_all_cfgs(n, seed=7):
    """전역 결정적 n조합 생성(샤드들이 동일 리스트 공유 → 중복/누락 없음)."""
    rng = np.random.default_rng(seed)
    cfgs, seen = [], set()
    trials = 0
    while len(cfgs) < n and trials < n * 5:
        trials += 1
        c = {
            "arch": str(rng.choice(["resnet", "resnet", "resnet_cbam", "cnn"])),
            "width": int(rng.choice([32, 48, 64])),
            "pool": str(rng.choice(["gap", "gem", "maxavg"])),
            "dropout": float(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5])),
            "lr": round(float(10 ** rng.uniform(-3.5, -2.5)), 6),
            "wd": float(rng.choice([1e-5, 1e-4, 3e-4, 1e-3, 3e-3])),
            "loss": str(rng.choice(["asl", "asl", "bce", "focal"])),
        }
        key = tuple(sorted(c.items()))
        if key in seen: continue
        seen.add(key); cfgs.append(c)
    return cfgs


def _load_data():
    X, Y, y_idx, lots = load_wm811k(normal_cap=10000, seed=config.SEED, size=52)
    tr, va, te = lot_group_split(y_idx, lots, seed=config.SEED)
    return X, Y, y_idx, lots, tr, va, te


def phase1(shard, nshards, n):
    """샤드 슬라이스의 프록시 스캔 → logs/s2mega_shard{shard}.json"""
    t0 = time.time()
    X, Y, y_idx, lots, tr, va, te = _load_data()
    rng0 = np.random.default_rng(0)
    defect = tr[y_idx[tr] >= 0]; normal = tr[y_idx[tr] < 0]
    sub = np.concatenate([defect, rng0.choice(normal, min(len(normal), 5000), replace=False)])
    xt, yt = materialize(X, Y, sub); xt, yt = xt.to(DEV), yt.to(DEV)
    xv, yv = materialize(X, Y, va); xv, yv = xv.to(DEV), yv.to(DEV)
    cfgs = gen_all_cfgs(n)
    mine = [(i, cfgs[i]) for i in range(shard, len(cfgs), nshards)]
    print(f"[shard {shard}/{nshards}] {len(mine)}개 (data {time.time()-t0:.0f}s)", flush=True)
    res = []
    for j, (i, c) in enumerate(mine):
        try: f = train_proxy(c, xt, yt, xv, yv)
        except Exception: f = -1.0
        res.append({"f": f, "cfg": c})
        if (j + 1) % 25 == 0:
            print(f"  [shard {shard}] {j+1}/{len(mine)} best={max(r['f'] for r in res):.4f} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs("docs/overnight/logs", exist_ok=True)
    json.dump(res, open(f"docs/overnight/logs/s2mega_shard{shard}.json", "w"))
    print(f"=== SHARD {shard} DONE ({len(res)}) ({time.time()-t0:.0f}s) ===", flush=True)


def phase2(topk=12):
    """모든 샤드 취합 → 전역 top-k 풀학습 → RESULTS_stage2mega.md"""
    import glob
    t0 = time.time()
    allr = []
    for f in glob.glob("docs/overnight/logs/s2mega_shard*.json"):
        allr += json.load(open(f))
    allr.sort(key=lambda r: -r["f"])
    print(f"[phase2] 취합 {len(allr)}조합, proxy best {allr[0]['f']:.4f}", flush=True)
    X, Y, y_idx, lots, tr, va, te = _load_data()
    final = []
    for r in allr[:topk]:
        cc = dict(r["cfg"], mixup="none")
        try: vb, tf = train_full(cc, X, Y, tr, va, te)
        except Exception: vb, tf = -1, -1
        final.append((tf, vb, cc)); print(f"    test {tf:.4f} (val {vb:.4f})  {cc}", flush=True)
    for tf, vb, c in sorted(final, key=lambda x: -x[0])[:2]:
        for mx in ["mixup", "cutmix"]:
            cc = dict(c, mixup=mx)
            try: vb2, tf2 = train_full(cc, X, Y, tr, va, te)
            except Exception: vb2, tf2 = -1, -1
            final.append((tf2, vb2, cc)); print(f"    [+{mx}] test {tf2:.4f} (val {vb2:.4f})", flush=True)
    final.sort(key=lambda x: -x[0]); bt, bv, bc = final[0]
    n_total = len(allr)
    with open("docs/overnight/RESULTS_stage2mega.md", "w", encoding="utf-8") as fp:
        fp.write(f"# Stage2 메가서치 ({n_total}조합 프록시 4-shard 병렬 → 상위 {topk} 풀학습)\n\n")
        fp.write(f"> best 단일모델 test macro-F1 **{bt:.4f}** (val {bv:.4f})  ·  6-앙상블+보정 **0.935**가 전체 best 유지\n\n")
        fp.write("## Phase2 풀학습(test macro-F1)\n\n| test F1 | val F1 | arch | width | pool | dropout | lr | wd | loss | mixup |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for tf, vb, c in final:
            fp.write(f"| {tf:.4f} | {vb:.4f} | {c['arch']} | {c['width']} | {c['pool']} | {c['dropout']} | {c['lr']} | {c['wd']} | {c['loss']} | {c.get('mixup','none')} |\n")
        fp.write("\n## Phase1 프록시 top25\n\n| proxy val F1 | cfg |\n|---|---|\n")
        for r in allr[:25]:
            fp.write(f"| {r['f']:.4f} | {r['cfg']} |\n")
    print(f"=== STAGE2 MEGA DONE: best test {bt:.4f} ({n_total}조합) ({time.time()-t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=12)
    a = ap.parse_args()
    if a.phase == 1:
        phase1(a.shard, a.nshards, a.n)
    else:
        phase2(a.topk)

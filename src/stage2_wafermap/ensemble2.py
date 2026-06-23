"""Stage2 앙상블 고도화 — mean vs 가중 vs greedy vs 스태킹 (전부 leak-free: val 적합→test 평가).

기존 6-앙상블(0.935)을 넘는지 정직 검증.
- experiments/stage2_real_*/best.pt 자동 탐색 → state_dict로 arch/width/pool 추론(3ch onehot만).
- 멤버별 val/test 확률 → 4방법 비교, 각 per-class 임계보정.

실행:  python -m src.stage2_wafermap.ensemble2 [--tta] [--glob stage2_real_]
"""
from __future__ import annotations
import argparse, os, glob
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

import config
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.rigor import best_thresholds

CLS = config.WM_CLASSES
NC = len(CLS)
EXP = config.ROOT / "experiments" if hasattr(config, "ROOT") else None


def onehot(maps):
    return np.stack([(maps == 0), (maps == 1), (maps == 2)], axis=1).astype(np.float32)


def try_load(path, device):
    """state_dict로 arch/width/pool 추론 후 strict 로드. 실패 시 None."""
    sd = torch.load(path, map_location="cpu")
    keys = list(sd.keys())
    # in_ch / width
    if "stem.0.weight" in sd:
        in_ch, width = sd["stem.0.weight"].shape[1], sd["stem.0.weight"].shape[0]
        archs = ["resnet_cbam", "resnet"] if any(".sp." in k for k in keys) else ["resnet"]
    elif "features.0.0.weight" in sd:
        in_ch, width = sd["features.0.0.weight"].shape[1], sd["features.0.0.weight"].shape[0]
        archs = ["cnn"] if "features.0.3.weight" in sd else ["dilated"]
    elif "embed.weight" in sd:
        in_ch, width, archs = sd["embed.weight"].shape[1], 48, ["vit"]
    elif "conv1.weight" in sd:
        in_ch, width, archs = sd["conv1.weight"].shape[1], 64, ["tvresnet18", "tvresnet34"]
    else:
        return None
    if in_ch != 3:  # onehot 3ch만
        return None
    pool_opts = ["gem"] if any(k.endswith(".p") for k in keys) else ["gap", "maxavg"]
    for arch in archs:
        for pool in pool_opts:
            try:
                m = build_model(arch, 3, NC, width, pool=pool).to(device).eval()
                m.load_state_dict(sd)
                return m, f"{arch}_w{width}_{pool}"
            except Exception:
                continue
    return None


def predict(model, maps, device, tta, bs=512):
    tfs = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    acc = np.zeros((len(maps), NC), dtype=np.float64)
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


def macro_cal(Yv, pv, Yt, pt):
    """val 임계보정 → test macro-F1."""
    th = best_thresholds(Yv, pv)
    return f1_score(Yt, (pt >= th).astype(int), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--glob", default="stage2_real_")
    ap.add_argument("--max", type=int, default=24, help="최대 멤버 수")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y, y_idx, lots = load_wm811k(normal_cap=10000, seed=config.SEED)
    tr, va, te = lot_group_split(y_idx, lots, seed=config.SEED)
    Xv, Yv, Xt, Yt = X[va], Y[va], X[te], Y[te]

    root = os.path.join(os.getcwd(), "experiments")
    paths = sorted(glob.glob(os.path.join(root, args.glob + "*", "best.pt")))
    pv, pt, names = [], [], []
    for p in paths:
        if len(pv) >= args.max:
            break
        r = try_load(p, device)
        if r is None:
            continue
        m, desc = r
        pvi, pti = predict(m, Xv, device, args.tta), predict(m, Xt, device, args.tta)
        single = macro_cal(Yv, pvi, Yt, pti)
        pv.append(pvi); pt.append(pti)
        names.append((os.path.basename(os.path.dirname(p)), desc, single))
        print(f"  [{len(pv):2d}] {os.path.basename(os.path.dirname(p)):30s} {desc:18s} test보정={single:.4f}")
        del m; torch.cuda.empty_cache()
    n = len(pv)
    print(f"\n로드된 멤버 {n}개 | TTA={'ON' if args.tta else 'OFF'}")
    if n == 0:
        return
    PV, PT = np.stack(pv), np.stack(pt)  # (M,N,C)

    # 1) mean
    mean_f1 = macro_cal(Yv, PV.mean(0), Yt, PT.mean(0))

    # 2) greedy forward selection (Caruana) — val 보정 macro 최대화
    chosen, cur_v, cur_t = [], None, None
    best_val = -1
    for _ in range(n):
        bi, bval, bv, bt = -1, best_val, None, None
        for i in range(n):
            cand_v = PV[i] if cur_v is None else (cur_v * len(chosen) + PV[i]) / (len(chosen) + 1)
            cand_t = PT[i] if cur_t is None else (cur_t * len(chosen) + PT[i]) / (len(chosen) + 1)
            th = best_thresholds(Yv, cand_v)
            vF = f1_score(Yv, (cand_v >= th).astype(int), average="macro", zero_division=0)
            if vF > bval:
                bval, bi, bv, bt = vF, i, cand_v, cand_t
        if bi < 0:
            break
        chosen.append(bi); cur_v, cur_t, best_val = bv, bt, bval
    greedy_f1 = macro_cal(Yv, cur_v, Yt, cur_t)

    # 3) weighted — val 보정 macro에 대해 좌표상승(간단 탐색)
    w = np.ones(n) / n
    for _ in range(200):
        i = np.random.randint(n)
        for delta in (0.1, -0.1, 0.05, -0.05):
            w2 = w.copy(); w2[i] = max(0, w2[i] + delta)
            if w2.sum() == 0:
                continue
            w2 /= w2.sum()
            ev = np.tensordot(w2, PV, axes=([0], [0]))
            th = best_thresholds(Yv, ev)
            vF = f1_score(Yv, (ev >= th).astype(int), average="macro", zero_division=0)
            ev0 = np.tensordot(w, PV, axes=([0], [0]))
            th0 = best_thresholds(Yv, ev0)
            vF0 = f1_score(Yv, (ev0 >= th0).astype(int), average="macro", zero_division=0)
            if vF > vF0:
                w = w2
    weighted_f1 = macro_cal(Yv, np.tensordot(w, PV, axes=([0], [0])),
                            Yt, np.tensordot(w, PT, axes=([0], [0])))

    # 4) 스태킹 — per-class LogisticRegression (입력=멤버 확률), val 적합→test
    stv = np.zeros((len(Xv), NC)); stt = np.zeros((len(Xt), NC))
    for c in range(NC):
        Xtr = PV[:, :, c].T   # (Nval, M)
        Xte = PT[:, :, c].T
        if len(np.unique(Yv[:, c])) < 2:
            stv[:, c], stt[:, c] = PV[:, :, c].mean(0), PT[:, :, c].mean(0)
            continue
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Xtr, Yv[:, c])
        stv[:, c] = lr.predict_proba(Xtr)[:, 1]
        stt[:, c] = lr.predict_proba(Xte)[:, 1]
    stack_f1 = macro_cal(Yv, stv, Yt, stt)

    best_single = max(s for _, _, s in names)
    print("\n================ 결과 (test macro-F1, val 임계보정) ================")
    print(f"  최고 단일      : {best_single:.4f}")
    print(f"  mean ({n})      : {mean_f1:.4f}")
    print(f"  greedy ({len(chosen)})   : {greedy_f1:.4f}   선택={chosen}")
    print(f"  weighted       : {weighted_f1:.4f}")
    print(f"  stacking(LR)   : {stack_f1:.4f}")
    print(f"\n  기존 6-앙상블 기준선 0.935 대비 best = "
          f"{max(mean_f1, greedy_f1, weighted_f1, stack_f1):.4f}")


if __name__ == "__main__":
    main()

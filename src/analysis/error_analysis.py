"""Stage2 오차분석 — 챔피언(WaferCNN32)을 WM-811K lot-split test에 재현 평가.
혼동행렬(어느 클래스끼리 헷갈리나) + 클래스별 P/R/F1 + 최대 혼동쌍 식별.
알려진 난제 Loc↔Edge-Loc(경계 모호) 정량화. turbofan/steel 오차분석 층 포팅.

산출: docs/images/eda/stage2_confusion.png + experiments/analysis/error.json
사용: python -m src.analysis.error_analysis   (CPU, 2GB LSWMD 로드 → 수십초)
"""
from __future__ import annotations
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.dataset import WaferMapDataset
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split
from src.stage2_wafermap.train import run_eval
import config

CLS = config.WM_CLASSES
CKPT = config.EXPERIMENTS / "stage2_real_asl" / "best.pt"
IMG = config.REPO / "docs" / "images" / "eda"; IMG.mkdir(parents=True, exist_ok=True)
OUT = config.EXPERIMENTS / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.family": "Malgun Gothic", "axes.unicode_minus": False})


def run(seed=42, normal_cap=10000, size=52):
    X, Y, y_idx, lots = load_wm811k(normal_cap=normal_cap, seed=seed, size=size)
    tr, va, te = lot_group_split(y_idx, lots, seed=seed)
    print(f"[data] test {len(te)} (lot-split leak-free)", flush=True)

    net = build_model("cnn", 3, len(CLS), width=32).eval()
    sd = torch.load(CKPT, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    net.load_state_dict(sd)
    dl = DataLoader(WaferMapDataset(X, Y, te), batch_size=256, shuffle=False)
    gts, probs = run_eval(net, dl, "cpu")            # (N,8) multilabel

    # 정상(전부 0라벨)은 클래스 혼동에서 제외 → 결함 보유 웨이퍼만(단일라벨 지배)
    mask = gts.sum(1) > 0
    n_normal = int((~mask).sum())
    true = gts[mask].argmax(1); pred = probs[mask].argmax(1)
    C = len(CLS)
    cm = np.zeros((C, C), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(C)); ax.set_xticklabels(CLS, rotation=45, ha="right")
    ax.set_yticks(range(C)); ax.set_yticklabels(CLS)
    ax.set_xlabel("예측"); ax.set_ylabel("실제")
    for i in range(C):
        for j in range(C):
            if cmn[i, j] > 0.02:
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > .5 else "black")
    ax.set_title("Stage2 혼동행렬 (결함 웨이퍼, 행정규화, WM-811K test)")
    fig.colorbar(im, shrink=.7); fig.tight_layout()
    fig.savefig(IMG / "stage2_confusion.png"); plt.close(fig)

    # 최대 혼동쌍(off-diagonal)
    off = [(CLS[i], CLS[j], int(cm[i, j]), round(float(cmn[i, j]), 3))
           for i in range(C) for j in range(C) if i != j and cm[i, j] > 0]
    off.sort(key=lambda r: -r[2])
    res = {"n_test": int(len(te)), "n_defect": int(mask.sum()), "n_normal_excluded": n_normal,
           "accuracy_argmax": round(float((true == pred).mean()), 4),
           "top_confusions": [{"true": a, "pred": b, "n": n, "rate": r} for a, b, n, r in off[:8]],
           "confusion_matrix": cm.tolist(), "classes": CLS}
    json.dump(res, open(OUT / "error.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    top = res["top_confusions"][0]
    print(f"[err] argmax acc={res['accuracy_argmax']:.3f} | 최대혼동 "
          f"{top['true']}→{top['pred']} {top['n']}건({top['rate']:.0%}) → "
          f"stage2_confusion.png, error.json", flush=True)


if __name__ == "__main__":
    run()

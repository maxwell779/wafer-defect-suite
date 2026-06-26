"""고도화 EDA 시각화 — MixedWM38 웨이퍼맵(38k×52×52, 8 멀티라벨).
클래스 분포·동시발생·클래스별 몬타주·t-SNE(클래스 분리도). turbofan EDA 층 포팅.
산출: docs/images/eda/*.png + experiments/eda/summary.json

사용: python -m src.eda_viz
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config

CLS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]
IMG = config.REPO / "docs" / "images" / "eda"; IMG.mkdir(parents=True, exist_ok=True)
OUT = config.REPO / "experiments" / "eda"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "font.family": "Malgun Gothic",
                     "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": .25})


def run():
    d = np.load(config.MIXEDWM38_NPZ)
    X, Y = d["arr_0"], d["arr_1"]            # (N,52,52), (N,8)
    n = len(X)

    # 1) 클래스 분포(멀티라벨 합) + 정상/단일/복합 비율
    counts = Y.sum(0)
    nlab = Y.sum(1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(CLS, counts, color="#3b82f6"); ax[0].set_title("클래스별 출현 수 (멀티라벨)")
    ax[0].tick_params(axis="x", rotation=35)
    mix = [int((nlab == 0).sum()), int((nlab == 1).sum()), int((nlab >= 2).sum())]
    ax[1].bar(["정상(0)", "단일(1)", "복합(2+)"], mix, color=["#22c55e", "#3b82f6", "#ef4444"])
    ax[1].set_title("결함 개수별 웨이퍼 수")
    fig.tight_layout(); fig.savefig(IMG / "class_dist.png"); plt.close(fig)

    # 2) 동시발생 행렬
    co = Y.T @ Y
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log1p(co), cmap="Blues")
    ax.set_xticks(range(8)); ax.set_xticklabels(CLS, rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels(CLS)
    ax.set_title("결함 동시발생(log)"); fig.colorbar(im, shrink=.7)
    fig.tight_layout(); fig.savefig(IMG / "cooccurrence.png"); plt.close(fig)

    # 3) 클래스별 대표 몬타주
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for ci, (c, axx) in enumerate(zip(CLS, axes.ravel())):
        idx = np.where(Y[:, ci] == 1)[0]
        axx.imshow(X[idx[0]] if len(idx) else np.zeros((52, 52)), cmap="viridis")
        axx.set_title(f"{c} (n={int(counts[ci])})", fontsize=9); axx.axis("off")
    fig.suptitle("MixedWM38 클래스별 대표 웨이퍼맵"); fig.tight_layout()
    fig.savefig(IMG / "class_montage.png"); plt.close(fig)

    # 4) t-SNE (샘플 3000, 단일라벨만 색칠 → 분리도)
    rng = np.random.default_rng(42)
    single = np.where(nlab == 1)[0]
    sel = single[rng.choice(len(single), min(3000, len(single)), replace=False)]
    flat = X[sel].reshape(len(sel), -1).astype(np.float32)
    z = PCA(n_components=50, random_state=42).fit_transform(flat)
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(z)
    lab = Y[sel].argmax(1)
    fig, ax = plt.subplots(figsize=(7, 6))
    for ci in range(8):
        m = lab == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=.5, label=CLS[ci])
    ax.legend(fontsize=7, markerscale=2); ax.set_title("t-SNE — 단일결함 클래스 분리도(픽셀 PCA50)")
    fig.tight_layout(); fig.savefig(IMG / "tsne_classes.png"); plt.close(fig)

    summary = {"n": int(n), "map_shape": list(X.shape[1:]),
               "class_counts": {CLS[i]: int(counts[i]) for i in range(8)},
               "mix": {"normal": mix[0], "single": mix[1], "multi": mix[2]},
               "rarest": CLS[int(counts.argmin())], "most_common": CLS[int(counts.argmax())]}
    json.dump(summary, open(OUT / "mixedwm38.json", "w"), ensure_ascii=False, indent=2)
    print(f"[eda] N={n} | 최다={summary['most_common']} 최소={summary['rarest']} | "
          f"정상/단일/복합={mix} → 그림 4종 docs/images/eda/", flush=True)


if __name__ == "__main__":
    run()

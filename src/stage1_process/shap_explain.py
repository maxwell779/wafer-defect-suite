"""Stage1 이상탐지 SHAP 설명 — 어떤 공정변수가 이상점수를 끌어올리나(IsolationForest).
기존 z-gap 기여도(run.py)를 SHAP로 대체: 모델이 실제로 본 변수기여를 정량화.
TreeExplainer(IForest) → 전역 중요도(beeswarm/bar) + 결함 7건 개별 기여.

산출: docs/images/eda/stage1_shap_*.png + experiments/stage1_process/shap.json
사용: python -m src.stage1_process.shap_explain
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import config

FEATS = ["temperature_c", "pressure_torr", "gas_flow_sccm",
         "etch_rate_nm_min", "voltage_v", "current_ma"]
IMG = config.REPO / "docs" / "images" / "eda"; IMG.mkdir(parents=True, exist_ok=True)
OUT = config.EXPERIMENTS / "stage1_process"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.family": "Malgun Gothic", "axes.unicode_minus": False})


def run():
    df = pd.read_csv(config.MERUVA_CSV)
    y = df["defect_label"].to_numpy().astype(int)
    X = df[FEATS].to_numpy().astype(float)

    rng = np.random.RandomState(config.SEED)
    normal_idx = np.where(y == 0)[0]; defect_idx = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    cut = int(len(normal_idx) * 0.8)
    tr = normal_idx[:cut]
    ev = np.concatenate([normal_idx[cut:], defect_idx]); rng.shuffle(ev)

    sc = StandardScaler().fit(X[tr])
    Xtr, Xev, yev = sc.transform(X[tr]), sc.transform(X[ev]), y[ev]

    iforest = IsolationForest(random_state=config.SEED).fit(Xtr)
    # SHAP: score_samples 낮을수록 이상 → -score_samples(높을수록 이상)로 부호 통일
    expl = shap.TreeExplainer(iforest)
    sv = expl.shap_values(Xev)            # (N, F) 이상점수 기여(부호: 정상방향 점수)
    sv = -sv                              # 부호 뒤집어 '이상 기여(+면 이상↑)'로

    # 전역 중요도(평균 |shap|)
    glob = np.abs(sv).mean(0)
    order = np.argsort(-glob)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([FEATS[i] for i in order][::-1], glob[order][::-1], color="#3b82f6")
    ax.set_title("Stage1 SHAP 전역 중요도 (평균 |기여|, +면 이상↑)")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(IMG / "stage1_shap_global.png"); plt.close(fig)

    # beeswarm (shap 기본) — 변수값↔이상기여 관계
    shap_exp = shap.Explanation(values=sv, data=Xev, feature_names=FEATS)
    plt.figure(figsize=(7.5, 4.2))
    shap.plots.beeswarm(shap_exp, show=False, max_display=6)
    plt.title("Stage1 SHAP beeswarm (붉을수록 변수값 큼)")
    plt.tight_layout(); plt.savefig(IMG / "stage1_shap_beeswarm.png"); plt.close()

    # 결함 7건 개별 기여 — 어떤 변수 때문에 이상으로 잡혔나
    dmask = yev == 1
    defect_sv = sv[dmask]
    per_defect = []
    for r in defect_sv:
        ti = int(np.argmax(np.abs(r)))
        per_defect.append({"top_feature": FEATS[ti], "shap": round(float(r[ti]), 4)})
    defect_mean = {FEATS[i]: round(float(defect_sv[:, i].mean()), 4) for i in range(len(FEATS))}

    res = {"model": "IsolationForest", "n_eval": int(len(ev)), "n_defect": int(yev.sum()),
           "global_importance": {FEATS[i]: round(float(glob[i]), 4) for i in order},
           "top_feature": FEATS[int(order[0])],
           "defect_mean_shap": defect_mean,
           "per_defect_top": per_defect}
    json.dump(res, open(OUT / "shap.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[stage1-shap] 최강 이상기여 변수={res['top_feature']} | "
          f"결함{int(yev.sum())}건 주도변수 " +
          ", ".join(sorted({d['top_feature'] for d in per_defect})) +
          " → 그림 2종 docs/images/eda/, shap.json", flush=True)


if __name__ == "__main__":
    run()

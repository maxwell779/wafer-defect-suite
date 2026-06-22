"""Stage 1 — 공정 센서 이상탐지: ML vs DL 비교 (Meruva CSV).

결함 7/5000(0.14%) → 지도 분류 불가 → 정상만 학습하는 이상탐지(one-class).
모델: IsolationForest · OneClassSVM · LOF · Mahalanobis(EllipticEnvelope) vs AutoEncoder(DL).
평가(leak-free): 정상 80% 학습 / 정상 20%+결함 전체로 평가. PR-AUC·ROC-AUC·recall@k.
+ 변수기여(어떤 공정값이 위험) → 파라미터 조정 추천 근거.

실행:  python -m src.stage1_process.run
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import average_precision_score, roc_auc_score

import config

FEATS = ["temperature_c", "pressure_torr", "gas_flow_sccm",
         "etch_rate_nm_min", "voltage_v", "current_ma"]


def recall_at_k(y, score, k):
    order = np.argsort(-score)            # 이상점수 높은 순
    topk = y[order[:k]]
    return float(topk.sum() / y.sum())


def ae_scores(Xtr, Xev, device="cpu", epochs=200):
    import torch, torch.nn as nn
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, 4), nn.ReLU(), nn.Linear(4, 2), nn.ReLU(),
                        nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, d)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-5)
    lossf = nn.MSELoss()
    xt = t(Xtr)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(xt), xt); loss.backward(); opt.step()
    with torch.no_grad():
        rec = net(t(Xev)).cpu().numpy()
    return ((Xev - rec) ** 2).mean(axis=1)   # 재구성오차 = 이상점수


def main():
    df = pd.read_csv(config.MERUVA_CSV)
    y = df["defect_label"].to_numpy().astype(int)
    X = df[FEATS].to_numpy().astype(float)

    rng = np.random.RandomState(config.SEED)
    normal_idx = np.where(y == 0)[0]; defect_idx = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    cut = int(len(normal_idx) * 0.8)
    tr = normal_idx[:cut]                                   # 정상만 학습
    ev = np.concatenate([normal_idx[cut:], defect_idx])     # 평가 = 정상20%+결함전체
    rng.shuffle(ev)

    sc = StandardScaler().fit(X[tr])
    Xtr, Xev, yev = sc.transform(X[tr]), sc.transform(X[ev]), y[ev]
    print(f"[data] train(normal) {len(tr)} | eval {len(ev)} (결함 {int(yev.sum())})")

    models = {}
    models["IsolationForest"] = -IsolationForest(random_state=config.SEED).fit(Xtr).score_samples(Xev)
    models["OneClassSVM"]     = -OneClassSVM(nu=0.05, gamma="scale").fit(Xtr).score_samples(Xev)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtr)
    models["LOF"]             = -lof.score_samples(Xev)
    models["Mahalanobis"]     = -EllipticEnvelope(contamination=0.05, random_state=config.SEED).fit(Xtr).score_samples(Xev)
    models["AutoEncoder(DL)"] = ae_scores(Xtr, Xev)

    rows = []
    for name, s in models.items():
        rows.append(dict(model=name,
                         pr_auc=round(average_precision_score(yev, s), 4),
                         roc_auc=round(roc_auc_score(yev, s), 4),
                         recall_at_50=round(recall_at_k(yev, s, 50), 3),
                         recall_at_100=round(recall_at_k(yev, s, 100), 3)))
    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    print("\n===== Stage1: ML vs DL (이상탐지) =====")
    print(res.to_string(index=False))

    # 변수기여 (결함 vs 정상 표준화 평균차 = 위험 방향)
    contrib = {f: round(float(X[defect_idx, i].mean() - X[normal_idx, i].mean())
                        / (X[normal_idx, i].std() + 1e-9), 3) for i, f in enumerate(FEATS)}
    print("\n변수기여(z-gap, +면 결함시 높음):")
    for f, z in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
        arrow = "↑" if z > 0 else "↓"
        print(f"  {f:18s} {z:+.2f} {arrow}")

    out = config.EXPERIMENTS / "stage1_process"; out.mkdir(parents=True, exist_ok=True)
    json.dump({"comparison": rows, "feature_contrib": contrib,
               "n_eval": len(ev), "n_defect": int(yev.sum())},
              open(out / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[saved] {out/'results.json'}")


if __name__ == "__main__":
    main()

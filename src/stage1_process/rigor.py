"""Stage 1 엄밀 평가 — 반복 split CV (양성 7건 단일 split의 노이즈 제거).

정상을 N회 무작위 80/20 분할 → 매회 모델 fit/평가 → PR-AUC 분포(mean±std).
단일 split 수치가 운에 좌우되는 문제를 정직하게 해소.

실행:  python -m src.stage1_process.rigor --repeats 30
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from src.stage1_process.run import FEATS, recall_at_k, ae_scores


def eval_once(X, y, seed):
    rng = np.random.RandomState(seed)
    nidx = np.where(y == 0)[0]; didx = np.where(y == 1)[0]
    rng.shuffle(nidx); cut = int(len(nidx) * 0.8)
    tr, ev = nidx[:cut], np.concatenate([nidx[cut:], didx])
    sc = StandardScaler().fit(X[tr])
    Xtr, Xev, yev = sc.transform(X[tr]), sc.transform(X[ev]), y[ev]
    models = {
        "IsolationForest": -IsolationForest(random_state=seed).fit(Xtr).score_samples(Xev),
        "OneClassSVM": -OneClassSVM(nu=0.05, gamma="scale").fit(Xtr).score_samples(Xev),
        "LOF": -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtr).score_samples(Xev),
        "Mahalanobis": -EllipticEnvelope(contamination=0.05, random_state=seed).fit(Xtr).score_samples(Xev),
        "AutoEncoder(DL)": ae_scores(Xtr, Xev, epochs=200),
    }
    from scipy.stats import rankdata
    ens = np.mean([rankdata(models[m]) for m in ["LOF", "Mahalanobis", "OneClassSVM"]], axis=0)
    models["Ensemble(rank L+M+O)"] = ens
    out = {}
    for n, s in models.items():
        out[n] = (average_precision_score(yev, s), roc_auc_score(yev, s), recall_at_k(yev, s, 100))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()
    df = pd.read_csv(config.MERUVA_CSV)
    X = df[FEATS].to_numpy(float); y = df["defect_label"].to_numpy(int)

    agg = {}
    for r in range(args.repeats):
        for n, (pr, roc, rec) in eval_once(X, y, 1000 + r).items():
            agg.setdefault(n, []).append((pr, roc, rec))

    print(f"===== Stage1 반복 CV ({args.repeats} seeds) — PR-AUC mean±std =====")
    rows = []
    for n, v in agg.items():
        a = np.array(v)
        rows.append((n, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean()))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'model':18s}{'PR-AUC':>16}{'ROC-AUC':>9}{'rec@100':>9}")
    for n, prm, prs, roc, rec in rows:
        print(f"  {n:18s}{prm:8.3f} ± {prs:.3f}{roc:9.3f}{rec:9.3f}")
    best = rows[0][0]
    print(f"\n→ 최고: {best} (PR-AUC {rows[0][1]:.3f}±{rows[0][2]:.3f}). 단일 split 수치보다 신뢰 가능.")

    out = config.EXPERIMENTS / "stage1_process"; out.mkdir(parents=True, exist_ok=True)
    json.dump({"repeats": args.repeats, "results": [
        {"model": n, "pr_auc_mean": round(prm, 4), "pr_auc_std": round(prs, 4),
         "roc_auc_mean": round(roc, 4), "recall_at_100_mean": round(rec, 4)}
        for n, prm, prs, roc, rec in rows]},
        open(out / "rigor_cv.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {out/'rigor_cv.json'}")


if __name__ == "__main__":
    main()

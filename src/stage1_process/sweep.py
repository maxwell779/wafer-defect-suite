"""Stage 1 확장 실험 — AD 모델 8종 × 피처셋 비교 (30-seed CV, PR-AUC mean±std).

피처셋: base(6) / +engineered(비율·교호작용) / +step(one-hot)
모델: IForest·OCSVM·LOF·Mahalanobis·GMM·KDE·kNN·AE + rank-앙상블
실행:  python -m src.stage1_process.sweep --repeats 30
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, KernelDensity
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score, roc_auc_score
import config
from src.stage1_process.run import recall_at_k, ae_scores

BASE = ["temperature_c", "pressure_torr", "gas_flow_sccm", "etch_rate_nm_min", "voltage_v", "current_ma"]


from itertools import combinations


def make_features(df, kind):
    X = df[BASE].to_numpy(float)
    if kind == "base":
        return X
    cols = [df[k].to_numpy(float) for k in BASE]
    if kind in ("eng", "eng_step"):
        t, p, g, e, v, c = cols
        eng = np.c_[p / (e + 1e-6), t / (p + 1e-6), e * t, v * c, g / (p + 1e-6)]
        X = np.c_[X, eng]
    if kind in ("pair", "pair_step"):
        # 모든 쌍 곱·비율 (결함=센서 조합 → 조합 전수 명시화)
        feats = [X]
        for i, j in combinations(range(6), 2):
            feats.append((cols[i] * cols[j])[:, None])
            feats.append((cols[i] / (cols[j] + 1e-6))[:, None])
        X = np.concatenate(feats, axis=1)
    if kind in ("eng_step", "pair_step"):
        step = pd.get_dummies(df["process_step"]).to_numpy(float)
        X = np.c_[X, step]
    return X


def score_models(Xtr, Xev, seed):
    M = {}
    M["IForest"] = -IsolationForest(random_state=seed).fit(Xtr).score_samples(Xev)
    M["OCSVM"] = -OneClassSVM(nu=0.05, gamma="scale").fit(Xtr).score_samples(Xev)
    M["LOF"] = -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtr).score_samples(Xev)
    M["Mahalanobis"] = -EllipticEnvelope(contamination=0.05, random_state=seed).fit(Xtr).score_samples(Xev)
    from sklearn.covariance import LedoitWolf      # shrinkage: 고차원 공분산 안정화
    M["Maha-LW(shrink)"] = LedoitWolf().fit(Xtr).mahalanobis(Xev)
    M["GMM"] = -GaussianMixture(3, random_state=seed).fit(Xtr).score_samples(Xev)
    M["KDE"] = -KernelDensity(bandwidth=1.0).fit(Xtr).score_samples(Xev)
    nn = NearestNeighbors(n_neighbors=5).fit(Xtr)
    M["kNN"] = nn.kneighbors(Xev)[0].mean(1)
    M["AE(DL)"] = ae_scores(Xtr, Xev, epochs=200)
    M["Ensemble"] = np.mean([rankdata(M[k]) for k in ["LOF", "Mahalanobis", "OCSVM", "kNN"]], axis=0)
    return M


def eval_once(X, y, seed):
    rng = np.random.RandomState(seed)
    nidx = np.where(y == 0)[0]; didx = np.where(y == 1)[0]
    rng.shuffle(nidx); cut = int(len(nidx) * 0.8)
    tr, ev = nidx[:cut], np.concatenate([nidx[cut:], didx])
    sc = StandardScaler().fit(X[tr])
    Xtr, Xev, yev = sc.transform(X[tr]), sc.transform(X[ev]), y[ev]
    return {n: (average_precision_score(yev, s), roc_auc_score(yev, s), recall_at_k(yev, s, 100))
            for n, s in score_models(Xtr, Xev, seed).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()
    df = pd.read_csv(config.MERUVA_CSV); y = df["defect_label"].to_numpy(int)
    results = {}
    for fs in ["base", "eng_step", "pair", "pair_step"]:
        X = make_features(df, fs)
        agg = {}
        for r in range(args.repeats):
            for n, v in eval_once(X, y, 2000 + r).items():
                agg.setdefault(n, []).append(v)
        results[fs] = {n: float(np.mean([a[0] for a in v])) for n, v in agg.items()}
        best = max(results[fs].items(), key=lambda kv: kv[1])
        print(f"\n=== 피처셋 [{fs}] (dim {X.shape[1]}) — PR-AUC mean (30-seed) ===")
        for n, v in sorted(results[fs].items(), key=lambda kv: -kv[1]):
            print(f"  {n:12s} {v:.3f}")
    out = config.EXPERIMENTS / "stage1_process"; out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / "sweep.json", "w"), indent=2)
    print(f"\n[saved] {out/'sweep.json'}")


if __name__ == "__main__":
    main()

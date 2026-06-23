"""Stage1 초대형 LGB 랜덤서치 — 1000조합 광역(reps8) → 상위30 정밀(reps60) + 스태킹 최종.

오버나이트용. 표적피처+Maha 하이브리드. 양성7건이라 reps↑로 CI 확보가 핵심.
"""
import sys, os
sys.path.insert(0, os.getcwd())   # repo root에서 config import (직접 실행 대응)
import numpy as np, pandas as pd, warnings, random
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import config

FEATS = ["temperature_c", "pressure_torr", "gas_flow_sccm", "etch_rate_nm_min", "voltage_v", "current_ma"]
df = pd.read_csv(config.MERUVA_CSV); X = df[FEATS].to_numpy(float); y = df["defect_label"].to_numpy(int)


def tf(A, mu, sd):
    t, p, g, e, v, c = A.T
    eng = np.c_[A, p / (e + 1e-6), t / (p + 1e-6), e * t, v * c, g / (p + 1e-6)]
    zp = -(p - mu[1]) / sd[1]; zt = (t - mu[0]) / sd[0]; ze = (e - mu[3]) / sd[3]
    return np.c_[eng, np.maximum(zp, 0) * np.maximum(zt + ze, 0), zp * ze, zp * zt]


def prep(reps):
    cache = []
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            trN = tr[y[tr] == 0]; mu, sd = X[trN].mean(0), X[trN].std(0)
            sc = StandardScaler().fit(tf(X[tr], mu, sd))
            Ztr, Zte, ZtrN = sc.transform(tf(X[tr], mu, sd)), sc.transform(tf(X[te], mu, sd)), sc.transform(tf(X[trN], mu, sd))
            ee = EllipticEnvelope(contamination=0.05).fit(ZtrN)
            cache.append((np.c_[Ztr, -ee.score_samples(Ztr)], y[tr], np.c_[Zte, -ee.score_samples(Zte)], y[te], r))
    return cache


def evalc(build, cache):
    a = np.array([average_precision_score(yte, build(r).fit(Htr, ytr).predict_proba(Hte)[:, 1])
                  for Htr, ytr, Hte, yte, r in cache])
    return a.mean(), a.std(), 1.96 * a.std() / np.sqrt(len(a))


def mk(p):
    return lambda r: lgb.LGBMClassifier(
        n_estimators=p["n"], max_depth=p["d"], num_leaves=p["l"], learning_rate=p["lr"],
        min_child_samples=p["mcs"], subsample=p["ss"], subsample_freq=1, colsample_bytree=p["cs"],
        reg_lambda=p["rl"], reg_alpha=p["ra"], min_split_gain=p["msg"],
        class_weight="balanced", verbose=-1, random_state=r, n_jobs=1)


def main():
    print(f"rows {len(X)} pos {y.sum()}", flush=True)
    space = dict(n=[100,150,200,300,400,500,600,800,1000,1200,1500], d=[2,3,4,5,6,8,-1],
                 l=[7,15,23,31,47,63,95,127], lr=[0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2],
                 mcs=[2,3,5,8,10,15,20,30,50], ss=[0.5,0.6,0.7,0.8,0.9,1.0], cs=[0.5,0.6,0.7,0.8,1.0],
                 rl=[0.0,0.5,1.0,2.0,5.0,10.0,20.0], ra=[0.0,0.5,1.0,2.0], msg=[0.0,0.0,0.01,0.1])
    random.seed(7)
    cacheA = prep(8)
    print("=== Phase A: LGB 1000조합 광역 (reps=8) ===", flush=True)
    seen = set(); results = []
    while len(results) < 1000:
        p = {k: random.choice(v) for k, v in space.items()}
        key = tuple(sorted(p.items()))
        if key in seen: continue
        seen.add(key)
        m, s, _ = evalc(mk(p), cacheA); results.append((m, s, p))
        if len(results) % 100 == 0:
            print(f"  [{len(results)}/1000] running best={max(r[0] for r in results):.3f}", flush=True)
    results.sort(key=lambda x: -x[0])
    print("  -- A top10 (reps=8) --", flush=True)
    for m, s, p in results[:10]:
        print(f"    {m:.3f}±{s:.3f}  n={p['n']} d={p['d']} l={p['l']} lr={p['lr']} rl={p['rl']}", flush=True)

    print("\n=== Phase B: 상위 30개 정밀 (reps=60, CI) ===", flush=True)
    cacheB = prep(60)
    fine = []
    for m8, s8, p in results[:30]:
        m, s, ci = evalc(mk(p), cacheB); fine.append((m, s, ci, p))
    fine.sort(key=lambda x: -x[0])
    for m, s, ci, p in fine[:10]:
        print(f"    {m:.3f}±{s:.3f} (CI±{ci:.3f})  n={p['n']} d={p['d']} l={p['l']} lr={p['lr']} mcs={p['mcs']} ss={p['ss']} cs={p['cs']} rl={p['rl']} ra={p['ra']}", flush=True)
    bestp = fine[0][3]; best_lgb = mk(bestp)
    print(f"  >>> LGB 최종 best: {fine[0][0]:.3f}±{fine[0][1]:.3f} (CI±{fine[0][2]:.3f})", flush=True)

    print("\n=== Phase C: 최종 스태킹 (best LGB + ET + RF, reps=60) ===", flush=True)
    def et(r): return ExtraTreesClassifier(500, class_weight="balanced", random_state=r, n_jobs=1)
    def rf(r): return RandomForestClassifier(500, class_weight="balanced", random_state=r, n_jobs=1)
    for name, bs in [("lgb+et+rf", ["lgb","et","rf"]), ("lgb+et", ["lgb","et"])]:
        bfn = {"lgb": best_lgb, "et": et, "rf": rf}
        build = lambda r, bs=bs: StackingClassifier([(b, bfn[b](r)) for b in bs],
                                  final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"), cv=3, n_jobs=1)
        m, s, ci = evalc(build, cacheB)
        print(f"    스태킹 {name}: {m:.3f}±{s:.3f} (CI±{ci:.3f})", flush=True)
    print("\n=== MEGASEARCH DONE ===", flush=True)


if __name__ == "__main__":
    main()

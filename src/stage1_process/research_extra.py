"""Stage1 추가 연구 실험 — ECOD/COPOD(직접구현) 피처 + PU learning 프레이밍.

pyod는 numba 충돌로 ECOD/COPOD를 직접 구현(꼬리 로그확률 합).
- ECOD: 각 차원 경험적 CDF의 좌/우 꼬리 -log 확률 합 (분포기반, 파라미터-free)
- COPOD: 동일 꼴(copula 근사) — 여기선 좌+우 양측 합산본
비지도 점수를 지도 LGB의 추가 피처로 → Maha 대비/결합 효과를 reps↑로 측정.
PU: 양품을 unlabeled로 보고, 비지도 점수 하위(확실 정상)만 신뢰음성으로 재학습.
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import config

FEATS = ["temperature_c", "pressure_torr", "gas_flow_sccm", "etch_rate_nm_min", "voltage_v", "current_ma"]


def tf(A, mu, sd):
    t, p, g, e, v, c = A.T
    eng = np.c_[A, p / (e + 1e-6), t / (p + 1e-6), e * t, v * c, g / (p + 1e-6)]
    zp = -(p - mu[1]) / sd[1]; zt = (t - mu[0]) / sd[0]; ze = (e - mu[3]) / sd[3]
    return np.c_[eng, np.maximum(zp, 0) * np.maximum(zt + ze, 0), zp * ze, zp * zt]


def ecod_fit(Xn):
    """train normal로 정렬값 보관."""
    return [np.sort(Xn[:, j]) for j in range(Xn.shape[1])]


def _tail_logp(sorted_col, x, side):
    n = len(sorted_col)
    if side == "left":
        rank = np.searchsorted(sorted_col, x, side="right")
        p = (rank + 1) / (n + 2)
    else:
        rank = np.searchsorted(sorted_col, x, side="left")
        p = (n - rank + 1) / (n + 2)
    return -np.log(np.clip(p, 1e-12, 1))


def ecod_score(model, X, both=False):
    s = np.zeros(len(X))
    for j, sc in enumerate(model):
        left = _tail_logp(sc, X[:, j], "left")
        right = _tail_logp(sc, X[:, j], "right")
        s += (left + right) if both else np.maximum(left, right)
    return s


def maha_score(Ztr_normal, Z):
    ee = EllipticEnvelope(contamination=0.05).fit(Ztr_normal)
    return -ee.score_samples(Z)


def best_lgb(r):
    return lgb.LGBMClassifier(n_estimators=800, max_depth=4, num_leaves=31, learning_rate=0.05,
                              min_child_samples=10, subsample=0.9, subsample_freq=1, reg_lambda=1.0,
                              class_weight="balanced", verbose=-1, random_state=r, n_jobs=1)


def evaluate(X, y, feat_fn, reps=40):
    aps = []
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            trN = tr[y[tr] == 0]; mu, sd = X[trN].mean(0), X[trN].std(0)
            sc = StandardScaler().fit(tf(X[tr], mu, sd))
            Ztr, Zte, ZtrN = sc.transform(tf(X[tr], mu, sd)), sc.transform(tf(X[te], mu, sd)), sc.transform(tf(X[trN], mu, sd))
            Htr, Hte = feat_fn(Ztr, Zte, ZtrN, X[tr], X[te], X[trN])
            m = best_lgb(r).fit(Htr, y[tr])
            aps.append(average_precision_score(y[te], m.predict_proba(Hte)[:, 1]))
    a = np.array(aps); return a.mean(), a.std(), 1.96 * a.std() / np.sqrt(len(a))


def main():
    df = pd.read_csv(config.MERUVA_CSV)
    X = df[FEATS].to_numpy(float); y = df["defect_label"].to_numpy(int)
    print(f"rows {len(X)} pos {y.sum()}", flush=True)

    def f_none(Ztr, Zte, ZtrN, *_): return Ztr, Zte
    def f_maha(Ztr, Zte, ZtrN, *_):
        ee = EllipticEnvelope(contamination=0.05).fit(ZtrN)
        return np.c_[Ztr, -ee.score_samples(Ztr)], np.c_[Zte, -ee.score_samples(Zte)]
    def f_ecod(Ztr, Zte, ZtrN, *_):
        m = ecod_fit(ZtrN)
        return np.c_[Ztr, ecod_score(m, Ztr)], np.c_[Zte, ecod_score(m, Zte)]
    def f_copod(Ztr, Zte, ZtrN, *_):
        m = ecod_fit(ZtrN)
        return np.c_[Ztr, ecod_score(m, Ztr, both=True)], np.c_[Zte, ecod_score(m, Zte, both=True)]
    def f_maha_ecod(Ztr, Zte, ZtrN, *_):
        ee = EllipticEnvelope(contamination=0.05).fit(ZtrN); m = ecod_fit(ZtrN)
        return (np.c_[Ztr, -ee.score_samples(Ztr), ecod_score(m, Ztr)],
                np.c_[Zte, -ee.score_samples(Zte), ecod_score(m, Zte)])
    def f_all3(Ztr, Zte, ZtrN, *_):
        ee = EllipticEnvelope(contamination=0.05).fit(ZtrN); m = ecod_fit(ZtrN)
        return (np.c_[Ztr, -ee.score_samples(Ztr), ecod_score(m, Ztr), ecod_score(m, Ztr, both=True)],
                np.c_[Zte, -ee.score_samples(Zte), ecod_score(m, Zte), ecod_score(m, Zte, both=True)])

    print("=== Stage1 비지도 피처 확장 (LGB 튜닝본, reps=40, CI) ===", flush=True)
    for name, fn in [("피처없음(LGB만)", f_none), ("+Maha", f_maha), ("+ECOD", f_ecod),
                     ("+COPOD", f_copod), ("+Maha+ECOD", f_maha_ecod), ("+Maha+ECOD+COPOD", f_all3)]:
        m, s, ci = evaluate(X, y, fn)
        print(f"  {name:20s}: {m:.3f}±{s:.3f} (CI±{ci:.3f})", flush=True)

    # PU 프레이밍: 비지도 점수 하위 60%만 '신뢰 정상'으로 (양품 라벨 불신) → 재학습
    print("\n=== PU 프레이밍 (신뢰음성만 사용, +Maha, reps=40) ===", flush=True)
    def f_pu(Ztr, Zte, ZtrN, Xtr_raw, Xte_raw, XtrN_raw):
        ee = EllipticEnvelope(contamination=0.05).fit(ZtrN)
        sc_tr = -ee.score_samples(Ztr)
        return np.c_[Ztr, sc_tr], np.c_[Zte, -ee.score_samples(Zte)]
    aps = []
    for r in range(40):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            trN = tr[y[tr] == 0]; mu, sd = X[trN].mean(0), X[trN].std(0)
            sc = StandardScaler().fit(tf(X[tr], mu, sd))
            Ztr, Zte, ZtrN = sc.transform(tf(X[tr], mu, sd)), sc.transform(tf(X[te], mu, sd)), sc.transform(tf(X[trN], mu, sd))
            ee = EllipticEnvelope(contamination=0.05).fit(ZtrN)
            sneg = -ee.score_samples(Ztr)
            ytr = y[tr].copy()
            neg_idx = np.where(ytr == 0)[0]
            thr = np.quantile(sneg[neg_idx], 0.6)             # 하위60% 신뢰음성
            keep = (ytr == 1) | (sneg <= thr)
            Htr = np.c_[Ztr, sneg][keep]; Hte = np.c_[Zte, -ee.score_samples(Zte)]
            m = best_lgb(r).fit(Htr, ytr[keep])
            aps.append(average_precision_score(y[te], m.predict_proba(Hte)[:, 1]))
    a = np.array(aps)
    print(f"  PU(신뢰음성60%)+Maha: {a.mean():.3f}±{a.std():.3f} (CI±{1.96*a.std()/np.sqrt(len(a)):.3f})", flush=True)
    print("\n=== RESEARCH DONE ===", flush=True)


if __name__ == "__main__":
    main()

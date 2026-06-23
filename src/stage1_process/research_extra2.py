"""Stage1 추가 연구 2 — SMOTE 오버샘플링 · 커스텀 focal LGB · MI 피처선택.

모두 표적피처+Maha 하이브리드 기반, reps=40 CI로 '진짜 향상' 판정.
양성 7건이라 SMOTE k_neighbors<=5, focal은 불균형 표적 커스텀 objective.
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import config

FEATS = ["temperature_c", "pressure_torr", "gas_flow_sccm", "etch_rate_nm_min", "voltage_v", "current_ma"]


def tf(A, mu, sd):
    t, p, g, e, v, c = A.T
    eng = np.c_[A, p / (e + 1e-6), t / (p + 1e-6), e * t, v * c, g / (p + 1e-6)]
    zp = -(p - mu[1]) / sd[1]; zt = (t - mu[0]) / sd[0]; ze = (e - mu[3]) / sd[3]
    return np.c_[eng, np.maximum(zp, 0) * np.maximum(zt + ze, 0), zp * ze, zp * zt]


def base_lgb(r, **kw):
    d = dict(n_estimators=800, max_depth=4, num_leaves=31, learning_rate=0.05, min_child_samples=10,
             subsample=0.9, subsample_freq=1, reg_lambda=1.0, class_weight="balanced", verbose=-1,
             random_state=r, n_jobs=1)
    d.update(kw); return lgb.LGBMClassifier(**d)


def focal_obj(gamma=2.0, alpha=0.9):
    """LGB 이진 focal objective (grad/hess). alpha=양성 가중."""
    def obj(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        a = np.where(y_true == 1, alpha, 1 - alpha)
        pt = np.where(y_true == 1, p, 1 - p)
        # focal grad/hess (근사, 표준 유도)
        g = a * (1 - pt) ** gamma * (gamma * pt * np.log(np.clip(pt, 1e-8, 1)) + pt - 1) * np.where(y_true == 1, 1, -1)
        h = np.abs(a * (1 - pt) ** gamma) * pt * (1 - pt) * (gamma + 1)
        return g, h
    return obj


def prep_hybrid(reps):
    cache = []
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            trN = tr[y[tr] == 0]; mu, sd = X[trN].mean(0), X[trN].std(0)
            sc = StandardScaler().fit(tf(X[tr], mu, sd))
            Ztr, Zte, ZtrN = sc.transform(tf(X[tr], mu, sd)), sc.transform(tf(X[te], mu, sd)), sc.transform(tf(X[trN], mu, sd))
            ee = EllipticEnvelope(contamination=0.05).fit(ZtrN)
            Htr = np.c_[Ztr, -ee.score_samples(Ztr)]; Hte = np.c_[Zte, -ee.score_samples(Zte)]
            cache.append((Htr, y[tr], Hte, y[te], r))
    return cache


def report(name, aps):
    a = np.array(aps)
    print(f"  {name:26s}: {a.mean():.3f}±{a.std():.3f} (CI±{1.96*a.std()/np.sqrt(len(a)):.3f})", flush=True)


def main():
    global X, y
    df = pd.read_csv(config.MERUVA_CSV)
    X = df[FEATS].to_numpy(float); y = df["defect_label"].to_numpy(int)
    print(f"rows {len(X)} pos {y.sum()}", flush=True)
    cache = prep_hybrid(40)

    # baseline
    report("LGB+Maha (baseline)", [average_precision_score(yte, base_lgb(r).fit(Htr, ytr).predict_proba(Hte)[:, 1])
                                    for Htr, ytr, Hte, yte, r in cache])

    # SMOTE 오버샘플링
    from imblearn.over_sampling import SMOTE
    print("=== SMOTE (k_neighbors=5) ===", flush=True)
    aps = []
    for Htr, ytr, Hte, yte, r in cache:
        try:
            Xs, ys = SMOTE(k_neighbors=min(5, int(ytr.sum()) - 1), random_state=r).fit_resample(Htr, ytr)
            m = base_lgb(r, class_weight=None).fit(Xs, ys)
            aps.append(average_precision_score(yte, m.predict_proba(Hte)[:, 1]))
        except Exception:
            aps.append(average_precision_score(yte, base_lgb(r).fit(Htr, ytr).predict_proba(Hte)[:, 1]))
    report("SMOTE+LGB+Maha", aps)

    # 커스텀 focal objective
    print("=== 커스텀 focal LGB ===", flush=True)
    for g in (1.0, 2.0):
        aps = []
        for Htr, ytr, Hte, yte, r in cache:
            m = lgb.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                   objective=focal_obj(gamma=g), n_jobs=1, verbose=-1, random_state=r)
            m.fit(Htr, ytr)
            raw = m.predict(Hte, raw_score=True); prob = 1 / (1 + np.exp(-raw))
            aps.append(average_precision_score(yte, prob))
        report(f"focal(γ={g})+Maha", aps)

    # MI 피처선택 (top-k)
    print("=== MI 피처선택 ===", flush=True)
    for k in (8, 12, 16):
        aps = []
        for Htr, ytr, Hte, yte, r in cache:
            mi = mutual_info_classif(Htr, ytr, random_state=r)
            top = np.argsort(mi)[-k:]
            m = base_lgb(r).fit(Htr[:, top], ytr)
            aps.append(average_precision_score(yte, m.predict_proba(Hte[:, top])[:, 1]))
        report(f"MI top-{k}+Maha", aps)
    print("\n=== RESEARCH2 DONE ===", flush=True)


if __name__ == "__main__":
    main()

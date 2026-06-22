"""WM-811K(실데이터, LSWMD.pkl) 로더 + lot 단위 leak-free 분할.

- 8개 단일결함(+옵션 'none' 정상) 추출
- 가변크기 맵 → 52x52 nearest 리사이즈(범주값 {0,1,2} 보존)
- 멀티핫 라벨(정상=all-zero) → MixedWM38 모델과 동일 출력공간
- ★ split은 lotName 그룹 단위(StratifiedGroupKFold): 같은 lot이 train/val/test에
  섞이지 않게 = leak-free (웨이퍼 1~25개가 한 lot이라 상관 높음)
"""
from __future__ import annotations
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config


def _flat(x):
    a = np.array(x).ravel()
    return a[0] if a.size else "NA"


def load_wm811k(pkl_path=None, classes=None, include_normal=True,
                normal_cap=10000, size=52, seed=42):
    """returns X(N,52,52 int8 {0,1,2}), Y(N,8 multi-hot), y_idx(N,), lots(N,)."""
    pkl_path = pkl_path or config.LSWMD_PKL
    classes = classes or config.WM_CLASSES
    rng = np.random.RandomState(seed)

    df = pd.read_pickle(pkl_path)
    ftv = df["failureType"].map(_flat).to_numpy()
    lotv = df["lotName"].map(_flat).to_numpy()
    wm = df["waferMap"].to_numpy()

    sel_pos, sel_cls = [], []
    for ci, name in enumerate(classes):           # 결함 8종
        pos = np.where(ftv == name)[0]
        sel_pos += list(pos); sel_cls += [ci] * len(pos)
    if include_normal:                             # 정상('none')
        npos = np.where(ftv == "none")[0]
        if normal_cap and len(npos) > normal_cap:
            npos = rng.choice(npos, normal_cap, replace=False)
        sel_pos += list(npos); sel_cls += [-1] * len(npos)

    X = np.zeros((len(sel_pos), size, size), dtype=np.int8)
    for k, p in enumerate(sel_pos):
        m = np.asarray(wm[p]).astype(np.uint8)
        X[k] = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    y_idx = np.array(sel_cls)
    lots = lotv[np.array(sel_pos)]
    Y = np.zeros((len(y_idx), len(classes)), dtype=np.float32)
    for k, c in enumerate(y_idx):
        if c >= 0:
            Y[k, c] = 1.0
    del df, wm
    return X, Y, y_idx, lots


def load_wm811k_unlabeled(pkl_path=None, cap=150000, size=52, seed=42):
    """미라벨(failureType=='NA') 맵 추출 → SSL 사전학습용. returns X(N,52,52 int8)."""
    pkl_path = pkl_path or config.LSWMD_PKL
    df = pd.read_pickle(pkl_path)
    ftv = df["failureType"].map(_flat).to_numpy()
    wm = df["waferMap"].to_numpy()
    pos = np.where(ftv == "NA")[0]
    rng = np.random.RandomState(seed)
    if cap and len(pos) > cap:
        pos = rng.choice(pos, cap, replace=False)
    X = np.zeros((len(pos), size, size), dtype=np.int8)
    for k, p in enumerate(pos):
        m = np.asarray(wm[p]).astype(np.uint8)
        if m.ndim != 2 or m.size == 0:
            continue
        X[k] = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    del df, wm
    return X


def lot_group_split(y_idx, lots, seed=42, n_splits=7):
    """lot 그룹 + 클래스 stratify. returns (train_idx, val_idx, test_idx).
    test≈1/7, val≈1/7, train≈5/7. 같은 lot은 한 split에만."""
    strat = y_idx.copy()
    strat[strat < 0] = y_idx.max() + 1            # 'none'도 한 stratum으로
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(np.zeros(len(strat)), strat, lots)]
    test_idx, val_idx = folds[0], folds[1]
    train_idx = np.concatenate(folds[2:])
    # leak 검증: split 간 lot 교집합 0 이어야 함
    s_tr, s_va, s_te = set(lots[train_idx]), set(lots[val_idx]), set(lots[test_idx])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "LOT LEAK!"
    return train_idx, val_idx, test_idx

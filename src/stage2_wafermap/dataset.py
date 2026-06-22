"""MixedWM38 멀티라벨 데이터셋 + leak-free 분할.

웨이퍼맵 값 {0:다이없음, 1:정상다이, 2:불량다이} → 3채널 one-hot 입력.
분할은 38개 라벨조합(combo) 기준 stratify (seed 고정).
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_mixedwm38(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["arr_0"].astype(np.int8)      # (N,52,52) in {0,1,2}
    Y = z["arr_1"].astype(np.float32)   # (N,8) multi-hot
    return X, Y


def combo_ids(Y):
    """8비트 멀티핫 → 고유 정수 id (결정적). stratify/그룹용."""
    bits = (1 << np.arange(Y.shape[1])).astype(np.int64)
    return (Y.astype(np.int64) * bits).sum(axis=1)


def make_splits(Y, seed=42, val=0.15, test=0.15):
    """combo 기준 stratified 3분할. returns (train_idx, val_idx, test_idx)."""
    cid = combo_ids(Y)
    idx = np.arange(len(Y))
    tr, tmp = train_test_split(idx, test_size=val + test, random_state=seed, stratify=cid)
    te_frac = test / (val + test)
    va, te = train_test_split(tmp, test_size=te_frac, random_state=seed, stratify=cid[tmp])
    return tr, va, te


def _augment_map(m, rng, max_noise=0.15):
    """실제 도메인 모사 증강: 회전/플립 + 배경 노이즈(정상다이 일부→불량).
    라벨은 방향·노이즈에 불변이라 안전. (A 실험: 노이즈가 전이갭 원인인지 검증)"""
    if rng.random() < 0.5:
        m = m[:, ::-1]
    if rng.random() < 0.5:
        m = m[::-1, :]
    m = np.rot90(m, int(rng.integers(0, 4))).copy()
    p = float(rng.random()) * max_noise          # 0~max_noise 비율의 정상다이를 불량으로
    flip = (m == 1) & (rng.random(m.shape) < p)
    m[flip] = 2
    return np.ascontiguousarray(m)


class WaferMapDataset(Dataset):
    def __init__(self, X, Y, idx, augment=False, seed=0):
        self.X = X[idx]
        self.Y = Y[idx]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        m = self.X[i]  # (52,52) in {0,1,2}
        if self.augment:
            m = _augment_map(m, self.rng)
        oh = np.stack([(m == 0), (m == 1), (m == 2)], axis=0).astype(np.float32)  # 3ch one-hot
        return torch.from_numpy(oh), torch.from_numpy(self.Y[i])


def pos_weight_from(Y, idx):
    """BCE 불균형 보정용 pos_weight = (#neg / #pos) per class."""
    y = Y[idx]
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)

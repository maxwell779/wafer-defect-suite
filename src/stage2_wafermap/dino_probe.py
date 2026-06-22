"""DINOv2 특징 프로브 — 자연이미지 사전학습 ViT가 웨이퍼맵에 전이되나?

웨이퍼맵 {0,1,2} → grayscale 3ch → 98×98(14의 배수) → DINOv2(frozen) CLS 특징(384)
→ MLP head(멀티라벨) 학습 → test macro-F1. CNN(0.88)과 비교.
가설: 웨이퍼맵은 추상 격자라 자연이미지 특징 전이 약할 것 (정직 검증).

실행:  python -m src.stage2_wafermap.dino_probe --cap 8000
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import f1_score

import config
from src.stage2_wafermap.dataset_wm811k import load_wm811k, lot_group_split

IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_img(maps, size=98):
    """(N,52,52){0,1,2} → (N,3,size,size) ImageNet 정규화."""
    g = (maps.astype(np.float32) / 2.0)          # 0,0.5,1
    out = np.zeros((len(maps), size, size), np.float32)
    for i, m in enumerate(g):
        out[i] = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    t = torch.from_numpy(out)[:, None].repeat(1, 3, 1, 1)
    return (t - IMN_M) / IMN_S


@torch.no_grad()
def extract(dino, maps, device, bs=256):
    feats = []
    for i in range(0, len(maps), bs):
        x = to_img(maps[i:i + bs]).to(device)
        feats.append(dino(x).cpu())
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=8000, help="train 특징 추출 상한(속도)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES

    try:
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
    except Exception as e:
        print(f"[중단] DINOv2 로드 실패(인터넷/캐시 필요): {e}")
        return

    X, Y, y_idx, lots = load_wm811k(normal_cap=10000, seed=config.SEED)
    tr, va, te = lot_group_split(y_idx, lots, seed=config.SEED)
    rng = np.random.default_rng(config.SEED)
    if args.cap and len(tr) > args.cap:
        tr = rng.choice(tr, args.cap, replace=False)
    print(f"[추출] DINOv2 특징 train {len(tr)} / test {len(te)} ...")
    Ftr = extract(dino, X[tr], device); Fte = extract(dino, X[te], device)
    Ytr = torch.tensor(Y[tr]); Yte = Y[te]

    head = nn.Sequential(nn.Linear(Ftr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, len(cls))).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Ftr_d, Ytr_d = Ftr.to(device), Ytr.to(device)
    for ep in range(80):
        head.train(); opt.zero_grad()
        loss = lossf(head(Ftr_d), Ytr_d); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pt = torch.sigmoid(head(Fte.to(device))).cpu().numpy()
    f1 = f1_score(Yte, (pt >= 0.5).astype(int), average="macro", zero_division=0)
    print(f"\n===== DINOv2(frozen) + MLP head =====")
    print(f"  test macro-F1 = {f1:.4f}  (CNN 베이스라인 0.880과 비교)")
    print("  → 낮으면: 자연이미지 사전학습이 추상 웨이퍼맵엔 전이 약함(정직 검증)")


if __name__ == "__main__":
    main()

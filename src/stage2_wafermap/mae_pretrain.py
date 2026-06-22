"""MAE 류 자기지도 — 마스킹된 웨이퍼맵 복원으로 CNN encoder 사전학습.

SimCLR(대조)가 실패 → 복원 기반(MAE)이 격자 데이터에 더 적합한지 검증.
입력 one-hot 일부 패치를 마스킹 → 원래 클래스(0/1/2) 복원(CE). encoder = WaferCNN.features.
이후 train_real --arch cnn --init <encoder.pt> 로 파인튜닝.

실행:  python -m src.stage2_wafermap.mae_pretrain --epochs 25 --cap 150000
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from src.common.seed import set_seed
from src.stage2_wafermap.model import WaferCNN
from src.stage2_wafermap.dataset_wm811k import load_wm811k_unlabeled


class MAE(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.encoder = WaferCNN(3, 8, width).features      # (w*4, 6, 6)
        w = width
        self.decoder = nn.Sequential(
            nn.Conv2d(w * 4, w * 2, 3, 1, 1), nn.ReLU(inplace=True), nn.Upsample(size=13),
            nn.Conv2d(w * 2, w, 3, 1, 1), nn.ReLU(inplace=True), nn.Upsample(size=26),
            nn.Conv2d(w, w, 3, 1, 1), nn.ReLU(inplace=True), nn.Upsample(size=52),
            nn.Conv2d(w, 3, 3, 1, 1))                       # 3 클래스 logit/픽셀

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MaskDS(Dataset):
    def __init__(self, X, seed=0, mask_ratio=0.5, block=8):
        self.X = X; self.rng = np.random.default_rng(seed)
        self.mask_ratio = mask_ratio; self.block = block

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        m = self.X[i].astype(np.int64)                     # (52,52) {0,1,2}
        oh = np.stack([(m == 0), (m == 1), (m == 2)], 0).astype(np.float32)
        masked = oh.copy()
        nb = 52 // self.block
        for by in range(nb):
            for bx in range(nb):
                if self.rng.random() < self.mask_ratio:
                    masked[:, by*self.block:(by+1)*self.block, bx*self.block:(bx+1)*self.block] = 0
        return torch.from_numpy(masked), torch.from_numpy(m)  # 입력(마스킹), 타깃(클래스)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--cap", type=int, default=150000)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = config.EXPERIMENTS / "stage2_mae"; out.mkdir(parents=True, exist_ok=True)

    print(f"[load] 미라벨 {args.cap} ...")
    X = load_wm811k_unlabeled(cap=args.cap, seed=config.SEED)
    dl = DataLoader(MaskDS(X, seed=config.SEED), batch_size=args.batch, shuffle=True,
                    num_workers=0, drop_last=True, pin_memory=(device == "cuda"))
    net = MAE(args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for ep in range(1, args.epochs + 1):
        net.train(); t = time.time(); tot = 0
        for xm, y in dl:
            xm, y = xm.to(device), y.to(device)
            opt.zero_grad(); loss = lossf(net(xm), y); loss.backward(); opt.step()
            tot += loss.item() * len(xm)
        print(f"[ep {ep:02d}] recon-CE {tot/len(X):.4f} ({time.time()-t:.0f}s)")
    enc = out / "encoder.pt"
    torch.save(net.encoder.state_dict(), enc)   # WaferCNN.features 에 로드 가능
    print(f"[saved] {enc}")


if __name__ == "__main__":
    main()

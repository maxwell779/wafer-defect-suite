"""Stage 2-C — WM-811K 미라벨(63.8만)로 SimCLR 자기지도 사전학습.

WaferCNN.features 와 동일 백본을 대조학습으로 사전학습 → encoder.pt 저장.
이후 train_real.py --init 으로 파인튜닝(특히 저라벨/희귀클래스에서 이득 기대).
(버스바 ReConPatch 대조학습 경험의 연장선)

실행:
    python -m src.stage2_wafermap.ssl_pretrain --epochs 30 --cap 150000
"""
from __future__ import annotations
import argparse, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config
from src.common.seed import set_seed
from src.stage2_wafermap.model import _block
from src.stage2_wafermap.dataset import _augment_map
from src.stage2_wafermap.dataset_wm811k import load_wm811k_unlabeled


class SimCLRNet(nn.Module):
    """WaferCNN.features 와 동일 백본 + projection head (대조학습용)."""
    def __init__(self, width=32, proj=128):
        super().__init__()
        self.features = nn.Sequential(_block(3, width), _block(width, width * 2),
                                      _block(width * 2, width * 4))
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 4, width * 4), nn.ReLU(inplace=True),
            nn.Linear(width * 4, proj),
        )

    def forward(self, x):
        return F.normalize(self.proj(self.features(x)), dim=1)


class SSLPairDataset(Dataset):
    """같은 맵의 두 증강 뷰 반환 (대조학습 양성쌍)."""
    def __init__(self, X, seed=0):
        self.X = X
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def _oh(self, m):
        return np.stack([(m == 0), (m == 1), (m == 2)], 0).astype(np.float32)

    def __getitem__(self, i):
        m = self.X[i]
        v1 = self._oh(_augment_map(m, self.rng))
        v2 = self._oh(_augment_map(m, self.rng))
        return torch.from_numpy(v1), torch.from_numpy(v2)


def nt_xent(z1, z2, temp=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)                      # (2N,D) L2-normalized
    sim = (z @ z.t()) / temp
    sim.masked_fill_(torch.eye(2 * N, device=z.device, dtype=torch.bool), -9e15)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cap", type=int, default=150000, help="미라벨 표본 수")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = config.EXPERIMENTS / "stage2_ssl"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] WM-811K 미라벨 {args.cap} ...")
    t = time.time()
    X = load_wm811k_unlabeled(cap=args.cap, seed=config.SEED)
    print(f"  {len(X)} maps ({time.time()-t:.0f}s)")
    dl = DataLoader(SSLPairDataset(X, seed=config.SEED), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=(device == "cuda"))

    net = SimCLRNet(width=args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        net.train(); t = time.time(); tot = 0.0
        for v1, v2 in dl:
            v1, v2 = v1.to(device), v2.to(device)
            opt.zero_grad()
            loss = nt_xent(net(v1), net(v2), temp=args.temp)
            loss.backward(); opt.step()
            tot += loss.item() * v1.size(0)
        sched.step()
        print(f"[ep {ep:02d}] nt-xent {tot/len(X):.4f} ({time.time()-t:.0f}s)")

    enc = out / "encoder.pt"
    torch.save(net.features.state_dict(), enc)   # WaferCNN.features 에 그대로 로드 가능
    print(f"[saved] {enc}")


if __name__ == "__main__":
    main()

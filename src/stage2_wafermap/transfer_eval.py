"""합성→실제 전이 평가.

MixedWM38(합성)로 학습한 모델을 WM-811K(실제, LSWMD.pkl) 단일결함 맵에 적용.
- 가변크기 맵 → 52x52 nearest 리사이즈(범주값 {0,1,2} 보존)
- 학습과 동일 3채널 one-hot 입력
- 평가: per-class(멀티라벨 지표) + top-1 정확도 + 혼동행렬
  (도메인 갭으로 0.5 임계가 안 맞을 수 있어, 임계 무관한 top-1/AP를 함께 본다)

실행:
    python -m src.stage2_wafermap.transfer_eval --ckpt experiments/stage2_asl_w32/best.pt
    python -m src.stage2_wafermap.transfer_eval --cap 1000        # 클래스당 표본 상한(속도)
    python -m src.stage2_wafermap.transfer_eval --include-normal  # 'none'으로 오탐(FP)도 측정
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import cv2
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix

import config
from src.common.metrics import multilabel_report, format_report
from src.stage2_wafermap.model import WaferCNN


def _flat(x):
    a = np.array(x).ravel()
    return a[0] if a.size else "NA"


def to_input(m, size=52):
    """가변크기 {0,1,2} 맵 → (3,size,size) one-hot float32 (nearest 리사이즈)."""
    m = np.asarray(m).astype(np.uint8)
    r = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.stack([(r == 0), (r == 1), (r == 2)], axis=0).astype(np.float32)


def load_real_single(pkl_path, classes, cap=0, include_normal=False):
    """WM-811K에서 단일결함(+옵션 normal) 맵 추출. returns X(N,3,52,52), y_idx(N,), names."""
    df = pd.read_pickle(pkl_path)
    ft = df["failureType"].map(_flat)
    wanted = list(classes) + (["none"] if include_normal else [])
    X, y_idx = [], []
    for ci, name in enumerate(wanted):
        idx = df.index[ft == name]
        if cap:
            idx = idx[:cap]
        for i in idx:
            X.append(to_input(df["waferMap"].loc[i]))
            y_idx.append(ci if name != "none" else -1)  # normal → -1
        print(f"    {name:10s} {len(idx)}")
    del df
    return np.stack(X), np.array(y_idx), wanted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/stage2_asl_w32/best.pt")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--cap", type=int, default=0, help="클래스당 표본 상한(0=전체)")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--include-normal", action="store_true")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES
    out = config.EXPERIMENTS / "stage2_transfer"
    out.mkdir(parents=True, exist_ok=True)

    print("[load] WM-811K 단일결함 추출 + 52x52 리사이즈 ...")
    t = time.time()
    X, y_idx, wanted = load_real_single(config.LSWMD_PKL, cls,
                                        cap=args.cap, include_normal=args.include_normal)
    print(f"  total {len(X)} maps  ({time.time()-t:.0f}s)")

    # ── 추론 ──────────────────────────────────────────────────────────
    model = WaferCNN(in_ch=3, n_classes=len(cls), width=args.width).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            xb = torch.from_numpy(X[i:i + args.batch]).to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    probs = np.concatenate(probs)

    # ── 결함 표본만으로 per-class / top-1 ─────────────────────────────
    dmask = y_idx >= 0  # 결함(=단일라벨)만
    yd, pd_ = y_idx[dmask], probs[dmask]
    Y = np.zeros_like(pd_); Y[np.arange(len(yd)), yd] = 1.0  # multi-hot(단일)

    summ, per = multilabel_report(Y, pd_, cls, thresh=args.thresh)
    pred_top1 = pd_.argmax(1)
    top1 = float((pred_top1 == yd).mean())

    lines = ["===== 합성→실제 전이 (WM-811K 단일결함) =====",
             format_report(summ, per),
             f"\nTop-1 accuracy (argmax==true): {top1:.4f}  (n={len(yd)})",
             "\n혼동행렬 (행=실제, 열=예측 argmax):",
             "        " + " ".join(f"{c[:4]:>5}" for c in cls)]
    cm = confusion_matrix(yd, pred_top1, labels=list(range(len(cls))))
    for i, c in enumerate(cls):
        lines.append(f"  {c[:7]:8s}" + " ".join(f"{cm[i, j]:5d}" for j in range(len(cls))))

    if args.include_normal:  # 'none'에 대한 오탐(FP) 측정
        nmask = y_idx < 0
        if nmask.any():
            fp = (probs[nmask] >= args.thresh).any(1).mean()
            lines.append(f"\nnormal('none') 오탐율 @thr {args.thresh}: {fp:.4f}  (n={int(nmask.sum())})")

    report = "\n".join(lines)
    print("\n" + report)
    (out / "transfer_report.txt").write_text(report, encoding="utf-8")
    print(f"\n[saved] {out/'transfer_report.txt'}")


if __name__ == "__main__":
    main()

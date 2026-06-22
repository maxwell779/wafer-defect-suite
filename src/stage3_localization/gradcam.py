"""Stage 3-B — 실데이터 결함 위치탐지 (Grad-CAM, 합성 미사용).

Stage2 실데이터 모델(WaferCNN)에 Grad-CAM을 걸어 WM-811K 실제 맵의
"어느 부분이 결함인지"를 클래스별 히트맵으로 표시 → "where(실데이터)".

실행: python -m src.stage3_localization.gradcam --ckpt experiments/stage2_real_asl/best.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.dataset_wm811k import load_wm811k

WMAP_CMAP = ListedColormap(["#e5e7eb", "#9cc3ff", "#ef4444"])


def gradcam(model, x, target, layer=None):
    """x:(1,3,52,52). returns cam (52,52) in [0,1] for target class."""
    acts, grads = {}, {}
    if layer is None:                       # CNN 기본: block2 마지막 ReLU
        layer = model.features[2][5]
    def fwd(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))
    h = layer.register_forward_hook(fwd)
    model.zero_grad()
    logit = model(x)[0, target]
    logit.backward()
    h.remove()
    A, G = acts["a"][0], grads["g"][0]       # (C,13,13)
    w = G.mean(dim=(1, 2))                    # (C,)
    cam = F.relu((w[:, None, None] * A).sum(0))
    cam = F.interpolate(cam[None, None], size=(52, 52), mode="bilinear",
                        align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/stage2_real_asl_resnet/best.pt")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--arch", choices=["cnn", "resnet", "resnet_cbam"], default="resnet")
    ap.add_argument("--per-class", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = config.WM_CLASSES
    out = config.EXPERIMENTS / "stage3_localization"; out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.arch, in_ch=3, n_classes=len(cls), width=args.width).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    layer = model.layers[-1] if args.arch != "cnn" else None  # ResNet 마지막 블록

    print("[load] WM-811K 실제 맵 (클래스별 샘플) ...")
    X, Y, y_idx, lots = load_wm811k(include_normal=False, seed=config.SEED)

    picks = []
    for c in range(len(cls)):
        idx = np.where(y_idx == c)[0]
        picks += [(c, int(i)) for i in idx[:args.per_class]]

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))
    for k, (c, i) in enumerate(picks):
        m = X[i]
        oh = np.stack([(m == 0), (m == 1), (m == 2)], 0).astype(np.float32)
        x = torch.from_numpy(oh)[None].to(device)
        cam = gradcam(model, x, c, layer)
        axes[0, k].imshow(m, cmap=WMAP_CMAP, vmin=0, vmax=2)
        axes[0, k].set_title(cls[c], fontsize=8); axes[0, k].axis("off")
        axes[1, k].imshow(m, cmap=WMAP_CMAP, vmin=0, vmax=2)
        axes[1, k].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, k].axis("off")
        # 개별 저장(웹 자산)
        f2, a2 = plt.subplots(figsize=(2.2, 2.2))
        a2.imshow(m, cmap=WMAP_CMAP, vmin=0, vmax=2); a2.imshow(cam, cmap="jet", alpha=0.5); a2.axis("off")
        f2.savefig(out / f"cam_{cls[c]}.png", dpi=80, bbox_inches="tight", pad_inches=0); plt.close(f2)
    axes[0, 0].set_ylabel("map"); axes[1, 0].set_ylabel("Grad-CAM")
    fig.suptitle("실데이터(WM-811K) 결함 위치탐지 — 상:원본 / 하:Grad-CAM(빨강=결함근거)")
    fig.tight_layout(); fig.savefig(out / "localization_montage.png", dpi=90, bbox_inches="tight")
    print(f"[saved] {out}/localization_montage.png + cam_*.png ({n}장)")


if __name__ == "__main__":
    main()

"""백엔드용 샘플 맵 추출 → backend/samples.npz (런타임에 2GB pkl 미로딩).
실데이터 8클래스 1개씩 + 합성 일부. 실행: python -m backend.prep_samples
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.stage2_wafermap.dataset_wm811k import load_wm811k
from src.stage2_wafermap.dataset import load_mixedwm38

OUT = Path(__file__).resolve().parent / "samples.npz"


def main():
    cls = config.WM_CLASSES
    X, Y, y_idx, lots = load_wm811k(include_normal=False, seed=config.SEED)
    real_maps, real_lbl = [], []
    for c in range(len(cls)):
        idx = np.where(y_idx == c)[0]
        if len(idx):
            real_maps.append(X[idx[0]].astype(np.int8)); real_lbl.append(c)

    Xs, Ys = load_mixedwm38(config.MIXEDWM38_NPZ)
    nlab = Ys.sum(1); synth_maps, synth_lbl = [], []
    for c in range(8):
        idx = np.where((nlab == 1) & (Ys[:, c] == 1))[0]
        if len(idx):
            synth_maps.append(Xs[idx[0]].astype(np.int8)); synth_lbl.append([c])

    np.savez_compressed(OUT,
                        real_maps=np.stack(real_maps), real_lbl=np.array(real_lbl),
                        synth_maps=np.stack(synth_maps),
                        synth_lbl=np.array([l[0] for l in synth_lbl]))
    print(f"[saved] {OUT}  real {len(real_maps)} / synth {len(synth_maps)}")


if __name__ == "__main__":
    main()

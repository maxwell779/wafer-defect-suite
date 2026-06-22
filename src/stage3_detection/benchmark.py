"""Stage 3-A (부록) — ELLIMAC 검출: 라벨 정제 + 제공 bestV2.pt 벤치마크.

★ ELLIMAC은 Roboflow 생성(합성) 데이터 → 실전 일반화 보장 없음(명시).
정제: 폴리곤(세그) 라벨 → bbox 변환, 범위초과 cls6 라인 제거.
벤치마크: 제공 bestV2.pt 로 test mAP@0.5.

실행: python -m src.stage3_detection.benchmark
"""
from __future__ import annotations
import glob, shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

import config

SRC = config.ELLIMAC_DIR
CKPT = SRC / "model/Model/bestV2.pt"
NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Scratch"]


def poly_to_bbox(line):
    t = line.split()
    if not t:
        return None
    c = int(float(t[0]))
    if c > 5:                       # nc=6(0~5) 범위초과 = 라벨노이즈 → 제거
        return None
    co = list(map(float, t[1:]))
    if len(co) == 4:               # 이미 bbox
        return f"{c} {co[0]} {co[1]} {co[2]} {co[3]}"
    xs, ys = co[0::2], co[1::2]    # 폴리곤 → 외접 bbox
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    clip = lambda v: max(0.0, min(1.0, v))
    return f"{c} {clip(cx):.6f} {clip(cy):.6f} {clip(w):.6f} {clip(h):.6f}"


def build_clean_test(ds):
    """test split을 정제(bbox)해 ds/{images,labels}/val 로 구성."""
    (ds / "images/val").mkdir(parents=True, exist_ok=True)
    (ds / "labels/val").mkdir(parents=True, exist_ok=True)
    imgs = sorted(glob.glob(str(SRC / "images/images/test/*.jpg")))
    kept_lines = dropped = 0
    for ip in imgs:
        stem = Path(ip).stem
        shutil.copy(ip, ds / "images/val" / Path(ip).name)
        lp = SRC / "labels/labels/test" / f"{stem}.txt"
        out = []
        if lp.exists():
            for line in open(lp):
                b = poly_to_bbox(line)
                if b:
                    out.append(b); kept_lines += 1
                elif line.split():
                    dropped += 1
        (ds / "labels/val" / f"{stem}.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"  test 이미지 {len(imgs)} | bbox {kept_lines} | 제거(cls6 등) {dropped}")
    return len(imgs)


def main():
    out = config.EXPERIMENTS / "stage3_detection"
    ds = out / "ds_test"; ds.mkdir(parents=True, exist_ok=True)
    print("[정제] 폴리곤→bbox + cls6 제거 (test) ...")
    build_clean_test(ds)

    yml = ds / "data.yaml"
    yaml.safe_dump({"path": str(ds.resolve()), "train": "images/val", "val": "images/val",
                    "names": {i: n for i, n in enumerate(NAMES)}},
                   open(yml, "w", encoding="utf-8"), allow_unicode=True)

    print(f"[벤치마크] bestV2.pt val ...")
    model = YOLO(str(CKPT))
    r = model.val(data=str(yml), project=str(out), name="bestV2_eval",
                  exist_ok=True, verbose=False)
    print("\n===== ELLIMAC bestV2 (test, 합성데이터) =====")
    print(f"  mAP@0.5      : {r.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {r.box.map:.4f}")
    print(f"  precision    : {r.box.mp:.4f}  recall: {r.box.mr:.4f}")
    for i, n in enumerate(NAMES):
        try:
            print(f"    {n:10s} AP50={r.box.ap50[i]:.3f}")
        except Exception:
            pass
    print("\n※ ELLIMAC=Roboflow 합성 → 실전 일반화 보장 없음(부록·스킬 데모).")


if __name__ == "__main__":
    main()

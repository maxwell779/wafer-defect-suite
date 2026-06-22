"""Stage 3-A — ELLIMAC 표준 6클래스로 YOLO11 정식 학습 (bestV2 0.739와 비교).

데이터 그대로 = 패키지 dataV2.yaml의 6클래스(Center/Donut/Edge-Loc/Edge-Ring/Loc/Scratch),
커뮤니티 표준 사용법. 폴리곤→bbox 변환, drop stray cls6.
(데이터셋 자체 내부 불일치 메모는 docs 참고; 여기선 표준대로만)

실행:  python -m src.stage3_detection.train_yolo --epochs 50
"""
from __future__ import annotations
import argparse, glob, shutil
from pathlib import Path
import yaml
from ultralytics import YOLO
import config

SRC = config.ELLIMAC_DIR
# 표준 6클래스 = 패키지 dataV2.yaml (커뮤니티·bestV2 동일). 데이터 그대로 사용.
NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Scratch"]


def poly_to_bbox(line):
    t = line.split()
    if not t:
        return None
    c = int(float(t[0]))
    if not (0 <= c <= 5):          # 6-class 표준(yaml과 동일), 드문 stray cls6 제외
        return None
    co = list(map(float, t[1:]))
    if len(co) == 4:
        return f"{c} {co[0]} {co[1]} {co[2]} {co[3]}"
    xs, ys = co[0::2], co[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    clip = lambda v: max(0.0, min(1.0, v))
    return f"{c} {clip((x0+x1)/2):.6f} {clip((y0+y1)/2):.6f} {clip(x1-x0):.6f} {clip(y1-y0):.6f}"


def build(ds):
    splits = {"train": "train", "valid": "val", "test": "test"}
    for srcs, dst in splits.items():
        (ds / f"images/{dst}").mkdir(parents=True, exist_ok=True)
        (ds / f"labels/{dst}").mkdir(parents=True, exist_ok=True)
        for ip in glob.glob(str(SRC / f"images/images/{srcs}/*.jpg")):
            stem = Path(ip).stem
            shutil.copy(ip, ds / f"images/{dst}" / Path(ip).name)
            lp = SRC / f"labels/labels/{srcs}/{stem}.txt"
            out = []
            if lp.exists():
                for line in open(lp):
                    b = poly_to_bbox(line)
                    if b:
                        out.append(b)
            (ds / f"labels/{dst}" / f"{stem}.txt").write_text("\n".join(out), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50, help="조기종료(개선없을시)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--name", default="yolo11")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reuse-ds", action="store_true", help="ds7 이미 있으면 빌드 스킵")
    args = ap.parse_args()
    out = config.EXPERIMENTS / "stage3_detection"
    ds = out / "ds7"
    if args.reuse_ds and (ds / "data.yaml").exists():
        print("[정제] ds7 재사용(스킵)")
    else:
        print("[정제] 6클래스 폴리곤→bbox (전 split) ...")
        build(ds)
    yml = ds / "data.yaml"
    yaml.safe_dump({"path": str(ds.resolve()), "train": "images/train", "val": "images/val",
                    "test": "images/test", "names": {i: n for i, n in enumerate(NAMES)}},
                   open(yml, "w", encoding="utf-8"), allow_unicode=True)
    print(f"[학습] YOLO11 {args.epochs}ep imgsz{args.imgsz} (7클래스) ...")
    model = YOLO(args.model)
    # busbar parity 최대성능: 200ep+patience 조기종료, cos_lr, imgsz1280, 기본 강증강(mosaic 등)
    model.train(data=str(yml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, cos_lr=True, optimizer="auto", iou=0.45,
                project=str(out), name=args.name, exist_ok=True, verbose=False, plots=False)
    m = model.val(data=str(yml), split="test", project=str(out), name=args.name + "_test", exist_ok=True, verbose=False)
    print(f"\n===== {args.name} (ELLIMAC 6클래스, test) =====")
    print(f"  mAP@0.5={m.box.map50:.4f}  mAP@0.5:0.95={m.box.map:.4f}  P={m.box.mp:.3f} R={m.box.mr:.3f}")
    for i, n in enumerate(NAMES):
        try:
            print(f"    {n:18s} AP50={m.box.ap50[i]:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

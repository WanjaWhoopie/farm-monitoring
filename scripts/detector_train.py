"""Fine-tune a nano YOLO detector on the exported pig frames, score it, and track pigs with ByteTrack.

Steps:
1. train: start from pretrained weights (default yolo11n.pt), train on the split's train list,
   validate on its val list; keep best.pt and the metrics.
2. track: run the trained detector with ByteTrack over every clip video present on this machine
   and write one row per detected box: clip_id, frame, track_id, x, y, w, h, conf
   (frame index = annotation index in the label table).
3. time: measure milliseconds per frame on the GPU and on the CPU, for the resource story.

Training runs live under data/processed/vision/yolo_runs (ignored by git); a summary, the
weights and the track CSVs are copied to reports/detector_<date>/.
Usage: python scripts/detector_train.py --data data/processed/vision/yolo_pig/split_time.yaml [--epochs 30]
       python scripts/detector_train.py --track-only --weights reports/detector_<date>/best.pt
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import DATA_ROOT, ROOT, SEED  # noqa: E402

RUNS = ROOT / "data" / "processed" / "vision" / "yolo_runs"


def clips_with_video() -> list[str]:
    return sorted(p.parent.parent.name + "/" + p.parent.name for p in (DATA_ROOT / "annotated").glob("*/*/color.mp4"))


def train(args, out: Path) -> Path:
    from ultralytics import YOLO
    model = YOLO(args.weights)
    res = model.train(data=str(ROOT / args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=SEED,
                      deterministic=True, project=str(RUNS), name=args.name, exist_ok=True, patience=args.patience,
                      device=0 if torch.cuda.is_available() else "cpu", workers=2, verbose=False, plots=True)
    best = Path(res.save_dir) / "weights" / "best.pt"
    shutil.copy(best, out / "best.pt")
    metrics = YOLO(str(best)).val(data=str(ROOT / args.data), imgsz=args.imgsz, batch=args.batch, plots=False, verbose=False,
                                  project=str(RUNS), name=args.name + "_val", exist_ok=True)
    summary = {"map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4),
               "precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4),
               "epochs": args.epochs, "imgsz": args.imgsz, "weights_start": args.weights, "data": args.data,
               "train_images": sum(1 for _ in open(ROOT / Path(args.data).parent / (Path(args.data).stem + "_train.txt"))),
               "val_images": sum(1 for _ in open(ROOT / Path(args.data).parent / (Path(args.data).stem + "_val.txt")))}
    for f in ("results.csv", "results.png", "confusion_matrix.png", "val_batch0_pred.jpg"):
        if (Path(res.save_dir) / f).exists():
            shutil.copy(Path(res.save_dir) / f, out / f)
    print("validation:", summary)
    return out / "best.pt", summary


def track(weights: Path, out: Path, imgsz: int, conf: float) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    stats = {}
    for clip in clips_with_video():
        rows = []
        t0 = time.time()
        for n, r in enumerate(model.track(source=str(DATA_ROOT / "annotated" / clip / "color.mp4"), tracker="bytetrack.yaml",
                                          persist=True, stream=True, conf=conf, imgsz=imgsz, verbose=False)):
            if r.boxes is None or r.boxes.id is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy(); ids = r.boxes.id.cpu().numpy().astype(int); cf = r.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), tid, c in zip(xyxy, ids, cf):
                rows.append({"clip_id": clip, "frame": n, "track_id": int(tid), "x": float(x0), "y": float(y0),
                             "width": float(x1 - x0), "height": float(y1 - y0), "conf": float(c)})
        df = pd.DataFrame(rows)
        df.to_csv(out / f"track_{clip.replace('/', '_')}.csv", index=False)
        stats[clip] = {"frames": n + 1, "boxes": len(df), "track_ids": int(df["track_id"].nunique()) if len(df) else 0,
                       "boxes_per_frame": round(len(df) / (n + 1), 2), "seconds": round(time.time() - t0, 1)}
        print("tracked", clip, stats[clip])
    return stats


def timing(weights: Path, imgsz: int, n_frames: int = 60) -> dict:
    from ultralytics import YOLO
    import cv2
    clip = clips_with_video()[0]
    cap = cv2.VideoCapture(str(DATA_ROOT / "annotated" / clip / "color.mp4"))
    frames = [cap.read()[1] for _ in range(n_frames)]
    cap.release()
    out = {}
    for dev in (["0"] if torch.cuda.is_available() else []) + ["cpu"]:
        model = YOLO(str(weights))
        model.predict(frames[0], imgsz=imgsz, device=dev, verbose=False)  # warm-up
        t0 = time.time()
        for f in frames:
            model.predict(f, imgsz=imgsz, device=dev, verbose=False)
        out["gpu" if dev == "0" else "cpu"] = round(1000 * (time.time() - t0) / n_frames, 1)
    print("ms per frame:", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/vision/yolo_pig/split_time.yaml")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--name", default="pig_nano")
    ap.add_argument("--track-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "reports" / f"detector_{datetime.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)

    info = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "clips": clips_with_video()}
    if args.track_only:
        weights = Path(args.weights)
    else:
        weights, info["validation"] = train(args, out)
    info["tracking"] = track(weights, out, args.imgsz, args.conf)
    info["ms_per_frame"] = timing(weights, args.imgsz)
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("saved to", out)


if __name__ == "__main__":
    main()

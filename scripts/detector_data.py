"""Export frames and YOLO-format box labels from the Edinburgh clips that are on this machine.

One image per frame on the 2 Hz label grid (every third captured frame, where the human labels
sit), with one box per pig taken from the per-frame label table. Rows the label gate rejected
(the pig moved between two human records, so the interpolated box is unreliable) are still
written, but the manifest records how many per frame so training can be limited to clean frames.

Output (under data/, ignored by git):
  data/processed/vision/yolo_pig/images/<clip>_<frame>.jpg
  data/processed/vision/yolo_pig/labels/<clip>_<frame>.txt      class 0 = pig, normalised cx cy w h
  data/processed/vision/yolo_pig/manifest.csv                   image, clip, frame, n_pigs, n_rejected
  data/processed/vision/yolo_pig/<split-name>.yaml + train/val lists

Usage: python scripts/detector_data.py [--every 3] [--split time|clip] [--val-clips CLIP,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import DATA_ROOT, ROOT, H, W, load_labels  # noqa: E402

OUT = ROOT / "data" / "processed" / "vision" / "yolo_pig"


def tag(clip: str) -> str:
    return clip.replace("/", "_")


def export_clip(clip: str, labels: pd.DataFrame, every: int, quality: int) -> list[dict]:
    rows = []
    sub = labels[labels["clip_id"] == clip]
    wanted = sorted(set(sub["annotation_index"][sub["annotation_index"] % every == 0]))
    cap = cv2.VideoCapture(str(DATA_ROOT / "annotated" / clip / "color.mp4"))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n in wanted:
            fr = sub[sub["annotation_index"] == n]
            name = f"{tag(clip)}_{n:04d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            lines = []
            for _, r in fr.iterrows():
                x0, y0 = max(r["x"], 0), max(r["y"], 0)
                x1, y1 = min(r["x"] + r["width"], W), min(r["y"] + r["height"], H)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                lines.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            (OUT / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            rows.append({"image": f"images/{name}.jpg", "clip_id": clip, "frame": n, "n_pigs": len(lines),
                         "n_rejected": int((fr["tier"] == "rejected").sum())})
        n += 1
    cap.release()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=3, help="keep every k-th captured frame (3 = the 2 Hz label grid)")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--split", choices=["time", "clip"], default="time",
                    help="time: first 70%% of each clip trains, last 30%% validates (same day!); clip: hold out whole clips")
    ap.add_argument("--val-clips", default="", help="comma-separated clip ids for --split clip")
    ap.add_argument("--clean-only", action="store_true", help="train only on frames where no pig's box was rejected by the gate")
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    clips = [c for c in sorted(labels["clip_id"].unique()) if (DATA_ROOT / "annotated" / c / "color.mp4").exists()]
    print("clips with video:", clips)
    rows = []
    for clip in clips:
        r = export_clip(clip, labels, args.every, args.quality)
        print(f"  {clip}: {len(r)} frames, {sum(x['n_rejected'] == 0 for x in r)} clean")
        rows += r
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)

    if args.split == "time":
        train, val = [], []
        for clip, g in man.groupby("clip_id"):
            g = g.sort_values("frame")
            cut = int(len(g) * 0.7)
            train += list(g["image"][:cut]); val += list(g["image"][cut:])
        name = "split_time"
    else:
        vc = set(args.val_clips.split(",")) if args.val_clips else set()
        train = list(man[~man["clip_id"].isin(vc)]["image"]); val = list(man[man["clip_id"].isin(vc)]["image"])
        name = "split_clip"
    if args.clean_only:
        clean = set(man[man["n_rejected"] == 0]["image"])
        train = [t for t in train if t in clean]
    (OUT / f"{name}_train.txt").write_text("\n".join(str(OUT / t) for t in train) + "\n", encoding="utf-8")
    (OUT / f"{name}_val.txt").write_text("\n".join(str(OUT / t) for t in val) + "\n", encoding="utf-8")
    (OUT / f"{name}.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: {name}_train.txt\nval: {name}_val.txt\nnames:\n  0: pig\n", encoding="utf-8")
    print(f"wrote {name}.yaml: {len(train)} train, {len(val)} val images")


if __name__ == "__main__":
    main()

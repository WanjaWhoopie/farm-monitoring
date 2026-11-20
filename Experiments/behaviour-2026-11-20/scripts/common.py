"""Annotated clips, the date split, and where a local run writes its tables.

Point ``--data`` at the folder of annotated clips. Each clip folder contains
``output.json``, ``times.txt``, and ``color.mp4``. The public annotated set is
the Edinburgh pig behaviour clips. 2019-12-02 is never used for training or
for choosing a model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

CLASSES = ["sleep", "investigating", "eat", "standing", "lying", "walk", "drink"]
BOX_FEATURES = [
    "cx",
    "cy",
    "w",
    "h",
    "log_area",
    "aspect",
    "speed_px_s",
    "nearest_pig_px",
    "max_overlap_other",
]
HELD_OUT_DATE = "2019_12_02"
SEED = 42
SPLIT_ORDER = ["train", "validation", "test"]
SPLIT_DATES = {
    "train": ["2019_11_05", "2019_11_15", "2019_11_22", "2019_11_28", "2019_12_10"],
    "validation": ["2019_11_11"],
    "test": ["2019_12_02"],
}
DATE_TO_SPLIT = {date: split for split, dates in SPLIT_DATES.items() for date in dates}
ACCEPTED_SOURCES = {"stored", "trailing_hold"}
EXPECTED_BASELINE_COUNTS = np.array(
    [
        [979, 159, 935],
        [2984, 705, 569],
        [231, 236, 514],
        [1594, 93, 140],
        [467, 223, 17],
        [2226, 239, 272],
        [134, 29, 48],
    ]
)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PAD_COLOR = (124, 116, 104)
PIG_VIDEO = Path(__file__).resolve().parents[1]


def add_run_arguments(parser: argparse.ArgumentParser, default_name: str) -> None:
    parser.add_argument(
        "--data",
        required=True,
        help="Folder of annotated clips. Each clip has output.json, times.txt, and color.mp4.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PIG_VIDEO / "outputs" / default_name,
        help="Folder for history.csv, validation_report.csv, and summary.json.",
    )


def resolve_run(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.data).expanduser().resolve()
    if not root.is_dir() or not any(root.rglob("output.json")):
        raise FileNotFoundError(f"No output.json under {root}.")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return root, out


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reconstruct(root: Path) -> pd.DataFrame:
    """One row per pig per annotation slot: 12 clips x 8 pigs x 600 slots."""
    rows = []
    for annotation_path in sorted(root.rglob("output.json")):
        clip_dir = annotation_path.parent
        clip_id = f"{clip_dir.parent.name}/{clip_dir.name}"
        with annotation_path.open(encoding="utf-8") as handle:
            annotation = json.load(handle)
        with (clip_dir / "times.txt").open(encoding="utf-8") as handle:
            timestamps = pd.DatetimeIndex(pd.to_datetime([line.strip() for line in handle if line.strip()]))
        for track in annotation["objects"]:
            entries = sorted(track["frames"], key=lambda entry: entry["frameNumber"])
            stored_indices = np.array([entry["frameNumber"] for entry in entries])
            target_indices = np.arange(600)
            previous = np.searchsorted(stored_indices, target_indices, side="right") - 1
            if (previous < 0).any():
                raise ValueError(f"Missing starting label: {clip_id}")
            boxes = {
                key: np.interp(target_indices, stored_indices, [entry["bbox"][key] for entry in entries])
                for key in ("x", "y", "width", "height")
            }
            stored_set = set(stored_indices.tolist())
            last_stored = int(stored_indices[-1])
            for index in target_indices:
                index = int(index)
                if index in stored_set:
                    source = "stored"
                elif index > last_stored:
                    source = "trailing_hold"
                else:
                    source = "interpolated_box_forward_label"
                rows.append(
                    {
                        "clip_id": clip_id,
                        "pig_id": str(track["id"]),
                        "annotation_index": index,
                        "raw_frame_index": index * 3,
                        "behavior": entries[int(previous[index])]["behaviour"],
                        "x": float(boxes["x"][index]),
                        "y": float(boxes["y"][index]),
                        "width": float(boxes["width"][index]),
                        "height": float(boxes["height"][index]),
                        "annotation_source": source,
                    }
                )
    labels = pd.DataFrame(rows)
    if len(labels) != 57600:
        raise ValueError(f"Expected 57600 reconstructed rows, found {len(labels)}.")
    labels["observation_id"] = [f"obs_{index:06d}" for index in range(len(labels))]
    labels["recording_date"] = labels["clip_id"].str.split("/").str[0]
    labels["split"] = labels["recording_date"].map(DATE_TO_SPLIT)
    if labels["split"].isna().any():
        raise ValueError("A recording date is outside the candidate 11 split.")
    return labels


def accepted_baseline(labels: pd.DataFrame, stored_only: bool = False) -> pd.DataFrame:
    sources = {"stored"} if stored_only else set(ACCEPTED_SOURCES)
    baseline = labels.loc[labels["annotation_source"].isin(sources) & labels["behavior"].isin(CLASSES)].copy()
    counts = (
        pd.crosstab(baseline["behavior"], baseline["split"]).reindex(index=CLASSES, columns=SPLIT_ORDER).fillna(0).astype(int)
    )
    if not stored_only and not np.array_equal(counts.to_numpy(), EXPECTED_BASELINE_COUNTS):
        raise ValueError("Baseline counts do not match the notebook checkpoint.")
    if not counts.gt(0).all().all():
        raise ValueError("A class is missing from a partition.")
    return baseline


def video_map(root: Path, frame: pd.DataFrame) -> dict:
    videos = sorted(root.rglob("color.mp4"))
    mapping = {}
    for clip_id in sorted(frame["clip_id"].unique()):
        parts = tuple(str(clip_id).replace("\\", "/").strip("/").split("/"))
        matches = [path for path in videos if tuple(path.parent.parts[-len(parts) :]) == parts]
        if len(matches) != 1:
            raise ValueError(f"{clip_id}: expected one color.mp4, found {len(matches)}.")
        mapping[clip_id] = matches[0]
    return mapping


def crop_box(frame, row):
    height, width = frame.shape[:2]
    x1 = max(0, math.floor(row.x))
    y1 = max(0, math.floor(row.y))
    x2 = min(width, math.ceil(row.x + row.width))
    y2 = min(height, math.ceil(row.y + row.height))
    if row.width <= 0 or row.height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class PadToSquare:
    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), color=PAD_COLOR)
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


def write_summary(out: Path, metrics: dict) -> None:
    (out / "summary.json").write_text(json.dumps({"metrics": metrics}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

"""Label gate, the nine box numbers, and the single-date box model.

The single-date run uses the same stored and trailing-hold rows as the crop
models: 8615 train, 1684 validation from 2019-11-11. Speed uses times.txt
across two annotation slots. No crop is encoded.

    python box_features.py --data <annotated-clips>

The leave-one-day-out scripts import ``load_tracks`` from this file. That
table drops 2019-12-02, keeps a row when it is stored, or an interpolated box
whose neighbouring human boxes are at most 20 raw frames apart, overlap by at
least half, and share a label, or a trail of at most 10 raw frames.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BOX_FEATURES,
    CLASSES,
    HELD_OUT_DATE,
    SEED,
    accepted_baseline,
    add_run_arguments,
    reconstruct,
    resolve_run,
    video_map,
    write_summary,
)


def _pair_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    ax2, ay2 = left[:, 0] + left[:, 2], left[:, 1] + left[:, 3]
    bx2, by2 = right[:, 0] + right[:, 2], right[:, 1] + right[:, 3]
    width = np.clip(np.minimum(ax2, bx2) - np.maximum(left[:, 0], right[:, 0]), 0, None)
    height = np.clip(np.minimum(ay2, by2) - np.maximum(left[:, 1], right[:, 1]), 0, None)
    intersection = width * height
    union = left[:, 2] * left[:, 3] + right[:, 2] * right[:, 3] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def apply_gate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["clip_id", "pig_id", "raw_frame_index"]).reset_index(drop=True)
    gap = np.full(len(frame), np.nan)
    trail = np.full(len(frame), np.nan)
    overlap = np.full(len(frame), np.nan)
    agree = np.zeros(len(frame), dtype=bool)
    has_right = np.zeros(len(frame), dtype=bool)
    for _, index in frame.groupby(["clip_id", "pig_id"], sort=False).indices.items():
        group = frame.iloc[index]
        raw = group["raw_frame_index"].to_numpy(int)
        stored = group["annotation_source"].eq("stored").to_numpy()
        if not stored.any():
            continue
        stored_raw = raw[stored]
        stored_box = group.loc[stored, ["x", "y", "width", "height"]].to_numpy(float)
        stored_behavior = group.loc[stored, "behavior"].to_numpy()
        left = np.clip(np.searchsorted(stored_raw, raw, side="right") - 1, 0, len(stored_raw) - 1)
        right = np.searchsorted(stored_raw, raw, side="left")
        has = right < len(stored_raw)
        right_clip = np.clip(right, 0, len(stored_raw) - 1)
        gap[index] = np.where(has, stored_raw[right_clip] - stored_raw[left], np.nan)
        trail[index] = np.where(has, np.nan, raw - stored_raw[left])
        overlap[index] = np.where(has, _pair_iou(stored_box[left], stored_box[right_clip]), np.nan)
        agree[index] = stored_behavior[left] == stored_behavior[right_clip]
        has_right[index] = has
    frame = frame.copy()
    stored_row = frame["annotation_source"].eq("stored")
    interpolated_ok = has_right & (gap <= 20) & (overlap >= 0.5) & agree
    trail_ok = ~has_right & (trail <= 10)
    frame["tier"] = np.where(stored_row, "stored", np.where(interpolated_ok | trail_ok, "gated", "rejected"))
    return frame


def attach_elapsed(frame: pd.DataFrame, videos: dict) -> pd.DataFrame:
    elapsed = np.zeros(len(frame))
    for clip_id, index in frame.groupby("clip_id").indices.items():
        stamps = pd.to_datetime(
            [line.strip() for line in (videos[clip_id].parent / "times.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        )
        origin = stamps[0]
        raw = frame.loc[frame.index[index], "raw_frame_index"].to_numpy(int)
        elapsed[index] = np.array([(stamps[i] - origin).total_seconds() for i in raw])
    frame = frame.copy()
    frame["elapsed_seconds"] = elapsed
    return frame


def _nearest_and_overlap(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nearest = np.full(len(frame), np.nan)
    overlap = np.full(len(frame), np.nan)
    for _, index in frame.groupby(["clip_id", "annotation_index"], sort=False).indices.items():
        boxes = frame.iloc[index][["x", "y", "width", "height"]].to_numpy(float)
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        distances = np.hypot(centers[:, None, 0] - centers[None, :, 0], centers[:, None, 1] - centers[None, :, 1])
        np.fill_diagonal(distances, np.inf)
        right = boxes[:, 0] + boxes[:, 2]
        bottom = boxes[:, 1] + boxes[:, 3]
        intersection_w = np.clip(
            np.minimum(right[:, None], right[None]) - np.maximum(boxes[:, 0][:, None], boxes[:, 0][None]), 0, None
        )
        intersection_h = np.clip(
            np.minimum(bottom[:, None], bottom[None]) - np.maximum(boxes[:, 1][:, None], boxes[:, 1][None]), 0, None
        )
        intersection = intersection_w * intersection_h
        area = boxes[:, 2] * boxes[:, 3]
        iou = intersection / np.maximum(area[:, None] + area[None] - intersection, 1e-9)
        np.fill_diagonal(iou, 0)
        nearest[index] = distances.min(axis=1)
        overlap[index] = iou.max(axis=1)
    return nearest, overlap


def nine_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Nine box numbers on the gated 2 Hz table. Speed requires both ends kept."""
    features = pd.DataFrame(index=frame.index)
    features["cx"] = (frame["x"] + frame["width"] / 2) / 1280
    features["cy"] = (frame["y"] + frame["height"] / 2) / 720
    features["w"] = frame["width"] / 1280
    features["h"] = frame["height"] / 720
    features["log_area"] = np.log(np.maximum(frame["width"] * frame["height"], 1))
    features["aspect"] = frame["width"] / np.maximum(frame["height"], 1)
    speed = np.full(len(frame), np.nan)
    reliable = frame["tier"].ne("rejected").to_numpy()
    for _, index in frame.groupby(["clip_id", "pig_id"], sort=False).indices.items():
        group = frame.iloc[index]
        center_x = (group["x"] + group["width"] / 2).to_numpy()
        center_y = (group["y"] + group["height"] / 2).to_numpy()
        clock = group["elapsed_seconds"].to_numpy()
        ok = reliable[index]
        count = len(group)
        earlier = np.clip(np.arange(count) - 2, 0, count - 1)
        later = np.clip(np.arange(count) + 2, 0, count - 1)
        delta_t = clock[later] - clock[earlier]
        distance = np.hypot(center_x[later] - center_x[earlier], center_y[later] - center_y[earlier])
        speed[index] = np.where(ok[earlier] & ok[later] & (delta_t > 0), distance / np.maximum(delta_t, 1e-6), np.nan)
    features["speed_px_s"] = speed
    nearest, overlap = _nearest_and_overlap(frame)
    features["nearest_pig_px"] = nearest
    features["max_overlap_other"] = overlap
    return features


def one_day_features(frame: pd.DataFrame, videos: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nine box numbers on every annotation slot, before the stored-row filter."""
    frame = frame.sort_values(["clip_id", "pig_id", "annotation_index"]).reset_index(drop=True)
    frame = attach_elapsed(frame, videos)
    features = pd.DataFrame(index=frame.index)
    features["cx"] = (frame["x"] + frame["width"] / 2) / 1280
    features["cy"] = (frame["y"] + frame["height"] / 2) / 720
    features["w"] = frame["width"] / 1280
    features["h"] = frame["height"] / 720
    features["log_area"] = np.log(np.maximum(frame["width"] * frame["height"], 1))
    features["aspect"] = frame["width"] / np.maximum(frame["height"], 1)
    speed = np.full(len(frame), np.nan)
    for _, index in frame.groupby(["clip_id", "pig_id"], sort=False).indices.items():
        group = frame.iloc[index]
        center_x = (group["x"] + group["width"] / 2).to_numpy()
        center_y = (group["y"] + group["height"] / 2).to_numpy()
        clock = group["elapsed_seconds"].to_numpy()
        count = len(group)
        earlier = np.clip(np.arange(count) - 2, 0, count - 1)
        later = np.clip(np.arange(count) + 2, 0, count - 1)
        delta_t = clock[later] - clock[earlier]
        distance = np.hypot(center_x[later] - center_x[earlier], center_y[later] - center_y[earlier])
        speed[index] = np.where(delta_t > 0, distance / delta_t, np.nan)
    features["speed_px_s"] = speed
    nearest, overlap = _nearest_and_overlap(frame)
    features["nearest_pig_px"] = nearest
    features["max_overlap_other"] = overlap
    return frame, features


def load_tracks(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Six-day 2 Hz table used by the budget, sequence, and fusion scripts."""
    labels = reconstruct(dataset_root)
    videos = video_map(dataset_root, labels)
    frame = labels.loc[labels["recording_date"].ne(HELD_OUT_DATE)].reset_index(drop=True)
    frame = apply_gate(frame)
    frame = attach_elapsed(frame, videos)
    features = nine_features(frame)
    frame["kept"] = frame["tier"].ne("rejected") & frame["behavior"].isin(CLASSES)
    print("Tier counts:", frame["tier"].value_counts().to_dict(), flush=True)
    print(f"Kept rows excluding {HELD_OUT_DATE}: {int(frame['kept'].sum())}", flush=True)
    return frame, features, videos


def run_one_day(dataset_root: Path, out: Path) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
    from sklearn.utils.class_weight import compute_sample_weight
    import joblib

    labels = reconstruct(dataset_root)
    videos = video_map(dataset_root, labels)
    accepted_ids = set(accepted_baseline(labels)["observation_id"])
    frame, features = one_day_features(labels, videos)
    kept = frame["observation_id"].isin(accepted_ids)
    frame = frame.loc[kept].reset_index(drop=True)
    features = features.loc[kept].reset_index(drop=True)
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("validation").to_numpy()
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150, random_state=SEED)
    started = time.perf_counter()
    model.fit(
        features.loc[train, BOX_FEATURES],
        frame.loc[train, "behavior"],
        sample_weight=compute_sample_weight("balanced", frame.loc[train, "behavior"]),
    )
    train_pred = model.predict(features.loc[train, BOX_FEATURES])
    val_pred = model.predict(features.loc[validation, BOX_FEATURES])
    train_f1 = float(
        precision_recall_fscore_support(
            frame.loc[train, "behavior"], train_pred, labels=CLASSES, average="macro", zero_division=0
        )[2]
    )
    val_f1 = float(
        precision_recall_fscore_support(
            frame.loc[validation, "behavior"], val_pred, labels=CLASSES, average="macro", zero_division=0
        )[2]
    )
    val_accuracy = float(accuracy_score(frame.loc[validation, "behavior"], val_pred))
    report = classification_report(
        frame.loc[validation, "behavior"],
        val_pred,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(out / "validation_report.csv")
    pd.DataFrame(
        {
            "epoch": [1],
            "epoch_seconds": [time.perf_counter() - started],
            "train_loss": [0.0],
            "train_macro_f1": [train_f1],
            "validation_loss": [0.0],
            "validation_macro_f1": [val_f1],
            "validation_accuracy": [val_accuracy],
        }
    ).to_csv(out / "history.csv", index=False)
    pd.DataFrame(
        {
            "observation_id": frame.loc[validation, "observation_id"].tolist(),
            "behavior": frame.loc[validation, "behavior"].tolist(),
            "prediction": list(val_pred),
        }
    ).to_csv(out / "validation_predictions.csv", index=False)
    joblib.dump(model, out / "best.pt")
    print(f"Geometry validation macro-F1={val_f1:.4f} accuracy={val_accuracy:.4f} train_f1={train_f1:.4f} rows={int(validation.sum())}")
    write_summary(out, {"validation_macro_f1": val_f1, "validation_accuracy": val_accuracy, "best_epoch": 1})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the nine box numbers on one validation date.")
    add_run_arguments(parser, "g1-geometry-hgb")
    args = parser.parse_args()
    dataset_root, out = resolve_run(args)
    print(f"Writing tables to {out}")
    run_one_day(dataset_root, out)


if __name__ == "__main__":
    main()

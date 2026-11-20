"""Frozen MobileNetV3-Small embeddings joined to the nine box features.

The image network stays at its ImageNet weights. Each kept crop becomes 576
numbers. Those numbers are concatenated with the nine box features, then a
logistic regression (C=0.02, balanced class weights, median imputation and
standard scaling) is trained leave-one-day-out. The network itself is not trained.

    python frozen_fusion.py --data <annotated-clips>

This reads color.mp4 and downloads ImageNet weights on the first run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CLASSES, MEAN, STD, PadToSquare, crop_box


def embed_kept(frame, videos: dict, device) -> np.ndarray:
    import cv2

    network = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device).eval()
    transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    embeddings = np.zeros((len(frame), 576), dtype=np.float32)
    failures = 0
    for clip_id, group in frame.groupby("clip_id", sort=True):
        capture = cv2.VideoCapture(str(videos[clip_id]))
        wanted = {int(raw): positions.index.tolist() for raw, positions in group.groupby("raw_frame_index")}
        pending = set(wanted)
        current = 0
        images = []
        positions = []

        def flush():
            if not images:
                return
            tensor = torch.stack(images).to(device)
            with torch.no_grad():
                encoded = network.avgpool(network.features(tensor)).flatten(1).cpu().numpy()
            embeddings[positions] = encoded
            images.clear()
            positions.clear()

        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {videos[clip_id]}")
            while pending:
                ok, image = capture.read()
                if not ok:
                    break
                if current in pending:
                    for position in wanted[current]:
                        crop = crop_box(image, frame.iloc[position])
                        if crop is None:
                            failures += 1
                            continue
                        rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        images.append(transform(rgb))
                        positions.append(int(position))
                        if len(images) == 64:
                            flush()
                    pending.remove(current)
                current += 1
            flush()
        finally:
            capture.release()
        if pending:
            raise RuntimeError(f"{clip_id}: missing {len(pending)} frames during embedding")
    if failures:
        raise RuntimeError(f"Embedding crops failed for {failures} rows")
    print(f"Frozen embeddings: {len(frame)} crops", flush=True)
    return embeddings


def leave_one_day_out(embeddings: np.ndarray, features: np.ndarray, labels: np.ndarray, days: np.ndarray) -> np.ndarray:
    combined = np.hstack([embeddings, features])
    prediction = np.empty(len(labels), dtype=object)
    for day in sorted(set(days)):
        train = days != day
        test = days == day
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.02, class_weight="balanced", max_iter=400),
        )
        model.fit(combined[train], labels[train])
        prediction[test] = model.predict(combined[test])
        print(f"Fusion fold {day}: train rows {int(train.sum())}", flush=True)
    return prediction


def main() -> None:
    from activity_budget import score_column, write_tables
    from box_features import load_tracks
    from common import BOX_FEATURES, add_run_arguments, resolve_run

    parser = argparse.ArgumentParser(description="Fit frozen embeddings joined to the nine box numbers.")
    add_run_arguments(parser, "g4-frozen-fusion")
    args = parser.parse_args()
    dataset_root, out = resolve_run(args)
    print(f"Writing tables to {out}")
    frame, features, videos = load_tracks(dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kept_mask = frame["kept"].to_numpy()
    kept = frame.loc[kept_mask].reset_index(drop=True)
    matrix = features[BOX_FEATURES].to_numpy(float)
    prediction = leave_one_day_out(
        embed_kept(kept, videos, device),
        matrix[kept_mask],
        kept["behavior"].to_numpy(object),
        kept["recording_date"].to_numpy(),
    )
    frame["pred_fusion"] = ""
    frame.loc[kept_mask, "pred_fusion"] = prediction
    row, folds, report = score_column(frame, "fusion", "pred_fusion")
    write_tables(out, [row], folds, [report])


if __name__ == "__main__":
    main()

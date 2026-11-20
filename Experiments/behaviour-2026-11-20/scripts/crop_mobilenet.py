"""Crop MobileNetV3-Small runs that produced the single-date tables.

Each setting uses the same 8615 training crops and the same 1684 validation
crops from 2019-11-11. The crop is padded to a square, resized to 224, and
normalized with ImageNet mean and standard deviation. The loss is class-weighted
cross-entropy. Patience is 4 epochs and the maximum is 15. 2019-12-02 is not scored.

Run from this folder, with the annotated clips (output.json, times.txt, color.mp4):

    python crop_mobilenet.py --data <annotated-clips> --setting frozen
    python crop_mobilenet.py --data <annotated-clips> --setting full
    python crop_mobilenet.py --data <annotated-clips> --setting split
    python crop_mobilenet.py --data <annotated-clips> --setting pixel_mean
    python crop_mobilenet.py --data <annotated-clips> --setting feature_mean

Tables are written to ``--out`` (history.csv, validation_report.csv, summary.json).
The first run downloads ImageNet weights.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CLASSES,
    MEAN,
    STD,
    PadToSquare,
    accepted_baseline,
    add_run_arguments,
    reconstruct,
    resolve_run,
    set_seed,
    video_map,
    write_summary,
)

SETTINGS = {
    "frozen": {
        "name": "e1-frozen-backbone",
        "freeze_backbone": True,
        "learning_rate": 1e-4,
        "backbone_learning_rate": None,
        "offsets": [0],
        "aggregation": "pixel_mean",
    },
    "full": {
        "name": "baseline-full-finetune",
        "freeze_backbone": False,
        "learning_rate": 1e-4,
        "backbone_learning_rate": None,
        "offsets": [0],
        "aggregation": "pixel_mean",
    },
    "split": {
        "name": "e3-differential-lr",
        "freeze_backbone": False,
        "learning_rate": 1e-4,
        "backbone_learning_rate": 1e-5,
        "offsets": [0],
        "aggregation": "pixel_mean",
    },
    "pixel_mean": {
        "name": "w1-temporal-window",
        "freeze_backbone": True,
        "learning_rate": 1e-4,
        "backbone_learning_rate": None,
        "offsets": [-2, -1, 0, 1, 2],
        "aggregation": "pixel_mean",
    },
    "feature_mean": {
        "name": "w2-feature-pool",
        "freeze_backbone": True,
        "learning_rate": 1e-4,
        "backbone_learning_rate": None,
        "offsets": [-2, -1, 0, 1, 2],
        "aggregation": "feature_mean",
    },
}


def average_window(crops, size: int = 160) -> np.ndarray:
    import cv2

    acc = np.zeros((size, size, 3), dtype=np.float32)
    for crop in crops:
        acc += cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(acc / len(crops)), 0, 255).astype(np.uint8)


def _save_crop(frame, row, destination: Path, failures: list) -> None:
    height, width = frame.shape[:2]
    x1 = max(0, int(np.floor(row.x)))
    y1 = max(0, int(np.floor(row.y)))
    x2 = min(width, int(np.ceil(row.x + row.width)))
    y2 = min(height, int(np.ceil(row.y + row.height)))
    if row.width <= 0 or row.height <= 0 or x2 <= x1 or y2 <= y1:
        failures.append(row.observation_id)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95]):
        failures.append(row.observation_id)


def extract_single_crops(baseline: pd.DataFrame, videos: dict, crop_root: Path, out: Path) -> pd.DataFrame:
    import cv2

    records = []
    failures = []
    for clip_id, group in baseline.groupby("clip_id", sort=True):
        capture = cv2.VideoCapture(str(videos[clip_id]))
        wanted = {int(frame): rows for frame, rows in group.groupby("raw_frame_index")}
        pending = set(wanted)
        current = 0
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {videos[clip_id]}")
            while pending:
                ok, frame = capture.read()
                if not ok:
                    break
                if current in pending:
                    for _, row in wanted[current].iterrows():
                        destination = crop_root / row.split / f"{row.observation_id}.jpg"
                        before = len(failures)
                        _save_crop(frame, row, destination, failures)
                        if len(failures) == before:
                            record = row.to_dict()
                            record["crop_path"] = str(destination)
                            records.append(record)
                    pending.remove(current)
                current += 1
            for index in pending:
                failures.extend(wanted[index]["observation_id"].tolist())
        finally:
            capture.release()
    if failures or len(records) != len(baseline):
        raise RuntimeError(f"Crop extraction incomplete. Failures: {len(failures)}.")
    manifest = pd.DataFrame(records)
    manifest.to_csv(out / "crop_manifest.csv", index=False)
    return manifest


def extract_window_crops(baseline, videos, labels, offsets, aggregation, crop_root: Path, out: Path) -> pd.DataFrame:
    import cv2

    boxes = {(row.clip_id, str(row.pig_id), int(row.annotation_index)): row for row in labels.itertuples(index=False)}
    plans = []
    needed = {}
    for row in baseline.itertuples(index=False):
        neighbors = []
        for offset in offsets:
            index = min(599, max(0, int(row.annotation_index) + offset))
            key = (row.clip_id, str(row.pig_id), index)
            if key not in boxes:
                raise ValueError(f"Missing window frame {key}")
            neighbor = boxes[key]
            neighbors.append(neighbor)
            slot = needed.setdefault(row.clip_id, {}).setdefault(int(neighbor.raw_frame_index), {})
            slot[(str(neighbor.pig_id), int(neighbor.annotation_index))] = neighbor
        plans.append((row, neighbors))

    part_root = crop_root / "parts"
    part_root.mkdir(parents=True, exist_ok=True)
    failures = []
    written = set()
    for clip_id, frames in needed.items():
        capture = cv2.VideoCapture(str(videos[clip_id]))
        pending = set(frames)
        current = 0
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {videos[clip_id]}")
            while pending:
                ok, frame = capture.read()
                if not ok:
                    break
                if current in pending:
                    for neighbor in frames[current].values():
                        destination = part_root / f"{clip_id.replace('/', '_')}__{neighbor.pig_id}__{int(neighbor.annotation_index)}.jpg"
                        crop = None
                        height, width = frame.shape[:2]
                        x1 = max(0, int(np.floor(neighbor.x)))
                        y1 = max(0, int(np.floor(neighbor.y)))
                        x2 = min(width, int(np.ceil(neighbor.x + neighbor.width)))
                        y2 = min(height, int(np.ceil(neighbor.y + neighbor.height)))
                        if neighbor.width > 0 and neighbor.height > 0 and x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                        if crop is None or not cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            failures.append(f"{clip_id}:{neighbor.pig_id}:{int(neighbor.annotation_index)}")
                        else:
                            written.add((clip_id, str(neighbor.pig_id), int(neighbor.annotation_index)))
                    pending.remove(current)
                current += 1
        finally:
            capture.release()
        failures.extend(f"{clip_id}:frame:{index}" for index in pending)

    records = []
    for row, neighbors in plans:
        crops = []
        for neighbor in neighbors:
            key = (row.clip_id, str(neighbor.pig_id), int(neighbor.annotation_index))
            path = part_root / f"{row.clip_id.replace('/', '_')}__{neighbor.pig_id}__{int(neighbor.annotation_index)}.jpg"
            if key not in written or not path.is_file():
                failures.append(row.observation_id)
                crops = []
                break
            image = cv2.imread(str(path))
            if image is None:
                failures.append(row.observation_id)
                crops = []
                break
            crops.append(image)
        if not crops:
            continue
        destination_dir = crop_root / row.split
        destination_dir.mkdir(parents=True, exist_ok=True)
        if aggregation == "feature_mean":
            saved = True
            for offset, image in zip(offsets, crops):
                destination = destination_dir / f"{row.observation_id}__o{offset}.jpg"
                if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    failures.append(row.observation_id)
                    saved = False
                    break
            if saved:
                record = row._asdict()
                record["crop_path"] = str(destination_dir / f"{row.observation_id}__o0.jpg")
                record["window_offsets"] = ",".join(str(offset) for offset in offsets)
                records.append(record)
            continue
        destination = destination_dir / f"{row.observation_id}.jpg"
        if not cv2.imwrite(str(destination), average_window(crops), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            failures.append(row.observation_id)
            continue
        record = row._asdict()
        record["crop_path"] = str(destination)
        record["window_offsets"] = ",".join(str(offset) for offset in offsets)
        records.append(record)
    if part_root.exists():
        shutil.rmtree(part_root)
    if failures or len(records) != len(baseline):
        raise RuntimeError(f"Window crop extraction incomplete. Failures: {len(failures)}.")
    manifest = pd.DataFrame(records)
    manifest.to_csv(out / "crop_manifest.csv", index=False)
    return manifest


class PigCropDataset(Dataset):
    def __init__(self, frame, transform, class_to_idx):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["crop_path"]) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.class_to_idx[row["behavior"]]


def transform_window(images, training, image_size):
    flip = training and random.random() < 0.5
    brightness = 1.0 + random.uniform(-0.1, 0.1) if training else 1.0
    contrast = 1.0 + random.uniform(-0.1, 0.1) if training else 1.0
    saturation = 1.0 + random.uniform(-0.1, 0.1) if training else 1.0
    processed = []
    for image in images:
        image = PadToSquare()(image)
        image = transforms.functional.resize(image, [image_size, image_size])
        if flip:
            image = transforms.functional.hflip(image)
        if training:
            image = transforms.functional.adjust_brightness(image, brightness)
            image = transforms.functional.adjust_contrast(image, contrast)
            image = transforms.functional.adjust_saturation(image, saturation)
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, MEAN, STD)
        processed.append(image)
    return torch.stack(processed)


class PigWindowDataset(Dataset):
    def __init__(self, frame, class_to_idx, offsets, image_size, training):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.offsets = offsets
        self.image_size = image_size
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        folder = Path(row["crop_path"]).parent
        images = []
        for offset in self.offsets:
            path = folder / f"{row.observation_id}__o{offset}.jpg"
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return transform_window(images, self.training, self.image_size), self.class_to_idx[row["behavior"]]


def pooled_logits(model, images, freeze_backbone):
    if images.ndim == 4:
        return model(images)
    batch, steps, channels, height, width = images.shape
    flat = images.reshape(batch * steps, channels, height, width)
    if freeze_backbone:
        with torch.no_grad():
            features = model.avgpool(model.features(flat))
            features = torch.flatten(features, 1)
    else:
        features = model.avgpool(model.features(flat))
        features = torch.flatten(features, 1)
    pooled = features.reshape(batch, steps, -1).mean(dim=1)
    return model.classifier(pooled)


def run_epoch(model, loader, criterion, optimizer, device, training, freeze_backbone):
    model.train(training)
    if freeze_backbone:
        model.features.eval()
    running_loss = 0.0
    seen = 0
    targets_all = []
    predictions_all = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = pooled_logits(model, images, freeze_backbone)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        batch = targets.size(0)
        running_loss += loss.item() * batch
        seen += batch
        targets_all.extend(targets.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        targets_all,
        predictions_all,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )
    metrics = {
        "loss": running_loss / seen,
        "accuracy": float(accuracy_score(targets_all, predictions_all)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }
    return metrics, targets_all, predictions_all


def train(setting: dict, dataset_root: Path, out: Path, seed: int = 42) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    labels = reconstruct(dataset_root)
    baseline = accepted_baseline(labels)
    videos = video_map(dataset_root, baseline)
    crop_root = out / "crops"
    offsets = list(setting["offsets"])
    aggregation = setting["aggregation"]
    print(f"Window offsets: {offsets}")
    print(f"Aggregation: {aggregation}")
    if offsets == [0]:
        manifest = extract_single_crops(baseline, videos, crop_root, out)
    else:
        manifest = extract_window_crops(baseline, videos, labels, offsets, aggregation, crop_root, out)
    train_df = manifest.loc[manifest["split"].eq("train")].reset_index(drop=True)
    val_df = manifest.loc[manifest["split"].eq("validation")].reset_index(drop=True)
    print(f"Train crops: {len(train_df)} Validation crops: {len(val_df)}")

    image_size = 224
    batch_size = 32
    train_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    class_to_idx = {name: index for index, name in enumerate(CLASSES)}
    use_feature_window = aggregation == "feature_mean" and offsets != [0]
    if use_feature_window:
        train_loader = DataLoader(
            PigWindowDataset(train_df, class_to_idx, offsets, image_size, True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            PigWindowDataset(val_df, class_to_idx, offsets, image_size, False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            PigCropDataset(train_df, train_transform, class_to_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            PigCropDataset(val_df, eval_transform, class_to_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as exc:
        raise RuntimeError("Could not load pretrained MobileNetV3-Small weights.") from exc
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    freeze_backbone = bool(setting["freeze_backbone"])
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model = model.to(device)

    counts = train_df["behavior"].value_counts().reindex(CLASSES)
    weights = len(train_df) / (len(CLASSES) * counts.to_numpy(dtype=float))
    train_criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    val_criterion = nn.CrossEntropyLoss()
    learning_rate = float(setting["learning_rate"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    backbone_learning_rate = setting["backbone_learning_rate"]
    if backbone_learning_rate is None:
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    else:
        backbone_ids = {id(parameter) for parameter in model.features.parameters() if parameter.requires_grad}
        backbone = [parameter for parameter in trainable if id(parameter) in backbone_ids]
        head = [parameter for parameter in trainable if id(parameter) not in backbone_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": float(backbone_learning_rate)},
                {"params": head, "lr": learning_rate},
            ],
            weight_decay=1e-4,
        )
        print(f"Differential LR: backbone {float(backbone_learning_rate):g}, head {learning_rate:g}")

    max_epochs = 15
    patience = 4
    best_f1 = -1.0
    best_accuracy = None
    best_epoch = None
    stale = 0
    history = []
    checkpoint_path = out / "best.pt"
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        train_metrics, _, _ = run_epoch(model, train_loader, train_criterion, optimizer, device, True, freeze_backbone)
        val_metrics, _, _ = run_epoch(model, val_loader, val_criterion, optimizer, device, False, freeze_backbone)
        history.append(
            {
                "epoch": epoch,
                "epoch_seconds": time.perf_counter() - started,
                "train_loss": train_metrics["loss"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": val_metrics["loss"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_accuracy": val_metrics["accuracy"],
            }
        )
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"Epoch {epoch:02d}/{max_epochs} train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, val_targets, val_predictions = run_epoch(model, val_loader, val_criterion, optimizer, device, False, freeze_backbone)
    report = classification_report(
        val_targets,
        val_predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(out / "validation_report.csv")
    if len(val_targets) == len(val_df):
        pd.DataFrame(
            {
                "observation_id": val_df["observation_id"].tolist(),
                "behavior": [CLASSES[index] for index in val_targets],
                "prediction": [CLASSES[index] for index in val_predictions],
            }
        ).to_csv(out / "validation_predictions.csv", index=False)
    write_summary(
        out,
        {"validation_macro_f1": best_f1, "validation_accuracy": best_accuracy, "best_epoch": best_epoch},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one MobileNet crop setting and write its tables.")
    add_run_arguments(parser, "crop")
    parser.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    args = parser.parse_args()
    setting = SETTINGS[args.setting]
    if args.out == parser.get_default("out"):
        args.out = args.out.parent / setting["name"]
    dataset_root, out = resolve_run(args)
    print(f"Writing tables to {out}")
    train(setting, dataset_root, out)


if __name__ == "__main__":
    main()

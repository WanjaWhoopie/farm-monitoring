"""Eight-step GRU on the nine box features.

Each example is the current box plus up to seven earlier annotation steps of
the same pig, about four seconds at 2 Hz. Missing steps repeat the earliest
available step. The network is one GRU layer of size 32 and a seven-class
head, trained for five epochs with Adam at 1e-3. The held-out day is not used
to stop training. Median imputation uses the training days of that fold only.

    python sequence_model.py --data <annotated-clips>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CLASSES


def sequences(features: np.ndarray, groups: list[np.ndarray], length: int = 8) -> np.ndarray:
    windows = np.zeros((len(features), length, features.shape[1]), dtype=np.float32)
    for index in groups:
        block = features[index]
        for step, row_index in enumerate(index):
            start = max(0, step - length + 1)
            window = block[start : step + 1]
            windows[row_index, -len(window) :] = window
            if len(window) < length:
                windows[row_index, : length - len(window)] = window[0]
    return windows


class BoxSequence(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.gru = nn.GRU(features, 32, batch_first=True)
        self.head = nn.Linear(32, len(CLASSES))

    def forward(self, inputs):
        output, _ = self.gru(inputs)
        return self.head(output[:, -1])


def leave_one_day_out(features: np.ndarray, labels: np.ndarray, days: np.ndarray, train_mask: np.ndarray, groups: list[np.ndarray], device) -> np.ndarray:
    class_to_index = {name: index for index, name in enumerate(CLASSES)}
    targets = np.array([class_to_index.get(label, -1) for label in labels])
    prediction = np.empty(len(labels), dtype=object)
    filled_windows = None
    for day in sorted(set(days)):
        train = train_mask & (days != day) & (targets >= 0)
        present = [name for name in CLASSES if (labels[train] == name).any()]
        class_weight = np.ones(len(CLASSES), dtype=np.float32)
        balanced = compute_class_weight("balanced", classes=np.array(present), y=labels[train])
        for name, weight in zip(present, balanced):
            class_weight[class_to_index[name]] = float(weight)
        median = np.nanmedian(features[train], axis=0)
        filled = features.copy()
        missing = np.isnan(filled)
        filled[missing] = np.take(median, np.where(missing)[1])
        filled_windows = sequences(filled, groups)
        model = BoxSequence(features.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device))
        loader = DataLoader(
            TensorDataset(torch.tensor(filled_windows[train]), torch.tensor(targets[train], dtype=torch.long)),
            batch_size=512,
            shuffle=True,
        )
        model.train()
        for _epoch in range(5):
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss_fn(model(batch_x.to(device)), batch_y.to(device)).backward()
                optimizer.step()
        model.eval()
        test = days == day
        with torch.no_grad():
            chosen = []
            test_x = torch.tensor(filled_windows[test])
            for start in range(0, len(test_x), 1024):
                chosen.append(model(test_x[start : start + 1024].to(device)).argmax(dim=1).cpu())
        prediction[test] = np.array(CLASSES, dtype=object)[torch.cat(chosen).numpy()]
        print(f"Sequence fold {day}: train rows {int(train.sum())}", flush=True)
    return prediction


def main() -> None:
    from activity_budget import score_column, write_tables
    from box_features import load_tracks
    from common import BOX_FEATURES, add_run_arguments, resolve_run

    parser = argparse.ArgumentParser(description="Train the eight-step GRU leave-one-day-out.")
    add_run_arguments(parser, "g3-box-sequence")
    args = parser.parse_args()
    dataset_root, out = resolve_run(args)
    print(f"Writing tables to {out}")
    frame, features, _videos = load_tracks(dataset_root)
    labels = frame["behavior"].to_numpy(object)
    days = frame["recording_date"].to_numpy()
    groups = [np.asarray(index) for _, index in frame.groupby(["clip_id", "pig_id"], sort=False).indices.items()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame["pred_sequence"] = leave_one_day_out(
        features[BOX_FEATURES].to_numpy(float), labels, days, frame["kept"].to_numpy(), groups, device
    )
    row, folds, report = score_column(frame, "sequence", "pred_sequence")
    write_tables(out, [row], folds, [report])


if __name__ == "__main__":
    main()

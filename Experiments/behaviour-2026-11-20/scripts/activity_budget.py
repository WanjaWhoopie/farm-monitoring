"""Leave-one-day-out box model and the pen activity budget.

Train histogram gradient boosting on the kept rows of every day except one,
predict the held-out day, and repeat. Macro-F1 is computed on stored human
rows. The budget is the share of kept rows in each behaviour, and the error
is the mean absolute difference in percentage points. The model is depth 4,
learning rate 0.08, 150 iterations, seed 42, with balanced sample weights.
2019-12-02 is omitted.

    python activity_budget.py --data <annotated-clips>

Add ``--sequence`` to include the eight-step GRU, and ``--fusion`` to include
frozen MobileNet embeddings joined to the nine box numbers. Those two need
the same flags to reproduce the combined table in ``results/farm-protocol-suite``.
``--fusion`` reads color.mp4 and downloads ImageNet weights on the first run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))

from box_features import load_tracks
from common import BOX_FEATURES, CLASSES, SEED, add_run_arguments, resolve_run, write_summary


def leave_one_day_out(features: np.ndarray, labels: np.ndarray, days: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(labels), dtype=object)
    for day in sorted(set(days)):
        train = train_mask & (days != day)
        test = days == day
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150, random_state=SEED)
        model.fit(features[train], labels[train], sample_weight=compute_sample_weight("balanced", labels[train]))
        prediction[test] = model.predict(features[test])
        print(f"Geometry fold {day}: train rows {int(train.sum())}", flush=True)
    return prediction


def activity_share(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = pd.crosstab([frame["clip_id"], frame["pig_id"]], frame[column]).reindex(columns=CLASSES, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def copied_training_share(truth: pd.DataFrame) -> pd.DataFrame:
    days = truth.index.get_level_values("clip_id").str.split("/").str[0]
    copied = truth.copy()
    for day in sorted(set(days)):
        copied.loc[days == day] = truth.loc[days != day].mean().to_numpy()
    return copied


def mean_absolute_points(truth: pd.DataFrame, predicted: pd.DataFrame) -> float:
    difference = (truth - predicted.reindex(truth.index)).abs() * 100
    return float(difference.to_numpy().mean())


def pen_budget_error(frame: pd.DataFrame, prediction_column: str) -> tuple[float, float, float]:
    kept = frame.loc[frame["kept"]]
    truth = activity_share(kept, "behavior")
    predicted = activity_share(kept, prediction_column)
    pen_truth = truth.groupby(level="clip_id").mean()
    pen_predicted = predicted.groupby(level="clip_id").mean()
    pen_copied = copied_training_share(truth).groupby(level="clip_id").mean()
    return (
        mean_absolute_points(truth, predicted),
        mean_absolute_points(pen_truth, pen_predicted),
        mean_absolute_points(pen_truth, pen_copied),
    )


def stored_macro_f1(labels: np.ndarray, prediction: np.ndarray, days: np.ndarray, stored: np.ndarray) -> tuple[float, float, list]:
    scores = []
    for day in sorted(set(days)):
        selected = stored & (days == day)
        if selected.any():
            scores.append((day, float(f1_score(labels[selected], prediction[selected], labels=CLASSES, average="macro", zero_division=0))))
    values = [score for _, score in scores]
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), spread, scores


def score_column(frame: pd.DataFrame, name: str, column: str) -> tuple[dict, list, pd.DataFrame]:
    scored = frame if name != "fusion" else frame.loc[frame["kept"]].copy()
    mean, spread, scores = stored_macro_f1(
        scored["behavior"].to_numpy(object),
        scored[column].to_numpy(object),
        scored["recording_date"].to_numpy(),
        scored["tier"].eq("stored").to_numpy() & scored["behavior"].isin(CLASSES),
    )
    pig_error, pen_error, copied_error = pen_budget_error(frame, column)
    row = {
        "model": name,
        "stored_macro_f1_mean": mean,
        "stored_macro_f1_std": spread,
        "budget_mae_pp_clip_pig": pig_error,
        "budget_mae_pp_pen": pen_error,
        "copied_budget_mae_pp_pen": copied_error,
    }
    folds = [{"model": name, "day": day, "stored_macro_f1": score} for day, score in scores]
    stored_rows = scored["tier"].eq("stored") & scored["behavior"].isin(CLASSES)
    report = classification_report(
        scored.loc[stored_rows, "behavior"],
        scored.loc[stored_rows, column],
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    report_frame.insert(0, "model", name)
    print(f"{name}: stored macro-F1 {mean:.4f} ± {spread:.4f}, pen budget error {pen_error:.2f} points", flush=True)
    return row, folds, report_frame


def write_tables(out: Path, rows: list, folds: list, reports: list) -> None:
    history = pd.DataFrame(rows)
    history.to_csv(out / "history.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "lodo_folds.csv", index=False)
    pd.concat(reports).to_csv(out / "validation_report.csv")
    geometry = history.loc[history["model"].eq("geometry")]
    metrics = {
        "geometry_macro_f1": float(geometry["stored_macro_f1_mean"].iloc[0]) if len(geometry) else None,
        "geometry_pen_budget_mae": float(geometry["budget_mae_pp_pen"].iloc[0]) if len(geometry) else None,
        "copied_pen_budget_mae": float(geometry["copied_budget_mae_pp_pen"].iloc[0]) if len(geometry) else None,
    }
    for name, key in (("sequence", "sequence"), ("fusion", "fusion")):
        selected = history.loc[history["model"].eq(name)]
        if len(selected):
            metrics[f"{key}_macro_f1"] = float(selected["stored_macro_f1_mean"].iloc[0])
            metrics[f"{key}_pen_budget_mae"] = float(selected["budget_mae_pp_pen"].iloc[0])
    write_summary(out, metrics)


def run(dataset_root: Path, out: Path, with_sequence: bool, with_fusion: bool) -> None:
    frame, features, videos = load_tracks(dataset_root)
    labels = frame["behavior"].to_numpy(object)
    days = frame["recording_date"].to_numpy()
    train_mask = frame["kept"].to_numpy()
    matrix = features[BOX_FEATURES].to_numpy(float)
    frame["pred_geometry"] = leave_one_day_out(matrix, labels, days, train_mask)
    columns = [("geometry", "pred_geometry")]
    if with_sequence:
        from sequence_model import leave_one_day_out as sequence_leave_one_day_out

        groups = [np.asarray(index) for _, index in frame.groupby(["clip_id", "pig_id"], sort=False).indices.items()]
        device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
        frame["pred_sequence"] = sequence_leave_one_day_out(matrix, labels, days, train_mask, groups, device)
        columns.append(("sequence", "pred_sequence"))
    if with_fusion:
        from frozen_fusion import embed_kept, leave_one_day_out as fusion_leave_one_day_out

        device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
        kept_mask = frame["kept"].to_numpy()
        kept = frame.loc[kept_mask].reset_index(drop=True)
        embeddings = embed_kept(kept, videos, device)
        fusion_prediction = fusion_leave_one_day_out(
            embeddings,
            matrix[kept_mask],
            kept["behavior"].to_numpy(object),
            kept["recording_date"].to_numpy(),
        )
        frame["pred_fusion"] = ""
        frame.loc[kept_mask, "pred_fusion"] = fusion_prediction
        columns.append(("fusion", "pred_fusion"))
    rows, folds, reports = [], [], []
    for name, column in columns:
        row, fold_rows, report = score_column(frame, name, column)
        rows.append(row)
        folds.extend(fold_rows)
        reports.append(report)
    write_tables(out, rows, folds, reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the box model by leave-one-day-out pen budget.")
    add_run_arguments(parser, "farm-protocol-suite")
    parser.add_argument("--sequence", action="store_true", help="Also train the eight-step GRU.")
    parser.add_argument("--fusion", action="store_true", help="Also fit frozen embeddings joined to the box numbers.")
    args = parser.parse_args()
    dataset_root, out = resolve_run(args)
    print(f"Writing tables to {out}")
    run(dataset_root, out, args.sequence, args.fusion)


if __name__ == "__main__":
    main()

"""Frame-rate ablation on the box track (research question 2, vision side). Runs locally; no videos needed.

The label table has one box per pig per captured frame (6.04 frames per second), with human
labels on every third frame (2 per second) and the two frames in between carried forward.
This script thins the track to every k-th frame (k = 1, 3, 6, 12 -> 6, 2, 1, 0.5 boxes per
second), recomputes the nine box features at that rate, retrains and re-scores the geometry
model leave-one-day-out, and rebuilds the activity budgets. Train and test always run at the
same rate, as a deployed system would.

What is held fixed: the label gate is computed once on the full-rate track and the tier of each
row is kept, so only the frame rate changes. The speed feature keeps the notebook's definition
(displacement over +-2 steps), so its time window grows as the rate drops; that is what a
system that only sees the thinner track could do.

Outputs: reports/fps_ablation_<date>/ with fps_ablation.csv, per-behaviour recall per rate,
a figure and run_info.json. Usage: python scripts/fps_ablation.py [--steps 1,3,6,12]
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import (CLASSES7, LABELS_CSV, ROOT, SEED, budget, build_features,  # noqa: E402
                             copied_training_budget, load_labels, lodo_predict_all, md5_of, score, set_seeds)

FULL_RATE_HZ = 6.04
PALETTE = ["#2a78d6", "#eb6834"]  # series 1 and 2 of the validated palette
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#5a5955", "#e1e0d9"


def evaluate_at(df_full: pd.DataFrame, k: int) -> tuple[dict, pd.Series]:
    """Thin the track to every k-th frame (keeping the human-label grid), recompute features, score."""
    sub = df_full[df_full["annotation_index"] % k == 0].reset_index(drop=True)
    feats = build_features(sub)
    feat_cols = list(feats.columns)
    sub = pd.concat([sub, feats], axis=1)
    kept = ((sub["tier"] != "rejected") & sub["behavior"].isin(CLASSES7)).to_numpy()
    X = sub[feat_cols].to_numpy(float)
    days = sub["recording_date"].to_numpy()
    y = sub["behavior"].to_numpy(object)
    sub["pred7"] = lodo_predict_all(X, y, days, kept)

    stored = (sub["tier"] == "stored").to_numpy() & kept
    f1_days = []
    for d in sorted(set(days)):
        m = stored & (days == d)
        f1_days.append(f1_score(y[m], sub["pred7"].to_numpy()[m], labels=np.unique(y[m]), average="macro", zero_division=0))
    cm = confusion_matrix(y[stored], sub["pred7"].to_numpy()[stored], labels=CLASSES7)
    recall = pd.Series(np.diag(cm) / np.maximum(cm.sum(1), 1), index=CLASSES7)

    kdf = sub[kept]
    truth = budget(kdf, "behavior", CLASSES7)
    mae_model, _ = score(truth, budget(kdf, "pred7", CLASSES7))
    mae_copy, _ = score(truth, copied_training_budget(truth))
    t_pen = truth.groupby(level="clip_id").mean()
    mae_pen, _ = score(t_pen, budget(kdf, "pred7", CLASSES7).reindex(truth.index).groupby(level="clip_id").mean())
    mae_copy_pen, _ = score(t_pen, copied_training_budget(truth).groupby(level="clip_id").mean())

    row = {"step": k, "boxes_per_second": round(FULL_RATE_HZ / k, 2), "rows": int(len(sub)), "rows_kept": int(kept.sum()),
           "rows_stored": int(stored.sum()), "macro_f1_stored_mean": round(float(np.mean(f1_days)), 3),
           "macro_f1_stored_std": round(float(np.std(f1_days, ddof=1)), 3),
           "budget_mae_pp_clip_pig": round(mae_model, 2), "budget_mae_pp_pen": round(mae_pen, 2),
           "copied_budget_mae_pp_clip_pig": round(mae_copy, 2), "copied_budget_mae_pp_pen": round(mae_copy_pen, 2),
           "boxes_per_pig_per_minute": round(60 * FULL_RATE_HZ / k, 1)}
    return row, recall.round(3)


def draw_figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    x = table["boxes_per_second"]
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
        ax.minorticks_off(); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xlabel("boxes per pig per second", color=INK2, fontsize=9)
    a = axes[0]
    a.plot(x, table["macro_f1_stored_mean"], color=PALETTE[0], linewidth=2, marker="o", markersize=7, label="geometry model")
    a.fill_between(x, table["macro_f1_stored_mean"] - table["macro_f1_stored_std"], table["macro_f1_stored_mean"] + table["macro_f1_stored_std"],
                   color=PALETTE[0], alpha=0.12, linewidth=0)
    for xi, yi in zip(x, table["macro_f1_stored_mean"]):
        a.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK)
    a.set_ylim(0, 0.5); a.set_title("Per-frame score: macro-F1 on human rows over 7 held-out days\n(band = spread across days)", loc="left", fontsize=9, color=INK)
    b = axes[1]
    b.plot(x, table["budget_mae_pp_pen"], color=PALETTE[0], linewidth=2, marker="o", markersize=7, label="geometry model")
    b.plot(x, table["copied_budget_mae_pp_pen"], color=PALETTE[1], linewidth=2, marker="o", markersize=7, label="training-days average (no model)")
    for xi, yi in zip(x, table["budget_mae_pp_pen"]):
        b.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=INK)
    b.set_ylim(0, max(table["copied_budget_mae_pp_pen"].max(), table["budget_mae_pp_pen"].max()) * 1.3)
    b.set_title("Pen-level activity-budget error\n(percentage points, lower is better)", loc="left", fontsize=9, color=INK)
    b.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Frame-rate ablation on the box track (human boxes, geometry model)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="1,3,6,12", help="comma-separated thinning factors")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    out = args.out or ROOT / "reports" / f"fps_ablation_{datetime.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)
    set_seeds()

    print("Loading labels; gate computed once at the full rate ...")
    labels = load_labels()
    rows, recalls = [], {}
    for k in steps:
        print(f"step {k}: {FULL_RATE_HZ / k:.2f} boxes per second ...", flush=True)
        row, rec = evaluate_at(labels, k)
        rows.append(row); recalls[f"{row['boxes_per_second']:g} per s"] = rec
        print("  ", {a: b for a, b in row.items() if a in ("rows_kept", "macro_f1_stored_mean", "budget_mae_pp_clip_pig", "budget_mae_pp_pen")})
    table = pd.DataFrame(rows)
    rec_df = pd.DataFrame(recalls).T
    rec_df.index.name = "rate"
    print("\n" + table.to_string(index=False))
    print("\nRecall per behaviour (human rows):")
    print(rec_df.to_string())
    table.to_csv(out / "fps_ablation.csv", index=False)
    rec_df.to_csv(out / "recall_by_rate.csv")
    draw_figure(table, out / "fps_ablation_figure.png")
    info = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED, "steps": steps,
            "full_rate_hz": FULL_RATE_HZ, "input_csv": {"path": str(LABELS_CSV.relative_to(ROOT)), "md5": md5_of(LABELS_CSV)},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

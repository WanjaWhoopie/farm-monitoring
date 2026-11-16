"""Activity-budget evaluation for the v2 geometry model. Runs locally; needs no videos.

An activity budget is the share of time a pig spends in each behaviour over a period.
Here the period is one five-minute clip and the unit is one pig, so there are
12 clips x 8 pigs = 96 budgets. The script:

1. rebuilds the label gate, the nine box features and the leave-one-day-out
   predictions exactly as Experiments/notebook_v2_marimo.py does (geometry model only);
2. also predicts every frame of the held-out day, including rows the gate rejected,
   because a deployed system has no gate;
3. optionally smooths predictions with a majority vote over a short window per pig;
4. turns true labels and predictions into budgets and scores them with the mean
   absolute difference in percentage points, against two cheap baselines:
   "always investigating" and the average budget of the training days copied over.

Outputs go to reports/activity_budget_<date>/ (tables as CSV, a figure, run_info.json).
Usage: python scripts/activity_budget.py [--smooth-frames 13] [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = ROOT / "data" / "processed" / "vision" / "reconstructed_annotations.csv"
DATA_ROOT = ROOT / "data" / "raw" / "edinburgh"

# Same configuration as the notebook.
SEED = 42
W, H = 1280, 720
CLASSES7 = ["sleep", "investigating", "eat", "standing", "lying", "walk", "drink"]
COARSE = {"sleep": "resting", "lying": "resting", "standing": "stationary",
          "walk": "locomotion", "eat": "feeding", "drink": "drinking",
          "investigating": "exploring"}
COARSE_CLASSES = ["resting", "stationary", "locomotion", "feeding", "drinking", "exploring"]
MAX_GAP, MIN_IOU, MAX_TRAIL = 20, 0.5, 10
REFERENCE_MACRO_F1 = 0.321  # geometry model, stored rows, run 2 of 16 Nov 2026


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- data and gate
def capture_seconds(clip_id: str) -> np.ndarray:
    t = pd.to_datetime(pd.read_csv(DATA_ROOT / "annotated" / clip_id / "times.txt", header=None)[0])
    return (t - t.iloc[0]).dt.total_seconds().to_numpy()


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def add_gate_columns(df: pd.DataFrame) -> pd.DataFrame:
    gap = np.full(len(df), np.nan); ext = np.full(len(df), np.nan)
    iou_lr = np.full(len(df), np.nan); agree = np.zeros(len(df), bool)
    has_right = np.zeros(len(df), bool)
    for _, ix in df.groupby(["clip_id", "pig_id"], sort=False).indices.items():
        g = df.iloc[ix]
        idx = g["annotation_index"].to_numpy()
        st = (g["annotation_source"] == "stored").to_numpy()
        sidx = idx[st]
        sbox = g.loc[st, ["x", "y", "width", "height"]].to_numpy(float)
        sbeh = g.loc[st, "behavior"].to_numpy()
        L = np.searchsorted(sidx, idx, side="right") - 1
        R = np.searchsorted(sidx, idx, side="left")
        hr = R < len(sidx)
        L = np.clip(L, 0, len(sidx) - 1); Rc = np.clip(R, 0, len(sidx) - 1)
        gap[ix] = np.where(hr, sidx[Rc] - sidx[L], np.nan)
        ext[ix] = np.where(hr, np.nan, idx - sidx[L])
        iou_lr[ix] = np.where(hr, box_iou(sbox[L], sbox[Rc]), np.nan)
        agree[ix] = sbeh[L] == sbeh[Rc]
        has_right[ix] = hr
    out = df.copy()
    out["gap"], out["trail_ext"], out["iou_lr"] = gap, ext, iou_lr
    out["label_agree"], out["has_right"] = agree, has_right
    return out


def assign_tier(df: pd.DataFrame) -> np.ndarray:
    stored = df["annotation_source"].eq("stored")
    inter_ok = (df["has_right"] & (df["gap"] <= MAX_GAP) & (df["iou_lr"] >= MIN_IOU) & df["label_agree"])
    trail_ok = (~df["has_right"]) & (df["trail_ext"] <= MAX_TRAIL)
    return np.where(stored, "stored", np.where(inter_ok | trail_ok, "gated", "rejected"))


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV, dtype={"pig_id": str})
    labels["recording_date"] = labels["clip_id"].str.split("/").str[0]
    labels["raw_frame_index"] = labels["annotation_index"]
    labels["elapsed_seconds"] = np.nan
    for clip, ix in labels.groupby("clip_id").indices.items():
        secs = capture_seconds(clip)
        labels.loc[labels.index[ix], "elapsed_seconds"] = secs[labels["raw_frame_index"].to_numpy()[ix]]
    labels = labels.sort_values(["clip_id", "pig_id", "annotation_index"]).reset_index(drop=True)
    labels = add_gate_columns(labels)
    labels["tier"] = assign_tier(labels)
    return labels


# ----------------------------------------------------------------------------- features
def pair_iou_matrix(b: np.ndarray) -> np.ndarray:
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(b[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(b[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    area = b[:, 2] * b[:, 3]
    return inter / np.maximum(area[:, None] + area[None] - inter, 1e-09)


def build_features(df: pd.DataFrame, half_window: int = 2) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    f = pd.DataFrame(index=df.index)
    f["cx"] = (df["x"] + df["width"] / 2) / W
    f["cy"] = (df["y"] + df["height"] / 2) / H
    f["w"], f["h"] = df["width"] / W, df["height"] / H
    f["log_area"] = np.log(df["width"] * df["height"])
    f["aspect"] = df["width"] / df["height"]
    speed = np.full(len(df), np.nan)
    reliable = (df["tier"] != "rejected").to_numpy()
    for _, ix in df.groupby(["clip_id", "pig_id"], sort=False).indices.items():
        g = df.iloc[ix]
        cx, cy = (g["x"] + g["width"] / 2).to_numpy(), (g["y"] + g["height"] / 2).to_numpy()
        t = g["elapsed_seconds"].to_numpy()
        ok = reliable[ix]
        n = len(g)
        lo = np.clip(np.arange(n) - half_window, 0, n - 1)
        hi = np.clip(np.arange(n) + half_window, 0, n - 1)
        dt = t[hi] - t[lo]
        d = np.hypot(cx[hi] - cx[lo], cy[hi] - cy[lo])
        speed[ix] = np.where(ok[lo] & ok[hi] & (dt > 0), d / np.where(dt > 0, dt, 1), np.nan)
    f["speed_px_s"] = speed
    nn = np.full(len(df), np.nan); mx = np.full(len(df), np.nan)
    for _, ix in df.groupby(["clip_id", "annotation_index"]).indices.items():
        b = df.iloc[ix][["x", "y", "width", "height"]].to_numpy(float)
        c = b[:, :2] + b[:, 2:] / 2
        d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        np.fill_diagonal(d, np.inf)
        iou = pair_iou_matrix(b)
        np.fill_diagonal(iou, 0)
        nn[ix] = d.min(1); mx[ix] = iou.max(1)
    f["nearest_pig_px"], f["max_overlap_other"] = nn, mx
    return f


# ----------------------------------------------------------------------------- model
def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150, random_state=SEED)


def lodo_predict_all(X: np.ndarray, y: np.ndarray, days: np.ndarray, train_mask: np.ndarray, n_jobs: int = 7) -> np.ndarray:
    """Leave-one-day-out. Train on kept rows of the other days; predict EVERY row of the held-out day."""
    day_list = sorted(set(days))

    def one(day):
        tr = train_mask & (days != day)
        te = days == day
        model = make_model()
        model.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        return te, model.predict(X[te])

    P = np.empty(len(y), dtype=object)
    for te, p in Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(d) for d in day_list):
        P[te] = p
    return P


def smooth_predictions(df: pd.DataFrame, col: str, window: int, classes: list[str]) -> np.ndarray:
    """Majority vote over a centred window of `window` frames, per pig, in time order."""
    out = np.empty(len(df), dtype=object)
    for _, ix in df.groupby(["clip_id", "pig_id"], sort=False).indices.items():
        p = df[col].to_numpy()[ix]
        votes = np.stack([pd.Series(p == c, dtype=float).rolling(window, center=True, min_periods=1).sum().to_numpy()
                          for c in classes], axis=1)
        out[ix] = np.array(classes, dtype=object)[votes.argmax(1)]
    return out


# ----------------------------------------------------------------------------- budgets
def budget(df: pd.DataFrame, col: str, classes: list[str]) -> pd.DataFrame:
    """Share of frames per (clip, pig) in each class. Frames are evenly spaced, so this is a time share."""
    counts = pd.crosstab([df["clip_id"], df["pig_id"]], df[col]).reindex(columns=classes, fill_value=0)
    return counts.div(counts.sum(1), axis=0)


def copied_training_budget(truth: pd.DataFrame) -> pd.DataFrame:
    """For each clip-pig, the mean true budget of all clip-pigs on OTHER days (what you would predict with no model)."""
    days = truth.index.get_level_values("clip_id").str.split("/").str[0]
    out = truth.copy()
    for d in sorted(set(days)):
        out.loc[days == d] = truth.loc[days != d].mean().to_numpy()
    return out


def score(truth: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, pd.Series]:
    """Mean absolute difference in percentage points: overall (mean over classes and budgets) and per class."""
    diff = (truth - pred.reindex(truth.index)).abs() * 100
    return float(diff.to_numpy().mean()), diff.mean(0)


# ----------------------------------------------------------------------------- figure
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # validated, fixed order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#5a5955", "#e1e0d9"


def draw_figure(pen: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(pen)
    clips = list(pen[methods[0]].index)
    n_m = len(methods)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(clips) * n_m + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bar_h, gap = 0.78 / n_m, 0.06
    ypos = {}
    for i, clip in enumerate(clips):
        for j, m in enumerate(methods):
            y = i * 1.0 + j * (bar_h + gap / n_m)
            ypos[(clip, m)] = y
            left = 0.0
            for k, cls in enumerate(CLASSES7):
                v = float(pen[m].loc[clip, cls]) * 100
                if v <= 0:
                    continue
                ax.barh(y, v, left=left, height=bar_h, color=PALETTE[k], edgecolor=SURFACE, linewidth=1.5)
                if v >= 9:
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                            color=INK if cls == "standing" else "#ffffff")
                left += v
            ax.text(-1.0, y, m, ha="right", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks([i * 1.0 + 0.39 for i in range(len(clips))])
    ax.set_yticklabels(clips, fontsize=8.5, color=INK)
    ax.tick_params(axis="y", length=0, pad=62)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the clip, % of frames (each clip is about five minutes; pen level = mean of 8 pigs)", color=INK2, fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[k]) for k in range(len(CLASSES7))]
    ax.legend(handles, CLASSES7, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8.5)
    ax.set_title("Activity budget per clip: true labels versus leave-one-day-out predictions", loc="left", fontsize=11, color=INK, pad=28)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smooth-frames", type=int, default=13, help="majority-vote window in frames (13 frames is about 2 s)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    out = args.out or ROOT / "reports" / f"activity_budget_{datetime.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)
    set_seeds()

    print("Loading labels and gate ...")
    labels = load_labels()
    print("  rows by tier:", labels["tier"].value_counts().to_dict())
    feats = build_features(labels)
    feat_cols = list(feats.columns)
    df = pd.concat([labels.reset_index(drop=True), feats], axis=1)
    kept = ((df["tier"] != "rejected") & df["behavior"].isin(CLASSES7)).to_numpy()
    print("  kept rows for training:", int(kept.sum()))

    print("Leave-one-day-out, geometry model, predicting every frame of the held-out day ...")
    X = df[feat_cols].to_numpy(float)
    days = df["recording_date"].to_numpy()
    y7 = df["behavior"].to_numpy(object)
    df["pred7"] = lodo_predict_all(X, y7, days, kept)
    df["pred7_smoothed"] = smooth_predictions(df, "pred7", args.smooth_frames, CLASSES7)

    # Check against the notebook's headline number (stored rows, macro-F1 per day, then mean).
    stored = (df["tier"] == "stored").to_numpy() & kept
    per_day = []
    for d in sorted(set(days)):
        m = stored & (days == d)
        per_day.append(f1_score(y7[m], df["pred7"].to_numpy()[m], labels=np.unique(y7[m]), average="macro", zero_division=0))
    f1_check = float(np.mean(per_day))
    f1_smooth = float(np.mean([f1_score(y7[stored & (days == d)], df["pred7_smoothed"].to_numpy()[stored & (days == d)],
                                        labels=np.unique(y7[stored & (days == d)]), average="macro", zero_division=0) for d in sorted(set(days))]))
    print(f"  per-frame macro-F1 on stored rows: {f1_check:.3f} (reference run 2: {REFERENCE_MACRO_F1}); smoothed: {f1_smooth:.3f}")

    # ---- budgets on rows with a trusted label (truth is only known there)
    k = df[kept].copy()
    truth = budget(k, "behavior", CLASSES7)
    preds = {
        "always investigating": pd.DataFrame(0.0, index=truth.index, columns=CLASSES7).assign(investigating=1.0),
        "training-days average": copied_training_budget(truth),
        "geometry": budget(k, "pred7", CLASSES7),
        "geometry, smoothed": budget(k, "pred7_smoothed", CLASSES7),
        "geometry, smoothed, all frames": budget(df, "pred7_smoothed", CLASSES7),
    }
    rows, per_class = [], {}
    for name, p in preds.items():
        overall, pc = score(truth, p)
        # pen level: average the 8 pigs of each clip first
        t_pen = truth.groupby(level="clip_id").mean()
        p_pen = p.reindex(truth.index).groupby(level="clip_id").mean()
        pen_overall, _ = score(t_pen, p_pen)
        # merged groups: map each 7-class budget onto the 6 coarse groups
        t_c = truth.T.groupby(truth.columns.map(COARSE)).sum().T.reindex(columns=COARSE_CLASSES)
        p_c = p.reindex(truth.index).T.groupby(p.columns.map(COARSE)).sum().T.reindex(columns=COARSE_CLASSES)
        coarse_overall, _ = score(t_c, p_c)
        rows.append({"method": name, "mae_pp_7class": round(overall, 2), "mae_pp_6groups": round(coarse_overall, 2),
                     "mae_pp_pen_level_7class": round(pen_overall, 2)})
        per_class[name] = pc.round(2)
    summary = pd.DataFrame(rows)
    per_class_df = pd.DataFrame(per_class).T
    per_class_df.index.name = "method"

    print("\nMean absolute difference between predicted and true budgets, percentage points (lower is better):")
    print(summary.to_string(index=False))
    print("\nPer behaviour (7 classes, per clip-pig):")
    print(per_class_df.to_string())

    # ---- save
    df[["clip_id", "pig_id", "annotation_index", "elapsed_seconds", "annotation_source", "tier", "behavior",
        "pred7", "pred7_smoothed"]].to_csv(out / "predictions_lodo_geometry.csv", index=False)
    long = pd.concat({"true": truth, **preds}, names=["method"]).stack().rename("share").reset_index()
    long.columns = ["method", "clip_id", "pig_id", "behavior", "share"]
    long.to_csv(out / "budgets_per_clip_pig.csv", index=False)
    summary.to_csv(out / "budget_mae_summary.csv", index=False)
    per_class_df.to_csv(out / "budget_mae_per_behavior.csv")
    pen = {"true": truth.groupby(level="clip_id").mean(),
           "geometry": preds["geometry"].groupby(level="clip_id").mean(),
           "smoothed": preds["geometry, smoothed"].groupby(level="clip_id").mean()}
    pd.concat(pen, names=["method"]).to_csv(out / "budgets_pen_level.csv")
    if not args.no_figure:
        draw_figure(pen, out / "activity_budget_figure.png")
    info = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "input_csv": {"path": str(LABELS_CSV.relative_to(ROOT)), "md5": md5_of(LABELS_CSV), "rows": int(len(df))},
        "gate": {"MAX_GAP": MAX_GAP, "MIN_IOU": MIN_IOU, "MAX_TRAIL": MAX_TRAIL},
        "rows_by_tier": {str(a): int(b) for a, b in labels["tier"].value_counts().items()},
        "rows_kept_7class": int(kept.sum()),
        "smooth_frames": args.smooth_frames,
        "per_frame_macro_f1_stored": {"geometry": round(f1_check, 4), "geometry_smoothed": round(f1_smooth, 4), "reference_run2": REFERENCE_MACRO_F1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

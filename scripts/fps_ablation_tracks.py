"""Frame-rate ablation on the PREDICTED box track (detector + ByteTrack), the counterpart of
fps_ablation.py, which did the same on human boxes.

For each rate (6, 2, 1, 0.5 boxes per pig per second = every 1st, 3rd, 6th, 12th captured frame)
and each clip with a track file:
  - thin the predicted track and the human label table to that rate;
  - match predicted boxes to human boxes frame by frame (overlap >= 0.5) for precision, recall
    and identity switches;
  - train the geometry model on HUMAN boxes of the other recording days at the same rate, score
    it on the predicted track (labels inherited from the matched human box, human-labelled rows
    only) and, beside it, on the clip's human boxes;
  - pen-level activity budget from the predicted track versus the true one.
The summary averages the clips the detector never saw; the detector's training clip is kept
separate. At the full rate the human boxes between labelled frames are carried-forward copies,
so precision and recall at 6 per second are slightly understated.

Usage: python scripts/fps_ablation_tracks.py --tracks reports/detector_2026-11-18 [--steps 1,3,6,12]
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import ROOT, SEED, capture_seconds, load_labels, set_seeds  # noqa: E402
from detector_eval import detection_stats, geometry_on_tracks, match_clip, tracking_stats  # noqa: E402

FULL_RATE_HZ = 6.04
PALETTE = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#5a5955", "#e1e0d9"
SUMMARY_COLS = ["precision", "recall", "identity_switches", "mean_track_ids_per_pig", "macro_f1_human_boxes",
                "macro_f1_predicted_boxes", "pen_budget_mae_pp_human_boxes", "pen_budget_mae_pp_predicted_boxes"]


def evaluate_rate(labels: pd.DataFrame, tracks: Path, k: int, train_clip: str) -> list[dict]:
    rows = []
    for f in sorted(tracks.glob("track_*.csv")):
        s = f.stem[len("track_"):]
        clip = s[:10] + "/" + s[11:]
        tr = pd.read_csv(f)
        gt = labels[(labels["clip_id"] == clip) & (labels["annotation_index"] % k == 0)]
        pr = tr[tr["frame"] % k == 0].copy()
        matched = match_clip(gt, pr)
        secs = capture_seconds(clip)
        src = pd.DataFrame({"clip_id": clip, "pig_id": pr["track_id"].astype(str), "annotation_index": pr["frame"],
                            "x": pr["x"], "y": pr["y"], "width": pr["width"], "height": pr["height"],
                            "elapsed_seconds": secs[pr["frame"].to_numpy()], "tier": "stored"}).sort_values(["pig_id", "annotation_index"]).reset_index(drop=True)
        d = {"step": k, "boxes_per_second": round(FULL_RATE_HZ / k, 2), "clip_id": clip,
             "role": "train (detector)" if clip == train_clip else "unseen day"}
        d.update(detection_stats(gt, matched)); d.update(tracking_stats(matched))
        d.update(geometry_on_tracks(labels, clip, matched, src, step=k))
        rows.append(d)
        print(f"  {clip} {d['role']}: P {d['precision']} R {d['recall']} switches {d['identity_switches']} "
              f"F1 human {d['macro_f1_human_boxes']} predicted {d['macro_f1_predicted_boxes']} "
              f"budget {d['pen_budget_mae_pp_human_boxes']} / {d['pen_budget_mae_pp_predicted_boxes']}", flush=True)
    return rows


def draw_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    x = summary["boxes_per_second"]
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
    a.plot(x, summary["macro_f1_human_boxes"], color=PALETTE[0], linewidth=2, marker="o", markersize=7, label="human boxes")
    a.plot(x, summary["macro_f1_predicted_boxes"], color=PALETTE[1], linewidth=2, marker="o", markersize=7, label="predicted track")
    for xi, yi in zip(x, summary["macro_f1_predicted_boxes"]):
        a.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color=PALETTE[1])
    a.set_ylim(0, 0.5); a.legend(frameon=False, fontsize=8, loc="upper right")
    a.set_title("Per-frame score: macro-F1 on human-labelled rows\n(mean of the three unseen clips)", loc="left", fontsize=9, color=INK)
    b = axes[1]
    b.plot(x, summary["pen_budget_mae_pp_human_boxes"], color=PALETTE[0], linewidth=2, marker="o", markersize=7, label="human boxes")
    b.plot(x, summary["pen_budget_mae_pp_predicted_boxes"], color=PALETTE[1], linewidth=2, marker="o", markersize=7, label="predicted track")
    for xi, yi in zip(x, summary["pen_budget_mae_pp_predicted_boxes"]):
        b.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=PALETTE[1])
    b.set_ylim(0, max(summary["pen_budget_mae_pp_predicted_boxes"].max(), summary["pen_budget_mae_pp_human_boxes"].max()) * 1.4)
    b.legend(frameon=False, fontsize=8, loc="upper right")
    b.set_title("Pen-level activity-budget error\n(percentage points, lower is better)", loc="left", fontsize=9, color=INK)
    fig.suptitle("Frame-rate ablation on the predicted track (detector + ByteTrack, geometry model)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, required=True, help="folder with track_<clip>.csv files")
    ap.add_argument("--steps", default="1,3,6,12")
    ap.add_argument("--train-clip", default="2019_12_02/000005")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    out = args.out or ROOT / "reports" / f"fps_ablation_tracks_{datetime.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)
    set_seeds()
    labels = load_labels()
    rows = []
    for k in steps:
        print(f"step {k}: {FULL_RATE_HZ / k:.2f} boxes per second ...", flush=True)
        rows += evaluate_rate(labels, args.tracks, k, args.train_clip)
    table = pd.DataFrame(rows).drop(columns=["budget_true", "budget_predicted_track"])
    table.to_csv(out / "fps_ablation_tracks_per_clip.csv", index=False)
    unseen = table[table["role"] == "unseen day"]
    summary = unseen.groupby(["step", "boxes_per_second"], as_index=False)[SUMMARY_COLS].mean().round(3)
    summary.to_csv(out / "fps_ablation_tracks.csv", index=False)
    print("\nMean over the unseen clips:\n" + summary.to_string(index=False))
    draw_figure(summary, out / "fps_ablation_tracks_figure.png")
    info = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED, "steps": steps,
            "tracks": str(args.tracks), "train_clip": args.train_clip,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved to", out)


if __name__ == "__main__":
    main()

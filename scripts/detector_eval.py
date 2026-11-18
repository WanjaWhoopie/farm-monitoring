"""Score the detector's box tracks against the human boxes, then run the geometry model on the
predicted track and compare with the same model on human boxes.

Per clip (on the 2 Hz label grid, every third captured frame):
  detection : predicted boxes matched to human boxes by best overlap (one to one, IoU >= 0.5);
              precision, recall, mean IoU of matches, boxes per frame.
  tracking  : for each pig, how often the matched track id changes between consecutive labelled
              frames (identity switches) and how many distinct track ids it gets.
  behaviour : the nine box numbers computed from the PREDICTED track (track id in place of pig id),
              classified by a geometry model trained on HUMAN boxes from the other recording days;
              each predicted box inherits the label of the human box it matches; macro-F1 on
              human-labelled rows. Beside it, the same model scored on the human boxes of that clip.
  budgets   : pen-level activity budget from the predicted track versus the true one.

Usage: python scripts/detector_eval.py --tracks reports/detector_<date>
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import (CLASSES7, ROOT, SEED, build_features, capture_seconds, compute_sample_weight,  # noqa: E402
                             load_labels, make_model, set_seeds)

STEP = 3  # the 2 Hz label grid


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    return inter / np.maximum(a[:, 2:4].prod(1)[:, None] + b[:, 2:4].prod(1)[None] - inter, 1e-9)


def match_clip(gt: pd.DataFrame, pr: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """One-to-one matching per frame. Returns the predicted rows with matched pig_id / label / tier (NaN if unmatched)."""
    out = []
    for frame, g in gt.groupby("annotation_index"):
        p = pr[pr["frame"] == frame]
        if len(p) == 0:
            continue
        m = iou_matrix(p[["x", "y", "width", "height"]].to_numpy(float), g[["x", "y", "width", "height"]].to_numpy(float))
        ri, ci = linear_sum_assignment(-m)
        rec = p.copy()
        gt_pig, gt_beh, gt_tier, iou = [None] * len(p), [None] * len(p), [None] * len(p), [np.nan] * len(p)
        for r, c in zip(ri, ci):
            if m[r, c] >= thr:
                gt_pig[r], gt_beh[r], gt_tier[r], iou[r] = str(g.iloc[c]["pig_id"]), g.iloc[c]["behavior"], g.iloc[c]["tier"], m[r, c]
        rec["gt_pig"], rec["gt_behavior"], rec["gt_tier"], rec["iou"] = gt_pig, gt_beh, gt_tier, iou
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def detection_stats(gt: pd.DataFrame, matched: pd.DataFrame) -> dict:
    n_gt = len(gt); n_pr = len(matched); n_hit = int(matched["gt_pig"].notna().sum())
    return {"human_boxes": n_gt, "predicted_boxes": n_pr, "matched": n_hit,
            "precision": round(n_hit / max(n_pr, 1), 3), "recall": round(n_hit / max(n_gt, 1), 3),
            "mean_iou_matched": round(float(matched["iou"].mean()), 3),
            "boxes_per_frame": round(n_pr / gt["annotation_index"].nunique(), 2)}


def tracking_stats(matched: pd.DataFrame) -> dict:
    switches, ids_per_pig = 0, []
    for pig, g in matched.dropna(subset=["gt_pig"]).groupby("gt_pig"):
        seq = g.sort_values("frame")["track_id"].to_numpy()
        switches += int((seq[1:] != seq[:-1]).sum())
        ids_per_pig.append(len(set(seq)))
    return {"identity_switches": switches, "mean_track_ids_per_pig": round(float(np.mean(ids_per_pig)), 1) if ids_per_pig else None}


def geometry_on_tracks(labels: pd.DataFrame, clip: str, matched: pd.DataFrame, pr_feats_src: pd.DataFrame, step: int = STEP) -> dict:
    """Train on human boxes of other days (every step-th frame, kept rows); score on human boxes and on the predicted track of this clip."""
    day = clip.split("/")[0]
    grid = labels[labels["annotation_index"] % step == 0].reset_index(drop=True)
    feats = build_features(grid)
    cols = list(feats.columns)
    grid = pd.concat([grid, feats], axis=1)
    kept = (grid["tier"] != "rejected") & grid["behavior"].isin(CLASSES7)
    tr = kept & (grid["recording_date"] != day)
    model = make_model()
    model.fit(grid.loc[tr, cols].to_numpy(float), grid.loc[tr, "behavior"], sample_weight=compute_sample_weight("balanced", grid.loc[tr, "behavior"]))

    # human boxes of this clip, stored rows only (the notebook's headline scoring set)
    te_h = (grid["clip_id"] == clip) & (grid["tier"] == "stored") & grid["behavior"].isin(CLASSES7)
    yh = grid.loc[te_h, "behavior"].to_numpy(); ph = model.predict(grid.loc[te_h, cols].to_numpy(float))
    f1_h = f1_score(yh, ph, labels=np.unique(yh), average="macro", zero_division=0)

    # predicted track of this clip: features from predicted boxes, label from the matched human box
    pf = build_features(pr_feats_src)
    pr = pd.concat([pr_feats_src.reset_index(drop=True), pf], axis=1)
    key = matched[["frame", "track_id", "gt_pig", "gt_behavior", "gt_tier"]].assign(track_id=matched["track_id"].astype(str))
    pr = pr.merge(key, left_on=["annotation_index", "pig_id"], right_on=["frame", "track_id"], how="left")
    te_p = (pr["gt_tier"] == "stored") & pr["gt_behavior"].isin(CLASSES7)
    yp = pr.loc[te_p, "gt_behavior"].to_numpy(); pp = model.predict(pr.loc[te_p, cols].to_numpy(float))
    f1_p = f1_score(yp, pp, labels=np.unique(yp), average="macro", zero_division=0)

    # pen-level budgets: truth from kept human rows of this clip; predicted from every predicted box
    truth = grid.loc[(grid["clip_id"] == clip) & kept, "behavior"].value_counts(normalize=True).reindex(CLASSES7, fill_value=0)
    pred_all = pd.Series(model.predict(pr[cols].to_numpy(float))).value_counts(normalize=True).reindex(CLASSES7, fill_value=0)
    pred_h = pd.Series(model.predict(grid.loc[(grid["clip_id"] == clip) & kept, cols].to_numpy(float))).value_counts(normalize=True).reindex(CLASSES7, fill_value=0)
    return {"macro_f1_human_boxes": round(float(f1_h), 3), "macro_f1_predicted_boxes": round(float(f1_p), 3),
            "n_scored_human": int(te_h.sum()), "n_scored_predicted": int(te_p.sum()),
            "pen_budget_mae_pp_human_boxes": round(float((truth - pred_h).abs().mean() * 100), 2),
            "pen_budget_mae_pp_predicted_boxes": round(float((truth - pred_all).abs().mean() * 100), 2),
            "budget_true": truth.round(3).to_dict(), "budget_predicted_track": pred_all.round(3).to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, required=True, help="folder with track_<clip>.csv files (reports/detector_<date>)")
    ap.add_argument("--train-clip", default="2019_12_02/000005", help="clip the detector was trained on (reported separately)")
    args = ap.parse_args()
    set_seeds()
    labels = load_labels()
    rows = []
    for f in sorted(args.tracks.glob("track_*.csv")):
        s = f.stem[len("track_"):]          # 2019_12_02_000005
        clip = s[:10] + "/" + s[11:]        # 2019_12_02/000005
        tr = pd.read_csv(f)
        if len(tr) == 0:
            print(clip, "no boxes"); continue
        gt = labels[(labels["clip_id"] == clip) & (labels["annotation_index"] % STEP == 0)]
        pr = tr[tr["frame"] % STEP == 0].copy()
        matched = match_clip(gt, pr)
        # predicted track as a label-table look-alike for build_features
        secs = capture_seconds(clip)
        src = pd.DataFrame({"clip_id": clip, "pig_id": pr["track_id"].astype(str), "annotation_index": pr["frame"],
                            "x": pr["x"], "y": pr["y"], "width": pr["width"], "height": pr["height"],
                            "elapsed_seconds": secs[pr["frame"].to_numpy()], "tier": "stored"}).sort_values(["pig_id", "annotation_index"]).reset_index(drop=True)
        d = {"clip_id": clip, "role": "train (detector)" if clip == args.train_clip else "held-out day"}
        d.update(detection_stats(gt, matched)); d.update(tracking_stats(matched)); d.update(geometry_on_tracks(labels, clip, matched, src))
        rows.append(d)
        print(clip, {k: d[k] for k in ("role", "precision", "recall", "mean_iou_matched", "identity_switches", "macro_f1_human_boxes", "macro_f1_predicted_boxes", "pen_budget_mae_pp_human_boxes", "pen_budget_mae_pp_predicted_boxes")})
    table = pd.DataFrame(rows)
    flat = table.drop(columns=["budget_true", "budget_predicted_track"])
    flat.to_csv(args.tracks / "detector_eval.csv", index=False)
    (args.tracks / "detector_eval_budgets.json").write_text(json.dumps(
        {r["clip_id"]: {"true": r["budget_true"], "predicted_track": r["budget_predicted_track"]} for r in rows}, indent=2), encoding="utf-8")
    print("\n" + flat.to_string(index=False))
    print("\nsaved to", args.tracks, datetime.datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()

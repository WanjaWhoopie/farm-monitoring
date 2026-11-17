# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.3.0",
#     "mcp>=2.0.0",
#     "nbformat==5.11.1",
#     "openai==3.51.0",
#     "pytest==9.2.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pig behaviour baseline, v2 (continues the first notebook)

    This notebook runs on molab. The data-restore cell below fetches the videos and checks that the label file `reconstructed_annotations.csv` is in place.

    What changes compared with v1:
    1. The 43,232 interpolated rows are no longer thrown away wholesale. They are kept only when the label and box are likely to be right (gap, IoU and label-agreement gate). Each row gets a tier: `stored`, `gated`, or `rejected`.
    2. Evaluation is leave-one-day-out over all 7 recording days. There is no single validation day.
    3. Cheap baselines come first: majority class, box geometry and motion features, then frozen-embedding logistic regression. Fine-tuning has to beat these before it is worth the GPU time.
    4. A coarse ethogram (from your `behavior_mapping_draft.json`) is scored next to the 7 classes. It is a draft and needs domain review.

    **Status (16 Nov 2026).** This notebook was converted to marimo and run end to end on molab, with the Edinburgh videos under `/home/marimo/pig_data` and the label file under `/marimo/pig_work`. The numbers quoted in the explanation cells come from the second run that day, on a label file rebuilt from the 2 Hz annotations (md5 `0f2be06d97076cf6bace2ccb1d1f75ed`, seed 42). Every result table is followed by a cell that explains what it shows and what the numbers mean; a cell after the configuration explains what makes a run reproducible; sections 5 and 6 compare v1 with v2 and set out what comes next.
    """)
    return


@app.cell
def _(mo):
    import os, json, math, warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.utils.class_weight import compute_sample_weight
    warnings.filterwarnings("ignore")
    # Show DataFrames and figures as rich cell output.
    display = mo.output.append

    WORK = Path(os.environ.get("AI4E_WORK", "/marimo/pig_work"))
    DATA_ROOT = Path(os.environ.get(
        "AI4E_DATA", "/home/marimo/pig_data"))
    SEED = 42

    def set_seeds(seed=SEED):
        """Seed every random source used in this notebook. Models below also take random_state=SEED."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    set_seeds()
    W, H = 1280, 720

    CLASSES7 = ["sleep", "investigating", "eat", "standing", "lying", "walk", "drink"]
    # Draft grouping from behavior_mapping_draft.json (needs domain review).
    COARSE = {"sleep": "resting", "lying": "resting", "standing": "stationary",
              "walk": "locomotion", "eat": "feeding", "drink": "drinking",
              "investigating": "exploring"}

    # Gate for rows that were not manually stored. Tune with the audit sheet below.
    MAX_GAP = 20      # annotation steps between bracketing stored records (1 step = 1 frame, ~6.4 Hz, so 20 steps is about 3 s)
    MIN_IOU = 0.5     # box overlap between the two bracketing stored records
    MAX_TRAIL = 10    # steps a last stored record may be held forward (about 1.6 s)
    return (
        CLASSES7,
        COARSE,
        DATA_ROOT,
        DummyClassifier,
        H,
        HistGradientBoostingClassifier,
        LogisticRegression,
        MAX_GAP,
        MAX_TRAIL,
        MIN_IOU,
        Path,
        SEED,
        SimpleImputer,
        StandardScaler,
        W,
        WORK,
        accuracy_score,
        compute_sample_weight,
        confusion_matrix,
        display,
        f1_score,
        json,
        make_pipeline,
        math,
        np,
        os,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Reproducibility: what is fixed and what is not

    Every random choice in this notebook is tied to `SEED = 42` in the configuration cell. `set_seeds()` seeds Python's `random`, NumPy and PyTorch (CPU and GPU), and every scikit-learn model that uses randomness receives `random_state=SEED`: the stratified-random baseline, gradient boosting's internal validation split, and the 24-panel audit sample. The leave-one-day-out folds, the nine features and logistic regression are deterministic by construction.

    Two runs give identical tables only when three things match. `run_info.json` in the results folder records all three, so any two runs can be told apart:

    1. **The input file.** The md5 checksum of `reconstructed_annotations.csv` is written to `run_info.json`. Two label files with the same row counts can still differ in where the boxes sit, and that changes every number downstream. This happened on 16 Nov 2026: a first run on the Kaggle export and a second on a locally rebuilt file gave different scores from identical source counts.
    2. **The seed**, as above.
    3. **Library versions and device.** Python, NumPy, pandas, scikit-learn, PyTorch and torchvision versions, and whether a GPU was used, are recorded. GPU convolution kernels can shift embedding values by tiny amounts; the embeddings are cached under `emb_v2`, so re-runs in the same sandbox reuse exactly the same vectors, while a fresh sandbox recomputes them.

    To check reproducibility, re-run the notebook from the configuration cell and compare the new `lodo_summary.csv` with the previous copy. With the same file, seed and versions the tables should match to the third decimal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to read this notebook (terms used throughout)

    This notebook is written so that someone new to computer vision can follow it. Every result table is followed by a cell that explains what the numbers mean. The terms below come up repeatedly.

    | Term | Meaning here |
    |---|---|
    | **Frame** | One still image from a video. Each clip has 1,800 frames captured at about 6.4 per second (the file header says 30 fps, but the real camera clock in `times.txt` says about 6.4 Hz). |
    | **Bounding box** | A rectangle around one pig on one frame, stored as `x`, `y`, `width`, `height` in pixels of a 1280 x 720 image. |
    | **Label / behaviour** | The behaviour name an annotator attached to a box (`eat`, `sleep`, `walk`, ...). Labels are the answers a model learns from. |
    | **Stored row** | A box and label that a human actually placed. Only 12,646 of the 57,600 pig-moments in the data are stored. |
    | **Interpolated row** | A box the annotation tool filled in between two human placements. These are cheap but may be wrong if the pig moved. |
    | **Annotation step** | One row in the sequence for one pig. In this export one step is one frame, so 1 step is about 0.16 s, 20 steps about 3 s. |
    | **Macro-F1** | The main score. F1 is computed for each class separately and then averaged, so a rare behaviour counts as much as a common one. A model that only predicts the commonest class scores badly on macro-F1 even if its accuracy looks fine. |
    | **Accuracy** | Share of rows predicted correctly. Easy to inflate by predicting the commonest class, which is why it is not the headline number. |
    | **Recall** | For one class: of all rows that truly are that class, the share the model found. Recall 0.15 for `lying` means the model found 15% of the lying rows. |
    | **Confusion matrix** | A table with true classes as rows and predicted classes as columns. Diagonal cells are correct predictions; off-diagonal cells show which classes get mixed up. |
    | **Leave-one-day-out (LODO)** | Each of the 7 recording days is held out once as the test set while the model trains on the other 6. Seven scores are then averaged. A form of cross-validation. |
    | **Baseline** | A deliberately simple model every real model must beat: always predict the commonest class, or guess at random. |
    | **Gradient boosting** | A model built from hundreds of small decision trees, each correcting the previous ones. The standard choice for tabular (rows and columns) data. |
    | **Feature** | One input number given to a model, for example the box width. Choosing them by hand is feature engineering. |
    | **Embedding** | The 576 numbers a pretrained image network produces internally for a crop. Using them unchanged is a "frozen" embedding; adjusting the network on our data is "fine-tuning". |
    | **Crop** | The pixels inside a bounding box, cut out and resized, which is what v1 fed to its image model. |

    **Note on the gate units.** `MAX_GAP = 20` and `MAX_TRAIL = 10` are counted in annotation steps. In this export one step is one frame at about 6.4 Hz, so the gap limit is about 3 seconds and the trail limit about 1.6 seconds, not the "10 s" the original comment assumed.
    """)
    return


@app.cell
def _(DATA_ROOT, Path, WORK, mo, os):
    # Restore data after a sandbox restart.
    # molab keeps only /marimo between sessions. The videos (about 3 GB) are downloaded again
    # into DATA_ROOT each session; the label CSV and every output live under WORK, which persists.
    # Kaggle credentials come from a file named kaggle.env in the notebook folder (/marimo), which
    # persists across restarts. Create it from the file panel with two lines:
    #   KAGGLE_USERNAME=your_kaggle_username
    #   KAGGLE_KEY=your_kaggle_api_key
    # Never paste them into a cell. Keep the notebook private while that file exists.
    import subprocess, sys, importlib

    def _load_env_file(path):
        """Load KEY=value lines into the environment without overriding values already set."""
        if not path.exists():
            return 0
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
                n += 1
        return n

    for _envf in (Path("/marimo/kaggle.env"), Path("/marimo/.env")):
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            _load_env_file(_envf)
    KAGGLE_DATASET = "jackbyte/edinburgh-pig-behaviour-annotated"
    WORK.mkdir(parents=True, exist_ok=True)

    def _find_clip_root(base):
        """Folder under which clips sit as <date>/<clip>/color.mp4, whatever the archive's top level is."""
        for p in sorted(Path(base).rglob("color.mp4")):
            return p.parent.parent.parent
        return None

    _have_videos = bool(DATA_ROOT.exists() and _find_clip_root(DATA_ROOT) == DATA_ROOT)
    _have_csv = (WORK / "reconstructed_annotations.csv").exists()
    _msgs = []

    if _have_videos:
        _msgs.append(f"Videos present at `{DATA_ROOT}`.")
    elif os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        try:
            importlib.import_module("kaggle")
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
            importlib.invalidate_caches()
        from kaggle.api.kaggle_api_extended import KaggleApi
        _api = KaggleApi(); _api.authenticate()
        _dl = Path("/home/marimo/pig_download"); _dl.mkdir(parents=True, exist_ok=True)
        print("Downloading", KAGGLE_DATASET, "to", _dl, "(a few minutes)...")
        _api.dataset_download_files(KAGGLE_DATASET, path=str(_dl), unzip=True, quiet=False)
        _root = _find_clip_root(_dl)
        if _root is None:
            raise RuntimeError(f"No color.mp4 found under {_dl} after download; check the dataset layout.")
        DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
        if not DATA_ROOT.exists():
            DATA_ROOT.symlink_to(_root, target_is_directory=True)
        _have_videos = True
        _msgs.append(f"Videos downloaded to `{_root}` and linked as `{DATA_ROOT}`.")
    else:
        _msgs.append("**Videos missing.** Create `kaggle.env` in the notebook folder (file panel, top icon) with the two lines `KAGGLE_USERNAME=...` and `KAGGLE_KEY=...`, then re-run this cell.")

    if _have_csv:
        _msgs.append(f"Label file present at `{WORK / 'reconstructed_annotations.csv'}`.")
    else:
        _msgs.append(f"**Label file missing.** Upload `reconstructed_annotations.csv` (the v1 Kaggle notebook's output) into `{WORK}` with molab's file panel, then re-run this cell.")

    DATA_READY = _have_videos and _have_csv
    _msgs.append("**Data ready.** The cells below can run." if DATA_READY else "**Data not ready.** The cells below will stop until both items above are in place.")
    mo.md("\n\n".join("- " + m for m in _msgs))
    return (DATA_READY,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Gate the reconstructed rows

    A reconstructed row sits between two stored (human) records. It is probably fine when the two records are close in time, the pig barely moved between them, and the label is the same on both sides. It is risky otherwise. Rows held forward past the last record have no right-hand check, so they only get a length cap.
    """)
    return


@app.cell
def _(
    CLASSES7,
    DATA_READY,
    DATA_ROOT,
    MAX_GAP,
    MAX_TRAIL,
    MIN_IOU,
    WORK,
    display,
    np,
    pd,
):
    assert DATA_READY, "Run the data-restore cell above first (videos and label CSV must both be present)."
    labels = pd.read_csv(WORK / "reconstructed_annotations.csv", dtype={"pig_id": str})
    labels["recording_date"] = labels["clip_id"].str.split("/").str[0]

    # This export has one annotation step per video frame (1,800 per clip), so the
    # frame to seek is the annotation index. Real capture times come from times.txt
    # (about 6.4 Hz), not from the 30 fps video clock.
    def capture_seconds(clip_id):
        t = pd.to_datetime(pd.read_csv(DATA_ROOT / clip_id / "times.txt", header=None)[0])
        return (t - t.iloc[0]).dt.total_seconds().to_numpy()

    if "raw_frame_index" not in labels:
        labels["raw_frame_index"] = labels["annotation_index"]
    if "elapsed_seconds" not in labels:
        labels["elapsed_seconds"] = np.nan
        for _clip, _ix in labels.groupby("clip_id").indices.items():
            _secs = capture_seconds(_clip)
            labels.loc[labels.index[_ix], "elapsed_seconds"] = _secs[labels["raw_frame_index"].to_numpy()[_ix]]
    labels = labels.sort_values(["clip_id", "pig_id", "annotation_index"]).reset_index(drop=True)

    def box_iou(a, b):
        """a, b: (n,4) arrays of x, y, w, h. Row-wise IoU."""
        ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
        bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
        ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
        inter = iw * ih
        union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
        return np.where(union > 0, inter / union, 0.0)

    def add_gate_columns(df):
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

    def assign_tier(df, max_gap=MAX_GAP, min_iou=MIN_IOU, max_trail=MAX_TRAIL):
        stored = df["annotation_source"].eq("stored")
        inter_ok = (df["has_right"] & (df["gap"] <= max_gap)
                    & (df["iou_lr"] >= min_iou) & df["label_agree"])
        trail_ok = (~df["has_right"]) & (df["trail_ext"] <= max_trail)
        return np.where(stored, "stored", np.where(inter_ok | trail_ok, "gated", "rejected"))

    labels = add_gate_columns(labels)
    labels["tier"] = assign_tier(labels)

    print("Rows by tier:")
    display(labels["tier"].value_counts().rename("rows"))
    print("\nStored+gated rows per baseline class (stored only in brackets):")
    b7 = labels[labels["behavior"].isin(CLASSES7)]
    display(pd.DataFrame({
        "kept": b7[b7.tier != "rejected"].groupby("behavior").size(),
        "stored_only": b7[b7.tier == "stored"].groupby("behavior").size(),
        "v1_accepted_incl_trailing": b7[b7.annotation_source != "interpolated_box_forward_label"]
            .groupby("behavior").size(),
    }).reindex(CLASSES7))
    return assign_tier, labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the two tables above show

    **Rows by tier.** Every pig-moment is placed in one of three tiers:

    - `stored`: a human placed this box and label. Always trusted.
    - `gated`: an interpolated row that passed three checks: the two human records around it are at most `MAX_GAP` steps apart, their boxes overlap by at least `MIN_IOU` (so the pig barely moved), and both carry the same label. Rows held forward after a pig's last record are kept only up to `MAX_TRAIL` steps. Treated as trustworthy.
    - `rejected`: everything else. Dropped from training and scoring.

    **Rows per class.** For the 7 behaviours used in modelling, the three columns are:

    - `kept`: stored plus gated rows, the training set for v2.
    - `stored_only`: human rows only, the scoring set for the headline numbers.
    - `v1_accepted_incl_trailing`: rows that are not tagged as interpolated, which is the rule v1 used.

    **Why this matters.** v1 threw away all 43,232 interpolated rows. Because annotators placed new boxes mostly when pigs moved, the discarded rows were disproportionately still behaviours, and v1's training set was left with 12,794 rows of which fewer than a thousand were `sleep`. The gate recovers a large share. In the run of 16 Nov 2026 the kept set for the 7 classes was 46,367 rows (11,073 human, 35,294 gated), with `sleep` at 1,720 and `lying` at 2,646.

    **A caution about the third column.** The label file used here lists every video frame (1,800 per clip), but humans labelled only every third frame. The two frames in between carry the previous box and label and are tagged `trailing_forward`, so on this file the `v1_accepted_incl_trailing` column counts far more rows than v1 actually trained on. Use 12,794, taken from the v1 notebook itself, as v1's number.

    **What the 16 Nov 2026 run then found.** Scoring on all kept rows gave macro-F1 0.335 against 0.321 on human rows only. That gap is smaller than the day-to-day spread, so the gated rows are consistent with the human rows but do not change the picture. The gate is worth keeping for the extra `sleep` and `lying` examples; it is not where the gains come from.
    """)
    return


@app.cell
def _(CLASSES7, MAX_TRAIL, assign_tier, display, labels, pd):
    # Sensitivity of the gate: how much data survives, and how many rows per class per day.
    rows = []
    for mg in (5, 10, 20, 40):
        for mi in (0.3, 0.5, 0.7):
            t = assign_tier(labels, mg, mi, MAX_TRAIL)
            k = labels[(t != "rejected") & labels["behavior"].isin(CLASSES7)]
            per_day = k.groupby(["behavior", "recording_date"]).size().unstack(fill_value=0)
            rows.append({"max_gap": mg, "min_iou": mi, "kept_rows": len(k),
                         "min_class_total": int(per_day.sum(1).reindex(CLASSES7, fill_value=0).min()),
                         "day_class_cells_empty": int((per_day.reindex(CLASSES7, fill_value=0) == 0).sum().sum())})
    display(pd.DataFrame(rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the sensitivity table

    Each row is one setting of the gate. The columns tell you what that setting costs or buys:

    - `max_gap`, `min_iou`: the two thresholds tried. Looser values (larger gap, lower overlap) keep more rows but trust more guesses.
    - `kept_rows`: how many rows survive for the 7 classes under that setting.
    - `min_class_total`: the size of the rarest class after gating. If this is tiny, one behaviour has almost no examples.
    - `day_class_cells_empty`: out of 7 days x 7 classes = 49 day-and-class combinations, how many have zero rows. Each empty cell is a day on which a behaviour can never be scored. Lower is better.

    Use this table to pick thresholds that keep the rarest class and the empty-cell count reasonable without opening the gate so wide that the audit sheet below fails. The current setting (`MAX_GAP = 20`, `MIN_IOU = 0.5`) is a middle choice, not a tuned one. Since the gated rows barely changed the scores in the 16 Nov run, there is little reason to loosen it further.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Audit the gate by eye

    Before trusting the gate, look at a random sample of `gated` rows and count how many boxes really sit on the labelled pig. Report that pass rate in the paper. If it is low, tighten `MAX_GAP` and `MIN_IOU`.
    """)
    return


@app.cell
def _(DATA_ROOT, SEED, WORK, display, labels, math, mo):
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def audit_sheet(df, n=24, seed=SEED, out=WORK / "gated_audit_sheet.png"):
        sample = df[df["tier"] == "gated"].sample(n, random_state=seed)
        cols = 6; rows_ = math.ceil(n / cols)
        fig, axes = plt.subplots(rows_, cols, figsize=(cols * 3.2, rows_ * 2.4))
        for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
            cap = cv2.VideoCapture(str(DATA_ROOT / r["clip_id"] / "color.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["raw_frame_index"]))
            ok, fr = cap.read(); cap.release()
            if ok:
                ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                ax.add_patch(Rectangle((r["x"], r["y"]), r["width"], r["height"],
                                       fill=False, edgecolor="red", linewidth=1.5))
            ax.set_title(f"{r['clip_id'][-17:]} p{r['pig_id']} i{int(r['annotation_index'])}\n{r['behavior']}", fontsize=6)
            ax.axis("off")
        plt.tight_layout(); plt.savefig(out, dpi=110); plt.show()
        print("Saved", out, "| count correct boxes and record the pass rate.")

    if (DATA_ROOT).exists():
        audit_sheet(labels)
    elif (WORK / "gated_audit_sheet.png").exists():
        print("Videos not present; showing the audit sheet saved in", WORK)
        display(mo.image(src=str(WORK / "gated_audit_sheet.png")))
    return (cv2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What to do with the audit sheet

    The figure above is a random sample of 24 `gated` rows. Each panel shows the frame with the gated box drawn in red and the label that was carried across. The gate is a rule, not a human, so this is the only check that the carried-over boxes and labels are right.

    **The manual step.** Count how many of the 24 red boxes sit on the labelled pig doing the labelled behaviour. That count divided by 24 is the **audit pass rate**. Write it in the paper next to the gate thresholds. Above about 90% the gate can be trusted as is; well below that, tighten `MAX_GAP` and `MIN_IOU` and re-run.

    **Status.** The sheet was first generated on 16 Nov 2026 but the pass rate has not yet been counted. It needs someone's eyes on the real frames. Type the count into the cell below; it is saved with the notebook and written into the results folder.
    """)
    return


@app.cell
def _(mo):
    # Record the audit result here so it persists with the notebook.
    # Count the panels in the sheet above whose red box sits on the labelled pig doing the labelled behaviour.
    # 18 Nov 2026: counted on the three-frame sheet made by scripts/audit_gate.py (seed 42, 24 gated rows
    # from clip 2019_12_02/000005, the only clip with video on the local machine). 23 boxes correct,
    # 1 doubtful (row 15); no label contradicted by the neighbouring human frames. Rows and verdicts are in
    # reports/gate_audit_2026-11-18/audit_rows.csv. Repeat across all 12 clips on molab before the paper.
    AUDIT_CORRECT = 23
    AUDIT_TOTAL = 24
    AUDIT_SAMPLE = "24 gated rows from clip 2019_12_02/000005, three-frame sheet of 18 Nov 2026 (one clip only)"

    if AUDIT_CORRECT is None:
        _txt = "**Audit pass rate: not counted yet.** Open the audit sheet above, count the correct panels, and set `AUDIT_CORRECT` in this cell."
    else:
        _rate = AUDIT_CORRECT / AUDIT_TOTAL
        _verdict = ("the gate can be trusted as set." if _rate >= 0.9
                    else "tighten `MAX_GAP` and `MIN_IOU` in the configuration cell and re-run.")
        _txt = f"**Audit pass rate: {AUDIT_CORRECT} of {AUDIT_TOTAL} = {_rate:.0%}** ({AUDIT_SAMPLE}). At this level {_verdict} Report this number next to the gate thresholds in the paper, and say which sample it came from."
    mo.md(_txt)
    return AUDIT_CORRECT, AUDIT_SAMPLE, AUDIT_TOTAL


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Box geometry and motion features

    No pixels here. These are posture and activity cues that a single crop cannot give: box shape (lying vs standing), speed from the real capture timestamps (not the 30 fps video clock), position in the pen (feeder and drinker zones), and crowding (huddles).

    Speed is computed only from reliable boxes (`stored` or `gated`), so a long interpolated stretch does not fake a smooth velocity.
    """)
    return


@app.cell
def _(CLASSES7, COARSE, H, W, display, labels, np, pd):
    def pair_iou_matrix(b):
        x2, y2 = (b[:, 0] + b[:, 2], b[:, 1] + b[:, 3])
        iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(b[:, 0][:, None], b[:, 0][None]), 0, None)
        ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(b[:, 1][:, None], b[:, 1][None]), 0, None)
        inter = iw * ih
        area = b[:, 2] * b[:, 3]
        return inter / np.maximum(area[:, None] + area[None] - inter, 1e-09)

    def build_features(df, half_window=2):
        df = df.reset_index(drop=True)
        _f = pd.DataFrame(index=df.index)
        _f['cx'] = (df['x'] + df['width'] / 2) / W
        _f['cy'] = (df['y'] + df['height'] / 2) / H
        _f['w'], _f['h'] = (df['width'] / W, df['height'] / H)
        _f['log_area'] = np.log(df['width'] * df['height'])
        _f['aspect'] = df['width'] / df['height']
        speed = np.full(len(df), np.nan)
        reliable = (df['tier'] != 'rejected').to_numpy()
        for _, ix in df.groupby(['clip_id', 'pig_id'], sort=False).indices.items():
            g = df.iloc[ix]
            cx, cy = ((g['x'] + g['width'] / 2).to_numpy(), (g['y'] + g['height'] / 2).to_numpy())
            t = g['elapsed_seconds'].to_numpy()
            ok = reliable[ix]
            n = len(g)
            lo = np.clip(np.arange(n) - half_window, 0, n - 1)
            hi = np.clip(np.arange(n) + half_window, 0, n - 1)
            dt = t[hi] - t[lo]
            d = np.hypot(cx[hi] - cx[lo], cy[hi] - cy[lo])
            v = np.where(ok[lo] & ok[hi] & (dt > 0), d / np.where(dt > 0, dt, 1), np.nan)
            speed[ix] = v
        _f['speed_px_s'] = speed
        nn = np.full(len(df), np.nan)
        mx = np.full(len(df), np.nan)
        for _, ix in df.groupby(['clip_id', 'annotation_index']).indices.items():
            b = df.iloc[ix][['x', 'y', 'width', 'height']].to_numpy(float)
            c = b[:, :2] + b[:, 2:] / 2
            d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
            np.fill_diagonal(d, np.inf)
            iou = pair_iou_matrix(b)
            np.fill_diagonal(iou, 0)
            nn[ix] = d.min(1)
            mx[ix] = iou.max(1)
        _f['nearest_pig_px'], _f['max_overlap_other'] = (nn, mx)
        return _f

    # Features need the whole series (including rejected rows) for speed and crowding.
    full_feats = build_features(labels)
    feat_cols = list(full_feats.columns)
    labels_feat = pd.concat([labels.reset_index(drop=True), full_feats], axis=1)
    kept = labels_feat[(labels_feat['tier'] != 'rejected') & labels_feat['behavior'].isin(CLASSES7)].reset_index(drop=True)
    kept['coarse'] = kept['behavior'].map(COARSE)
    print('Kept rows for modelling:', len(kept))
    display(kept[feat_cols].describe().T[['mean', 'std', 'min', 'max']].round(3))
    return feat_cols, kept


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The nine features, one by one

    The table above summarises the nine numbers computed for every pig at every moment. None of them uses a single pixel from the image; they come from the box and from the other pigs' boxes.

    | Feature | What it is | What it helps with |
    |---|---|---|
    | `cx` | Horizontal centre of the box as a fraction of frame width (0 = left wall, 1 = right wall). | Feeder and drinker are fixed places, so position hints at `eat` and `drink`. |
    | `cy` | Vertical centre as a fraction of frame height. | Same, top to bottom of the pen. |
    | `w` | Box width as a fraction of frame width. | Size and posture. |
    | `h` | Box height as a fraction of frame height. | Size and posture. |
    | `log_area` | Log of width x height in pixels. | A lying pig covers more floor than a standing one seen from above. Log scale keeps big and small boxes comparable. |
    | `aspect` | Width divided by height. | Long thin box versus squat box: different postures. |
    | `speed_px_s` | Pixels per second moved by the box centre over a 4-step window, using the real timestamps from `times.txt`. Only computed when both ends of the window are `stored` or `gated` boxes. | The main cue separating `walk` and `investigating` from still behaviours. |
    | `nearest_pig_px` | Distance in pixels to the nearest other pig's centre on the same frame. | Pigs sleep in huddles, so small distances hint at `sleep` and `lying`. |
    | `max_overlap_other` | Largest intersection-over-union with another pig's box on the same frame (0 = no overlap, 1 = identical boxes). | Crowding and huddling. |

    Features 1 to 6 need only the pig's own box on one frame. `speed_px_s` needs the same pig on nearby frames. The last two need all eight pigs on the same frame. A deployed system therefore needs a tracker (to link one pig across frames) as well as a detector (to find all pigs on a frame) before it can compute these.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Leave-one-day-out evaluation

    Each of the 7 recording days is held out once. The primary score is on `stored` rows only (human-placed records). A second score on all kept rows shows what the gate adds. Macro-F1 uses only the classes present in the held-out day, so a missing class does not silently count as zero.
    """)
    return


@app.cell
def _(
    DummyClassifier,
    HistGradientBoostingClassifier,
    SEED,
    accuracy_score,
    compute_sample_weight,
    confusion_matrix,
    display,
    f1_score,
    feat_cols,
    kept,
    np,
    pd,
):
    from joblib import Parallel, delayed

    class Weighted:
        """Balanced sample weights for models without class_weight."""

        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            self.model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
            return self

        def predict(self, X):
            return self.model.predict(X)

    def lodo(X, y, groups, tiers, make_model, name, n_jobs=7):
        groups, tiers, y = (np.asarray(groups), np.asarray(tiers), np.asarray(y))
        folds, P = ([], np.empty(len(y), dtype=object))
        days = sorted(set(groups))
        masks = [groups == day for day in days]

        def fit_predict(te):
            model = make_model()
            model.fit(X[~te], y[~te])
            return model.predict(X[te])

        # Folds are independent, so fitting them in parallel threads gives the same predictions.
        fold_preds = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(fit_predict)(te) for te in masks)
        for day, te, p in zip(days, masks, fold_preds):
            P[te] = p
            for scope, m in (('stored', te & (tiers == 'stored')), ('all_kept', te)):
                if m.sum():
                    labs = np.unique(y[m])
                    folds.append({'model': name, 'day': day, 'scope': scope, 'n': int(m.sum()), 'macro_f1': f1_score(y[m], P[m], labels=labs, average='macro', zero_division=0), 'accuracy': accuracy_score(y[m], P[m])})
        folds = pd.DataFrame(folds)
        return (folds, P)

    def summarize(folds_list):
        allf = pd.concat(folds_list)
        return allf.groupby(['model', 'scope']).agg(mean_fold_macro_f1=('macro_f1', 'mean'), std_fold=('macro_f1', 'std'), mean_fold_accuracy=('accuracy', 'mean')).round(3).reset_index()

    def recall_table(y, P, tiers, classes):
        m = np.asarray(tiers) == 'stored'
        cm_ = confusion_matrix(np.asarray(y)[m], np.asarray(P)[m], labels=classes)
        return (pd.DataFrame(cm_, index=classes, columns=classes), pd.Series(np.diag(cm_) / np.maximum(cm_.sum(1), 1), index=classes, name='recall'))
    y7, y_coarse = (kept['behavior'].to_numpy(), kept['coarse'].to_numpy())
    groups, tiers = (kept['recording_date'].to_numpy(), kept['tier'].to_numpy())
    Xg = kept[feat_cols].to_numpy(float)
    all_folds, preds = ([], {})
    for _task, _y in (('7class', y7), ('coarse', y_coarse)):
        for _nm, mk in (('majority', lambda: DummyClassifier(strategy='most_frequent')), ('stratified_random', lambda: DummyClassifier(strategy='stratified', random_state=SEED)), ('geometry_motion_HGB', lambda: Weighted(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150, random_state=SEED)))):
            _f, _P = lodo(Xg, _y, groups, tiers, mk, f'{_task}:{_nm}')
            all_folds.append(_f)
            preds[f'{_task}:{_nm}'] = _P
    display(summarize(all_folds))
    return all_folds, lodo, preds, recall_table, summarize, tiers, y7


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the leave-one-day-out summary

    Each row of the table is one model scored one way:

    - `model`: `7class:` rows use the 7 behaviours, `coarse:` rows use the merged groups (`resting`, `stationary`, `locomotion`, `feeding`, `drinking`, `exploring`).
    - `scope`: `stored` scores only human-placed rows (the headline). `all_kept` also scores the gated rows, which shows whether the gate adds anything.
    - `mean_fold_macro_f1`: macro-F1 averaged over the 7 held-out days. The number to compare.
    - `std_fold`: how much that score swings from day to day. Large values mean one lucky or unlucky day could move the result a lot.
    - `mean_fold_accuracy`: plain accuracy, for reference only.

    The three models are the two baselines (`majority` always predicts the commonest class; `stratified_random` guesses in proportion to class frequency) and `geometry_motion_HGB`, gradient boosting on the nine features with balanced class weights.

    **Results from the run on 16 Nov 2026** (7 classes, stored rows, macro-F1 over 7 days):

    | Model | 7 classes | Coarse classes |
    |---|---|---|
    | Majority class | 0.069 | 0.082 |
    | Stratified random | 0.124 | 0.151 |
    | Geometry + motion (gradient boosting) | **0.321 ± 0.057** | **0.387** |

    **What this means.**

    - The geometry model scores more than four times the majority baseline and about two and a half times random guessing. It has learned something real about behaviour from box shape, position and movement alone.
    - The ± 0.057 spread means individual days ranged from 0.27 (2019-11-22) to 0.40 (2019-11-11). v1 reported one validation day with no error bar; this spread shows how misleading a single day can be.
    - On 2019-11-11, the only day v1 was validated on, the geometry model reached macro-F1 0.40 against v1's 0.286. The row sets differ slightly, so read this as a clear direction rather than an exact gain.
    - Merging confusable behaviours into coarse groups lifts the score to 0.387. That is a reasonable headline for the paper, with the 7-class number alongside it.
    - Scoring on `all_kept` rows gave 0.335, close to the 0.321 on `stored`. The gated rows are consistent with the human rows but add little signal.

    **A note on the input file.** An earlier run the same day, on the label file exported by the v1 Kaggle notebook, scored 0.352 for this model. The file used here was rebuilt from the 2 Hz annotations, and in it a box only moves every third frame, so the speed feature is coarser. The 0.03 difference sits inside one day's swing, but it shows that the geometry model depends on how the box track is built. The checksum of the file behind each run is recorded in `run_info.json`.
    """)
    return


@app.cell
def _(CLASSES7, display, preds, recall_table, tiers, y7):
    # Where does the geometry model fail? (stored rows, pooled over days)
    _cm, _rec = recall_table(y7, preds['7class:geometry_motion_HGB'], tiers, CLASSES7)
    display(_cm)
    display(_rec.round(2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the confusion matrix and recall

    The first table is the confusion matrix for the geometry model, pooled over all 7 held-out days and scored on human-placed rows only. Rows are the true behaviour, columns are what the model predicted. The diagonal is correct; everything else is a mistake, and reading along a row shows what each behaviour gets mistaken for. The second table is recall per class: the diagonal count divided by the row total.

    **What the 16 Nov 2026 run showed.**

    - The weakest classes were the still ones: recall 0.14 for `lying`, 0.23 for `standing` and 0.28 for `sleep`. A box that does not move looks the same whether the pig is lying, sleeping or standing still, and the nine features contain no posture detail beyond the box shape.
    - Still pigs were often predicted as moving ones: 228 of the 530 human `sleep` rows were called `walk`, and `lying` went to `investigating` (190) and `walk` (177) more often than to itself (84). In this file a box jumps every third frame rather than gliding, so a resting pig's speed is a train of small steps instead of zero. The speed feature is the most fragile of the nine, and the most sensitive to how the boxes are produced.
    - `investigating`, the largest class, was mostly confused with `walk` (1,171) and `standing` (629). All three involve an upright pig; what separates them is head movement and speed over a longer window than 4 steps.
    - Behaviours tied to a place did well: recall 0.77 for `eat` and 0.60 for `drink`, with `walk` at 0.61 thanks to the speed feature.

    **What it tells us to do next.** These failures are the mirror image of v1's. v1 (a single image crop) could not tell lying from sleep because a still image has no motion; v2 (box geometry) cannot tell them apart because a motionless box has no posture detail. The fix is to combine them: box motion over a longer, smoothed window (several seconds per pig) plus a small amount of appearance information for the resting poses. The combined model in section 4 is a first test of exactly that.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Frozen-embedding baseline

    Pretrained MobileNetV3-Small features (the same backbone as v1) with no fine-tuning, then logistic regression. If this matches the fine-tuned model, the fine-tuning in v1 adds nothing but overfitting. Embeddings are computed once per clip and cached. Turn on the GPU in Kaggle first.

    Crops use the same recipe as v1: box plus 10% margin, padded to square, 224 px.
    """)
    return


@app.cell
def _(DATA_ROOT, WORK, cv2, kept, math, np):
    EMB_DIR = WORK / 'emb_v2'
    EMB_DIR.mkdir(exist_ok=True)
    MARGIN = 0.1
    PAD_RGB = (124, 116, 104)

    def crop_square(frame_bgr, x, y, w, h, size=224):
        H_, W_ = frame_bgr.shape[:2]
        x0 = int(max(0, math.floor(x - MARGIN * w)))
        y0 = int(max(0, math.floor(y - MARGIN * h)))
        x1 = int(min(W_, math.ceil(x + w * (1 + MARGIN))))
        y1 = int(min(H_, math.ceil(y + h * (1 + MARGIN))))
        if x1 <= x0 or y1 <= y0:
            return None
        c = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        s = max(c.shape[:2])
        canvas = np.empty((s, s, 3), np.uint8)
        canvas[:] = PAD_RGB
        t, l = ((s - c.shape[0]) // 2, (s - c.shape[1]) // 2)
        canvas[t:t + c.shape[0], l:l + c.shape[1]] = c
        return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    def make_torch_embedder():
        import torch
        from torchvision import models
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)  # (n, 224, 224, 3) RGB uint8
        net.classifier = torch.nn.Identity()
        net.eval().to(dev)
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

        @torch.no_grad()
        def embed(batch_uint8):
            x = torch.from_numpy(batch_uint8).to(dev).permute(0, 3, 1, 2).float() / 255
            return net((x - mean) / std).cpu().numpy().astype(np.float32)
        print('Embedding device:', dev)
        return embed

    def embed_rows(df, embed_fn, batch=128):
        """Returns (n, d) embeddings aligned with df rows. Reads each video once, in order."""
        out = {}
        for clip_id, g in df.groupby('clip_id'):
            cache = EMB_DIR / (clip_id.replace('/', '_') + '.npz')
            if cache.exists():
                z = np.load(cache, allow_pickle=True)
                have = dict(zip(map(tuple, z['keys']), z['emb']))
            else:
                have = {}
            need = g[[(r.pig_id, int(r.annotation_index)) not in have for r in g.itertuples()]]
            if len(need):
                by_frame = {}
                for r in need.itertuples():
                    by_frame.setdefault(int(r.raw_frame_index), []).append(r)
                cap = cv2.VideoCapture(str(DATA_ROOT / clip_id / 'color.mp4'))
                keys, crops, fi = ([], [], 0)
                last = max(by_frame)
                while fi <= last:
                    if fi in by_frame:
                        ok, fr = cap.read()
                        if not ok:
                            break
                        for r in by_frame[fi]:
                            c = crop_square(fr, r.x, r.y, r.width, r.height)
                            if c is not None:
                                keys.append((r.pig_id, int(r.annotation_index)))
                                crops.append(c)
                    elif not cap.grab():
                        break
                    fi += 1
                cap.release()
                embs = [embed_fn(np.stack(crops[i:i + batch])) for i in range(0, len(crops), batch)]
                for k, e in zip(keys, np.concatenate(embs) if embs else []):
                    have[k] = e
                kk = list(have.keys())
                np.savez(cache, keys=np.array(kk, dtype=object), emb=np.stack([have[k] for k in kk]))
            d = next(iter(have.values())).shape[0]
            for r in g.itertuples():
                out[r.Index] = have.get((r.pig_id, int(r.annotation_index)), np.full(d, np.nan, np.float32))
        return np.stack([out[i] for i in df.index])
    RUN_EMBEDDINGS = DATA_ROOT.exists()
    if RUN_EMBEDDINGS:
        emb = embed_rows(kept, make_torch_embedder())
        ok_rows = ~np.isnan(emb).any(1)
        print('Embedded', int(ok_rows.sum()), 'of', len(kept), 'rows')
    return RUN_EMBEDDINGS, emb, ok_rows


@app.cell
def _(
    CLASSES7,
    LogisticRegression,
    RUN_EMBEDDINGS,
    SimpleImputer,
    StandardScaler,
    all_folds,
    display,
    emb,
    feat_cols,
    kept,
    lodo,
    make_pipeline,
    np,
    ok_rows,
    preds,
    recall_table,
    summarize,
):
    if RUN_EMBEDDINGS:
        k2 = kept[ok_rows].reset_index(drop=True)
        E = emb[ok_rows]
        g2, t2 = (k2['recording_date'].to_numpy(), k2['tier'].to_numpy())
        X_emb = E
        X_both = np.hstack([E, k2[feat_cols].to_numpy(float)])
        mk_lr = lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.02, class_weight='balanced', max_iter=400))
        emb_folds = []
        for _task, _y in (('7class', k2['behavior'].to_numpy()), ('coarse', k2['coarse'].to_numpy())):
            for _nm, X in (('frozen_emb_LR', X_emb), ('frozen_emb+geometry_LR', X_both)):
                _f, _P = lodo(X, _y, g2, t2, mk_lr, f'{_task}:{_nm}')
                emb_folds.append(_f)
                preds[f'{_task}:{_nm}'] = _P
        display(summarize(all_folds + emb_folds))
        _cm, _rec = recall_table(k2['behavior'].to_numpy(), preds['7class:frozen_emb+geometry_LR'], t2, CLASSES7)
        display(_cm)
        display(_rec.round(2))
    return (emb_folds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the embedding results

    This section asks a simple question: does the image itself add anything that the nine box numbers do not already give? Two models are added to the summary table:

    - `frozen_emb_LR`: the pretrained MobileNetV3-Small from v1, not fine-tuned, turned into 576 numbers per crop (an embedding), then a logistic regression classifier with balanced class weights.
    - `frozen_emb+geometry_LR`: the same 576 numbers plus the nine geometry features.

    The confusion matrix and recall below the table are for the combined model.

    **Results from the run on 16 Nov 2026** (7 classes, stored rows, macro-F1 over 7 days):

    | Model | 7 classes | Coarse classes |
    |---|---|---|
    | Frozen embedding alone | 0.320 ± 0.083 | 0.406 |
    | Frozen embedding + geometry | **0.344 ± 0.094** | **0.434** |
    | Geometry + motion alone (from section 3) | 0.321 ± 0.057 | 0.387 |

    **What this means.**

    - Nine box numbers and 576 image numbers score the same on their own, and the two together score best on both class sets. They carry different information: the image shows posture, the boxes show place and movement.
    - The combined model's recall confirms the split. `sleep` rises from 0.28 to 0.47 and `standing` from 0.23 to 0.33 once the image is included, while `walk` falls from 0.61 to 0.37. Appearance helps the resting poses; motion helps the moving ones.
    - Treat the ordering with care. The gap between the combined model and geometry alone (0.344 against 0.321) is smaller than the ± 0.09 day-to-day swing, and the earlier run the same day on the v1 Kaggle export ranked the three the other way round (geometry 0.352, embedding 0.245, combined 0.292). What held in both runs: all three beat the baselines by a wide margin, and the still postures stay the weak spot.
    - This also reinterprets v1. v1 fine-tuned the same backbone on crops and reached 0.286 on one day. The frozen version, with no training of the network at all, reaches 0.320 averaged over 7 days. Fine-tuning added nothing beyond what the pretrained features already had, and the v1 training curve (0.86 on training days, 0.29 on the validation day) shows the rest was memorisation.
    - The next model should add a longer, smoothed time window per pig before it adds more pixels: that is where both feature sets are weakest.
    """)
    return


@app.cell
def _(
    AUDIT_CORRECT,
    AUDIT_SAMPLE,
    AUDIT_TOTAL,
    CLASSES7,
    MAX_GAP,
    MAX_TRAIL,
    MIN_IOU,
    RUN_EMBEDDINGS,
    SEED,
    WORK,
    all_folds,
    emb_folds,
    json,
    kept,
    labels,
    mo,
    np,
    pd,
    preds,
    recall_table,
    summarize,
    tiers,
    y7,
):
    # Save every result table under WORK so it survives a sandbox restart,
    # together with a reproducibility manifest: seed, input checksum, versions, device.
    import datetime
    import hashlib as _hashlib
    import platform as _platform
    import sklearn as _sklearn
    RESULTS_DIR = WORK / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _csv_path = WORK / "reconstructed_annotations.csv"
    _h = _hashlib.md5()
    with _csv_path.open("rb") as _fh:
        for _chunk in iter(lambda: _fh.read(1 << 20), b""):
            _h.update(_chunk)
    INPUT_MD5 = _h.hexdigest()

    _versions = {"python": _platform.python_version(), "numpy": np.__version__,
                 "pandas": pd.__version__, "scikit-learn": _sklearn.__version__}
    _device = "cpu"
    try:
        import torch as _torch
        import torchvision as _tv
        _versions["torch"], _versions["torchvision"] = _torch.__version__, _tv.__version__
        _device = ("cuda: " + _torch.cuda.get_device_name(0)) if _torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    _fold_list = all_folds + (emb_folds if RUN_EMBEDDINGS else [])
    pd.concat(_fold_list).to_csv(RESULTS_DIR / "lodo_folds.csv", index=False)
    summarize(_fold_list).to_csv(RESULTS_DIR / "lodo_summary.csv", index=False)
    _cm, _rec = recall_table(y7, preds["7class:geometry_motion_HGB"], tiers, CLASSES7)
    _cm.to_csv(RESULTS_DIR / "confusion_geometry_7class.csv")
    _rec.round(4).to_csv(RESULTS_DIR / "recall_geometry_7class.csv")
    RUN_INFO = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "input_csv": {"path": str(_csv_path), "md5": INPUT_MD5, "rows": int(len(labels))},
        "gate": {"MAX_GAP": MAX_GAP, "MIN_IOU": MIN_IOU, "MAX_TRAIL": MAX_TRAIL},
        "rows_by_tier": {str(k): int(v) for k, v in labels["tier"].value_counts().items()},
        "rows_kept_7class": int(len(kept)),
        "embeddings_run": bool(RUN_EMBEDDINGS),
        "device": _device,
        "versions": _versions,
        "audit": {"correct": AUDIT_CORRECT, "total": AUDIT_TOTAL, "sample": AUDIT_SAMPLE},
    }
    with (RESULTS_DIR / "run_info.json").open("w") as _fh:
        json.dump(RUN_INFO, _fh, indent=2)
    _vtxt = ", ".join(f"{k} {v}" for k, v in _versions.items())
    mo.md(f"""Saved to `{RESULTS_DIR}` (persists across sandbox restarts):

    - `lodo_summary.csv`: the summary table (one row per model and scope)
    - `lodo_folds.csv`: every per-day fold score behind the summary
    - `confusion_geometry_7class.csv`, `recall_geometry_7class.csv`: the geometry model's confusion matrix and recall
    - `run_info.json`: the manifest below plus gate thresholds, row counts and the audit count

    **Reproducibility manifest.** Seed {SEED}. Input `{_csv_path.name}`, md5 `{INPUT_MD5}`, {len(labels):,} rows. Device: {_device}. Versions: {_vtxt}.

    The audit sheet itself is at `{WORK / 'gated_audit_sheet.png'}`.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. v1 versus v2

    ### What each version did and why

    **v1** took the standard first approach in computer vision. For every human-placed box it cut the pig out of the frame (a crop) and trained an image model, MobileNetV3-Small, to name the behaviour from that crop alone. MobileNet is a small convolutional neural network pretrained on ImageNet, chosen because it is fast and transfer learning from it usually works on small datasets. Training used weighted cross-entropy so rare behaviours were not ignored, and the days were split 5 for training, 1 for validation (2019-11-11) and 1 for test (2019-12-02).

    **v2** bet that posture, movement and location matter more than appearance. It computes nine numbers per pig per moment from the boxes (position, size, shape, speed, distance to and overlap with the nearest pig) and trains gradient-boosted trees on them. It keeps interpolated rows that pass a gate instead of discarding all of them, scores every day by leave-one-day-out, and adds trivial baselines plus a frozen-embedding baseline so every number has something to beat.

    ### Setup side by side

    | | v1 (Kaggle notebook) | v2 (this notebook, run 16 Nov 2026) |
    |---|---|---|
    | Input to the model | One image crop per pig, from human boxes | Nine box numbers: position, size, aspect, speed, nearest pig, overlap |
    | Model | MobileNetV3-Small, fine-tuned, weighted cross-entropy | Gradient-boosted trees, balanced class weights |
    | Label rows used | 12,794 (human rows plus held-forward; all interpolated rows dropped) | 46,367 (human rows plus gated interpolated rows) |
    | Evaluation | One fixed split: 5 days train, 2019-11-11 validate, 2019-12-02 test | Leave-one-day-out over all 7 days |
    | Baselines reported | None | Majority, random, frozen MobileNet, frozen + geometry |
    | Reproducibility | No seed recorded | Seed 42 throughout; input checksum, versions and device written to `run_info.json` |
    | Hardware | CPU, about 4 min per epoch | GPU; geometry evaluation about 1 min for all 7 folds |

    ### Scores side by side (7 classes, macro-F1)

    | | v1 | v2 geometry + motion |
    |---|---|---|
    | Same day, 2019-11-11 | 0.286 | 0.40 |
    | Average over 7 held-out days | not measured | 0.321 ± 0.057 |
    | Accuracy on 2019-11-11 | 0.323 | not the headline metric in v2 |
    | Majority-class baseline | about 0.42 accuracy, **above** the model | 0.069 macro-F1, well **below** the model |
    | Coarse classes | not tried | 0.387 (0.434 with image features added) |

    The 2019-11-11 row is the only like-for-like comparison and even that is approximate: v2 scores human rows only while v1 also scored held-forward rows, and v1 trained on 5 days while v2's fold trains on 6. Read it as "roughly 0.29 became roughly 0.40 on that day".

    ### Where each version fails

    | Behaviour | v1 | v2 |
    |---|---|---|
    | `lying` | 147 of 223 predicted as `sleep` | Recall 0.14, still the weakest class |
    | `sleep` | Merged with `lying` in practice | Recall 0.28; 228 of 530 rows predicted as `walk` |
    | `standing` | Absorbed `eat` rows (118 of 236 eat predicted as standing) | Recall 0.23 |
    | `investigating` | Dominant class, over-predicted | Confused with `walk` and `standing` |

    The same three still behaviours fail in both. v1 failed because a single crop shows no motion; v2 fails because a motionless box shows no posture detail. That overlap is the clearest argument for combining them, and the combined model in section 4 (macro-F1 0.344, `sleep` recall 0.47) is the first evidence that it works.

    ### What v2 settles that v1 could not

    - v1's single validation day gave one number with no error bar. v2 shows a day-to-day spread of about 0.06, so v1's 0.286 could have moved a lot on a different day.
    - v1 had no baseline, so nobody could see it was below majority-class accuracy. v2 puts every model against two trivial baselines.
    - v1's pixel model was never tested against a non-pixel alternative. v2 shows that frozen image features (0.320) and nine box numbers (0.321) do equally well, and that the fine-tuning in v1 was not what mattered.
    - v2 uses every day as a test day, so the 2019-12-02 day that v1 kept untouched is no longer special. The paper should state that leave-one-day-out replaced the fixed split, and why.
    - v2 records what it ran on. Two runs on 16 Nov 2026 with different label files gave different numbers from identical row counts; from now on the file checksum, seed and versions travel with every result table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. What this means for the project

    **Are we building the model?** Yes, and v2 is the first version that works: a classifier that beats every trivial baseline on days it has never seen. It is not the final system, for one reason: it is fed human-drawn boxes. A deployed system has no humans. It needs a **detector** (a model that finds every pig on a raw frame and draws the box) and a **tracker** (which links the same pig across frames so speed and history can be measured). Every v2 number is therefore an upper bound: what the behaviour model could achieve if detection and tracking were perfect.

    **The research story has sharpened.** The contribution is no longer "fine-tune a CNN on crops". It is that nine cheap numbers from a box track do as well as a pretrained image network for pen behaviour, that the two together do best, that the gated label recovery is safe but not decisive, and that the still postures (`lying`, `sleep`, `standing`) are where the remaining difficulty lives. One caveat belongs in the paper: the geometry scores moved by 0.03 between two label files with identical row counts, so the box track itself is part of the method and must be reported with its checksum.

    **Open items, in order:**

    1. Count the audit pass rate on `gated_audit_sheet.png` (section 1) and record it with the gate thresholds.
    2. Settle the label file. Either recover the per-frame export from the v1 Kaggle notebook or rebuild it from the 2 Hz annotations with boxes interpolated frame by frame, record its md5, and re-run so that every number in this notebook comes from one documented file.
    3. Add a temporal model: speed and box statistics over a window of several seconds per pig, to separate `lying`, `sleep` and `standing`, and `investigating` from `walk`.
    4. Only then consider adding appearance back, with the backbone frozen for the first epochs, label smoothing, stronger augmentation, and every 3rd or 4th step per pig to avoid near-duplicate neighbours.
    5. Build the detector and tracker (a fine-tuned pig detector with ByteTrack), then re-score v2 on detected boxes. That is the honest end-to-end number for the final paper.
    6. The fps ablation (6, 2, 1, 0.5 fps), INT8 versus FP32, and byte logging remain for the resource-cost part of the story.

    **Note for anyone re-running this notebook.** molab keeps only the notebook folder `/marimo` between sessions; the home folder is wiped. The data-restore cell near the top downloads the videos again from Kaggle (credentials as molab secrets, never in a cell) and checks that the label CSV is under `/marimo/pig_work`. Outputs, the audit sheet and the saved result tables all live under `/marimo/pig_work`, so they survive. The numbers quoted in the explanation cells are from the second run on 16 Nov 2026 (label file md5 `0f2be06d97076cf6bace2ccb1d1f75ed`, seed 42); the saved tables in `/marimo/pig_work/results` are the record of the latest run, and `run_info.json` there says which file, seed and versions produced them.
    """)
    return


if __name__ == "__main__":
    app.run()

# Resource-aware multimodal monitoring for low-connectivity farms

Resource-aware monitoring of pig welfare for low-connectivity farms. The idea is an offline-first
edge pipeline that fuses two cheap signals, pen video and pen audio, into compact event summaries
that can reach a farmer over a poor connection. 

Proposal asks three questions: how accurate small behaviour and distress models are on days they
have never seen, how much accuracy is lost when the models are compressed or fed fewer frames per
second, and how to deliver event-based alerts over a weak connection. Video and audio are evaluated
as separate branches, because no dataset has both recorded together, and combined by rules on top.

## Data

The two sources are public:

| Branch | Dataset                                                                                                                                                                                                                                                                      | Where to get it                                                                                                                                                                                      |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vision | Edinburgh Pig Behaviour Video Dataset (University of Edinburgh, School of Informatics). 12 annotated clips over 7 days in one pen, 8 pigs, about one hour in total, with per-pig bounding boxes and behaviour labels in `output.json` and capture timestamps in `times.txt`. | Dataset page: https://homepages.inf.ed.ac.uk/rbf/PIGDATA/ . The notebooks read the Kaggle mirror of the annotated subset: https://www.kaggle.com/datasets/jackbyte/edinburgh-pig-behaviour-annotated |
| Audio  | Soundwel: 6,887 labelled pig calls with valence and context labels.                                                                                                                                                                                                          | Zenodo record: https://zenodo.org/records/8252482 . Paper: Briefer et al., Scientific Reports 2022, https://doi.org/10.1038/s41598-022-07174-8                                                       |

Things to know about it,  (details in `IMPLEMENTATION_NOTES.txt`):

- The video files say 30 frames per second, but the recorded timestamps show about 6 frames per
  second. Use the timestamps in `times.txt`, not the video's own clock, for anything involving
  duration or speed.
- The annotators only wrote down a box and a label when something changed. Between those
  moments the box has to be filled in by interpolation and the label carried forward, so a filled-in
  row is not a new human label. Annotation slot `f` corresponds to video frame `3*f`.
- The notebooks work from one table with a row per pig per video frame (172,800 rows: 1,800 frames x
  8 pigs x 12 clips), built from the raw annotation files by the first notebook. Two versions of
  that table can have the same row counts and still differ in where the boxes sit, which changes
  the results. So every run records a fingerprint of the exact file it used (its md5 checksum) in
  `run_info.json`. The numbers reported below come from the version of
  `reconstructed_annotations.csv` with fingerprint `0f2be06d97076cf6bace2ccb1d1f75ed`, rebuilt on
  16 Nov 2026 from the 2 Hz annotation table; a rebuilt copy is the same file only if its
  fingerprint matches.
- The Kaggle mirror is downloaded with the Kaggle API. Put your Kaggle username and API key in a
  `kaggle.env` file next to the notebook (two lines, `KAGGLE_USERNAME=...` and `KAGGLE_KEY=...`).
  That file is ignored by git; never paste the key into a notebook cell.

## Layout

```
AI4E_Project_Proposal_Submitted.pdf   the proposal
AI4E_Proposal_template/               LaTeX template for the final paper (main.tex, references.bib)
IMPLEMENTATION_NOTES.txt              what the data probe found and the implementation plan (9 Nov 2026)
scripts/                              data probe: small bounded download, preprocessing, checks, summary figure
scripts/activity_budget.py            activity-budget evaluation of the v2 geometry model (runs locally, no videos needed)
scripts/audit_gate.py                 three-frame audit sheet for checking recovered label rows by eye (needs the videos)
scripts/fps_ablation.py               geometry model and budgets at 6, 2, 1 and 0.5 boxes per second (runs locally)
requirements-probe.txt                packages for the scripts; versions used are in reports/environment.json
Experiments/
  behaviour-2026-11-20/               crop ablations and the six-day activity budget: notebooks, scripts, and tables
  notebook_v1_kadima.ipynb            v1 (Kaggle): an image network looks at one cut-out of each pig; one fixed test day
  notebook_v2_improved.ipynb          v2 (Jupyter draft): recovers usable interpolated rows, uses box geometry, tests every day in turn
  notebook_v2_marimo.py               v2 as a marimo notebook, the version that runs on molab (current)
  gated_audit_sheet.png               24 random recovered rows, to be checked by eye to see how often the recovery is right
reports/
  acoustic_baseline.json              audio baseline (near chance when one recording team is held out for testing)
  *_probe_summary.json, verification.json, data_probe_findings.png   probe outputs
  v2_molab_2026-11-16/                v2 results: per-day scores, confusion matrix, recall per behaviour, v2_results.xlsx
    run2_rebuilt_csv/                 the run the numbers below come from (seed 42, label file fingerprint 0f2be06d...)
    run3_reproducibility_check.md     the attempted repeat run and what went wrong
  v2_molab_2026-11-18_rerun/          the repeat run that reproduced run 2 exactly (tables, manifest, comparison note)
  activity_budget_2026-11-16/         how well predicted time budgets match the true ones (tables, figure, run_info.json)
  gate_audit_2026-11-18/              audit sheets and verdicts for 24 recovered rows (one clip): 23 of 24 boxes correct
  gate_audit_2026-11-18_all_clips/    four sheets and verdicts for 48 recovered rows, four per clip, all 12 clips: 45 correct, 3 doubtful
  fps_ablation_2026-11-18/            scores and budgets at 6, 2, 1 and 0.5 boxes per second (table, recall by rate, figure)
  detector_2026-11-18/                detector weights (best.pt), training curves, one track file per clip, detector_eval.csv
  fps_ablation_tracks_2026-11-18/     the frame-rate ablation on the predicted track (per clip, mean over unseen clips, figure)
```

## Vision branch so far

**v1** cut each pig out of the frame and trained an image network (MobileNetV3-Small) to name the
behaviour from that cut-out alone. It reached a macro-F1 of 0.286 on a single test day, which is
worse than always guessing the commonest behaviour.

**v2** replaces the cut-out with nine numbers from the box track: where the box is, its size and
shape, how fast it moves, and how close to and how much it overlaps the nearest other pig. It keeps
interpolated rows that pass a sanity check instead of throwing them all away, and it tests every
day in turn (train on six days, test on the seventh, repeat seven times) so each score comes with
a spread across days. It also reports two trivial baselines next to every model so there is always
something to beat.

The score used throughout is macro-F1: a 0 to 1 score where every behaviour counts equally however
rare it is, so a model cannot look good by getting only the common behaviours right.

Reference run, 16 Nov 2026, scored on human-placed rows only, averaged over the 7 test days
(the ± is the spread from day to day):

| Model                                                | 7 behaviours      | 6 merged groups |
| ---------------------------------------------------- | ----------------- | --------------- |
| Always guess the commonest behaviour                 | 0.069             | 0.082           |
| Random guessing in proportion to how common each is  | 0.124             | 0.151           |
| Nine box numbers, gradient-boosted trees             | 0.321 ± 0.057     | 0.387           |
| Pretrained image network used as a fixed feature extractor, logistic regression | 0.320 ± 0.083 | 0.406 |
| Both together                                        | **0.344 ± 0.094** | **0.434**       |

The nine box numbers do as well as the pretrained image network, and the two together do best.
The still postures (lying, sleeping, standing) are the weak classes for every model, because a box
that does not move looks the same for all three. These numbers are the best case for the deployed
system: the boxes were placed by people, and the model that finds and follows pigs automatically
has not been run yet.

## Activity budgets

The deployed system is not meant to report a label per frame. It reports an activity budget: the
share of time a pig, or the pen, spends in each behaviour over a period. That is what a farmer
reads and what the alert rules compare against "normal". `scripts/activity_budget.py` checks how
close budgets built from the geometry model's predictions come to budgets built from the human
labels. The period is one five-minute clip and the unit is one pig (12 clips x 8 pigs = 96
budgets), using the same held-out-day predictions as the table above. The score is the mean
absolute difference in percentage points between the predicted and true share, averaged over the
seven behaviours (lower is better; 0 would be a perfect budget).

| Budget built from | Per pig and clip | Per clip, pen level (8 pigs averaged) |
|---|---|---|
| "Always investigating" | 19.3 | 19.3 |
| Average budget of the training days, copied over | 16.6 | 11.9 |
| Geometry model predictions | **11.8** | **9.2** |
| Geometry model, majority vote over 2 s | 11.9 | 9.2 |

Eating and drinking budgets are within 2 to 4 points of the truth. Sleeping (18 points off) and
investigating (22 points off) are the problem: the model splits still time between sleeping,
lying, standing and investigating, so a clip that is really 88% sleep comes out as 33% sleep.
The predicted budgets are flatter than the true ones. Smoothing the predictions over two seconds
helps the per-frame score (0.323 to 0.339) but not the budgets, which already average over the
whole clip. Details, per-behaviour errors and a figure are in `reports/activity_budget_2026-11-16/`.

## Fewer boxes per second

A system on a farm will not run at full frame rate. `scripts/fps_ablation.py` thins the box track
to 2, 1 and 0.5 boxes per pig per second (the camera captured 6), recomputes the nine box numbers,
retrains the geometry model and rebuilds the budgets at each rate. Same held-out-day scoring as
above; the label gate is computed once at full rate so only the rate changes.

| Boxes per second | Per-frame macro-F1 | Pen budget error (points) | Boxes per pig per minute |
|---|---|---|---|
| 6 (full rate) | 0.323 ± 0.051 | 9.2 | 362 |
| 2 | **0.360 ± 0.064** | **8.3** | 121 |
| 1 | 0.338 ± 0.069 | 8.4 | 60 |
| 0.5 | 0.332 ± 0.039 | 8.5 | 30 |

Two things stand out. First, 2 boxes per second beats the full rate. The human labels sit on
every third frame, so the full-rate track repeats each box twice before it jumps, which makes the
speed feature a train of small steps; at 2 per second every box is a real one and speed is
cleaner. Second, going from 2 to 0.5 per second cuts the data twelve-fold and costs about 0.03 in
per-frame score and almost nothing in the budgets. Walking and drinking suffer most at 0.5 per
second (short events and speed are what a slow track loses); eating, sleeping and lying do not.
For the budget use case, half a box per second looks sufficient. Figure and tables are in
`reports/fps_ablation_2026-11-18/`.

### The same ablation on the predicted track

`scripts/fps_ablation_tracks.py` repeats the frame-rate ablation with the detector's boxes in
place of the human ones: the predicted track is thinned to each rate, the geometry model is
trained on human boxes of the other days at that rate, and the pen budget is rebuilt from the
predicted boxes. Means over the three clips the detector never saw:

| Boxes per second | Precision | Recall | Id switches in 5 min | Macro-F1, human boxes | Macro-F1, predicted track | Pen budget error, human (pts) | Pen budget error, predicted (pts) |
|---|---|---|---|---|---|---|---|
| 6 (full rate) | 0.889 | 0.928 | 154 | 0.279 | 0.294 | 6.4 | 5.9 |
| 2 | 0.890 | 0.927 | 78 | 0.305 | 0.297 | 6.7 | 7.8 |
| 1 | 0.891 | 0.926 | 54 | 0.283 | 0.310 | 7.0 | 7.9 |
| 0.5 | 0.891 | 0.924 | 42 | 0.306 | 0.315 | 6.4 | 8.7 |

Finding pigs does not depend on the rate at all: precision and recall are flat, because each
frame is detected on its own. The per-frame behaviour score on the predicted track stays within
0.02 of itself across the four rates, which is inside the clip-to-clip noise with only three
clips. The pen budget from the predicted track drifts up from about 6 to about 9 points as the
rate drops, still well under the 12 points of the no-model baseline. Fewer boxes per second also
means fewer identity switches, but mostly because there are fewer frames at which to switch. The
practical reading is the same as on human boxes: half a box per second is enough for budgets, and
nothing in the detector stops the rate being lowered. Per-clip numbers and the figure are in
`reports/fps_ablation_tracks_2026-11-18/`; three unseen clips is a small sample, so treat the
differences between rates as indicative.

## Finding the pigs: detector and tracker

Everything above used the human-drawn boxes. A farm system has to find the pigs itself, so
`scripts/detector_train.py` fine-tunes a small detector (YOLO11n, 2.6 million parameters, starting
from weights pretrained on everyday photos) on 600 frames of one clip (2019_12_02/000005, the
2-per-second grid), and `scripts/detector_eval.py` measures how much is lost when the nine box
numbers come from predicted boxes instead of human ones. Only four clip videos are on this
machine (fetched with `scripts/fetch_clips.py`), so the detector saw one day and was tested on
the other three.

Detection on 1,800 frames from the three unseen days: mAP50 0.951, precision 0.941, recall 0.906.
In plain words: on a new day it finds about nine in ten pigs, nine in ten of its boxes are real
pigs, and a box counts as right when it overlaps the human one by at least half. Training took
about 25 minutes on a laptop GPU and stopped itself at epoch 24 (best epoch 14). One frame takes
13.5 ms on the GPU and 58 ms on the CPU, so a CPU alone manages 17 frames per second, far more
than the 2 per second the rest of the pipeline needs.

ByteTrack then follows each box from frame to frame and gives it a track id. The table scores each
clip on the 2-per-second grid: predicted boxes are matched one to one to human boxes (overlap at
least 0.5); the geometry model, trained on human boxes of the other days at the same rate, is run
on the predicted track; and its scores are set beside the same model on that clip's human boxes.

| Clip | Role | Precision | Recall | Id switches in 5 min | Track ids per pig | Macro-F1, human boxes | Macro-F1, predicted boxes | Pen budget error, human (pts) | Pen budget error, predicted (pts) |
|---|---|---|---|---|---|---|---|---|---|
| 2019_11_11/000028 | unseen day | 0.896 | 0.921 | 94 | 5.9 | 0.288 | 0.307 | 3.4 | 5.2 |
| 2019_11_22/000010 | unseen day | 0.934 | 0.909 | 83 | 6.1 | 0.278 | 0.283 | 6.5 | 7.7 |
| 2019_12_10/000078 | unseen day | 0.841 | 0.951 | 58 | 4.2 | 0.348 | 0.301 | 10.1 | 10.4 |
| 2019_12_02/000005 | detector's training clip | 0.997 | 0.999 | 0 | 1.0 | 0.354 | 0.390 | 4.9 | 7.5 |

What this says:

- Finding pigs transfers to new days; keeping their identities does not. On unseen days a pig
  changes track id 60 to 90 times in five minutes and collects 4 to 6 ids, so a per-pig history is
  not reliable yet. Counting pigs and the pen-level budget are.
- The behaviour score barely moves between human and predicted boxes (within 0.05, up on two
  clips, down on one), so the geometry model does not depend on perfect boxes. The pen budget
  error rises by one to three points.
- The training clip's row (no switches, near-perfect boxes) is optimistic: the detector has seen
  those frames. Read the unseen-day rows.

Caveats: one training clip and four clips in total; the human-box scores here differ from the
per-day numbers in the notebook because this model is trained at 2 per second and scored one clip
at a time. Outputs are in `reports/detector_2026-11-18/`.

## Reproducibility

`Experiments/notebook_v2_marimo.py` fixes every source of randomness from one seed (42). Each run
writes `results/run_info.json` saying which seed, which label file (its fingerprint and row count),
which sanity-check thresholds, how many rows of each kind, which library versions and which
hardware produced the tables.

Checked on 18 Nov 2026: the notebook was run again from scratch on a fresh molab sandbox (new
machine, newer libraries, videos downloaded afresh, same label file) and every table came out
identical to the reference run of 16 Nov, to the last decimal: the summary table, all 140
per-day rows, the confusion matrix and the recall per behaviour. The image-network scores
reproduced too, not only the geometry model. The two runs' tables and manifests are in
`reports/v2_molab_2026-11-16/run2_rebuilt_csv/` and `reports/v2_molab_2026-11-18_rerun/`, with
the comparison in the latter's `reproducibility_check.md`.

## Running v2

On molab (or any marimo server with a GPU): open `Experiments/notebook_v2_marimo.py`, put the
Kaggle credentials in `kaggle.env`, and run all cells. The first cell downloads the videos, the
label table is expected in the work folder, and results are written to a `results/` folder inside it.
Locally: `pip install marimo` then `marimo edit Experiments/notebook_v2_marimo.py`, with the
`AI4E_DATA` and `AI4E_WORK` environment variables pointing at the data and work folders.

## Open items

1. Audit of recovered rows: done. 24 rows from one clip (23 correct, 1 doubtful) and 48 rows from
   all 12 clips, four per clip (45 correct, 3 doubtful, none wrong): 68 of 72 boxes correct, no
   label contradicted. The three doubtful rows are ones where the human boxes before and after sit
   in the same odd place, or the pig moved within half a second. Verdicts in
   `reports/gate_audit_2026-11-18/` and `reports/gate_audit_2026-11-18_all_clips/`.
2. Reproducibility: done; the 18 Nov re-run matched the reference run exactly (see "Reproducibility").
3. Detector and tracker: done on four clips (see "Finding the pigs"). Train on more days once
   the other eight videos are fetched, and reduce identity switches (tune the tracker, or use the
   fact that the pen always holds eight pigs).
4. Frame-rate ablation: done on human boxes and on the predicted track (see "Fewer boxes per
   second"). Still to do: time a compressed detector (8-bit instead of 32-bit numbers).
5. Write the rules that combine video and audio events and test them on days with simulated
   problems.

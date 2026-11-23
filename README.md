# Resource-aware multimodal monitoring for low-connectivity farms

AI4E group project (Group 1, Agriculture). We are building a pig-welfare monitor that runs small
vision and audio models on the farm and sends only a tiny summary per pen, so it works on a 2G
link or by SMS. Video and audio are developed as two separate branches, because no public dataset
has both recorded together. Rules on top of the two summaries will combine them.

The proposal asks three questions:

1. How accurate are small behaviour and distress models on recording days and teams they have
   never seen?
2. How much accuracy is lost when the models are compressed or fed fewer frames per second?
3. Can event summaries and alerts be delivered over a weak connection?

This README is the map of the repo. Each section says what was done, what came out, and where
the files are. If you only read one part, read the next section.

## Where we are (20 November 2026)

| Piece | What it shows | Key number | Files |
| --- | --- | --- | --- |
| v1 crop model (vision) | An image network on a cut-out of each pig. Below the trivial baseline. | macro-F1 0.286 on one test day | `Experiments/notebook_v1_kadima.ipynb` |
| v2 box model (vision) | Nine numbers from the box track, gradient-boosted trees, tested on every day in turn. | macro-F1 0.321 ± 0.057 over 7 held-out days, 0.360 at 2 boxes per second | `Experiments/notebook_v2_marimo.py`, `reports/v2_molab_2026-11-16/` |
| Activity budgets | Share of time per behaviour from the model's predictions versus the human labels. | pen-level error 9.2 points, no-model copy 11.9 | `scripts/activity_budget.py`, `reports/activity_budget_2026-11-16/` |
| Label gate audit | Checking by eye that the interpolated rows we keep are right. | 68 of 72 boxes correct, none wrong | `scripts/audit_gate.py`, `reports/gate_audit_2026-11-18*/` |
| Frame-rate ablation | The box model at 6, 2, 1 and 0.5 boxes per second. | 2 per second is best, 0.5 per second loses only 0.03 | `scripts/fps_ablation.py`, `reports/fps_ablation_2026-11-18/` |
| Detector and tracker | YOLO11n finds pigs, ByteTrack follows them. Trained on one clip, tested on three unseen days. | mAP50 0.951, 58 ms per frame on CPU | `scripts/detector_*.py`, `reports/detector_2026-11-18/` |
| Ablation on predicted boxes | The frame-rate ablation again, on the detector's boxes. | behaviour score within 0.02 across rates | `scripts/fps_ablation_tracks.py`, `reports/fps_ablation_tracks_2026-11-18/` |
| Event summary and bytes per day | The record a pen monitor sends, and what it costs to send. | 24 bytes per record, 6.9 kB per day, 50 SMS | `scripts/event_summary.py`, `reports/deployability_2026-11-19/` |
| Reproducibility | The v2 notebook re-run on a fresh machine. | every table identical | `reports/v2_molab_2026-11-18_rerun/` |
| Audio branch | Call valence models on Soundwel, scored under three splits. | macro-F1 0.74 within seen teams, 0.52 on a new team | `Experiments/pigwatch-audio.ipynb` |

The one-line story so far. On video, nine numbers from the box track beat an image network, the
model gives useful pen-level activity budgets on days it has never seen, the detector finds pigs
on new days but loses their identities, and half a box per second is enough. On audio, the models
work within the recording teams they were trained on and drop to near chance on a new team. On the
systems side, one summary per pen every five minutes fits in a few SMS messages a day.

## How to run things

Everything in `scripts/` runs from the repo root with `python scripts/<name>.py`. The packages are
in `requirements-probe.txt` plus `torch`, `ultralytics` and `marimo` for the detector and the
notebook. The exact versions used are in `reports/environment.json` and in each run's
`run_info.json`.

Data is not in the repo. The scripts expect:

- `data/raw/edinburgh/annotated/<day>/<clip>/` with `output.json` and `times.txt` for all 12 clips,
  and `color.mp4` for the clips you want to run video scripts on. `scripts/fetch_clips.py` downloads
  single clips from the Edinburgh archive. Example: `python scripts/fetch_clips.py 2019_11_11/000028`.
- `data/processed/vision/reconstructed_annotations.csv`, the label table (one row per pig per video
  frame, 172,800 rows). Every result below was computed from the copy with md5 fingerprint
  `0f2be06d97076cf6bace2ccb1d1f75ed`. Ask Wanjiru for that file, and check the fingerprint before
  trusting a rebuilt one.

| Command | Needs videos | Time | What it writes |
| --- | --- | --- | --- |
| `python scripts/activity_budget.py` | no | 2 min | budgets and their errors, `reports/activity_budget_<date>/` |
| `python scripts/fps_ablation.py` | no | 5 min | scores at four frame rates, `reports/fps_ablation_<date>/` |
| `python scripts/audit_gate.py --n 24` | yes | 1 min | three-frame audit sheets and a CSV to fill in |
| `python scripts/detector_data.py --split clip --val-clips <a,b,c>` | yes | 2 min | frames and labels in YOLO format under `data/processed/vision/yolo_pig/` |
| `python scripts/detector_train.py --data data/processed/vision/yolo_pig/split_clip.yaml --weights yolo11n.pt` | yes, GPU | 25 min | weights, tracks per clip, timings, `reports/detector_<date>/` |
| `python scripts/detector_eval.py --tracks reports/detector_<date>` | no | 1 min | detection, tracking and behaviour scores on predicted boxes |
| `python scripts/fps_ablation_tracks.py --tracks reports/detector_<date>` | no | 5 min | the frame-rate ablation on predicted boxes |
| `python scripts/event_summary.py` | no | seconds | example record, bytes-per-day table and figure |

The v2 notebook runs on molab or any marimo server with a GPU. Open
`Experiments/notebook_v2_marimo.py`, put your Kaggle username and key in a file called
`kaggle.env` next to it (two lines, `KAGGLE_USERNAME=...` and `KAGGLE_KEY=...`), upload the label
table to the work folder, and press Run all. The first cell downloads the videos from the Kaggle
mirror. Results land in `results/` inside the work folder. Locally, `pip install marimo` then
`marimo edit Experiments/notebook_v2_marimo.py`, with `AI4E_DATA` and `AI4E_WORK` pointing at the
data and work folders. The `kaggle.env` file is ignored by git. Never paste the key into a cell.

The audio notebook `Experiments/pigwatch-audio.ipynb` runs on Kaggle with the Soundwel zip
attached. It writes its tables into the notebook outputs.

## Data

Both sources are public.

| Branch | Dataset | Where to get it |
| --- | --- | --- |
| Vision | Edinburgh Pig Behaviour Video Dataset (University of Edinburgh). 12 annotated clips of five minutes, one pen, 8 pigs, 7 recording days in November and December 2019, with per-pig boxes and behaviour labels in `output.json` and capture timestamps in `times.txt`. | Dataset page: https://homepages.inf.ed.ac.uk/rbf/PIGDATA/ . The notebooks read the Kaggle mirror of the annotated subset: https://www.kaggle.com/datasets/jackbyte/edinburgh-pig-behaviour-annotated |
| Audio | Soundwel. 6,887 labelled pig calls from six recording teams, each with a context and a positive or negative valence label. | Zenodo record: https://zenodo.org/records/8252482 . Paper: Briefer et al., Scientific Reports 2022, https://doi.org/10.1038/s41598-022-07174-8 |

Things to know about the video data (details in `IMPLEMENTATION_NOTES.txt`):

- The video files claim 30 frames per second, but the timestamps show about 6 per second. Use
  `times.txt` for anything involving duration or speed.
- The annotators wrote down a box and a label only when something changed, on every third frame.
  The frames in between are filled in by interpolating the box and carrying the label forward, so a
  filled-in row is not a new human label. Annotation slot `f` is video frame `3*f`.
- We keep a filled-in row only if it passes a gate. The box must overlap both neighbouring human
  boxes by at least half, the gap between them must be at most 20 frames, and a carried label may
  run for at most 10 steps. This keeps 39,180 of the 160,154 filled-in rows. We audited 72 kept
  rows by eye and found 68 boxes correct, 4 doubtful and none wrong.
- Two label tables with the same row counts can still differ in where the boxes sit, and that
  changes every number. Each run therefore records the md5 fingerprint of the table it used in
  `run_info.json`.
- Of the 16 behaviour labels, we score seven (sleep, investigating, eat, standing, lying, walk,
  drink). The other nine are too rare to score day by day. Sleep and investigating alone make up
  over half of all rows, which is why every score below is macro-F1 rather than accuracy.

### Data accounting

Every count, with its unit, so nobody has to guess what a number refers to.

| Count | Unit | What it is |
| --- | --- | --- |
| 172,800 | label rows | 12 clips x 8 pigs x 1,800 video frames, one row per pig per frame |
| 12,646 | label rows | a human placed the box and the label (every third frame, when something changed) |
| 43,232 | label rows | box interpolated between two human boxes, label carried forward |
| 116,922 | label rows | after a pig's last human box in the clip, box and label carried forward |
| 57,600 | label rows | the 2-per-second grid, 12 x 8 x 600, where every human label sits |
| 39,180 | label rows | filled-in rows kept by the gate (used for training only) |
| 120,974 | label rows | filled-in rows rejected by the gate (never used) |
| 600 | video frames | exported per clip for the detector, one per grid slot |
| 6,887 | calls | Soundwel, one WAV file per call |
| 6,150 | calls | calls with all ten supplied acoustic features; the other 737, all from one team, lack them, so the feature-based models use 6,150 and the spectrogram models use all 6,887 |

The 16 behaviour labels, their counts on the 2-per-second grid, and what we do with each:

| Label | Rows | Scored as | Merged group | Why |
| --- | --- | --- | --- | --- |
| sleep | 16,392 | sleep | resting | frequent, welfare-relevant (inactivity) |
| investigating | 14,296 | investigating | exploring | frequent, the normal active state |
| eat | 7,093 | eat | feeding | frequent, feeding is a core welfare measure |
| standing | 6,263 | standing | stationary | frequent still posture |
| lying | 5,088 | lying | resting | frequent still posture, awake |
| walk | 4,306 | walk | locomotion | frequent, locomotion |
| fight | 1,367 | not scored | none | welfare-relevant but absent on some days, so it cannot be scored day by day; candidate for an event flag |
| drink | 962 | drink | drinking | rare but happens in one place, so it is learnable |
| nose-to-nose, sitting, play with toy, run, nose-poke elsewhere | 75 to 929 | not scored | none | too rare for a per-day score |
| jump on top of, other, chase | 1 to 6 | not scored | none | almost never occur |

The merged groups follow the Welfare Quality grouping of pig activity into resting, stationary,
locomotion, feeding, drinking and exploring. Audio valence is a label derived from the context a
call was recorded in. Isolation, restraint and castration calls are negative, reunion, running and
enrichment calls are positive. It is a proxy for emotional state, not a welfare diagnosis.

## Layout

```
AI4E_Project_Proposal_Submitted.pdf   the proposal
AI4E_Proposal_template/               LaTeX template for the final paper
IMPLEMENTATION_NOTES.txt              what the data probe found and the implementation plan (9 Nov 2026)
requirements-probe.txt                packages for the scripts
scripts/
  fetch_data_probe.py, preprocess_*.py, verify_data_probe.py, plot_data_probe.py
                                      the first data probe: a small bounded download, checks and a summary figure
  fetch_clips.py                      downloads named clip videos from the Edinburgh archive
  activity_budget.py                  the v2 box model and its activity budgets, run locally (also a library for the others)
  audit_gate.py                       three-frame audit sheets for checking kept interpolated rows by eye
  fps_ablation.py                     the box model at 6, 2, 1 and 0.5 boxes per second
  detector_data.py                    exports frames and labels in the detector's format
  detector_train.py                   fine-tunes the detector, runs the tracker, times inference
  detector_eval.py                    scores predicted boxes and tracks, and the behaviour model on them
  fps_ablation_tracks.py              the frame-rate ablation on the predicted track
  event_summary.py                    the per-window event record and bytes per day
Experiments/
  notebook_v1_kadima.ipynb            v1 (Kaggle): image network on a cut-out of each pig, one fixed test day
  notebook_v2_improved.ipynb          v2 (Jupyter draft)
  notebook_v2_marimo.py               v2 as a marimo notebook, the version that runs on molab (current)
  pigwatch-audio.ipynb                the audio branch (Paul): Soundwel preprocessing, baselines, CNN and PANNs
  gated_audit_sheet.png               the first audit sheet (whole frames, superseded by the three-frame sheets)
reports/
  environment.json                    package versions
  *_probe_summary.json, verification.json, data_probe_findings.png   outputs of the first data probe
  acoustic_baseline.json              the probe's audio baseline under three splits
  v2_molab_2026-11-16/                v2 results (run2_rebuilt_csv/ is the reference run)
  v2_molab_2026-11-18_rerun/          the repeat run that matched the reference exactly
  activity_budget_2026-11-16/         budgets, errors per behaviour, figure
  gate_audit_2026-11-18/              24 audited rows from one clip, 23 correct
  gate_audit_2026-11-18_all_clips/    48 audited rows from all 12 clips, 45 correct, 3 doubtful
  fps_ablation_2026-11-18/            frame-rate ablation on human boxes
  detector_2026-11-18/                detector weights (best.pt), training curves, one track file per clip, detector_eval.csv
  fps_ablation_tracks_2026-11-18/     frame-rate ablation on predicted boxes
  deployability_2026-11-19/           example event record, bytes-per-day table and figure
```

Every results folder has a `run_info.json` that says when it was made, with which seed, which
input file and which package versions.

## Vision branch, step by step

### v1: an image network on pig cut-outs

v1 cut each pig out of the frame and trained MobileNetV3-Small to name the behaviour from that
cut-out alone. It reached macro-F1 0.286 on a single test day. Always guessing the commonest
behaviour does better on accuracy. The lesson was that a single still picture of a pig does not
say much about what it is doing.

### v2: nine numbers from the box track

v2 replaces the cut-out with nine numbers from the box track. Where the box is, its size and
shape, how fast it moves, how far it is from the nearest other pig and how much it overlaps it.
It keeps interpolated rows that pass the gate instead of throwing them all away. It tests every
day in turn, training on six days and testing on the seventh, seven times, so each score comes
with a spread across days. Two trivial baselines sit next to every model.

Macro-F1 is a 0 to 1 score where every behaviour counts equally however rare it is. A model cannot
look good by getting only the common behaviours right.

Reference run, 16 November 2026, scored on human-placed rows, averaged over the 7 test days. The ±
is the spread from day to day.

| Model | 7 behaviours | 6 merged groups |
| --- | --- | --- |
| Always guess the commonest behaviour | 0.069 | 0.082 |
| Random guessing in proportion to how common each is | 0.124 | 0.151 |
| Nine box numbers, gradient-boosted trees | 0.321 ± 0.057 | 0.387 |
| Pretrained image network as a fixed feature extractor, logistic regression | 0.320 ± 0.083 | 0.406 |
| Both together | **0.344 ± 0.094** | **0.434** |

The nine box numbers do as well as the pretrained image network, and the two together do best.
The still postures (lying, sleeping, standing) are the weak classes for every model, because a box
that does not move looks the same for all three. These numbers use boxes placed by people. The
section on the detector shows what changes when the boxes are predicted.

Per class, for the box model under the six-day protocol the paper will use (2 December excluded,
every other day held out in turn, human-labelled rows on the 2-per-second grid):

| Behaviour | Precision | Recall | F1 | Human-labelled rows |
| --- | --- | --- | --- | --- |
| walk | 0.55 | 0.67 | 0.61 | 2,464 |
| investigating | 0.53 | 0.41 | 0.46 | 3,638 |
| drink | 0.32 | 0.45 | 0.37 | 163 |
| eat | 0.40 | 0.26 | 0.32 | 467 |
| standing | 0.26 | 0.30 | 0.28 | 1,672 |
| sleep | 0.23 | 0.30 | 0.26 | 498 |
| lying | 0.17 | 0.17 | 0.17 | 584 |

Walking is the strongest class and drinking is found despite being small, because it happens in
one place. The confusion matrix is in `reports/v2_sixday_confusion_2026-11-23.csv`. Under this
protocol the mean macro-F1 is 0.314 ± 0.070 and the pen budget error 9.3 points against 12.2 for
no model.

### Activity budgets

The deployed system does not report a label per frame. It reports an activity budget, the share of
time a pig or the pen spends in each behaviour over a period. That is what a farmer reads and what
the alert rules compare against normal. `scripts/activity_budget.py` checks how close budgets built
from the model's predictions come to budgets built from the human labels. The period is one
five-minute clip and the unit is one pig, so 12 clips times 8 pigs gives 96 budgets. The score is
the mean absolute difference in percentage points between predicted and true share, averaged over
the seven behaviours. Lower is better.

| Budget built from | Per pig and clip | Per clip, pen level |
| --- | --- | --- |
| Always investigating | 19.3 | 19.3 |
| Average budget of the training days, copied over | 16.6 | 11.9 |
| Box model predictions | **11.8** | **9.2** |
| Box model, majority vote over 2 s | 11.9 | 9.2 |

Eating and drinking budgets are within 2 to 4 points of the truth. Sleeping (18 points off) and
investigating (22 points off) are the weak ones. The model splits still time between sleeping,
lying, standing and investigating, so a clip that is really 88% sleep comes out as 33% sleep.
Smoothing the predictions over two seconds helps the per-frame score (0.323 to 0.339) but not the
budgets, which already average over the whole clip. Per-behaviour errors and a figure are in
`reports/activity_budget_2026-11-16/`.

### Fewer boxes per second

A system on a farm will not run at full frame rate. `scripts/fps_ablation.py` thins the box track
to 2, 1 and 0.5 boxes per pig per second (the camera captured 6), recomputes the nine numbers,
retrains the model and rebuilds the budgets at each rate. The gate is computed once at full rate,
so only the rate changes.

| Boxes per second | Per-frame macro-F1 | Pen budget error (points) | Boxes per pig per minute |
| --- | --- | --- | --- |
| 6 (full rate) | 0.323 ± 0.051 | 9.2 | 362 |
| 2 | **0.360 ± 0.064** | **8.3** | 121 |
| 1 | 0.338 ± 0.069 | 8.4 | 60 |
| 0.5 | 0.332 ± 0.039 | 8.5 | 30 |

Two things stand out. First, 2 boxes per second beats the full rate. The human labels sit on every
third frame, so the full-rate track repeats each box twice before it jumps, which makes the speed
feature a train of small steps. At 2 per second every box is a real one. Second, going from 2 to
0.5 per second cuts the data twelve-fold and costs about 0.03 in per-frame score and almost nothing
in the budgets. Walking and drinking suffer most at 0.5 per second, because they are short events.
For the budget use case, half a box per second is enough. Figure and tables are in
`reports/fps_ablation_2026-11-18/`.

### Finding the pigs: detector and tracker

Everything above used the human-drawn boxes. A farm system has to find the pigs itself.
`scripts/detector_train.py` fine-tunes a small detector (YOLO11n, 2.6 million parameters, starting
from weights pretrained on everyday photos) on 600 frames of one clip (2019_12_02/000005), and
`scripts/detector_eval.py` measures how much is lost when the nine numbers come from predicted
boxes instead of human ones. Only four clip videos are on the local machine, so the detector saw
one day and was tested on the other three.

Detection on 1,800 frames from the three unseen days gives mAP50 0.951, precision 0.941 and recall
0.906. In plain words, on a new day it finds about nine in ten pigs, nine in ten of its boxes are
real pigs, and a box counts as right when it overlaps the human one by at least half. Training
took about 25 minutes on a laptop GPU. One frame takes 13.5 ms on the GPU and 58 ms on the CPU, so
a CPU alone manages 17 frames per second, far more than the 2 per second the pipeline needs.

ByteTrack then follows each box from frame to frame and gives it a track id. The table scores each
clip on the 2-per-second grid. Predicted boxes are matched one to one to human boxes (overlap at
least 0.5). The box model, trained on human boxes of the other days at the same rate, is run on the
predicted track, and its score sits beside the same model on that clip's human boxes.

| Clip | Role | Precision | Recall | Id switches in 5 min | Track ids per pig | Macro-F1, human boxes | Macro-F1, predicted boxes | Pen budget error, human | Pen budget error, predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019_11_11/000028 | unseen day | 0.896 | 0.921 | 94 | 5.9 | 0.288 | 0.307 | 3.4 | 5.2 |
| 2019_11_22/000010 | unseen day | 0.934 | 0.909 | 83 | 6.1 | 0.278 | 0.283 | 6.5 | 7.7 |
| 2019_12_10/000078 | unseen day | 0.841 | 0.951 | 58 | 4.2 | 0.348 | 0.301 | 10.1 | 10.4 |
| 2019_12_02/000005 | detector's training clip | 0.997 | 0.999 | 0 | 1.0 | 0.354 | 0.390 | 4.9 | 7.5 |

What this says:

- Finding pigs transfers to new days. Keeping their identities does not yet. On unseen days a pig
  changes track id 60 to 90 times in five minutes, so a per-pig history is not reliable. Counting
  pigs and the pen-level budget are.
- The behaviour score barely moves between human and predicted boxes (within 0.05), so the box
  model does not depend on perfect boxes. The pen budget error rises by one to three points.
- The training clip's row is optimistic, because the detector has seen those frames. Read the
  unseen-day rows.

The human-box scores here differ from the per-day numbers in the notebook because this model is
trained at 2 per second and scored one clip at a time. Outputs are in `reports/detector_2026-11-18/`.

### The frame-rate ablation on predicted boxes

`scripts/fps_ablation_tracks.py` repeats the frame-rate ablation with the detector's boxes in place
of the human ones. Means over the three clips the detector never saw:

| Boxes per second | Precision | Recall | Id switches in 5 min | Macro-F1, human boxes | Macro-F1, predicted track | Pen budget error, human | Pen budget error, predicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 (full rate) | 0.889 | 0.928 | 154 | 0.279 | 0.294 | 6.4 | 5.9 |
| 2 | 0.890 | 0.927 | 78 | 0.305 | 0.297 | 6.7 | 7.8 |
| 1 | 0.891 | 0.926 | 54 | 0.283 | 0.310 | 7.0 | 7.9 |
| 0.5 | 0.891 | 0.924 | 42 | 0.306 | 0.315 | 6.4 | 8.7 |

Finding pigs does not depend on the rate, because each frame is detected on its own. The behaviour
score on the predicted track stays within 0.02 across the four rates, which is inside the
clip-to-clip noise with only three clips. The pen budget from the predicted track drifts from about
6 to about 9 points as the rate drops, still well under the 12 points of the no-model baseline. The
reading is the same as on human boxes. Half a box per second is enough for budgets. Per-clip numbers
and the figure are in `reports/fps_ablation_tracks_2026-11-18/`.

## Audio branch

The audio notebook (`Experiments/pigwatch-audio.ipynb`, by Paul) classifies each Soundwel call as
positive or negative valence. Calls are resampled to 32 kHz, centred in a one-second window and
turned into 64-band log-mel spectrograms. Three kinds of model are compared. Gradient boosting on
the ten acoustic features that come with the corpus, a small CNN on the spectrograms, and the
pretrained PANNs networks (MobileNetV2 and Cnn14), either fine-tuned or frozen with a logistic
regression on top.

The split matters more than the model. Calls from one recording session or one recording team are
alike, so a random split lets the model recognise the setup rather than the valence. Every model is
therefore scored three ways. Random five-fold (optimistic), five folds grouped by session, and
leave-one-recording-team-out, which is what a new farm looks like.

| Model | Random split | Grouped by session | Recording team held out |
| --- | --- | --- | --- |
| Logistic regression on the supplied features | 0.69 | 0.58 | 0.51 |
| Gradient boosting on the supplied features | 0.83 | 0.63 | 0.52 |
| CNN on log-mel spectrograms | 0.82 | 0.74 | 0.48 |
| PANNs MobileNetV2, frozen, logistic regression on top | 0.80 | 0.69 | 0.50 |
| PANNs MobileNetV2, fine-tuned | 0.77 | 0.65 | 0.44 |

Scores are macro-F1, mean over folds. Every model loses a third to a half of its score between the
random split and a held-out team. The best score on a new team is 0.52, and for two of the four
teams the AUC is near 0.5. The audio branch works within the conditions it was trained on and does
not yet transfer to a new recording setup. That is the same pattern the vision branch shows across
days, and it is the honest result for research question 1.

## What leaves the farm: event summaries and bytes per day

The point of running the models on the farm is that only a small record has to leave it.
`scripts/event_summary.py` defines that record and costs it. One record per pen per time window
holds when and how long the window was, how many pigs the detector saw, the share of time in each
of the seven behaviours, how much the pigs moved, two audio fields (call count and share of
negative-valence calls, empty here because these pens have no paired audio), and any flags. The
example in `reports/deployability_2026-11-19/` is filled from the real outputs for one clip.

The same record encoded four ways. Readable JSON is 374 bytes, compact JSON 273, compressed 211,
and a fixed binary layout 24 bytes. Five binary records fit in one SMS.

Bytes per day for one pen, 24 hours, by what is sent:

| What is sent | Bytes per day | Hours on a 2G link | SMS per day |
| --- | --- | --- | --- |
| Raw video as recorded (mean of the 12 clips) | 73 GB | 4,078 | 524 million |
| Raw audio, 16 kHz 16-bit | 2.8 GB | 154 | 20 million |
| Audio compressed (Opus, 24 kbit/s) | 259 MB | 14 | 1.9 million |
| Box stream, CSV at 6 per second | 508 MB | 28 | 3.6 million |
| Box stream, binary, 2 per second | 11 MB | 0.6 | 82 thousand |
| Event summary, compact JSON every 5 min | 79 kB | under a minute | 562 |
| Event summary, 24-byte binary every 5 min | 6.9 kB | seconds | 50 |
| Event summary, 24-byte binary every hour | 576 B | seconds | 5 |

Assumptions, all in the script. A 2G link moves 5,000 bytes per second, an SMS carries 140 bytes
of payload, audio is 16 kHz 16-bit or Opus at 24 kbit/s. Video and box sizes are measured from the
real files. Raw video needs a link busy for 170 days to send one day. The box stream still needs a
fast link. The event summary fits in a handful of SMS messages a day with no data link at all. That
is the case for running the models on the farm.

## Reproducibility

`Experiments/notebook_v2_marimo.py` fixes every source of randomness from one seed (42). Each run
writes `results/run_info.json` with the seed, the label file's fingerprint and row count, the gate
thresholds, the row counts of each kind, the package versions and the hardware.

Checked on 18 November 2026. The notebook was run again from scratch on a fresh molab sandbox, on
a different GPU with newer libraries and freshly downloaded videos, and every table came out
identical to the reference run of 16 November, to the last decimal. The two runs' tables and
manifests are in `reports/v2_molab_2026-11-16/run2_rebuilt_csv/` and
`reports/v2_molab_2026-11-18_rerun/`, with the comparison in the latter's `reproducibility_check.md`.

A second, independent implementation of the box model by a teammate, on six days with 2 December
held out, lands in the same place (macro-F1 0.28 versus our 0.31 on the same rows, pen budget 10.4
versus 9.3 points, both against 12.2 for no model). It is on the branch
`report/behaviour-experiments-2026-11-20` and will be merged after review.

## What is next

The instructor's evaluation of 23 November and our point-by-point response are in
`reports/instructor_feedback_response_2026-11-23.md`. The items below reflect it.

1. Fusion rules on the event summary, for example less movement together with more negative calls,
   tested by injecting anomalies into real summaries and sweeping the alert threshold. Week 12,
   owed, and week 14 depends on it.
2. Compressed detector timing, 8-bit instead of 32-bit weights, to finish research question 2.
3. Systems track. Delivery under emulated 2G and 4G links, SMS fallback latency, solar uptime for
   three kit sizes, and timing on a small computer such as a Raspberry Pi.
4. Detector on all 12 clips, and fewer identity switches (tune the tracker, or use the fact that
   the pen always holds eight pigs).
5. Agree one evaluation protocol for the paper. The six-day protocol with 2 December held out is
   the stricter one.
6. Final presentation on 7 or 9 December, paper on 14 December.

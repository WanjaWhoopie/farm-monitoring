# Resource-aware multimodal monitoring for low-connectivity farms

Resource-aware monitoring of pig welfare for low-connectivity farms. The idea is an offline-first
edge pipeline that fuses two cheap signals, pen video and pen audio, into compact event summaries
that can reach a farmer over a poor connection. This is the AI4E group project (group of 5);
final presentations are on 7 and 9 December 2026 and the paper is due on 14 December 2026.

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

The data itself is not in this repository (the `data/` folder is ignored). Things to know about it,
found while probing it (details in `IMPLEMENTATION_NOTES.txt`):

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
requirements-probe.txt                packages for the scripts; versions used are in reports/environment.json
Experiments/
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

## Reproducibility

`Experiments/notebook_v2_marimo.py` fixes every source of randomness from one seed (42). Each run
writes `results/run_info.json` saying which seed, which label file (its fingerprint and row count),
which sanity-check thresholds, how many rows of each kind, which library versions and which
hardware produced the tables. Two runs with the same entries should give the same tables to three
decimals. That check is still to be done: see `reports/v2_molab_2026-11-16/run3_reproducibility_check.md`.

## Running v2

On molab (or any marimo server with a GPU): open `Experiments/notebook_v2_marimo.py`, put the
Kaggle credentials in `kaggle.env`, and run all cells. The first cell downloads the videos, the
label table is expected in the work folder, and results are written to a `results/` folder inside it.
Locally: `pip install marimo` then `marimo edit Experiments/notebook_v2_marimo.py`, with the
`AI4E_DATA` and `AI4E_WORK` environment variables pointing at the data and work folders.

## Open items

1. Check the 24 pictures in the audit sheet by eye and record how many recovered rows are right.
2. Run the notebook again and confirm it gives the same tables as the reference run.
3. Add the model that finds each pig in the frame and follows it between frames (YOLO-nano with
   ByteTrack), so the nine box numbers come from predicted boxes instead of human ones.
4. Repeat the vision results with fewer frames per second (2, 1 and 0.5) and with a compressed
   model (8-bit instead of 32-bit numbers), then write the rules that combine video and audio
   events and test them on days with simulated problems.

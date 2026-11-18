# Reproducibility check, 18 November 2026

The seeded v2 notebook (`Experiments/notebook_v2_marimo.py`, 30 cells, seed 42) was run from the
notebook UI ("Run all") on a fresh molab sandbox. The videos were downloaded again from Kaggle by
the data-restore cell; the label file was uploaded from this repository's local copy
(`data/processed/vision/reconstructed_annotations.csv`, md5 0f2be06d97076cf6bace2ccb1d1f75ed,
172,800 rows), the same file the reference run used.

Compared with the reference run (`reports/v2_molab_2026-11-16/run2_rebuilt_csv/`):

| Table | Rows | Result |
|---|---|---|
| lodo_summary.csv (20 model/scope rows) | 20 | identical, max abs difference 0.0 |
| lodo_folds.csv (every model on every held-out day) | 140 | identical, max abs difference 0.0 |
| confusion_geometry_7class.csv | 7 x 7 | identical |
| recall_geometry_7class.csv | 7 | identical |

Row counts by tier matched as well (stored 12,646; gated 39,180; rejected 120,974; 46,367 rows
kept for the seven classes).

What differed between the two runs and did not matter: the machine (this run: NVIDIA RTX PRO 6000
Blackwell), Python 3.13.11, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, torch 2.11.0. The
reference run's manifest predates the version fields, so its exact versions are unknown; it ran
on 16 November 2026 on the sandbox of that day.

Earlier, a local re-implementation of the geometry model (`scripts/activity_budget.py`,
scikit-learn 1.6.1) gave 0.323 against the notebook's 0.321 for the stored-row macro-F1, so a
different code path on a different machine lands within 0.002.

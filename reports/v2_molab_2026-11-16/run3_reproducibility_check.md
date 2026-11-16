# Run 3 (reproducibility check), 16 Nov 2026: not completed

Goal: re-run the v2 notebook unchanged, with `set_seeds()` (seed 42) and the new
`run_info.json` manifest, and check that `lodo_summary.csv` matches run 2
(`run2_rebuilt_csv/`, label file md5 0f2be06d97076cf6bace2ccb1d1f75ed) to three decimals.

What happened:
- 14:30 SAST: the run was queued from the scratchpad with ExecuteStaleCellsCommand. Nothing ran in 75 min.
- 15:49: the config cell was run explicitly; the gate re-ran (46,367 kept rows) and the audit sheet
  was regenerated, but the scratchpad request stream was cut while cells were running, which on
  molab interrupts the running cell. The results file kept the run-2 stamp (11:02:12).
- 17:13: the sandbox started answering HTTP 410 (gone). Nothing from run 3 was saved.

So run 2 is still unreproduced and the seeded notebook has not yet run end to end.

Files:
- `Experiments/notebook_v2_marimo.py`: the seeded notebook (30 cells), rebuilt locally from the
  12:10 backup plus the exact edits made on the kernel. Not yet executed on a kernel.
- `notebook_v2_marimo_1210_before_seeding.py` (this folder): the notebook as it was before those edits.

Next time, in a fresh sandbox:
1. Check that `/marimo/notebook.py` already has `def set_seeds` (the edits were saved there before the
   sandbox died). If not, upload `Experiments/notebook_v2_marimo.py`.
2. Make sure `/marimo/pig_work/reconstructed_annotations.csv` has md5 0f2be06d97076cf6bace2ccb1d1f75ed
   and `kaggle.env` is in place; the restore cell re-downloads the videos.
3. Run all from the notebook UI, not from a scratchpad request.
4. Compare `/marimo/pig_work/results/lodo_summary.csv` with `run2_rebuilt_csv/lodo_summary.csv`:
   max abs difference below 0.001 means reproduced; `run_info.json` says which seed, file and versions ran.

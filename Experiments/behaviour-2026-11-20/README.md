# Behaviour experiments, 20 Nov 2026

Crop ablations on one validation date, then the six-day activity budget. Start with `notebooks/00_index.ipynb`. The budget comparison is `notebooks/07_farm_protocol.ipynb`.

| | |
| --- | --- |
| Notebooks | `notebooks/` |
| Scripts | `scripts/`. Run from that folder. |
| Measured tables | `results/` |
| Write-ups | `notes/` |

2019-12-02 is not used for training or for choosing a model. The published dense-table scores use all seven days. These tables use six days and, for the budget, the 2 Hz annotation grid, so the two are not ranked against each other.

The result to build on is the box model: stored macro-F1 0.281 ± 0.085, pen budget error 10.40 points, against 12.24 for copying the other days with no model. The best crop result on 2019-11-11 is the five-frame pixel mean at 0.3205. The executed single-crop notebook on that date is 0.3097.

`--data` is the folder of annotated clips. Each clip has `output.json`, `times.txt`, and `color.mp4`. A run writes new tables under `outputs/` and does not overwrite `results/`.

```
python crop_mobilenet.py --data <annotated-clips> --setting frozen
python crop_mobilenet.py --data <annotated-clips> --setting full
python crop_mobilenet.py --data <annotated-clips> --setting split
python crop_mobilenet.py --data <annotated-clips> --setting pixel_mean
python crop_mobilenet.py --data <annotated-clips> --setting feature_mean
python box_features.py --data <annotated-clips>
python activity_budget.py --data <annotated-clips> --sequence --fusion
```

`activity_budget.py` alone scores the box model. `--sequence` adds the eight-step GRU. `--fusion` adds frozen MobileNet embeddings joined to the nine box numbers and reads the videos. Packages: numpy, pandas, scikit-learn, torch, torchvision, opencv-python, pillow.

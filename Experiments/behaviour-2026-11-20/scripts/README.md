# Scripts

These are the procedures that produced the tables in `../results/`. Run them from this folder. `--data` is the folder of annotated clips: each clip has `output.json`, `times.txt`, and `color.mp4`. Each command writes `history.csv`, `validation_report.csv`, and `summary.json` under `--out`. The default folder is `../outputs/<experiment>/`. The tables already measured are in `../results/` and are not overwritten.

```
python crop_mobilenet.py --data <annotated-clips> --setting frozen
python crop_mobilenet.py --data <annotated-clips> --setting full
python crop_mobilenet.py --data <annotated-clips> --setting split
python crop_mobilenet.py --data <annotated-clips> --setting pixel_mean
python crop_mobilenet.py --data <annotated-clips> --setting feature_mean
python box_features.py --data <annotated-clips>
python activity_budget.py --data <annotated-clips>
python sequence_model.py --data <annotated-clips>
python frozen_fusion.py --data <annotated-clips>
```

`activity_budget.py --sequence --fusion` writes the three models into one folder, which is the combined table in `../results/farm-protocol-suite/`. Crop training and frozen embeddings need the video files. The box budget does too, because speed is read from `times.txt` beside `color.mp4`. The first image-network run downloads ImageNet weights.

| file | experiment |
| --- | --- |
| `crop_mobilenet.py` | Frozen head, full fine-tune, split learning rate, five-frame pixel mean, embedding average |
| `box_features.py` | Label gate, nine box numbers, and the single-date box model |
| `activity_budget.py` | Leave-one-day-out gradient boosting and the pen budget |
| `sequence_model.py` | Eight-step GRU on the nine box numbers |
| `frozen_fusion.py` | Frozen MobileNet embeddings joined to the nine box numbers |

Packages: numpy, pandas, scikit-learn, torch, torchvision, opencv-python, pillow.

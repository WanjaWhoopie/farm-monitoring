# Pig-video experiment register

`keep` is the result to build on. `do-not-retry` is a measured failure: do not repeat that same change. The test date is never used for training or selection. The farm score is leave-one-day-out pen budget error. The 1684-crop date remains the check for crop models.

| id | change | validation macro-F1 | accuracy | best epoch | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| notebook | single crop, full fine-tune, executed notebook | 0.3097 | 0.3462 | 5 | reference score to beat |
| e1-frozen-backbone | freeze MobileNetV3-Small, train the head | 0.2773 | 0.2838 | 1 | keep as the model-only reference; below the window |
| baseline-full-finetune | unfreeze the backbone at 1e-4 | 0.2653 | 0.2506 | 1 | do-not-retry |
| e3-differential-lr | backbone 1e-5, head 1e-4 | 0.2725 | 0.2827 | 1 | do-not-retry |
| w1-temporal-window | frozen head, input is the mean of annotation offsets -2..2 | 0.3205 | 0.3557 | 4 | keep: best crop result, above the notebook |
| w2-feature-pool | same five crops, average frozen embeddings instead of pixels | 0.3042 | 0.3409 | 4 | do-not-retry |
| g1-geometry-hgb | nine box-track features, histogram gradient boosting, same date split | 0.4038 | 0.4460 | 1 | keep on this one date; it does not hold up across days |
| g2-lodo-budget | same box model, six-day leave-one-day-out | 0.281 ± 0.085 | — | — | keep: pen budget error 10.40, against 12.24 with no model |
| g3-box-sequence | eight-step GRU on the nine box features | 0.174 ± 0.036 | — | 5 | do-not-retry: pen budget error 14.13 |
| g4-frozen-fusion | frozen MobileNet embeddings plus the nine features | 0.284 ± 0.042 | — | — | do-not-retry: pen budget error 11.21 |

Do not retry:

- Full fine-tune of MobileNetV3-Small at 1e-4 with class-weighted loss and patience 4. Training loss matched the notebook at epoch 1 and validation fell from 0.265 to 0.197.
- Backbone learning rate 1e-5 with the head at 1e-4, same loss and patience. Validation peaked at 0.2725 on epoch 1 and did not beat the frozen head.
- The exact five-frame mean at offsets -2..2 with the frozen head. That run already completed and remains the best crop result.
- Averaging the frozen MobileNet embeddings of those same five crops. Validation macro-F1 was 0.3042, below the pixel mean, and walk and investigating lost the gain.
- The same nine box features with histogram gradient boosting, depth 4, learning rate 0.08, and 150 iterations, on the single validation date. That fit already completed.
- The eight-step GRU on those features, hidden size 32, five epochs. Pen budget error was 14.13, worse than copying the other days.
- Frozen MobileNet embeddings joined to the nine features and fit with logistic regression. Pen budget error was 11.21, worse than the box model alone.

Use:

- The leave-one-day-out box model is the result to build on. Stored macro-F1 is 0.281 ± 0.085 and pen budget error is 10.40 points, against 12.24 for a copied training-day budget. Tables are in `results/farm-protocol-suite/`. The write-up is `notes/farm-protocol-suite.md`.
- The single-date box model scored 0.4038 on 2019-11-11. That day is the high fold. Do not quote it as the score across days.
- The five-frame window stays the best crop result: validation macro-F1 0.3205 and accuracy 0.3557, best epoch 4. The gain there is mostly walk and `investigating`. Drink got worse.
- The frozen single-frame run remains the best model-only ablation, and it lost to both the notebook and the window. Do not go back to a single crop unless the new change is the input itself.

Notes are in `notes/`, scripts in `scripts/`, and the measured tables in `results/`.

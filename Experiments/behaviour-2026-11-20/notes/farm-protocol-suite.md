# Leave-one-day-out activity budget, 20 Nov 2026

The question is: on a day the model has not seen, how close is the pen's activity budget to the true share of time in each behaviour? Macro-F1 on human-placed rows is reported beside that. Lower budget error is better. Copying the average budget of the other days, with no model, scores 12.24 points.

2019-12-02 is not in training and not in the mean. The six days are 2019-11-05, 2019-11-11, 2019-11-15, 2019-11-22, 2019-11-28, and 2019-12-10. Rows are the 2 Hz annotation grid. A row is kept when it is a stored label, or an interpolated box whose neighbouring human boxes are at most 20 raw frames apart, overlap by at least 0.5, and carry the same label, or a trail of at most 10 raw frames. That leaves 15,177 kept rows and 9,486 stored rows. The published scores of 0.321 and 0.360 use a dense 6 Hz table and all seven days, so those numbers are not a ranking against the table below.

| model | stored macro-F1 | pen budget error | clip-pig budget error | verdict |
| --- | ---: | ---: | ---: | --- |
| Nine box features, gradient-boosted trees | 0.281 ± 0.085 | 10.40 | 12.78 | keep |
| Eight-step GRU on those features | 0.174 ± 0.036 | 14.13 | 18.03 | do-not-retry |
| Frozen MobileNet embeddings plus the nine features, logistic regression | 0.284 ± 0.042 | 11.21 | 13.51 | do-not-retry |
| No model: average budget of the other days |  | 12.24 |  | baseline |

## Box model

This is the result to build on for the farm score. The pen budget beats the no-model copy by 1.8 points. The same nine features on one date scored macro-F1 0.4038. That date is the easy fold here: 2019-11-11 is 0.358, and 2019-12-10 is 0.153. The single-date number does not hold up.

| day | stored macro-F1 |
| --- | ---: |
| 2019-11-05 | 0.308 |
| 2019-11-11 | 0.358 |
| 2019-11-15 | 0.336 |
| 2019-11-22 | 0.197 |
| 2019-11-28 | 0.335 |
| 2019-12-10 | 0.153 |

Recall on the pooled stored rows: walk 0.674, drink 0.466, investigating 0.401, standing 0.300, sleep 0.285, eat 0.216, lying 0.173. Eat is much weaker than on the published dense table, where the geometry recall for eat is 0.77. The still postures remain the weak classes.

## Sequence model

A GRU with hidden size 32 reads the previous eight annotation steps, about four seconds, and is trained for five epochs with no look at the held-out day. It loses to the box model on every day and loses to the no-model budget. Drink recall falls to 0.160 and eat to 0.069. Do not repeat this GRU.

## Frozen appearance joined to the boxes

The image network stays frozen. Its 576 numbers are concatenated with the nine box numbers and classified with logistic regression, the pairing in the v2 notebook. Macro-F1 moves from 0.281 to 0.284, inside the day-to-day swing. The pen budget gets worse, 11.21 against 10.40. Sleep recall rises from 0.285 to 0.514. Drink recall falls from 0.466 to 0.227. Appearance helps the resting posture and costs the budget. Do not replace the box model with this fusion.

## What this leaves open

The box model is ahead of both additions on the budget, and two days are still near 0.15 macro-F1. The published fusion gained sleep in the same way and was the best frame score on a different table. On this protocol the budget prefers the boxes alone.

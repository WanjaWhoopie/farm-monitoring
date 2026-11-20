# G1: box-track geometry model

Verdict: **keep**. This is the result to build on for this date split.

The model is a histogram gradient boosting classifier on nine box-track features: center, width, height, log area, aspect, speed, distance to the nearest other pig, and overlap with the nearest other pig. Depth 4, learning rate 0.08, 150 iterations, seed 42, balanced sample weights. No crop is encoded. The rows are the same stored and trailing-hold labels as the MobileNet runs: 8615 train, 1684 validation from 2019-11-11. The test date was not evaluated. Speed uses `times.txt` across two annotation slots on either side, about one second, which is coarser than the published dense frame grid.

| run | validation macro-F1 | validation accuracy | best epoch |
| --- | ---: | ---: | ---: |
| G1 box track | 0.4038 | 0.4460 | 1 |
| W1 five-frame pixel mean | 0.3205 | 0.3557 | 4 |
| Notebook single crop | 0.3097 | 0.3462 | 5 |

The fit itself took 2.3 seconds. Train macro-F1 was 0.8804. Train macro-F1 was 0.8804 on 1684 validation rows.

Per-class F1 on the validation date:

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| walk | 0.670 | 0.611 | 0.639 | 239 |
| investigating | 0.556 | 0.546 | 0.551 | 705 |
| drink | 0.469 | 0.517 | 0.492 | 29 |
| eat | 0.737 | 0.309 | 0.436 | 236 |
| sleep | 0.275 | 0.421 | 0.333 | 159 |
| lying | 0.447 | 0.152 | 0.227 | 223 |
| standing | 0.096 | 0.333 | 0.149 | 93 |

Drink is the class the crops kept missing. Its F1 moved from 0.068 on the five-frame mean to 0.492 here. Walk and investigating also improved. Lying and standing stay weak, which matches the published finding that a still box looks alike for those postures.

The published leave-one-day-out geometry score of 0.321 ± 0.057 is a different protocol: gated interpolated rows, every day held out in turn. This 0.4038 does not replace that number. It is the score of the same nine features on one validation date.

Do not resubmit this same fit. The five-frame pixel mean remains the crop result to beat if the next change is still an image model.

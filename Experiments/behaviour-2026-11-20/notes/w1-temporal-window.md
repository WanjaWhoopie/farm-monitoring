# W1: five-frame temporal mean

Verdict: **keep**. This is the crop result to build on.

The model is the frozen MobileNetV3-Small head from E1. The data change is the input. Each example is the mean of five crops of the same pig, at annotation offsets -2, -1, 0, 1, and 2. Each offset is three video frames. The label is still the center frame. Train and validation counts stay 8615 and 1684, so the score is on the same observations as the single-frame runs. The test date was not evaluated.

| run | validation macro-F1 | validation accuracy | best epoch |
| --- | ---: | ---: | ---: |
| W1 temporal window | 0.3205 | 0.3557 | 4 |
| Notebook single crop | 0.3097 | 0.3462 | 5 |
| E1 frozen single crop | 0.2773 | 0.2838 | 1 |

| epoch | train macro-F1 | validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.4624 | 0.2983 |
| 2 | 0.5822 | 0.2909 |
| 3 | 0.6285 | 0.3020 |
| 4 | 0.6499 | 0.3205 |
| 5 | 0.6670 | 0.2906 |
| 6 | 0.6781 | 0.2848 |
| 7 | 0.6989 | 0.2718 |
| 8 | 0.7011 | 0.2824 |

Per-class F1 on the best checkpoint:

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| walk | 0.642 | 0.569 | 0.603 | 239 |
| investigating | 0.679 | 0.277 | 0.393 | 705 |
| eat | 0.578 | 0.284 | 0.381 | 236 |
| sleep | 0.255 | 0.522 | 0.342 | 159 |
| lying | 0.245 | 0.350 | 0.288 | 223 |
| standing | 0.107 | 0.387 | 0.168 | 93 |
| drink | 0.045 | 0.138 | 0.068 | 29 |

Walk and `investigating` are where the window helps. Drink, with 29 validation crops, is weaker than the frozen single-frame head. Do not rerun this exact five-frame mean; use it as the input for the next data-prep change.

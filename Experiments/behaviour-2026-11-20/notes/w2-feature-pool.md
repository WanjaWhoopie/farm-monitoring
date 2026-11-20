# W2: average frozen embeddings, not pixels

Verdict: **do-not-retry**.

The five crops at offsets -2..2 stay separate. The frozen MobileNetV3-Small encodes each crop, the embeddings are averaged, and the same seven-class head is trained. The label is still the center frame. Validation is still the 1684 crops from 2019-11-11. The test date was not evaluated.

| run | validation macro-F1 | validation accuracy | best epoch |
| --- | ---: | ---: | ---: |
| W1 pixel mean | 0.3205 | 0.3557 | 4 |
| Notebook single crop | 0.3097 | 0.3462 | 5 |
| W2 feature mean | 0.3042 | 0.3409 | 4 |

| epoch | train macro-F1 | validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.4321 | 0.2760 |
| 2 | 0.5567 | 0.2759 |
| 3 | 0.6022 | 0.2811 |
| 4 | 0.6257 | 0.3042 |
| 5 | 0.6486 | 0.2993 |
| 6 | 0.6655 | 0.2688 |
| 7 | 0.6774 | 0.2634 |
| 8 | 0.6955 | 0.2889 |

Per-class F1 on the best checkpoint:

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| eat | 0.612 | 0.754 | 0.676 | 236 |
| walk | 0.414 | 0.364 | 0.388 | 239 |
| sleep | 0.208 | 0.579 | 0.306 | 159 |
| investigating | 0.590 | 0.186 | 0.283 | 705 |
| lying | 0.265 | 0.260 | 0.262 | 223 |
| standing | 0.115 | 0.269 | 0.161 | 93 |
| drink | 0.037 | 0.103 | 0.054 | 29 |

Eat improved. Walk fell from 0.603 on the pixel mean to 0.388, and `investigating` fell from 0.393 to 0.283. Drink stayed weak. Averaging embeddings did not repair the drink class and gave back the motion gain from the pixel mean. Do not rerun this pooling. The five-frame pixel mean remains the result to build on.

Per-sample validation predictions are in `results/w2-feature-pool/validation_predictions.csv`.

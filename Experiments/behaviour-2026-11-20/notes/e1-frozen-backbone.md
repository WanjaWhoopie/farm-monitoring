# E1: frozen MobileNetV3-Small backbone

The only change from the notebook recipe was freezing `model.features` and training the new 7-way head. Class weighting, the date split, accepted annotations, image size, and AdamW stayed the same. The test date was not evaluated.

| | validation macro-F1 | validation accuracy | best epoch |
| --- | ---: | ---: | ---: |
| Notebook full fine-tune | 0.3097 | 0.3462 | 5 |
| E1 frozen backbone | 0.2773 | 0.2838 | 1 |

E1 is lower on both validation metrics. The head fit the training crops and the validation score fell after epoch 1, so early stopping ended the run at epoch 5.

| epoch | train macro-F1 | validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.4056 | 0.2773 |
| 2 | 0.5090 | 0.2629 |
| 3 | 0.5556 | 0.2678 |
| 4 | 0.5847 | 0.2681 |
| 5 | 0.6066 | 0.2275 |

8615 training crops and 1684 validation crops. Device was CUDA. Each epoch took about 38 seconds after the first.

Per-class F1 on the best checkpoint, validation date 2019-11-11:

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| eat | 0.523 | 0.716 | 0.605 | 236 |
| lying | 0.290 | 0.390 | 0.333 | 223 |
| walk | 0.465 | 0.247 | 0.322 | 239 |
| sleep | 0.190 | 0.403 | 0.258 | 159 |
| standing | 0.091 | 0.333 | 0.143 | 93 |
| drink | 0.089 | 0.379 | 0.144 | 29 |
| investigating | 0.429 | 0.081 | 0.136 | 705 |

`investigating` is the largest validation class and the frozen head almost never predicts it. Eat is the only class above 0.6 F1.

Freezing the ImageNet backbone did not improve prediction. The next MobileNet run trains the same pipeline with the backbone unfrozen, so the comparison is against this entry as well as the notebook.

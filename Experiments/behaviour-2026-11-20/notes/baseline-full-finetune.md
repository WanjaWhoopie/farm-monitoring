# Baseline rerun: unfrozen MobileNetV3-Small

This is the notebook recipe: the backbone is trained, the loss stays class-weighted, and nothing else changes. The test date was not evaluated.

| run | validation macro-F1 | validation accuracy | best epoch |
| --- | ---: | ---: | ---: |
| Notebook full fine-tune | 0.3097 | 0.3462 | 5 |
| E1 frozen backbone | 0.2773 | 0.2838 | 1 |
| This unfrozen rerun | 0.2653 | 0.2506 | 1 |

Epoch 1 training loss is 1.129, against 1.126 in the notebook, and epoch 1 training macro-F1 is 0.459 against 0.447. The training trajectory matches. The validation score does not. The notebook dipped and then reached 0.3097 at epoch 5. This rerun fell from 0.2653 and early stopping ended it at epoch 5.

| epoch | train macro-F1 | validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.4586 | 0.2653 |
| 2 | 0.6608 | 0.2648 |
| 3 | 0.7356 | 0.2135 |
| 4 | 0.7831 | 0.2011 |
| 5 | 0.8199 | 0.1969 |

Per-class F1 on the best checkpoint:

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| drink | 0.211 | 0.897 | 0.342 | 29 |
| sleep | 0.201 | 0.597 | 0.301 | 159 |
| walk | 0.303 | 0.247 | 0.272 | 239 |
| investigating | 0.475 | 0.133 | 0.208 | 705 |
| standing | 0.120 | 0.559 | 0.198 | 93 |
| eat | 0.580 | 0.292 | 0.389 | 236 |
| lying | 0.186 | 0.121 | 0.147 | 223 |

Unfreezing the backbone at 1e-4 overwrites the ImageNet features in the first epoch. The frozen head was the better of the two results. The next run keeps this recipe and lowers only the backbone learning rate to 1e-5.

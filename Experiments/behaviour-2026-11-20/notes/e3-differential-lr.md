# E3: backbone learning rate 1e-5, head 1e-4

The backbone stays trainable. Its AdamW learning rate is 1e-5 and the classifier stays at 1e-4. Class weighting, the date split, and preprocessing stay as in the unfrozen rerun. The test date was not evaluated.

| run | validation macro-F1 | validation accuracy | best epoch |
| --- | ---: | ---: | ---: |
| Notebook full fine-tune | 0.3097 | 0.3462 | 5 |
| E1 frozen backbone | 0.2773 | 0.2838 | 1 |
| E3 differential learning rate | 0.2725 | 0.2827 | 1 |
| Unfrozen rerun, both at 1e-4 | 0.2653 | 0.2506 | 1 |

| epoch | train macro-F1 | validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.3846 | 0.2725 |
| 2 | 0.5180 | 0.2283 |
| 3 | 0.5793 | 0.1967 |
| 4 | 0.6216 | 0.2237 |
| 5 | 0.6555 | 0.1978 |

Training macro-F1 rose more slowly than the unfrozen rerun, and validation still peaked at epoch 1. Eat remains the strongest class (F1 0.586). `investigating`, the largest validation class, stays weak (F1 0.229, recall 0.146).

Across these three settings, freezing the backbone is the best validation macro-F1. None of them passed the notebook's 0.3097.

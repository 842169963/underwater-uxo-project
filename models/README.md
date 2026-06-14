# Model Weights

This folder intentionally tracks only the representative model weights that match the final project conclusions.

Most training runs produced additional `.pth` files. Those files remain local because they are reproducible training artifacts and would make the repository unnecessarily large.

## Tracked Models

| File | Size | Why it is included |
|---|---:|---|
| `underwater_v1_best_model_seed42_underwater_hn12_cons.pth` | ~42.7 MB | Best balanced single-split model used for the strongest fixed-split result. |
| `groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold1_best.pth` | ~42.7 MB | Fold 1 model from the best strict group-CV strategy. |
| `groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold2_best.pth` | ~42.7 MB | Fold 2 model from the best strict group-CV strategy. |
| `groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold3_best.pth` | ~42.7 MB | Fold 3 model from the best strict group-CV strategy. |

Total tracked model size is approximately `171 MB`.

## Selection Rationale

The selected weights support the two claims made in the project report:

- The best fixed-split model reached `test_acc = 0.8214`, `UXO recall = 0.8571`, and `macro F1 = 0.7888`.
- The best strict validation strategy was `recordings` auxiliary warmup plus a small `trash` train-stage supplement, with mean `test_acc = 0.6620`, mean `UXO recall = 0.5751`, and mean `macro F1 = 0.5870`.

Smoke-test weights and intermediate experiment weights are excluded from Git.


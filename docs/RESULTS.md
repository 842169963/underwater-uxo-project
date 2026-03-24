# Underwater UXO Project Results (Optical-only Baseline)

## 1. Experiment setup
- Task: binary classification on cropped/object-centered underwater images (`UXO` vs `non_UXO`).
- Data source: TU Clausthal subset prepared by [`prepare_underwater_dataset_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/prepare_underwater_dataset_v1.py).
- Optical-only policy: excluded `non optical` folder and filename pattern containing `BV`.
- Preprocess: isotropic resize + centered padding to `224x224` (no geometric stretch).
- Split policy: group-aware split by object instance (to reduce leakage risk).
- Final dataset size: `181` images.
- Train: `124` (`UXO=26`, `non_UXO=98`)
- Val: `29` (`UXO=8`, `non_UXO=21`)
- Test: `28` (`UXO=7`, `non_UXO=21`)

## 2. Training configuration
- Script: [`train_underwater_classifier_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/train_underwater_classifier_v1.py)
- Model: `ResNet18` (ImageNet pretrained), 2-class head.
- Loss: weighted cross-entropy.
- Sampling: `--balanced-sampling` enabled.
- Main metrics: `UXO recall`, `macro F1`, with `accuracy` as secondary.

## 3. Multi-seed results (test split)

| Seed | Test Acc | UXO Recall | UXO F1 | Macro F1 |
|---|---:|---:|---:|---:|
| 42  | 0.7500 | 0.7143 | 0.5882 | 0.7044 |
| 7   | 0.6071 | 1.0000 | 0.5600 | 0.6026 |
| 123 | 0.6786 | 0.8571 | 0.5714 | 0.6571 |
| Mean +- Std | **0.6786 +- 0.0583** | **0.8571 +- 0.1166** | **0.5732 +- 0.0116** | **0.6547 +- 0.0416** |

Source files:
- [`underwater_v1_metrics_seed42.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42.json)
- [`underwater_v1_metrics_seed7.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed7.json)
- [`underwater_v1_metrics_seed123.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed123.json)

## 4. Optimization experiments

### 4.1 Augmentation comparison (seed42)

| Setting | Test Acc | UXO Recall | Macro F1 |
|---|---:|---:|---:|
| `basic` | 0.7500 | 0.7143 | 0.7044 |
| `underwater` | 0.7500 | 0.8571 | 0.7212 |

Takeaway:
- The stronger underwater augmentation improved `UXO recall` and `macro F1` without reducing test accuracy.

Source files:
- [`underwater_v1_metrics_seed42_basic.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_basic.json)
- [`underwater_v1_metrics_seed42_underwater.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater.json)

### 4.2 Conservative hard-negative experiment

Conservative hard-negative mining was tested with:
- underwater augmentation
- hard-negative keywords: `branch`, `wreck`, `shipwreck`
- hard-negative factor: `1.2`

Seed-wise test results:

| Seed | Test Acc | UXO Recall | Macro F1 |
|---|---:|---:|---:|
| 42  | 0.8214 | 0.8571 | 0.7888 |
| 7   | 0.7500 | 0.7143 | 0.7044 |
| 123 | 0.6429 | 0.8571 | 0.6257 |
| Mean +- Std | **0.7381 +- 0.0734** | **0.8095 +- 0.0673** | **0.7063 +- 0.0666** |

Compared to the earlier seeded baseline:
- mean test accuracy improved from `0.6786` to `0.7381`
- mean macro F1 improved from `0.6547` to `0.7063`
- mean non_UXO recall improved from `0.6190` to `0.7143`
- mean UXO recall decreased slightly from `0.8571` to `0.8095`

Interpretation:
- the conservative hard-negative strategy improved overall balance
- it reduced some false positives on confusing non_UXO objects
- it is more suitable than the stronger hard-negative setting tested earlier

Source files:
- [`underwater_v1_metrics_seed42_underwater_hn12_cons.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater_hn12_cons.json)
- [`underwater_v1_metrics_seed7_underwater_hn12_cons.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed7_underwater_hn12_cons.json)
- [`underwater_v1_metrics_seed123_underwater_hn12_cons.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed123_underwater_hn12_cons.json)

## 5. Current best model choice
- Recommended report model: [`underwater_v1_best_model_seed42_underwater_hn12_cons.pth`](C:/Users/stephenxxy/Desktop/project/uxo_project/models/underwater_v1_best_model_seed42_underwater_hn12_cons.pth)
- Why: best single-model balance so far between `UXO recall`, `non_UXO recall`, test accuracy, and `macro F1`.
- Best single-model metrics:
- `test_acc = 0.8214`
- `UXO recall = 0.8571`
- `macro F1 = 0.7888`

## 6. Grad-CAM findings (seed42)
- Summary file: [`GRADCAM_FINDINGS.md`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs/GRADCAM_FINDINGS.md)
- Full visualizations:
- [`seed42_val_all`](C:/Users/stephenxxy/Desktop/project/uxo_project/outputs/gradcam/seed42_val_all)
- [`seed42_test_all`](C:/Users/stephenxxy/Desktop/project/uxo_project/outputs/gradcam/seed42_test_all)
- [`seed42_val_errors`](C:/Users/stephenxxy/Desktop/project/uxo_project/outputs/gradcam/seed42_val_errors)
- [`seed42_test_errors`](C:/Users/stephenxxy/Desktop/project/uxo_project/outputs/gradcam/seed42_test_errors)

Key observations:
- Val: 29 samples, 6 errors (`acc=0.7931`).
- Test: 28 samples, 7 errors (`acc=0.7500`).
- Dominant test error type: `non_UXO -> UXO` false positives (5 cases), especially branch/wreck-like elongated shapes.

## 7. What the baseline has learned
- The model can recover many UXO-like patterns under small-data conditions (high average UXO recall).
- It still confuses hard negatives with UXO-like geometry, limiting precision and stability.
- Performance variance across seeds indicates sensitivity to data scarcity and split composition.

## 8. Known limitations
- Small sample size and class imbalance (`UXO` much fewer than `non_UXO`).
- Single split evaluation can be optimistic or pessimistic depending on random seed.
- Hard negatives (branch/wreck/cylinder-like) remain the main challenge.

## 9. Next priority plan
1. Consolidate this baseline as project deliverable (this report + scripts + Grad-CAM evidence).
2. Extend the conservative hard-negative setting to more controlled comparisons.
3. Continue underwater-specific augmentation tuning.
4. Group-based cross-validation (`3-fold` first, then `5-fold`) for robust final claims.

Reference roadmap:
- [`OPTIMIZATION_ROADMAP.md`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs/OPTIMIZATION_ROADMAP.md)

## 10. Practical final statement
The project established a reproducible optical-only underwater `UXO/non_UXO` baseline with group-aware splitting and explainability support.
On a small dataset, the model shows strong UXO sensitivity and can be improved further through underwater-specific augmentation and conservative hard-negative training. The current optimized version offers a better balance between UXO sensitivity and non_UXO discrimination, while still highlighting the need for more data and more robust group-based validation.

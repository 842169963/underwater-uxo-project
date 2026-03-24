# Underwater UXO Project Progress

## 1. Project goal
- Build an optical-only underwater `UXO / non_UXO` image classification baseline.
- Keep the pipeline reproducible and research-friendly.
- Improve the system gradually through data preparation, hard negatives, augmentation, and validation discipline.

## 2. Initial data preparation
- Main source data was organized from [`tuc_images`](C:/Users/stephenxxy/Desktop/project/uxo_project/tuc_images).
- Non-optical images were excluded.
- Files containing `BV` were also excluded to keep the first version optical-only.
- The prepared dataset was built by [`prepare_underwater_dataset_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/prepare_underwater_dataset_v1.py).
- Preprocessing was standardized to isotropic resize plus centered padding to `224x224`.

Final v1 dataset:
- Total images: `181`
- Train: `124` (`UXO=26`, `non_UXO=98`)
- Val: `29` (`UXO=8`, `non_UXO=21`)
- Test: `28` (`UXO=7`, `non_UXO=21`)

Main outputs:
- Dataset: [`underwater_dataset_v1`](C:/Users/stephenxxy/Desktop/project/uxo_project/underwater_dataset_v1)
- Manifest: [`underwater_dataset_v1_manifest.csv`](C:/Users/stephenxxy/Desktop/project/uxo_project/manifests/underwater_dataset_v1_manifest.csv)

## 3. Baseline training
- Baseline script: [`train_underwater_classifier_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/train_underwater_classifier_v1.py)
- Model: `ResNet18` with ImageNet pretraining
- Loss: weighted cross entropy
- Sampling: balanced sampling enabled
- Main metrics: `UXO recall`, `macro F1`, and confusion matrix

Early multi-seed baseline results:

| Seed | Test Acc | UXO Recall | Macro F1 |
|---|---:|---:|---:|
| 42 | 0.7500 | 0.7143 | 0.7044 |
| 7 | 0.6071 | 1.0000 | 0.6026 |
| 123 | 0.6786 | 0.8571 | 0.6571 |
| Mean +- Std | 0.6786 +- 0.0583 | 0.8571 +- 0.1166 | 0.6547 +- 0.0416 |

Metric files:
- [`underwater_v1_metrics_seed42.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42.json)
- [`underwater_v1_metrics_seed7.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed7.json)
- [`underwater_v1_metrics_seed123.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed123.json)

## 4. First explainability pass
- Grad-CAM visualizations were generated to inspect what the model looked at.
- Main script: [`gradcam_underwater.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/gradcam_underwater.py)
- Key analysis summary: [`GRADCAM_FINDINGS.md`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs/GRADCAM_FINDINGS.md)

Main findings:
- The model learned some UXO-like geometry.
- The dominant failure mode was `non_UXO -> UXO` false positives.
- Branch-like, wreck-like, and elongated clutter were the main confusing negatives.

## 5. Augmentation and hard-negative optimization

### 5.1 Underwater augmentation
- A stronger augmentation profile was added to simulate blur, contrast loss, and turbidity.
- Comparison on seed42:

| Setting | Test Acc | UXO Recall | Macro F1 |
|---|---:|---:|---:|
| `basic` | 0.7500 | 0.7143 | 0.7044 |
| `underwater` | 0.7500 | 0.8571 | 0.7212 |

Useful file:
- [`underwater_v1_metrics_seed42_underwater.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater.json)

Interpretation:
- Stronger underwater augmentation improved `UXO recall` and `macro F1`.

### 5.2 Conservative hard negatives
- Hard-negative keywords were narrowed to confusing classes only:
- `branch`
- `wreck`
- `shipwreck`

The more conservative setting (`factor=1.2`) performed better than the earlier stronger version.

Multi-seed result:

| Seed | Test Acc | UXO Recall | Macro F1 |
|---|---:|---:|---:|
| 42 | 0.8214 | 0.8571 | 0.7888 |
| 7 | 0.7500 | 0.7143 | 0.7044 |
| 123 | 0.6429 | 0.8571 | 0.6257 |
| Mean +- Std | 0.7381 +- 0.0734 | 0.8095 +- 0.0673 | 0.7063 +- 0.0666 |

Best single model so far:
- [`underwater_v1_best_model_seed42_underwater_hn12_cons.pth`](C:/Users/stephenxxy/Desktop/project/uxo_project/models/underwater_v1_best_model_seed42_underwater_hn12_cons.pth)

Main metric file:
- [`underwater_v1_metrics_seed42_underwater_hn12_cons.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater_hn12_cons.json)

Interpretation:
- This setting gave the best balance so far between `UXO recall`, `non_UXO recall`, and `macro F1`.

## 6. More realistic evaluation: 3-fold Group CV
- A dedicated script was added:
- [`train_underwater_groupcv.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/train_underwater_groupcv.py)
- This used group-aware cross-validation instead of relying on one split.

Aggregate result:
- Test accuracy mean: `0.6409`
- Test UXO recall mean: `0.3388`
- Test macro F1 mean: `0.5259`

Metric file:
- [`underwater_groupcv_3fold_underwater_hn12_cons.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_groupcv_3fold_underwater_hn12_cons.json)

Interpretation:
- The single-split best result was optimistic.
- Under stricter group-aware evaluation, UXO generalization remained weak.
- This confirmed that the project still needed more robust data support and better shape-focused negatives.

## 7. External paper study and optimization planning
- Two paper-inspired notes were consolidated into a formal roadmap.
- Main roadmap:
- [`OPTIMIZATION_ROADMAP.md`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs/OPTIMIZATION_ROADMAP.md)

Main takeaway from paper review:
- Keep `ResNet18` as a valid baseline.
- Focus on data-side improvements first.
- Prioritize augmentation, hard negatives, and robust evaluation over more complex architectures.

## 8. New recordings dataset analysis
- A new dataset was added under [`recordings`](C:/Users/stephenxxy/Desktop/project/recordings).
- This dataset contains:
- optical `gopro` images
- sonar `aris_raw` frames
- optical labels for three UXO targets
- `test_cylinder` runs as a potential hard-negative source

Important findings:
- Current priority remains optical-only, so sonar is not yet used in the main pipeline.
- The optical labels were generated on SD frames, so coordinates must be multiplied by `3` to match the FHD images.
- The data consists of many near-duplicate video frames, so sparse run-level sampling is required.

## 9. Auxiliary optical dataset from recordings
- A new extraction script was built:
- [`prepare_recordings_aux_optical_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/prepare_recordings_aux_optical_v1.py)
- This script:
- uses only optical `gopro` frames
- applies `bbox * 3` for labelled UXO crops
- applies a weak crop for cylinder runs
- sparsely samples each run instead of exporting all frames

Current exported auxiliary dataset:
- Dataset: [`aux_recordings_optical_v1`](C:/Users/stephenxxy/Desktop/project/uxo_project/aux_recordings_optical_v1)
- Manifest: [`recordings_aux_optical_v1_manifest.csv`](C:/Users/stephenxxy/Desktop/project/uxo_project/manifests/recordings_aux_optical_v1_manifest.csv)

Current size:
- Total: `325`
- `UXO = 285`
- `non_UXO = 40`

Interpretation:
- This is small enough to be an auxiliary training set rather than a replacement dataset.
- It preserves the original TU evaluation setup.

## 10. Auxiliary pretraining experiment
- The main training script was extended to support:
- `--aux-data-root`
- `--aux-epochs`
- `--aux-lr`

This enables:
- auxiliary optical pretraining on `recordings`
- followed by fine-tuning on the original TU dataset

First real experiment:
- Metric file: [`underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec.json)
- Model: [`underwater_v1_best_model_seed42_underwater_hn12_cons_auxrec.pth`](C:/Users/stephenxxy/Desktop/project/uxo_project/models/underwater_v1_best_model_seed42_underwater_hn12_cons_auxrec.pth)

Compared with the previous best seed42 model:

| Setting | Test Acc | UXO Recall | non_UXO Recall | Macro F1 |
|---|---:|---:|---:|---:|
| `underwater + conservative hn12` | 0.8214 | 0.8571 | 0.8095 | 0.7888 |
| `aux pretrain + underwater + conservative hn12` | 0.7143 | 1.0000 | 0.6190 | 0.7005 |

Interpretation:
- The auxiliary data clearly changes the model behavior.
- It strengthens UXO sensitivity.
- However, in this first setting it biases the model too far toward `UXO`, which hurts overall balance.
- So the auxiliary data is promising, but its training strength must be reduced.

Follow-up milder experiment:
- Metric file: [`underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec1.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec1.json)
- Model: [`underwater_v1_best_model_seed42_underwater_hn12_cons_auxrec1.pth`](C:/Users/stephenxxy/Desktop/project/uxo_project/models/underwater_v1_best_model_seed42_underwater_hn12_cons_auxrec1.pth)
- Setting change: auxiliary pretraining reduced from `3` epochs to `1` epoch

Result compared with the stronger auxiliary stage:

| Setting | Test Acc | UXO Recall | non_UXO Recall | Macro F1 |
|---|---:|---:|---:|---:|
| `aux-epochs=3` | 0.7143 | 1.0000 | 0.6190 | 0.7005 |
| `aux-epochs=1` | 0.7857 | 0.8571 | 0.7619 | 0.7544 |

Interpretation:
- Reducing auxiliary strength made the model more balanced.
- The milder auxiliary stage kept the useful shape prior while reducing the over-bias toward `UXO`.
- It still did not beat the current best non-auxiliary model, but it was clearly better than the stronger auxiliary setting.

Additional learning-rate reduction experiments:
- [`underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec1_lr1e4.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec1_lr1e4.json)
- [`underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec1_lr5e5.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater_hn12_cons_auxrec1_lr5e5.json)

Comparison:

| Setting | Test Acc | UXO Recall | non_UXO Recall | Macro F1 |
|---|---:|---:|---:|---:|
| `aux-epochs=1, aux-lr=2e-4` | 0.7857 | 0.8571 | 0.7619 | 0.7544 |
| `aux-epochs=1, aux-lr=1e-4` | 0.6786 | 0.2857 | 0.8095 | 0.5492 |
| `aux-epochs=1, aux-lr=5e-5` | 0.6786 | 0.4286 | 0.7619 | 0.5902 |

Interpretation:
- Lowering the auxiliary learning rate too much did not help.
- In these runs the model lost too much UXO sensitivity.
- So the best auxiliary setting tested so far is still the milder warmup with `aux-epochs=1` and `aux-lr=2e-4`.

## 11. Current project status
What is already solid:
- reproducible data preparation
- baseline training and evaluation
- multi-seed comparison
- Grad-CAM error analysis
- conservative hard-negative improvement
- group-aware cross-validation
- auxiliary optical dataset extraction from recordings
- two-stage training pipeline support

What remains unstable:
- group-CV UXO generalization
- balance between UXO sensitivity and false positives
- effective use of auxiliary data without over-biasing the model

## 12. Current best model
Current best balanced single model remains:
- [`underwater_v1_best_model_seed42_underwater_hn12_cons.pth`](C:/Users/stephenxxy/Desktop/project/uxo_project/models/underwater_v1_best_model_seed42_underwater_hn12_cons.pth)

Why:
- It still gives the strongest overall balance between UXO recall and non_UXO discrimination.

## 13. Immediate next experiments
1. Reduce auxiliary pretraining strength.
2. Keep TU validation and test unchanged for fair comparison.
3. Consider whether auxiliary data should be used only as light warmup rather than as a stronger pretraining stage.
4. Consider changing auxiliary data composition rather than only changing learning rate.

## 14. Practical summary
The project has evolved from a small optical-only baseline into a more disciplined experimental pipeline with:
- data filtering
- structured hard negatives
- explainability
- stricter validation
- auxiliary-data integration

The main lesson so far is:
- better data and better evaluation matter more than adding model complexity too early
- and new auxiliary data can help, but only if it is introduced gently enough not to dominate the original task

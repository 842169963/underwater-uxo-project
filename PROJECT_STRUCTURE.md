# Project Structure

This repository is organized around one current pipeline: optical-only underwater `UXO` vs `non_UXO` classification.

The root directory contains runnable scripts. Supporting records are grouped into dedicated folders so that reports, manifests, metrics, and data are easy to locate.

## Root Scripts

| File | Purpose |
|---|---|
| `prepare_underwater_dataset_v1.py` | Builds the main TU optical-only dataset and manifest. |
| `prepare_recordings_aux_optical_v1.py` | Extracts the auxiliary `recordings` optical crops with the `bbox * 3` correction. |
| `prepare_combined_aux_optical_v1.py` | Builds combined auxiliary datasets using `recordings` plus selected `trash_ICRA19` hard negatives. |
| `train_underwater_classifier_v1.py` | Trains the single-split ResNet18 baseline. |
| `train_underwater_groupcv.py` | Runs stricter 3-fold group-aware cross-validation. |
| `gradcam_underwater.py` | Generates Grad-CAM visualizations for trained models. |
| `predict_underwater.py` | Runs command-line inference on one image. |
| `predict_underwater_ui.py` | Provides a small local UI for inference. |
| `download_all_images.py` | Utility script used during early data collection. |

## Documentation

| Path | Purpose |
|---|---|
| `docs/RESULTS.md` | Main result summary. |
| `docs/PROJECT_PROGRESS.md` | Full progress and experiment timeline. |
| `docs/REPORT_OUTLINE.md` | Presentation-ready project report outline. |
| `docs/GRADCAM_FINDINGS.md` | Grad-CAM observations and error patterns. |
| `docs/OPTIMIZATION_ROADMAP.md` | Possible next-step experiments. |

## Manifests

| Path | Purpose |
|---|---|
| `manifests/underwater_dataset_v1_manifest.csv` | Main TU optical dataset manifest. |
| `manifests/recordings_aux_optical_v1_manifest.csv` | Auxiliary `recordings` optical manifest. |
| `manifests/aux_combined_optical_v1_manifest.csv` | Full combined auxiliary manifest. |
| `manifests/aux_combined_optical_v1_lite_manifest.csv` | Lighter combined auxiliary manifest. |
| `manifests/trash_train_supplement_v1_manifest.csv` | Small hard-negative supplement used during main training. |
| `manifests/hard_negative_keywords.txt` | Earlier hard-negative keyword list. |
| `manifests/hard_negative_keywords_conservative.txt` | Conservative hard-negative keyword list used in the main experiments. |

## Metrics

The `metrics/` folder contains JSON outputs from real experiments. Temporary smoke-test metrics are ignored by `.gitignore` and should not be committed.

Important files:

| Path | Meaning |
|---|---|
| `metrics/underwater_v1_metrics_seed42_underwater_hn12_cons.json` | Best single-split seed-42 conservative hard-negative result. |
| `metrics/underwater_groupcv_3fold_underwater_hn12_cons.json` | Strict 3-fold baseline without auxiliary data. |
| `metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1.json` | Strict 3-fold result with `recordings` auxiliary warmup. |
| `metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxmix1.json` | Strict 3-fold result with full combined auxiliary warmup. |
| `metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxmix1_lite.json` | Strict 3-fold result with lighter combined auxiliary warmup. |
| `metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json` | Best strict strategy so far: `recordings` warmup plus small trash train supplement. |

## Model Weights

Only representative final model weights are tracked. Intermediate, exploratory, and smoke-test weights remain ignored.

| Path | Purpose |
|---|---|
| `models/README.md` | Explains why only selected weights are committed. |
| `models/underwater_v1_best_model_seed42_underwater_hn12_cons.pth` | Best balanced single-split model. |
| `models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold1_best.pth` | Fold 1 model from the best strict group-CV strategy. |
| `models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold2_best.pth` | Fold 2 model from the best strict group-CV strategy. |
| `models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold3_best.pth` | Fold 3 model from the best strict group-CV strategy. |

## Data Folders

### Tracked

| Path | Contents |
|---|---|
| `aux_combined_optical_v1/` | Compact full combined auxiliary dataset, `UXO=285`, `non_UXO=278`. |
| `aux_combined_optical_v1_lite/` | Compact lighter combined auxiliary dataset, `UXO=285`, `non_UXO=115`. |
| `models/` | Only the selected representative model weights listed above. |

### Local-only / ignored

| Path | Reason |
|---|---|
| `tuc_images/` | Original source image folders; too large and source-specific. |
| `underwater_dataset_v1/` | Generated main dataset split. |
| `aux_recordings_optical_v1/` | Generated auxiliary crops from local `recordings` source. |
| unselected `models/*.pth` | Intermediate and smoke-test trained weights. |
| `outputs/` | Grad-CAM and other generated visual outputs. |
| `archive/` | Old experiments and backups. |
| `__pycache__/` | Python cache files. |

## Organization Rules

- Keep runnable entry-point scripts in the root.
- Put experiment records in `metrics/`.
- Put data split and auxiliary-set tables in `manifests/`.
- Put project explanations, reports, and findings in `docs/`.
- Keep only representative final model weights in Git.
- Do not commit smoke-test files; use names matching `metrics/smoke_*.json` so they stay ignored.

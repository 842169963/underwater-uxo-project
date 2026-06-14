<div align="center">

# Underwater UXO Optical Classification

**A reproducible optical-only research baseline for underwater `UXO` vs `non_UXO` image classification.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Binary%20Classification-0F766E?style=for-the-badge)](#project-status)
[![Validation](https://img.shields.io/badge/Validation-Group%20CV-334155?style=for-the-badge)](#project-status)
[![Models](https://img.shields.io/badge/Models-Selected%20Weights-7C3AED?style=for-the-badge)](#model-weights)

<a href="#project-status">Status</a> |
<a href="#results-at-a-glance">Results</a> |
<a href="#reproducibility">Reproducibility</a> |
<a href="#model-weights">Models</a> |
<a href="#repository-layout">Layout</a> |
<a href="#workflow">Workflow</a> |
<a href="#current-best-references">References</a>

</div>

---

## Overview

This project builds a controlled experimental pipeline around a small underwater optical dataset. It covers data preparation, ResNet18 baseline training, hard-negative tuning, Grad-CAM analysis, group-aware validation, and auxiliary optical data experiments.

> This is a research baseline, not a deployment-ready detection system.

## Project Status

<table>
  <tr>
    <td><strong>Current stage</strong></td>
    <td>Finalized optical-only baseline with strict validation evidence</td>
  </tr>
  <tr>
    <td><strong>Main model</strong></td>
    <td>ImageNet-pretrained ResNet18, 2-class head</td>
  </tr>
  <tr>
    <td><strong>Primary task</strong></td>
    <td>Classify cropped underwater optical images as <code>UXO</code> or <code>non_UXO</code></td>
  </tr>
  <tr>
    <td><strong>Most important metric</strong></td>
    <td><code>UXO recall</code>, with <code>macro F1</code> used for balance</td>
  </tr>
  <tr>
    <td><strong>Main bottleneck</strong></td>
    <td>Data composition and shape-similar hard negatives</td>
  </tr>
</table>

## Results At A Glance

### Best Single-Split Result

| Setting | Test Accuracy | UXO Recall | Macro F1 |
|---|---:|---:|---:|
| Underwater augmentation + conservative hard negatives, seed 42 | `0.8214` | `0.8571` | `0.7888` |

Single-split results are useful for model development, but they can be optimistic because visually similar object instances may be split across train and test.

### Best Strict Group-CV Result

| Strategy | Mean Test Accuracy | Mean UXO Recall | Mean non_UXO Recall | Mean Macro F1 |
|---|---:|---:|---:|---:|
| `recordings` auxiliary warmup + small trash train-stage supplement | `0.6620` | `0.5751` | `0.6858` | `0.5870` |

The stricter result is the more reliable evidence. It shows that the project direction is valid, but generalization remains data-limited.

## Reproducibility

The repository is organized so the experiment record can be inspected and the selected final models can be reused directly. Exact end-to-end retraining from raw data still requires local source datasets that are not committed to Git.

| Goal | Supported from GitHub clone alone? | Notes |
|---|---|---|
| Inspect code, manifests, metrics, and reports | Yes | All scripts, result JSON files, and documentation are tracked. |
| Use the selected final model weights | Yes | Four representative `.pth` files are tracked under [models/](models/). |
| Reuse the compact combined auxiliary datasets | Yes | `aux_combined_optical_v1/` and `aux_combined_optical_v1_lite/` are tracked. |
| Re-run exact training from raw TU images | Requires local data | `tuc_images/` and `underwater_dataset_v1/` are intentionally ignored. |
| Rebuild the `recordings` auxiliary warmup set | Requires local data | `aux_recordings_optical_v1/` is generated from local `recordings` sources. |
| Reproduce the reported metrics exactly | Requires the same local data and split state | Metrics are tracked; exact reruns need the ignored source/generated datasets. |

In short: the GitHub repository now preserves the full project logic, result evidence, compact auxiliary data, and final representative model weights. A full raw-data retraining run still needs the original local datasets.

## Main Finding

The strongest lesson is that **data composition matters more than model complexity** at this stage.

Hard negatives such as branches, wreck-like shapes, cylinders, plastic objects, and biological clutter are the main source of instability. Auxiliary data helps only when it is introduced carefully:

| Auxiliary strategy | Observed behavior |
|---|---|
| Light `recordings` warmup | Improves UXO recall under strict validation |
| Full trash mix inside warmup | Pushes the model too strongly toward `non_UXO` |
| Smaller trash mix inside warmup | Still does not recover UXO recall |
| Small trash train-stage supplement | Best strict-evaluation balance so far |

## Repository Layout

```text
.
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- requirements.txt
|
|-- prepare_underwater_dataset_v1.py
|-- prepare_recordings_aux_optical_v1.py
|-- prepare_combined_aux_optical_v1.py
|
|-- train_underwater_classifier_v1.py
|-- train_underwater_groupcv.py
|-- gradcam_underwater.py
|-- predict_underwater.py
|-- predict_underwater_ui.py
|
|-- docs/
|-- manifests/
|-- metrics/
|-- models/
|-- aux_combined_optical_v1/
`-- aux_combined_optical_v1_lite/
```

The root directory is the command-entry layer. Scripts that are meant to be run directly stay in the root. Supporting records are grouped under `docs/`, `manifests/`, and `metrics/`.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the detailed file map.

## Component Map

| Area | Files | Purpose |
|---|---|---|
| Data preparation | [prepare_underwater_dataset_v1.py](prepare_underwater_dataset_v1.py), [prepare_recordings_aux_optical_v1.py](prepare_recordings_aux_optical_v1.py), [prepare_combined_aux_optical_v1.py](prepare_combined_aux_optical_v1.py) | Build the main dataset and auxiliary datasets |
| Training | [train_underwater_classifier_v1.py](train_underwater_classifier_v1.py), [train_underwater_groupcv.py](train_underwater_groupcv.py) | Run single-split and group-aware experiments |
| Explainability | [gradcam_underwater.py](gradcam_underwater.py) | Generate Grad-CAM visualizations |
| Inference | [predict_underwater.py](predict_underwater.py), [predict_underwater_ui.py](predict_underwater_ui.py) | Run CLI or local UI prediction |
| Documentation | [docs/](docs/) | Store results, progress notes, report outline, and findings |
| Experiment records | [metrics/](metrics/), [manifests/](manifests/) | Store JSON metrics and dataset manifests |
| Selected models | [models/](models/) | Store only the final representative `.pth` weights |

## Data Policy

### Included in Git

| Path | Contents |
|---|---|
| [aux_combined_optical_v1](aux_combined_optical_v1) | Full compact combined auxiliary set, `UXO=285`, `non_UXO=278` |
| [aux_combined_optical_v1_lite](aux_combined_optical_v1_lite) | Lighter compact combined auxiliary set, `UXO=285`, `non_UXO=115` |
| [models/](models/) | Selected representative model weights only |

### Local-only / Ignored

| Path | Reason |
|---|---|
| `tuc_images/` | Original source images |
| `underwater_dataset_v1/` | Generated main training split |
| `aux_recordings_optical_v1/` | Generated auxiliary crops from local recordings |
| unselected `models/*.pth` | Intermediate and smoke-test model weights |
| `outputs/` | Grad-CAM and other generated outputs |

## Model Weights

Only the model weights that support the final conclusions are committed. This keeps the repository usable while avoiding the full local model directory, which is about `1.41 GB`.

| Model | File | Role |
|---|---|---|
| Best single-split model | [models/underwater_v1_best_model_seed42_underwater_hn12_cons.pth](models/underwater_v1_best_model_seed42_underwater_hn12_cons.pth) | Supports the strongest fixed-split result: `0.8214` test accuracy, `0.8571` UXO recall, `0.7888` macro F1 |
| Best strict Group-CV fold 1 | [models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold1_best.pth](models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold1_best.pth) | Fold model from the best strict validation strategy |
| Best strict Group-CV fold 2 | [models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold2_best.pth](models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold2_best.pth) | Fold model from the best strict validation strategy |
| Best strict Group-CV fold 3 | [models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold3_best.pth](models/groupcv_underwater_hn12_cons_auxrec1_trsupp1/fold3_best.pth) | Fold model from the best strict validation strategy |

See [models/README.md](models/README.md) for the model selection rationale.

## Environment

| Requirement | Recommended version |
|---|---|
| Python | `3.10+` |
| PyTorch | `2.5+` |
| torchvision | `0.20+` |

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, if `python` points to an older interpreter:

```bash
py -3 -m pip install -r requirements.txt
```

## Workflow

| Step | Command / file | Output |
|---:|---|---|
| 1 | `prepare_underwater_dataset_v1.py` | Main TU optical dataset manifest |
| 2 | `prepare_recordings_aux_optical_v1.py` | Auxiliary `recordings` crops |
| 3 | `train_underwater_classifier_v1.py` | Single-split baseline metrics and model |
| 4 | `train_underwater_groupcv.py` | Strict group-aware metrics |
| 5 | `gradcam_underwater.py` | Visual explanation outputs |

<details>
<summary><strong>Common commands</strong></summary>

Build the main optical dataset:

```bash
python prepare_underwater_dataset_v1.py
```

Build the recordings auxiliary set:

```bash
python prepare_recordings_aux_optical_v1.py --clear-output
```

Train the single-split baseline:

```bash
python train_underwater_classifier_v1.py --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

Run strict 3-fold group-aware evaluation:

```bash
python train_underwater_groupcv.py --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

Reproduce the best strict strategy tested so far:

```bash
python train_underwater_groupcv.py --aux-data-root aux_recordings_optical_v1 --aux-epochs 1 --aux-lr 2e-4 --extra-train-manifest manifests/trash_train_supplement_v1_manifest.csv --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater --metrics-out metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json
```

</details>

## Interpretation Notes

| Point | Practical meaning |
|---|---|
| Single split can be optimistic | Use it for development, not final claims |
| Group-aware CV is stricter | It better tests generalization to unseen object instances |
| UXO recall matters | Missing UXO is more costly than producing some false positives |
| Hard negatives dominate errors | More careful negative composition is the next best improvement path |
| Auxiliary data must be controlled | Strong non_UXO auxiliary mixing can hurt UXO sensitivity |

## Current Best References

| File | Use |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Main result summary |
| [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) | Full progress log |
| [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md) | Presentation-ready report outline |
| [docs/GRADCAM_FINDINGS.md](docs/GRADCAM_FINDINGS.md) | Visual explanation findings |
| [models/README.md](models/README.md) | Selected model weight rationale |
| [metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json](metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json) | Best strict-evaluation metric file |

---

<div align="center">

**Final takeaway:** the pipeline is reproducible and the direction is promising, but robust underwater UXO recognition still depends on better independent UXO examples and more carefully selected hard negatives.

</div>

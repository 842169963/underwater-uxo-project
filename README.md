# Underwater UXO Optical Classification

This repository contains a reproducible optical-only baseline for underwater `UXO` vs `non_UXO` image classification.

The project starts with a small TU optical dataset and builds a controlled experimental pipeline around data preparation, ResNet18 training, hard-negative tuning, Grad-CAM analysis, group-aware validation, and auxiliary optical data experiments.

## Project Status

This is a research baseline, not a deployment-ready detection system.

The strongest single-split model reached:
- test accuracy: `0.8214`
- UXO recall: `0.8571`
- macro F1: `0.7888`

The stricter 3-fold group-aware evaluation shows that generalization is still limited. The best strict-evaluation strategy tested so far is:

`recordings auxiliary warmup + small trash train-stage hard-negative supplement`

It reached:
- mean test accuracy: `0.6620`
- mean UXO recall: `0.5751`
- mean non_UXO recall: `0.6858`
- mean macro F1: `0.5870`

The main conclusion is that data composition matters more than model complexity at this stage. Hard negatives such as branches, wreck-like shapes, cylinders, plastic objects, and biological clutter are the main source of instability.

## Repository Layout

```text
.
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- requirements.txt
|-- prepare_underwater_dataset_v1.py
|-- prepare_recordings_aux_optical_v1.py
|-- prepare_combined_aux_optical_v1.py
|-- train_underwater_classifier_v1.py
|-- train_underwater_groupcv.py
|-- gradcam_underwater.py
|-- predict_underwater.py
|-- predict_underwater_ui.py
|-- docs/
|-- manifests/
|-- metrics/
|-- aux_combined_optical_v1/
`-- aux_combined_optical_v1_lite/
```

The root directory is intentionally kept as the command-entry layer. Scripts that a user runs directly stay in the root. Supporting evidence and generated records are grouped under `docs/`, `manifests/`, and `metrics/`.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a more detailed file map.

## Main Components

### Data Preparation

- [prepare_underwater_dataset_v1.py](prepare_underwater_dataset_v1.py) builds the main optical TU dataset.
- [prepare_recordings_aux_optical_v1.py](prepare_recordings_aux_optical_v1.py) extracts auxiliary optical crops from the `recordings` dataset.
- [prepare_combined_aux_optical_v1.py](prepare_combined_aux_optical_v1.py) builds combined auxiliary datasets from `recordings` plus selected `trash_ICRA19` hard negatives.

### Training and Evaluation

- [train_underwater_classifier_v1.py](train_underwater_classifier_v1.py) trains the single-split ResNet18 baseline.
- [train_underwater_groupcv.py](train_underwater_groupcv.py) runs 3-fold group-aware cross-validation.
- [gradcam_underwater.py](gradcam_underwater.py) generates Grad-CAM visualizations.
- [predict_underwater.py](predict_underwater.py) runs command-line inference.
- [predict_underwater_ui.py](predict_underwater_ui.py) launches a small local prediction UI.

### Results and Documentation

- [docs/RESULTS.md](docs/RESULTS.md) summarizes the main baseline and optimization results.
- [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) records the full project timeline.
- [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md) provides a presentation-ready report structure.
- [docs/GRADCAM_FINDINGS.md](docs/GRADCAM_FINDINGS.md) summarizes visual explanation findings.
- [docs/OPTIMIZATION_ROADMAP.md](docs/OPTIMIZATION_ROADMAP.md) lists possible next experiments.
- [metrics/](metrics/) stores JSON result files from the main experiments.
- [manifests/](manifests/) stores dataset and experiment manifests.

## Data Included in Git

This repository includes two compact auxiliary datasets because they are part of the final experiment record:

- [aux_combined_optical_v1](aux_combined_optical_v1): full combined auxiliary set, `UXO=285`, `non_UXO=278`
- [aux_combined_optical_v1_lite](aux_combined_optical_v1_lite): lighter combined auxiliary set, `UXO=285`, `non_UXO=115`

Large local sources and generated artifacts are intentionally not tracked:

- `tuc_images/`
- `underwater_dataset_v1/`
- `aux_recordings_optical_v1/`
- `models/`
- `outputs/`
- `*.pth`

The repository therefore contains enough code, manifests, metrics, and compact auxiliary data to understand and reproduce the reported experiment flow, while excluding bulky original sources and trained model weights.

## Environment

Recommended:
- Python `3.10+`
- PyTorch and torchvision

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, if `python` points to an older interpreter, use:

```bash
py -3 -m pip install -r requirements.txt
```

## Typical Workflow

### 1. Build the main optical dataset

Requires the local TU image folders:

```bash
python prepare_underwater_dataset_v1.py
```

### 2. Build the recordings auxiliary set

Requires the local `recordings/` source outside this repository:

```bash
python prepare_recordings_aux_optical_v1.py --clear-output
```

The script applies the required `bbox * 3` correction for the optical labels and performs sparse run-level sampling.

### 3. Train the single-split baseline

```bash
python train_underwater_classifier_v1.py --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

### 4. Train with light auxiliary warmup

```bash
python train_underwater_classifier_v1.py --aux-data-root aux_recordings_optical_v1 --aux-epochs 1 --aux-lr 2e-4 --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

### 5. Run strict 3-fold group-aware evaluation

```bash
python train_underwater_groupcv.py --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

### 6. Reproduce the best strict strategy tested so far

```bash
python train_underwater_groupcv.py --aux-data-root aux_recordings_optical_v1 --aux-epochs 1 --aux-lr 2e-4 --extra-train-manifest manifests/trash_train_supplement_v1_manifest.csv --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater --metrics-out metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json
```

## Important Interpretation Notes

- Single-split results are useful for model development but can be optimistic.
- Group-aware cross-validation is the more reliable evidence because it reduces object-instance leakage.
- UXO recall is a priority metric, but false positives on hard negatives remain a major limitation.
- Auxiliary data helps only when introduced carefully. Adding too many non_UXO hard negatives during auxiliary warmup can push the model too strongly toward `non_UXO`.
- The most promising tested strategy is to use `recordings` as light auxiliary warmup and add a small `trash` subset only during the main training stage.

## Current Best References

- Main result summary: [docs/RESULTS.md](docs/RESULTS.md)
- Full progress log: [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md)
- Presentation outline: [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md)
- Best strict-evaluation metric file: [metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json](metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json)

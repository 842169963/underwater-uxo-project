# Underwater UXO Project

This repository contains an optical-only baseline for underwater `UXO / non_UXO` image classification.

The project focuses on:
- reproducible dataset preparation
- ResNet18 baseline training
- hard-negative tuning
- Grad-CAM analysis
- group-aware validation
- auxiliary optical training data from the `recordings` dataset

## What is included
- training scripts
- dataset preparation scripts
- prediction scripts
- progress and results documents
- manifests and lightweight metrics

## What is not included
Large local assets are intentionally excluded from Git:
- original image datasets
- generated dataset folders
- trained model weights
- Grad-CAM output folders

## Main scripts
- `prepare_underwater_dataset_v1.py`: build the main TU-based optical dataset
- `prepare_recordings_aux_optical_v1.py`: build a small auxiliary optical dataset from `recordings/`
- `train_underwater_classifier_v1.py`: main training script
- `train_underwater_groupcv.py`: 3-fold group-aware cross-validation
- `gradcam_underwater.py`: Grad-CAM visualization
- `predict_underwater.py`: CLI inference
- `predict_underwater_ui.py`: local UI inference

## Project documents
- `docs/RESULTS.md`: main experiment results
- `docs/PROJECT_PROGRESS.md`: timeline of project progress
- `docs/OPTIMIZATION_ROADMAP.md`: next-step optimization plan
- `docs/GRADCAM_FINDINGS.md`: Grad-CAM observations

## Environment
Recommended:
- Python 3.10
- PyTorch + torchvision

Install dependencies:

```bash
pip install -r requirements.txt
```

## Typical workflow

### 1. Prepare the main optical dataset
If the TU image folders are available locally:

```bash
python prepare_underwater_dataset_v1.py
```

### 2. Optional: prepare auxiliary optical crops from recordings
If the `recordings/` dataset is available locally:

```bash
python prepare_recordings_aux_optical_v1.py --clear-output
```

Note:
- the script already applies the required `bbox * 3` correction for the optical labels from `recordings`
- it performs sparse run-level sampling instead of exporting every video frame

### 3. Train the main model

```bash
python train_underwater_classifier_v1.py --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

### 4. Optional: light auxiliary warmup before main training

```bash
python train_underwater_classifier_v1.py --aux-data-root aux_recordings_optical_v1 --aux-epochs 1 --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

### 5. Run stricter group-aware evaluation

```bash
python train_underwater_groupcv.py --epochs 10 --batch-size 16 --balanced-sampling --augmentation-profile underwater
```

## Current status
The current best balanced single model in local experiments was obtained with:
- underwater augmentation
- conservative hard negatives
- seed 42

On the current single split, it reached approximately:
- test accuracy: `0.82`
- UXO recall: `0.86`
- macro F1: `0.79`

However, stricter 3-fold group-aware validation shows that generalization is still limited, so the project is still data-constrained and should be interpreted as a research baseline rather than a final system.

## Notes
- Many scripts currently contain Windows-style default paths used during local development.
- For reuse on another machine, prefer passing explicit CLI arguments instead of relying on defaults.
- The auxiliary `recordings` data is intended as training-only support, not as the main benchmark.

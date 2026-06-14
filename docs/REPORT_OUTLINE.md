# Underwater UXO Project Report Outline

## 1. Title
- Project title:
  - `Optical-only Underwater UXO vs non_UXO Classification Baseline`
- Short positioning:
  - Build a reproducible underwater optical classification pipeline and gradually improve it through augmentation, hard negatives, and stricter validation.

## 2. Presentation Goal
- Explain what problem the project solves.
- Show what has already been completed.
- Distinguish between optimistic single-split performance and more realistic group-aware performance.
- Highlight what has been learned from auxiliary data and hard-negative experiments.

## 3. Recommended Presentation Structure

### Slide 1. Background and Motivation
- Problem:
  - underwater UXO detection is challenging because images are scarce, degraded, and often visually similar to non-UXO clutter.
- Project goal:
  - build an optical-only `UXO / non_UXO` baseline first, before exploring more complex systems.
- Core challenge:
  - small dataset
  - class imbalance
  - hard negatives such as `branch`, `wreck`, `cylinder`, and elongated clutter

Suggested speaking point:
- "The project starts from a realistic baseline question: can we classify underwater optical images into UXO and non_UXO under small-data conditions, and what actually improves the model?"

### Slide 2. Data Preparation
- Main source:
  - TU optical dataset
- Cleaning policy:
  - excluded non-optical images
  - excluded files containing `BV`
- Preprocessing:
  - isotropic resize + centered padding to `224x224`
- Final dataset size:
  - total `181`
  - train `124`
  - val `29`
  - test `28`

Suggested speaking point:
- "The first step was to clean the data and define a reproducible optical-only training set."

### Slide 3. Baseline Pipeline
- Model:
  - `ResNet18` with ImageNet pretraining
- Training:
  - weighted cross-entropy
  - balanced sampling
- Evaluation:
  - accuracy
  - `UXO recall`
  - `macro F1`
  - confusion matrix
- Explainability:
  - Grad-CAM

Suggested speaking point:
- "The baseline was not only trained, but also made reproducible and interpretable."

### Slide 4. Baseline and First Optimization Results
- Multi-seed baseline mean:
  - `test_acc = 0.6786 +- 0.0583`
  - `UXO recall = 0.8571 +- 0.1166`
  - `macro F1 = 0.6547 +- 0.0416`
- First useful optimizations:
  - underwater-specific augmentation
  - conservative hard negatives (`branch`, `wreck`, `shipwreck`)
- Best single-split model:
  - `test_acc = 0.8214`
  - `UXO recall = 0.8571`
  - `macro F1 = 0.7888`

Suggested speaking point:
- "On the fixed split, the optimized model already achieved a fairly strong balance between UXO recall and non_UXO discrimination."

### Slide 5. Why Strict Validation Was Needed
- Problem:
  - single split may be optimistic
  - same object instance may appear in multiple similar views
- Solution:
  - `3-fold Group CV`
  - group-aware split by object instance
- Result without auxiliary data:
  - `test_acc mean = 0.6409`
  - `UXO recall mean = 0.3388`
  - `macro F1 mean = 0.5259`

Suggested speaking point:
- "This showed that the best single-split result is promising, but optimistic. Under stricter validation, generalization is still limited."

### Slide 6. Error Analysis
- Grad-CAM showed that the model learned some UXO-like geometry.
- Main failure mode:
  - `non_UXO -> UXO` false positives
- Typical confusing negatives:
  - `branch`
  - `wreck / shipwreck`
  - `cylinder-like` shapes

Suggested speaking point:
- "The model is not failing randomly. The main issue is confusion with shape-similar hard negatives."

### Slide 7. Auxiliary Data from Recordings
- New `recordings` dataset analyzed
- Only optical `gopro` frames used for now
- Important finding:
  - label coordinates had to be multiplied by `3`
- Auxiliary optical dataset built from:
  - multi-view UXO crops
  - `cylinder` hard negatives
- Auxiliary dataset size:
  - `UXO = 285`
  - `non_UXO = 40`

Suggested speaking point:
- "A second optical dataset was introduced to provide more UXO shape variation and cylinder-like hard negatives."

### Slide 8. Auxiliary-Data Strategy Comparison
- `recordings` as aux warmup helped under strict evaluation:
  - `test_acc mean = 0.6407`
  - `UXO recall mean = 0.5092`
  - `macro F1 mean = 0.5590`
- `trash` used inside aux warmup was too strong:
  - it pushed the model too much toward `non_UXO`
- Better strategy:
  - keep `recordings` as light aux warmup
  - use a small `trash` subset only as train-stage hard-negative supplement

Suggested speaking point:
- "The main lesson is that the usefulness of auxiliary data depends strongly on how it is introduced."

### Slide 9. Current Best Strict-Evaluation Result
- Best strict-evaluation strategy so far:
  - `recordings aux warmup + small trash train supplement`
- Result:
  - `test_acc mean = 0.6620`
  - `UXO recall mean = 0.5751`
  - `non_UXO recall mean = 0.6858`
  - `macro F1 mean = 0.5870`

Suggested speaking point:
- "This is currently the most promising direction under group-aware validation, because it improves both UXO recall and macro F1 compared with the earlier strict baseline."

### Slide 10. Current Conclusion
- What is already solid:
  - reproducible optical-only pipeline
  - baseline training and evaluation
  - multi-seed comparison
  - Grad-CAM analysis
  - group-aware validation
  - auxiliary-data integration experiments
- What remains limited:
  - group-CV UXO generalization
  - stability across difficult negative types
- Main bottleneck:
  - data quality and hard-negative composition, more than model complexity

Suggested speaking point:
- "The project direction is correct, but the main bottleneck is still data, especially better hard negatives and more independent UXO instances."

### Slide 11. Next Steps
- refine hard-negative composition
- use `trash` more selectively
- keep strict group-aware evaluation
- continue with controlled comparisons rather than larger model changes

Suggested speaking point:
- "The next step is not to jump to a more complex architecture, but to improve data composition under the same disciplined evaluation protocol."

## 4. Key Numbers to Remember
- Main dataset size:
  - `181`
- Best single-split model:
  - `0.8214` test accuracy
  - `0.8571` UXO recall
  - `0.7888` macro F1
- Strict `3-fold CV` baseline:
  - `0.6409` accuracy mean
  - `0.3388` UXO recall mean
  - `0.5259` macro F1 mean
- Best strict strategy so far:
  - `0.6620` accuracy mean
  - `0.5751` UXO recall mean
  - `0.5870` macro F1 mean

## 5. One-Minute Oral Summary
- "I built a reproducible optical-only baseline for underwater UXO vs non_UXO classification."
- "The pipeline includes data cleaning, group-aware splitting, ResNet18 baseline training, Grad-CAM analysis, and hard-negative optimization."
- "On a fixed split, the optimized model achieved strong results, but stricter 3-fold group-aware validation showed that generalization is still limited."
- "I then introduced auxiliary optical data from recordings and tested different ways of using external hard negatives."
- "The most promising result so far is to use recordings as light auxiliary warmup and use a small trash subset only as train-stage hard-negative supplement."
- "So the project is no longer just a baseline; it has already moved into structured optimization and strict validation."

## 6. Suggested Backup Files
- [`PROJECT_PROGRESS.md`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs/PROJECT_PROGRESS.md)
- [`RESULTS.md`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs/RESULTS.md)
- [`underwater_v1_metrics_seed42_underwater_hn12_cons.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_v1_metrics_seed42_underwater_hn12_cons.json)
- [`underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics/underwater_groupcv_3fold_underwater_hn12_cons_auxrec1_trsupp1.json)

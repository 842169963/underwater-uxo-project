# Project Structure

## Main working area
- [`tuc_images`](C:/Users/stephenxxy/Desktop/project/uxo_project/tuc_images): source optical and non-optical image folders.
- [`underwater_dataset_v1`](C:/Users/stephenxxy/Desktop/project/uxo_project/underwater_dataset_v1): current v1 dataset split used for training.
- [`prepare_underwater_dataset_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/prepare_underwater_dataset_v1.py): build v1 manifest and dataset.
- [`prepare_recordings_aux_optical_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/prepare_recordings_aux_optical_v1.py): extract a small optical auxiliary dataset from `recordings/` with `bbox x3` correction and sparse run sampling.
- [`train_underwater_classifier_v1.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/train_underwater_classifier_v1.py): train the current baseline.
- [`gradcam_underwater.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/gradcam_underwater.py): generate Grad-CAM visualizations.
- [`predict_underwater.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/predict_underwater.py): CLI inference.
- [`predict_underwater_ui.py`](C:/Users/stephenxxy/Desktop/project/uxo_project/predict_underwater_ui.py): local UI inference.

## Organized outputs
- [`models`](C:/Users/stephenxxy/Desktop/project/uxo_project/models): trained `.pth` files.
- [`metrics`](C:/Users/stephenxxy/Desktop/project/uxo_project/metrics): training and evaluation `.json` metrics.
- [`manifests`](C:/Users/stephenxxy/Desktop/project/uxo_project/manifests): dataset manifests and split tables.
- [`outputs/gradcam`](C:/Users/stephenxxy/Desktop/project/uxo_project/outputs/gradcam): Grad-CAM images and summary CSVs.
- [`docs`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs): reports, findings, and reading notes.
- [`docs/OPTIMIZATION_ROADMAP.md`](C:/Users/stephenxxy/Desktop/project/uxo_project/docs/OPTIMIZATION_ROADMAP.md): formal next-step optimization plan derived from paper reading and current experiments.

## Archived experiments
- [`archive/stage2`](C:/Users/stephenxxy/Desktop/project/uxo_project/archive/stage2): older stage2 scripts, datasets, and model artifacts kept for reference.

## Quick rule
- If it is part of the current optical-only v1 pipeline, keep it in the project root.
- If it is a generated result, put it under `models`, `metrics`, `manifests`, or `outputs`.
- If it is analysis text or paper notes, put it under `docs`.
- If it belongs to old experiments, move it to `archive`.

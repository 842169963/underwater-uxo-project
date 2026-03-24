# Grad-CAM Findings (seed42)

## 1) Error overview
- Val: 29 samples, 6 errors, accuracy 0.7931
- Test: 28 samples, 7 errors, accuracy 0.7500

## 2) Main error patterns
- Val top mistakes: UXO->non_UXO: 4, non_UXO->UXO: 2
- Test top mistakes: non_UXO->UXO: 5, UXO->non_UXO: 2

## 3) Suggested visual checks in exported Grad-CAM images
- Check whether heatmap is on elongated body contour (good) or on local bright haze/background texture (risk).
- Pay special attention to non_UXO -> UXO false positives in branch/plank/wreck-like shapes.
- Compare UXO true positives vs UXO false negatives to see if contrast/occlusion drives misses.

## 4) Example misclassified files (first 8)
- [val] true=non_UXO pred=UXO prob=0.839 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\val\non_UXO\cylinder2__cylinder2dSD.PNG
- [val] true=UXO pred=non_UXO prob=0.820 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\val\UXO\torpedo2__torpedo2aSD.PNG
- [val] true=non_UXO pred=UXO prob=0.673 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\val\non_UXO\cylinder2__cylinder2aSD.PNG
- [val] true=UXO pred=non_UXO prob=0.638 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\val\UXO\torpedo2__coralTorpedo2a4K.PNG
- [val] true=UXO pred=non_UXO prob=0.630 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\val\UXO\torpedo2__coralTorpedo2aSD.PNG
- [val] true=UXO pred=non_UXO prob=0.916 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\val\UXO\torpedo2__coralTorpedo2SD.PNG
- [test] true=non_UXO pred=UXO prob=0.898 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\test\non_UXO\branch1__branch1cSDWooden%20branch1_Bornholm%20Deep_depth93.9m.png
- [test] true=non_UXO pred=UXO prob=0.898 | C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1\test\non_UXO\branch1__branch1bSD.png
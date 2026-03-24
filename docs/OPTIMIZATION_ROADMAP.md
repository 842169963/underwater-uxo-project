# Optimization Roadmap

## Goal
- Improve the current optical-only `UXO / non_UXO` baseline without prematurely replacing the whole pipeline.

## Main conclusion from the two papers
- The current `ResNet18 + transfer learning` pipeline is a valid baseline.
- The biggest near-term bottlenecks are data quality, hard negatives, and underwater-specific augmentation.
- More complex feature extraction or hybrid classifiers should come later, after the baseline is strengthened.

## Priority order
1. Stronger underwater-specific augmentation
2. Hard negative reinforcement
3. Data cleaning and relabeling review
4. Stable evaluation with confusion matrix and class-wise metrics
5. Lightweight backbone comparison
6. `ResNet18` vs `ResNet18 + SVM`
7. More advanced feature extraction such as dilated or multi-scale variants

## Stage 1: Strengthen the baseline
- Keep `ResNet18` as the main reference model.
- Use stronger underwater augmentation:
- rotation
- translation and scale changes
- blur
- contrast and brightness shifts
- haze or turbidity-like degradation
- Reinforce hard negatives, especially:
- branch-like objects
- wreck-like objects
- cylinders
- box-like clutter
- Review labels and low-value samples.
- Keep reporting confusion matrix, precision, recall, and F1.

## Stage 2: Make the experiments more research-like
- Compare a few lightweight backbones:
- `ResNet18`
- `MobileNetV2`
- `ShuffleNet`
- `EfficientNet-B0`
- Compare:
- end-to-end `ResNet18`
- `ResNet18` feature extractor + `SVM`
- Report mean and variance across seeds.

## Stage 3: Improve features if needed
- Try dilated convolution.
- Try multi-scale feature extraction.
- Consider small architectural improvements only if Stage 1 and Stage 2 stop improving results.

## Not a priority right now
- GOA or other metaheuristic optimization
- complex feature selection pipelines
- YOLO detection pipeline
- multi-sensor fusion

## Practical takeaway
- Do not rebuild the project from scratch.
- First improve data-side robustness and evaluation discipline.
- Only then test more complex model variants.

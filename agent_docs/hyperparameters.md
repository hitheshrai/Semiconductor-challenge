---
name: Defect Classifier Hyperparameters
description: Detailed training hyperparameters, phase config, and tuning notes
type: project
---

# Hyperparameters

## Training Phases

| Phase | Epochs | Backbone | LR |
|-------|--------|----------|----|
| Phase 1 | 20 | Frozen | head: 3e-4 |
| Phase 2 | 20 | Last 3 blocks unfrozen | backbone: 3e-5 |

## Full Parameter Table

| Parameter    | Value   | Notes                          |
|--------------|---------|--------------------------------|
| IMG_SIZE     | 224     | EfficientNet-B0 input          |
| BATCH_SIZE   | 32      |                                |
| PHASE1_EP    | 20      | frozen backbone                |
| PHASE2_EP    | 20      | fine-tune last 3 blocks        |
| LR_HEAD      | 3e-4    |                                |
| LR_BACK      | 3e-5    | 10× lower for backbone         |
| EMBED_DIM    | 256     | L2-normalised embedding size   |
| class weight | 5:1 cap | defects:good max ratio         |

## Class Weight Fix (Critical)

Raw inverse-frequency weights reached 476:1 (defect9=82 vs good=0.17). Combined with
`WeightedRandomSampler`, the model had zero incentive to predict "good" — confusion matrix
showed 0/714 good samples correct.

**Fix:** Cap class weight ratio at 5:1 in `class_weights_tensor()`:
```python
capped = [min(wi, min_w * 5.0) for wi in raw]
```
Effective weights after fix: good=1.0, all defects=5.0.

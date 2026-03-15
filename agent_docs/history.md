---
name: Defect Classifier Run History
description: Prior training run outcomes, failures, and next steps
type: project
---

# Training Run History

## Run 1 — Broken (Pre-fix)
- **Result:** val accuracy 2%, good class 0%
- **Cause:** 476:1 class weights (defect9=82 vs good=0.17) combined with
  `WeightedRandomSampler` — model never predicted "good"

## Run 2 — Fixed (CPU, Phase 1 only)
- **Result:** Phase 1 ep20 val_acc=0.22, bal_acc=0.65
- **Fix applied:** Class weights capped at 5:1
- **Interrupted:** Moved to DGX before Phase 2

## Run 3 — Phase 1 + Phase 2 (DGX)
- **Best bal_acc:** 0.7936 (Phase 2 ep01)
- **Problem:** good class recall collapsed to 4.2% → overall accuracy = 8.6%
- **Root cause:** Model over-predicts defects; balanced_accuracy checkpoint criterion
  rewards defect recall without penalizing false alarms on the good class

## Run 5 — Phase 1 + Phase 2 (DGX, current best)
- **Best bal_acc:** 0.7884 (Phase 2 ep12)
- **Overall accuracy:** 62.8%, good recall = 62.1%
- **This is the saved model_best.pth**
- **Validation loss diverging:** train ~0.55, val ~1.7–3.0 → overfitting
- **Defect precision terrible:** 0.02–0.23, recall 0.5–1.0 (over-predicting defects)

## Status
Phase 1 + Phase 2 training is complete. The gap from 62.8% → 85% is a modelling problem:
1. Checkpoint selection on balanced_accuracy is the wrong metric
2. The good class needs better precision/recall balance
3. Significant overfitting (val loss diverges from train loss)

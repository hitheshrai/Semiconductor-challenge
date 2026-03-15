---
name: Defect Classifier Improvement Strategies
description: Researched strategies to improve defect recall; what's been tried and what's next
type: project
---

# Improvement Strategies

## What's Been Done (no retraining)

### Tau-Normalization (τ = 0.3) — APPLIED
Corrects minority-class weight-norm bias in the trained classifier.
Applied in `evaluate.py::tau_normalize()` and `classify.py::__init__`.

### Logit Adjustment (τ_LA = 0.1) — APPLIED
Subtracts `τ_LA × log(class_prior)` from each logit at inference, boosting rare class scores.
Applied in `evaluate.py::predict_all()` and `classify.py::predict()`.

**Tau sweep results (tau_la × tau_norm):**
- Best: tau_la=0.1, tau_norm=0.3 → overall=85.7%, bal_acc=0.56, avg_defect_rec=52.5%
- Baseline (no adjustment): overall=91.1%, bal_acc=0.28, avg_defect_rec=~20%

---

## Next Steps (require retraining)

### Day 2: Decoupled Classifier Retraining (cRT)
- Freeze entire backbone, reinitialize only `model.classifier`
- Retrain head ~20 epochs with `WeightedRandomSampler` (equal class frequency)
- Paper: "Decoupling Representation and Classifier for Long-Tailed Recognition" ICLR 2020
- Expected: significant defect recall improvement, minimal overall accuracy drop

### Day 3: Batch-Balanced Focal Loss + ContextMix
- Replace CrossEntropy with `FocalLoss(gamma=2.0)`
- Enforce equal class frequency per batch (batch_size=72, 8 per class)
- Replace augmentation with ContextMix (github.com/Hy2MK/ContextMix)
- Paper: "Batch-balanced Focal Loss" PMC 2023; "ContextMix" arXiv 2401.10050
- Retrain ~50 epochs from existing checkpoint

### Day 3-4: Per-class Autoencoder Augmentation
- Train lightweight conv autoencoder per defect class
- Inject noise into latent codes to generate 200-500 synthetic samples per class
- Paper: "Wafer Map Defect Classification Using Autoencoder-Based Augmentation" arXiv 2411.11029
- Reported: rare class recall 39% → 100%

### Day 4-5: MAE Domain Pretraining (highest ceiling)
- Pretrain ViT-Small as Masked Autoencoder on all 3,778 wafer images (no labels)
- Use 8×8 patch size (not default 16×16) per microelectronics paper
- Paper: "Masked Autoencoder Self Pre-Training for Defect Detection" arXiv 2504.10021
- Reported: +10.2% over ImageNet ViT-B on microelectronics defect detection

---

## Key Insight
The fundamental problem is training/inference distribution mismatch combined with
classifier weight-norm bias. Post-hoc calibration (done) addresses the symptom.
The cRT approach (next) addresses the root cause in the classifier layer.
MAE pretraining addresses the root cause in the backbone features.

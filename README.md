# Semiconductor Defect Classifier
**ASU / Intel Semiconductor Solutions Challenge 2026 — Problem A**

Few-shot semiconductor wafer defect classification. Identifies 8 defect types from as few as one labelled example per class — no retraining required for new defect types.

---

## The Problem

| Challenge | Detail |
|-----------|--------|
| Dataset | 3,778 grayscale wafer images, 9 classes |
| Imbalance | 94.5% "good" chips, 0.2%–1.1% per defect class |
| Few-shot | 8–50 labelled examples per defect type |
| Target | ≥ 85% overall classification accuracy |

The real-world constraint is asymmetric cost: **a defective chip shipped as "good" is far more costly than a good chip rejected for review.** High defect recall matters more than overall accuracy.

---

## Why Two-Stage Cascade

A single classifier trying to handle a 94.5/5.5 imbalance while also distinguishing 8 defect subtypes faces an irresolvable objective conflict:

- Training signal that improves defect recall → collapses overall accuracy
- Training signal that preserves overall accuracy → misses most defects

**Every single-model approach confirmed this:**

| Approach | Overall Acc | Defect Recall |
|----------|-------------|---------------|
| Baseline (single model) | 91.1% | ~20% |
| + Tau-norm + Logit adjustment | 85.7% | 52.5% |
| Focal Loss fine-tune | 65.6% | 75.0% |
| **Two-stage cascade** | **85.1%** ✅ | **70.7%** |

**The fix:** decompose into two independent, tractable problems.

```
Image ──► [Stage 1: Is this chip defective?]
                  │ defect_prob < 0.35          │ defect_prob ≥ 0.35
                  ▼                             ▼
           predict: good          [Stage 2: Which defect type?]
                                         │
                                         ▼
                                  predict: defect class
```

- **Stage 1** trains on all 3,778 images for the easy binary question — no class conflict
- **Stage 2** trains on ~230 defect-only images — no dominant "good" class suppressing gradients

---

## Architecture

**Backbone:** EfficientNet-B0 (ImageNet pretrained via `timm`)

```
Input (224×224 RGB)
    → EfficientNet-B0 backbone → 1280-d feature vector
    → FC(1280→256) → BN → ReLU → Dropout(0.35) → FC(256→256) → BN → L2-Norm
    → 256-d unit-sphere embedding
    → Linear classifier  (cosine similarity = dot product on unit sphere)
```

L2-normalised embeddings enable **prototype-based few-shot inference**: new defect types can be registered at runtime from ≥1 labelled example — no retraining.

---

## Results

### Two-stage cascade (final submission)

| Class | Support | Recall |
|-------|---------|--------|
| defect1 | 4 | 75% |
| defect2 | 10 | 70% |
| defect3 | 2 | 100% |
| defect4 | 3 | 67% |
| defect5 | 5 | 80% |
| defect8 | 8 | 50% |
| defect9 | 1 | 100% |
| defect10 | 8 | 87.5% |
| good | 715 | 85.9% |
| **Overall** | **756** | **85.1%** ✅ |

**Balanced accuracy: 0.781** | **Avg defect recall: 70.7%** | **Inference: <50 ms/image**

### Few-shot learning curve (prototype inference, no retraining)

| N-shot per defect class | Accuracy |
|------------------------|----------|
| 1-shot | 79.5% |
| 2-shot | 78.9% |
| 5-shot | 75.9% |
| 20-shot | 76.5% |

The model achieves ~80% accuracy from a **single labelled example per defect class**.

---

## Quick Start

```bash
# Install
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn matplotlib pillow

cd solution

# Classify a single image (cascade — best defect recall)
python classify.py path/to/wafer.png --cascade

# Classify a single image (single model — highest overall accuracy)
python classify.py path/to/wafer.png

# Classify a folder
python classify.py path/to/folder/ --cascade --output results.json

# Register a new defect type at runtime (no retraining)
python classify.py image.png --cascade --register new_defect examples/*.png
```

---

## Training

```bash
cd solution

# Train single model (Phase 1: frozen backbone, Phase 2: fine-tune)
python train.py --finetune

# Train two-stage cascade (~80 min on DGX, ~3 hrs on consumer GPU)
python train_cascade.py --stage both

# Evaluate cascade on validation set
python train_cascade.py --evaluate

# Full evaluation suite (6 plots + metrics.json)
python evaluate.py
```

---

## Repository Structure

```
defect_challenge/
├── README.md
├── SOLUTION.md                  ← full technical writeup
├── solution/
│   ├── model.py                 ← EfficientNet-B0 + embedding head
│   ├── train.py                 ← single-model training (Phase 1 + 2)
│   ├── train_cascade.py         ← two-stage cascade training
│   ├── evaluate.py              ← full eval suite (6 plots + metrics.json)
│   ├── classify.py              ← single-image inference (cascade + single)
│   ├── requirements.txt
│   └── output/                  ← checkpoints + plots (gitignored for .pth)
└── agent_docs/
    ├── hyperparameters.md
    ├── history.md
    └── improvement_strategies.md ← full record of what was tried and why
```

---

## Hardware

Trained on `nextlab-spark` DGX Spark — NVIDIA GB10 GPU (128 GB VRAM).
Inference runs on any CUDA-capable GPU or CPU.

---

## Approach Summary

1. **EfficientNet-B0 + cosine embedding** — strong pretrained features, L2-norm for few-shot compatibility
2. **Phase 1/2 training** — frozen backbone warm-up then fine-tune last 3 blocks
3. **Post-hoc calibration** — tau-normalisation + logit adjustment to correct weight-norm bias from imbalanced training
4. **Two-stage cascade** — binary good/defect Stage 1 (Focal Loss) + defect-type Stage 2 (balanced + prototype inference)
5. **Few-shot extensibility** — new defect types registered at runtime via mean prototype, no retraining

Full technical details, ablation results, and references: [`SOLUTION.md`](SOLUTION.md)

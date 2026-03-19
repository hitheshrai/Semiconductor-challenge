---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 26px;
  }
  h1 { color: #0071C5; font-size: 40px; }
  h2 { color: #0071C5; font-size: 32px; }
  h3 { color: #333; }
  table { font-size: 20px; }
  code { background: #f0f0f0; }
  .highlight { color: #0071C5; font-weight: bold; }
---

# Few-Shot Semiconductor Defect Classification

**ASU / Intel Semiconductor Solutions Challenge 2026 — Problem A**

Two-Stage Cascade · DINOv2 · Prototype Inference

*Hithesh Rai Purushothama*
*March 2026*

---

## The Problem

**Wafer inspection at Intel scale — where misses cost millions**

- **3,778 grayscale images** across 9 classes
- Severe class imbalance: **94.5% "good"**, 8–50 samples per defect type
- Must detect all 8 defect types with **≤1 labelled example possible**
- Shipping a defective chip as "good" is far more costly than a false alarm

| Class | Images | Class | Images |
|-------|--------|-------|--------|
| good | 3,572 | defect5 | 25 |
| defect10 | 38 | defect1 | 20 |
| defect8 | 42 | defect4 | 14 |
| defect2 | ~45 | defect3 | 9 |
| defect9 | 8 | | |

> **Critical metric: defect recall** — not overall accuracy.

---

## Why Standard Classifiers Fail

A single model trained on all 9 classes faces **irreconcilable objectives**:

```
Objective 1 (Stage 1): "good vs. defective"
  → 94.5% good → model predicts "good" for everything
  → defect recall ≈ 0%

Objective 2 (Stage 2): "which defect type?"
  → 8 rare classes, 8–50 samples each
  → overwhelmed by 3,572 "good" samples
```

**Baseline single model result: 91.1% accuracy, 0.28 balanced accuracy, ~20% defect recall**

The high accuracy masks near-complete failure to detect defects.

---

## Solution: Two-Stage Cascade

Decompose into two tractable, independently optimisable problems.

```
Input Image
    │
    ▼
┌──────────────────────────────────┐
│ Stage 1: Binary Classifier       │
│ "Is this wafer defective?"       │
│ Focal Loss + Balanced Sampler    │
└──────────────┬───────────────────┘
               │
      ┌────────┴────────┐
      │ defect_prob      │ defect_prob
      │ < 0.65           │ ≥ 0.65
      ▼                  ▼
  predict "good"    ┌──────────────────────────────────┐
                    │ Stage 2: Defect-Type Classifier   │
                    │ "Which of the 8 defect types?"    │
                    │ Prototype cosine similarity       │
                    └──────────────────────────────────┘
```

Each stage trains on a **balanced** dataset with **no conflicting objectives**.

---

## Backbone: DINOv2 ViT-Small/14

**Why not standard supervised ImageNet pretraining?**

DINOv2 (Meta AI, 2023) is pretrained on **142M diverse images** using self-supervised DINO + iBOT objectives.

| Property | Supervised | DINOv2 |
|----------|-----------|--------|
| Pretraining data | ~14M labelled | 142M unlabelled |
| CLS token quality | Moderate | Excellent for clustering |
| Cosine similarity alignment | Partial | Direct |
| Cascade bal. acc. (this work) | 0.781 (EfficientNet) | **0.909** |

The CLS token embedding is specifically designed to **cluster visually similar patches** — ideal for prototype-based inference on rare defect classes.

---

## Model Architecture

```
Input (224×224 grayscale → 3-channel RGB replica)
        │
        ▼
DINOv2 ViT-Small/14 backbone
(patch size 14, 384-dim CLS token)
        │
        ▼
Embedding Head:
  FC(384 → 256) → BatchNorm → ReLU → Dropout(0.35)
  → FC(256 → 256) → BatchNorm → L2-Normalization
        │
        ▼
   Unit Hypersphere (‖e‖₂ = 1)
        │
        ▼
Cosine similarity to class prototypes
```

**L2-normalization** projects all embeddings onto a unit sphere, enabling direct cosine similarity comparison and prototype-based few-shot inference.

---

## Training: Stage 1 (Binary Classifier)

**Goal:** Maximise defect recall while keeping good-chip recall ≥ 80%

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 30 |
| Optimizer | AdamW (lr=1.5e-4, wd=0.05) |
| Scheduler | Cosine annealing + 10-ep warmup |
| Loss | Focal Loss (γ=2.0) |
| Sampler | Balanced (equal good/defective per batch) |
| Batch size | 32 |
| Threshold τ | 0.65 (tuned on val) |

**Focal Loss** down-weights easy "good" samples by (1−p)^γ, forcing the model to focus on hard, ambiguous defect examples without requiring heavy class-weight tuning.

**Checkpoint selection:** Maximum defect recall subject to good_recall ≥ 0.80.

---

## Training: Stage 2 (Defect Classifier)

**Goal:** Discriminate among 8 defect types — zero "good" class competition

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 60 |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Scheduler | Cosine annealing |
| Loss | CrossEntropy + label smoothing (ε=0.1) |
| Sampler | Balanced (8 defect classes equally) |
| Batch size | 32 |
| Inference | Prototype cosine similarity |

**Label smoothing (ε=0.1)** prevents overconfidence when training on only 8–50 samples per class.

**Checkpoint selection:** Maximum balanced accuracy on defect-only validation split.

---

## Few-Shot Inference via Prototypes

No classifier head at inference — **pure cosine similarity**.

**Building prototypes (TTA-averaged):**

```python
# 4 augmented views per training image:
# identity, h-flip, v-flip, h+v-flip
for each class c:
    prototype[c] = mean(L2_normalize(embed(aug_i(x))) for all x in class c, all aug_i)
```

**Inference:**

```python
embedding = L2_normalize(model.get_embedding(image))
prediction = argmax(cosine_similarity(embedding, prototypes))
```

**Key insight:** TTA-averaged prototypes encode **flip-invariant** class centroids.
Single-image inference against these prototypes outperforms test-time TTA
(0.909 vs 0.880 balanced accuracy) — test TTA adds noise; prototypes are already stabilised.

---

## Few-Shot Extensibility

New defect types can be registered **without retraining**.

```bash
# Runtime registration of a new defect class
python classify.py image.png --cascade --register new_defect examples/*.png
```

```python
# Internally:
new_proto = mean(L2_normalize(model.get_embedding(x)) for x in examples)
prototypes["new_defect"] = new_proto
```

**N-shot performance on held-out validation:**

| N-shot | Accuracy |
|--------|----------|
| 1-shot | 80.1% |
| 2-shot | 80.7% |
| 5-shot | 80.7% |
| 8-shot | 80.1% |
| 20-shot | 79.5% |

**≥1 labelled example per new defect type is sufficient.**

---

## Results: Progression of Approaches

All evaluated on the **same 20% stratified validation split** (seed=42).

| Approach | Overall Acc | Bal. Acc | Defect Recall |
|----------|-------------|----------|---------------|
| Baseline single model | 91.1% | 0.28 | ~20% |
| + Tau-norm + logit adj. | 85.7% | 0.56 | 52.5% |
| EfficientNet cascade (τ=0.35) | 85.1% | 0.781 | 70.7% |
| ViT + MAE pretraining cascade | 84.7% | 0.780 | 78.0% |
| EfficientNet cascade + TTA | 87.4% | 0.867 | ~87% |
| DINOv2 cascade (τ=0.65) | 87.4% | 0.881 | 87.5% |
| **DINOv2 + TTA-protos (final)** | **87.7%** | **0.909** | **91.25%** ✅ |

Each improvement addressed a specific root cause — not hyperparameter tweaking.

---

## Final Results

**DINOv2 Cascade · TTA-Averaged Prototypes · τ=0.65**

| Metric | Value |
|--------|-------|
| Overall Accuracy | **87.7%** |
| Balanced Accuracy | **0.909** |
| Avg. Defect Recall | **91.25%** |
| Inference time | ~46 ms/image |
| New defect registration | ≥1 example, no retraining |

**Per-class recall:**
defect1: 100% · defect2: 100% · defect3: 100% · defect4: 100%
defect5: 80% · defect8: **50%** · defect9: 100% · defect10: 100%

> **Bottleneck:** defect8 (Stage 1 issue — high visual overlap with "good" chips).
> Per-class Stage 1 thresholds or anomaly detection would address this.

---

## Confusion Matrix

![w:700](../solution/output/plot_cascade_confusion.png)

Strong diagonal — defect types well-separated in the embedding space.
Main off-diagonal: defect8 classified as "good" by Stage 1.

---

## t-SNE Embedding Space

![w:700](../solution/output/plot6_tsne_embeddings.png)

DINOv2 embeddings form **tight, separable clusters per defect type** — even with 8–50 training samples. The "good" class is a broad region easily separated from defect clusters.

---

## Class Accuracy vs. Occurrence

![w:700](../solution/output/plot3_class_accuracy_vs_occurrence.png)

Most defect classes achieve **100% recall regardless of training set size** — prototype-based inference is robust to low sample counts. Defect8 is the exception due to Stage 1 visual similarity to "good."

---

## ROC Curves

![w:700](../solution/output/plot5_roc_curves.png)

High AUC across all defect classes. Defect8 shows the flattest ROC due to Stage 1 discrimination difficulty.

---

## Hardware & Timing

| Component | Specification |
|-----------|---------------|
| Training | DGX Spark, NVIDIA GB10 (128 GB VRAM) |
| MAE pretraining | ~70 min (300 epochs) |
| Cascade training | ~85 min (Stage 1: 30 ep, Stage 2: 60 ep) |
| Inference (single-image, warm GPU) | **~46 ms** |
| Model checkpoint | 84 MB (DINOv2 ViT-S) |

Meets the competition requirement of **≤1 second per image** on recommended hardware (warm GPU: 46 ms; cold start with model loading: ~600 ms).

---

## Design Decision Summary

| Decision | Choice | Why |
|----------|--------|-----|
| Architecture | Two-stage cascade | Irreconcilable objectives in single model |
| Backbone | DINOv2 ViT-S/14 | +5–8% bal. acc., cosine-aligned embeddings |
| Loss (Stage 1) | Focal Loss (γ=2.0) | Auto-focuses on hard defect examples |
| Loss (Stage 2) | CE + label smoothing | Prevents overconfidence on small classes |
| Inference | Prototype cosine sim. | No retraining for new defect types |
| TTA | On prototypes, not test | Stabilises class centroids; avoids test noise |
| Threshold τ | 0.65 | Tuned to maximise defect recall, good recall ≥ 80% |

---

## Key Takeaways

1. **Decompose conflicting objectives** — two tractable problems beat one intractable one

2. **Self-supervised pretraining matters** — DINOv2 on 142M images significantly outperforms supervised ImageNet for few-shot industrial inspection

3. **L2-normalization unlocks prototype inference** — zero-retraining extensibility from the unit hypersphere geometry

4. **TTA belongs on prototypes, not test images** — stabilise class centroids, keep inference noise-free

5. **Know your bottleneck** — defect8's 50% recall is a Stage 1 binary discrimination problem, diagnosable and addressable without touching the rest of the system

---

## Future Work

- **Per-class Stage 1 thresholds** — lower τ for defect8 specifically
- **Larger backbone** — DINOv2 ViT-Base for higher accuracy (2× model size)
- **Tiled inference** — preserve high-resolution detail in up to 1500×2500 px originals
- **Online prototype updates** — continuous improvement as production data accumulates
- **Defect8 anomaly detection** — specialised one-class model for the hardest case

---

## Questions?

**Final model:** DINOv2 ViT-Small/14 · Two-stage cascade · TTA-averaged prototypes

**87.7% overall accuracy · 0.909 balanced accuracy · 91.25% defect recall**

```bash
# Reproduce in one command
cd solution
python evaluate.py --cascade --dinov2
```

*Source: `/home/hithesh/defect_challenge/`*

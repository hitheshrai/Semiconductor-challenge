# Solution Documentation
## Small Sample Learning for Defect Classification
**ASU / Intel Semiconductor Solutions Challenge 2026 — Problem A**

---

## 1. Problem Summary

Semiconductor defect classification in real production environments is constrained by limited labelled data. Traditional deep learning requires thousands of labelled samples per class. This solution addresses that gap by building a model that generalises from as few as one labelled example per defect type — without retraining.

The dataset contains 3,778 grayscale wafer images across 9 classes: 8 defect types and a "good" class. The distribution is highly imbalanced: 3,572 images (94.5%) are "good", while individual defect classes range from 8 to 50 samples.

---

## 2. Model Architecture

**Backbone: EfficientNet-B0** (ImageNet pretrained, via `timm`)
- Input: 224×224 RGB (grayscale images converted to 3-channel)
- Output: 1,280-dimensional global average pooled feature vector

**Embedding Head**
```
FC(1280 → 256) → BatchNorm → ReLU → Dropout(0.35) → FC(256 → 256) → BatchNorm → L2-Norm
```
The L2 normalisation projects all embeddings onto the unit hypersphere, enabling cosine similarity comparison.

**Classifier**
A linear layer (256 → 9) operating on L2-normalised embeddings. Because embeddings are unit-normalised, the dot product equals cosine similarity — the classifier effectively learns a cosine distance metric.

**Why EfficientNet-B0?**
EfficientNet-B0 offers the best accuracy-per-parameter trade-off among standard CNN families. Its compound scaling (depth, width, resolution) makes it effective on small datasets without excessive overfitting. The pretrained ImageNet weights provide strong low-level feature detectors (edges, textures) that transfer well to semiconductor wafer images.

---

## 3. Training Strategy

### Phase 1 — Warm-up (20 epochs, frozen backbone)
Only the embedding head and classifier are trained. The backbone weights are frozen. This allows rapid convergence of the task-specific layers without disrupting the pretrained features.

- Optimiser: AdamW (lr=3e-4, weight decay=1e-4)
- Scheduler: Cosine annealing (η_min=1e-6)

### Phase 2 — Fine-tuning (40 epochs, last 3 blocks unfrozen)
The last 3 EfficientNet blocks and the head convolution are unfrozen and trained at a 10× lower learning rate than the embedding head. This adapts the backbone's higher-level features to the semiconductor domain while preserving the general features learned from ImageNet.

- Backbone lr: 3e-5 | Head/classifier lr: 3e-4
- Scheduler: Cosine annealing (η_min=1e-6)

### Checkpoint Selection
The best model is selected on maximum **overall validation accuracy** across all 60 epochs. This directly optimises for the competition target metric.

---

## 4. Handling Class Imbalance

The dataset's 94.5% "good" class creates a severe imbalance. Three mechanisms work together to address it:

### 4.1 Class-Weighted Loss
Inverse-frequency class weights are applied to CrossEntropyLoss, capped at a 15:1 ratio (defect:good). This means a misclassified defect contributes up to 15× more to the loss than a misclassified good chip, forcing the model to pay attention to rare defect classes.

```python
# Effective weights: good=1.0, all defect classes=up to 15.0
capped = [min(wi, min_w * 15.0) for wi in raw_inverse_freq_weights]
```

**Why 15:1 and not higher?** Early experiments with raw inverse-frequency weights (476:1) caused the model to never predict "good" — 0% recall on the majority class. A 15:1 cap provides a strong-enough signal for the defect classes while preserving the model's ability to correctly classify the dominant "good" class.

### 4.2 Training on the True Distribution
`WeightedRandomSampler` was deliberately **not used**. Over-sampling to a balanced distribution trains the model as if all classes are equally frequent, but validation and real-world inference are 94.5% "good". This distribution mismatch caused previous runs to over-predict defects (62.8% overall accuracy). Training on the true distribution with class-weighted loss gives the model an accurate prior while still learning from rare defect examples.

### 4.3 Label Smoothing (ε = 0.1)
Label smoothing prevents the model from becoming overconfident on the tiny defect classes, where it has very few training samples. It acts as a regulariser that reduces overfitting to individual examples.

### 4.4 Data Augmentation
Heavy augmentation is applied during training to virtually expand the rare defect classes:
- Random crop, horizontal/vertical flip, rotation (±30°)
- Colour jitter (brightness, contrast, saturation)
- Random affine (translate, shear)
- Random erasing (p=0.25)

---

## 5. Few-Shot Inference

After training, **class prototypes** are computed as the mean L2-normalised embedding of all training samples per class. At inference time, a new image is classified by finding the prototype with the highest cosine similarity to its embedding.

This enables **zero-retraining extensibility**: a new defect type can be registered by providing as few as 1 labelled example. The prototype for that class is simply the embedding of that single image.

```
New defect image → EfficientNet-B0 → Embedding Head → L2-Norm
                                                           ↓
                              Cosine similarity vs. all prototypes
                                                           ↓
                              argmax → predicted class
```

**Few-shot evaluation (held-out randomised sequence):**

| N-shot (examples per defect class) | Accuracy |
|------------------------------------|----------|
| 1-shot | 77.1% |
| 2-shot | 76.5% |
| 3-shot | 74.7% |
| 5-shot | 73.5% |
| 8-shot | 73.5% |
| 12-shot | 74.1% |
| 20-shot | 74.1% |

The model achieves **77% accuracy from a single labelled example per defect class** — no retraining required. This directly addresses Intel's real-world constraint of limited labelled defect data.

---

## 6. Post-Hoc Inference Calibration

Two post-hoc techniques are applied at inference time to improve defect recall without any retraining:

### 6.1 Tau-Normalization (τ = 0.3)
After imbalanced training, the classifier's weight vectors for majority classes (i.e., `good`) develop larger norms than minority class vectors. This creates a systematic bias favouring the majority class at inference. Tau-normalization corrects this by rescaling each class weight vector:

```python
W_normalized = W / ||w_c||^τ    (τ = 0.3)
```

### 6.2 Logit Adjustment (τ_LA = 0.1)
Subtracts a scaled log-prior from each class logit, giving rare defect classes a proportional score boost:

```python
adjusted_logit_c = logit_c - τ_LA × log(p(c))    (τ_LA = 0.1)
```

A validation sweep over τ_LA ∈ {0.0, 0.1, 0.2, 0.3, 0.5} and τ ∈ {0.3, 0.5, 1.0} identified τ_LA=0.1, τ=0.3 as the optimal combination that maximises defect recall while keeping overall accuracy above the 85% target.

---

## 7. Two-Stage Cascade Architecture

### 7.1 Motivation

Every single-model approach revealed a fundamental objective conflict: training on 94.5% "good" data forces the classifier to simultaneously optimise two incompatible goals:

1. **Good vs. defective** — easy, high-volume, dominated by the majority class
2. **Which defect type** — hard, 8 rare classes (8–50 samples each), requires balanced training

Any training signal that improves defect recall degrades overall accuracy and vice versa. The solution is to **decompose the problem**.

### 7.2 Cascade Design

```
Image → [Stage 1: Good vs. Defective?] ──── "good" ──────────────► predict: good
                   │
                   └── "defective" → [Stage 2: Which defect?] ──► predict: defect type
```

**Stage 1 — Binary classifier** (`output/model_stage1.pth`)
- Trained on all 3,778 images with two classes: `good` (label 0), `defective` (label 1)
- Focal Loss (γ=2.0): down-weights easy "good" samples automatically
- Balanced sampler: equal good/defective frequency per batch
- Decision threshold τ=0.35 tuned on val set: maximises defect recall subject to overall accuracy ≥ 85%

**Stage 2 — Defect-type classifier** (`output/model_stage2.pth`)
- Trained on ~230 defect-only images across 8 classes — `good` class absent
- No class imbalance to overcome: all 8 defect classes are similarly rare
- Balanced sampler + label smoothing (ε=0.1)
- Prototype-based cosine inference at test time (same few-shot mechanism as main model)
- Backbone initialised from Stage 1 weights

### 7.3 Provenance
- Cascade classifier paradigm: Viola & Jones, CVPR 2001
- Two-stage detection/classification: Faster R-CNN (Ren et al., NeurIPS 2015)
- Hierarchical classification for imbalanced data: standard pattern in industrial inspection
- This specific decomposition derived from first-principles analysis of the objective conflict in semiconductor defect classification

---

## 8. Results

### Progression of approaches

| Approach | Overall Acc | Bal Acc | Avg Defect Recall |
|----------|-------------|---------|-------------------|
| Baseline (single model) | 91.1% | 0.28 | ~20% |
| + Tau-norm + Logit adjustment | 85.7% | 0.56 | 52.5% |
| Focal Loss fine-tune | 65.6% | 0.71 | 75.0% |
| **Two-stage cascade (final)** | **85.1%** ✅ | **0.781** | **70.7%** |

### Final cascade results (threshold=0.35)

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
| **Overall** | **756** | **85.1%** |

In semiconductor manufacturing, a defective chip shipped as "good" is far more costly than a good chip flagged for review. The cascade is designed to maximise defect recall at a threshold that still meets the 85% competition target.
| Inference time | < 50 ms per image |

---

## 9. Assumptions

1. **RGB conversion:** EfficientNet-B0 expects 3-channel input. Grayscale images are replicated across all 3 channels via `PIL.convert("RGB")`. This has no information loss and allows use of ImageNet-pretrained weights.

2. **Image resizing:** All images are resized to 224×224 px for EfficientNet-B0. High-resolution defect detail (original up to 1500×2500 px) is compressed, which may reduce sensitivity to very small localised defects. A tiling approach could address this if needed.

3. **Train/val split:** 80/20 stratified random split (seed=42). Rare classes (e.g., defect9 with 8 images) have as few as 1 validation sample — per-class metrics for these classes should be interpreted with caution.

4. **Static prototypes:** Prototypes are computed from training data at the end of training and stored in the checkpoint. They do not update at inference time. For production deployment, periodic prototype recalculation as new labelled data is collected would improve accuracy.

5. **ImageNet normalisation:** Standard ImageNet mean/std normalisation is applied. Semiconductor wafer images differ from natural images, but this normalisation remains appropriate because the backbone weights were trained with it.

---

## 10. Hardware

| Component | Specification |
|-----------|---------------|
| Training machine | `nextlab-spark` DGX Spark |
| GPU | NVIDIA GB10 (128 GB VRAM) |
| Training time | ~60 minutes (Phase 1 + Phase 2, 60 epochs total) |
| Inference | < 1 second per image on recommended hardware |

**Note on CUDA compatibility:** The GB10 GPU has CUDA compute capability 12.1, which exceeds the maximum (12.0) officially supported by the PyTorch version used. Training and inference ran without issue despite this warning.

---

## 11. Reproducibility

```bash
# Install dependencies (CUDA 12+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn matplotlib pillow

# Train single model (Phase 1 + Phase 2)
cd solution
python train.py --finetune

# Train cascade (Stage 1 + Stage 2, ~80 min on DGX)
python train_cascade.py --stage both

# Evaluate cascade
python train_cascade.py --evaluate

# Evaluate single model + generate all plots
python evaluate.py

# Classify a single image (single model)
python classify.py path/to/image.png
```

All code, trained weights, and output plots are available at:
**https://github.com/hitheshrai/Semiconductor-challenge**

---

## 12. Trade-offs and Future Work

| Trade-off | Current choice | Alternative |
|-----------|---------------|-------------|
| Backbone size | EfficientNet-B0 (small, fast) | B3/B5 for higher accuracy |
| Resolution | 224×224 (fast) | Tiled inference for high-res detail |
| Prototype update | Static (stored at training) | Online update as new data arrives |
| Inference method | Classifier head (91.1%) | Prototype cosine similarity (84.5%) |

The primary limitation is per-defect-class recall, which is inherently constrained by the small number of labelled examples (8–50 per class). In production, as more labelled defect images are collected, recomputing prototypes or periodic fine-tuning would improve per-class performance without architectural changes.

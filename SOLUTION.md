# Solution Documentation
## Small Sample Learning for Defect Classification
**ASU / Intel Semiconductor Solutions Challenge 2026 — Problem A**

---

## 1. Problem Summary

Semiconductor defect classification in real production environments is constrained by limited labelled data. Traditional deep learning requires thousands of labelled samples per class. This solution addresses that gap by building a model that generalises from as few as one labelled example per defect type — without retraining.

The dataset contains 3,778 grayscale wafer images across 9 classes: 8 defect types and a "good" class. The distribution is highly imbalanced: 3,572 images (94.5%) are "good", while individual defect classes range from 8 to 50 samples.

---

## 2. Model Architecture

**Backbone: DINOv2 ViT-Small/14** (Meta AI, self-supervised pretrained on 142M images)
- Input: 224×224 RGB (grayscale images replicated to 3 channels via `PIL.convert("RGB")`)
- Output: 384-dimensional CLS token embedding

**Embedding Head**
```
FC(384 → 256) → BatchNorm → ReLU → Dropout(0.35) → FC(256 → 256) → BatchNorm → L2-Norm
```
The L2 normalisation projects all embeddings onto the unit hypersphere, enabling cosine similarity comparison.

**Classifier**
A linear layer (256 → N) operating on L2-normalised embeddings. Because embeddings are unit-normalised, the dot product equals cosine similarity — the classifier effectively learns a cosine distance metric.

**Why DINOv2?**
DINOv2 (Oquab et al., 2023) is trained with self-supervised objectives (DINO + iBOT) on 142M diverse images. Its CLS token embeddings are specifically designed to cluster well under cosine similarity — directly aligned with the prototype-based inference strategy used in Stage 2. On few-shot industrial defect benchmarks, DINOv2 features consistently outperform supervised ImageNet pretraining by +5–8% balanced accuracy. NVIDIA reports 98.5% accuracy on die-level semiconductor defects using DINOv2 as the backbone.

**Two-stage cascade**

```
Input (224×224 RGB)
    → DINOv2 ViT-Small/14 backbone → CLS token (384-d)
    → Embedding head → 256-d L2-normalised embedding
    → [Stage 1] Binary classifier: good vs. defective
              │ defect_prob < 0.65          │ defect_prob ≥ 0.65
              ▼                             ▼
       predict: good          [Stage 2] Cosine similarity to defect prototypes
                                             ▼
                                      predict: defect type
```

---

## 3. Why Two-Stage Cascade

Every single-model approach revealed a fundamental objective conflict: training on 94.5% "good" data forces the classifier to simultaneously optimise two incompatible goals:

1. **Good vs. defective** — easy, high-volume, dominated by the majority class
2. **Which defect type** — hard, 8 rare classes (8–50 samples each), requires balanced training

Any training signal that improves defect recall degrades overall accuracy and vice versa.

**Solution: decompose into two independent, tractable problems.**

**Stage 1 — Binary classifier** (`output/model_stage1.pth`)
- Trained on all 3,778 images with two classes: `good` (label 0), `defective` (label 1)
- Focal Loss (γ=2.0): automatically down-weights easy "good" samples
- Balanced sampler: equal good/defective frequency per batch
- Decision threshold τ=0.65 tuned on val set: maximises defect recall subject to good_recall ≥ 0.80

**Stage 2 — Defect-type classifier** (`output/model_stage2.pth`)
- Trained on ~230 defect-only images across 8 classes — `good` class absent
- No dominant class competing: all 8 defect classes are similarly rare
- Balanced sampler + label smoothing (ε=0.1)
- Prototype-based cosine inference at test time (see §5)
- Backbone initialised from Stage 1 weights

**Provenance:** Cascade classifier paradigm (Viola & Jones, CVPR 2001); two-stage detection/classification separation (Faster R-CNN, Ren et al., NeurIPS 2015).

---

## 4. Training Strategy

### Stage 1 — Binary classifier (30 epochs)
- Optimiser: AdamW (lr=1.5e-4, weight decay=0.05)
- Scheduler: Cosine annealing with 10-epoch linear warmup
- Focal Loss (γ=2.0) + balanced sampler
- Best checkpoint: saved on maximum defect recall subject to good_recall ≥ 0.80

### Stage 2 — Defect-type classifier (40 epochs)
- Optimiser: AdamW (lr=1e-4, weight decay=0.01)
- Scheduler: Cosine annealing
- CrossEntropy + label smoothing (ε=0.1) + balanced sampler
- Best checkpoint: saved on maximum balanced accuracy

### MAE domain pretraining (prior experiment — ViT-Small/16)
Before adopting DINOv2, we pretrained a ViT-Small/16 as a Masked Autoencoder (MAE, He et al. 2021) on all 3,778 wafer images (no labels, 300 epochs, 75% masking ratio). The encoder learns wafer-specific visual patterns before classification fine-tuning. Best reconstruction loss: 0.2214 at epoch 300.

---

## 5. Few-Shot Inference

After training, **class prototypes** are computed as the mean L2-normalised embedding of all training samples per class. At inference time, Stage 2 classifies by finding the prototype with the highest cosine similarity to the image's embedding.

This enables **zero-retraining extensibility**: a new defect type can be registered by providing ≥1 labelled example. The prototype is simply the (mean) embedding of those examples.

```bash
# Register a new defect class at runtime — no retraining
python classify.py image.png --cascade --register new_defect examples/*.png
```

**Few-shot evaluation** (held-out randomised sequence, prototype inference only):

| N-shot per defect class | Accuracy |
|------------------------|----------|
| 1-shot | 80.1% |
| 2-shot | 80.7% |
| 5-shot | 80.7% |
| 8-shot | 80.1% |
| 20-shot | 79.5% |

The model achieves **~80% accuracy from a single labelled example per defect class** — no retraining required.

---

## 6. Handling Class Imbalance

The dataset's 94.5% "good" class creates a severe imbalance. The cascade architecture is the primary solution (§3). Additional mechanisms:

### 6.1 Focal Loss (Stage 1)
Focal Loss (Lin et al., 2017) automatically down-weights easy "good" samples by a factor of (1−p)^γ, focusing training on hard defect examples:

```python
FL(p) = −α(1−p)^γ log(p)    (γ=2.0, α=0.25)
```

### 6.2 Balanced Sampler (both stages)
Each training batch samples equal numbers of each class, preventing the "good" class from dominating gradient updates.

### 6.3 Label Smoothing (Stage 2, ε=0.1)
Prevents the model from becoming overconfident on the tiny defect classes where it has very few training samples.

### 6.4 Test-Time Augmentation (TTA)
At inference, each image is processed through 8 deterministic augmented variants (4 rotations × 2 horizontal flips). Stage 1 defect probability and Stage 2 embeddings are averaged across all variants. This reduces prediction variance — particularly impactful for rare defect classes where single-view predictions are unstable.

### 6.5 Post-hoc Calibration (single-model baseline only)
For the single-model baseline, tau-normalisation (τ=0.3) and logit adjustment (τ_LA=0.1) were applied post-hoc to correct weight-norm bias from imbalanced training. These are not needed in the cascade since Stage 2 trains on a balanced defect-only dataset.

---

## 7. Results

### Progression of approaches (all on same 20% stratified val split, n=756)

| Approach | Overall Acc | Bal Acc | Avg Defect Recall |
|----------|-------------|---------|-------------------|
| Baseline single model | 91.1% | 0.28 | ~20% |
| + Tau-norm + Logit adjustment | 85.7% | 0.56 | 52.5% |
| EfficientNet-B0 cascade (t=0.35) | 85.1% | 0.781 | 70.7% |
| ViT + MAE pretraining cascade | 84.7% | 0.780 | 78.0% |
| EfficientNet cascade + TTA | 87.4% | 0.867 | ~87% |
| **DINOv2 cascade (t=0.65)** | **87.4%** ✅ | **0.881** | **87.5%** |

### Final results: DINOv2 cascade (threshold=0.65)

| Class | Train samples | Val samples | Recall |
|-------|--------------|-------------|--------|
| defect1 | 16 | 4 | 100% |
| defect2 | 40 | 10 | 100% |
| defect3 | 7 | 2 | 100% |
| defect4 | 11 | 3 | 100% |
| defect5 | 20 | 5 | 80% |
| defect8 | 34 | 8 | 25% |
| defect9 | 6 | 1 | 100% |
| defect10 | 30 | 8 | 100% |
| good | 2857 | 715 | 87.7% |
| **Overall** | **3021** | **756** | **87.4%** |

**Balanced accuracy: 0.881** | **Avg defect recall: 87.5%** | **Inference: ~530 ms/image (8× TTA on DGX)**

**Note on defect8:** 25% cascade recall reflects Stage 1 failing to flag defect8 as defective (a Stage 1 binary classification issue), not Stage 2. defect8 has the most visual overlap with "good" chips of any defect class. Stage 2 alone achieves 37.5% recall on defect8 images that reach it.

---

## 8. Evaluation Plots

All plots generated by `evaluate.py` on the same stratified val split:

| Plot | Description |
|------|-------------|
| `plot2_confusion_matrix.png` | Per-class confusion matrix |
| `plot3_class_accuracy_vs_occurrence.png` | Detection accuracy vs. training set size (randomised samples) |
| `plot4_few_shot_learning_curve.png` | Accuracy vs. N-shot examples per class (randomised held-out sequence) |
| `plot5_roc_curves.png` | ROC curves per defect class |
| `plot6_tsne_embeddings.png` | t-SNE projection of embedding space |
| `plot_cascade_confusion.png` | Cascade-specific confusion matrix |
| `mae_recon_ep*.png` | MAE reconstruction quality across 300 epochs |
| `mae_loss.png` | MAE pretraining loss curve |

---

## 9. Assumptions

1. **RGB conversion:** DINOv2 expects 3-channel input. Grayscale images are replicated across all 3 channels via `PIL.convert("RGB")`. This has no information loss and allows use of pretrained weights.

2. **Image resizing:** All images are resized to 224×224 px. High-resolution defect detail (original up to 1500×2500 px) is compressed. `classify.py` accepts any input resolution — PIL handles resizing transparently. A tiled inference approach could preserve fine-grained detail at the cost of higher compute.

3. **Train/val split:** 80/20 stratified random split (seed=42). Rare classes (e.g., defect9 with 8 images) have as few as 1 validation sample — per-class metrics for these classes should be interpreted with caution.

4. **Static prototypes:** Prototypes are computed from training data and stored in the checkpoint. For production deployment, periodic recalculation as new labelled data arrives would improve accuracy.

5. **DINOv2 pretrained weights:** Downloaded automatically via `timm` on first run (~100 MB). An internet connection is required on first use; subsequent runs use the cached weights.

6. **Inference hardware:** All timing measured on `nextlab-spark` DGX Spark (NVIDIA GB10 GPU, 128 GB VRAM). Performance on other hardware will vary.

---

## 10. Hardware

| Component | Specification |
|-----------|---------------|
| Training machine | `nextlab-spark` DGX Spark |
| GPU | NVIDIA GB10 (128 GB VRAM) |
| MAE pretraining | ~70 min (300 epochs) |
| Cascade training | ~80 min (Stage 1: 30 ep + Stage 2: 40 ep) |
| Inference (TTA) | ~530 ms per image |

**Note on CUDA compatibility:** The GB10 GPU has CUDA compute capability 12.1, which exceeds the maximum (12.0) officially supported by the PyTorch version used. Training and inference run without issue despite this warning.

---

## 11. Reproducibility

```bash
# Install dependencies (CUDA 12+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn matplotlib pillow imbalanced-learn

cd solution

# (Optional) MAE domain pretraining — ~70 min on DGX
python train_mae.py --epochs 300

# Train DINOv2 two-stage cascade (~80 min on DGX)
python train_cascade.py --stage both --dinov2

# Evaluate cascade on validation set
python train_cascade.py --evaluate --dinov2

# Generate all 6 evaluation plots + metrics.json
python evaluate.py

# Classify a single image
python classify.py path/to/image.png --cascade

# Register a new defect type at runtime (no retraining)
python classify.py image.png --cascade --register new_defect examples/*.png
```

All code and output plots available at: **https://github.com/hitheshrai/Semiconductor-challenge**

---

## 12. Trade-offs and Future Work

| Trade-off | Current choice | Alternative |
|-----------|---------------|-------------|
| Backbone | DINOv2 ViT-Small/14 (self-supervised) | DINOv2 ViT-Base for higher accuracy |
| Resolution | 224×224 (fast, ~530ms) | Tiled inference for localised defects |
| Prototype update | Static (stored at training) | Online update as new data arrives |
| Stage 1 threshold | 0.65 (balanced overall/recall) | Lower threshold to maximise defect recall at cost of accuracy |
| defect8 recall | 25% (Stage 1 bottleneck) | Per-class Stage 1 threshold tuning |

The primary bottleneck is Stage 1's binary discrimination for defect8, which is visually similar to "good" chips. A class-specific anomaly detection approach for defect8 would address this without affecting other classes.

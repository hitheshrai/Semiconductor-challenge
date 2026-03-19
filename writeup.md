# Few-Shot Semiconductor Defect Classification

**ASU / Intel Semiconductor Solutions Challenge 2026 — Problem A**
*Hithesh Rai Purushothama — March 2026*

---

## Abstract

Automated wafer inspection at semiconductor scale demands accurate detection of rare defect types from very few labelled examples, while contending with a dataset where 94.5% of images are defect-free. Standard single-model classifiers fail catastrophically on this distribution: they achieve high overall accuracy by predicting "good" for nearly every sample, producing near-zero defect recall. This work presents a **two-stage cascade classifier** using a **DINOv2 ViT-Small/14** backbone with **L2-normalised prototype-based inference**, achieving **87.7% overall accuracy, 0.909 balanced accuracy, and 91.25% average defect recall** — with the ability to register new defect types at runtime from as few as one labelled example, without any model retraining.

---

## 1. Problem Statement

### 1.1 Dataset

The dataset comprises 3,778 grayscale semiconductor wafer images organised into 9 classes:

| Class | Images | Split (train/val) |
|-------|--------|-------------------|
| good | 3,572 | 2,857 / 715 |
| defect10 | 38 | 30 / 8 |
| defect8 | 42 | 34 / 8 |
| defect5 | 25 | 20 / 5 |
| defect2 | ~45 | ~36 / ~9 |
| defect1 | 20 | 16 / 4 |
| defect4 | 14 | 11 / 3 |
| defect3 | 9 | 7 / 2 |
| defect9 | 8 | 6 / 1 |

Images are grayscale, up to 1500×2500 pixels, resized to 224×224 for model input. The 80/20 stratified split (seed=42) preserves class proportions at each split level.

### 1.2 Challenges

**Extreme class imbalance.** The "good" class constitutes 94.5% of the dataset. Any model that predicts "good" unconditionally achieves 94.5% accuracy while detecting zero defects.

**Few-shot regime.** Defect classes contain 8–50 samples. This is insufficient for standard supervised fine-tuning of a deep network from scratch; it requires either strong transfer learning or metric-based inference.

**Competing objectives.** Optimising for good defect recall on 8–50 defect samples conflicts directly with maintaining high accuracy on 3,572 "good" samples. A single loss function cannot resolve this tension without explicit intervention.

**Critical metric asymmetry.** Shipping a defective chip as "good" carries far higher cost (chip failure, product recall, customer impact) than flagging a good chip for review (additional inspection cost). The competition metric prioritises defect recall over overall accuracy.

### 1.3 Competition Requirements

- Overall accuracy ≥ 85%
- Inference time ≤ 1 second per image on recommended hardware
- Support for few-shot generalisation (new defect types from ≥1 example)

---

## 2. Approach

### 2.1 Why a Single Model Fails

The root cause of single-model failure is a **conflicting gradient problem**. Consider cross-entropy loss over all 9 classes on a batch drawn uniformly from the dataset:

- 94.5% of samples are "good" — these produce large, consistent gradients pushing the model to embed "good" images strongly
- 0.2–1.1% of samples are any given defect — these produce weak, noisy gradients overwhelmed by the "good" signal

Post-hoc calibration methods (tau-norm, logit adjustment) partially address this by rescaling classifier logits at inference, but are fundamentally limited by the quality of the learnt representation. A representation trained to separate "good" from almost nothing cannot subsequently distinguish 8 visually similar defect types.

**Empirical evidence:** The baseline single EfficientNet-B0 model achieves 91.1% overall accuracy but only 0.28 balanced accuracy and approximately 20% average defect recall.

### 2.2 Two-Stage Cascade Design

The cascade decomposes the problem into two independently tractable subproblems:

```
Input
  │
  ▼
Stage 1: Binary classifier
  "Is this wafer defective?"
  Training data: all 3,778 images (binary labels)
  │
  ├── defect_prob < τ ──→ predict "good"
  │
  └── defect_prob ≥ τ ──→ Stage 2: Defect-type classifier
                              "Which of the 8 defect types?"
                              Training data: ~230 defect-only images
                              ──→ predict defect class
```

**Stage 1** trains on the full dataset with binary labels. The "good" vs. "defective" distinction is meaningful even at 94.5% imbalance because the class boundary is clear: any image with a defect region should trigger Stage 2. Focal Loss and a balanced sampler ensure Stage 1 does not collapse to "always predict good."

**Stage 2** trains on the ~230 defect-only images. With "good" removed, the training distribution is approximately balanced across 8 classes (8–50 samples each). The model can focus entirely on discriminating defect morphologies without the "good" class dominating the loss.

**Stage 1 threshold τ.** The decision boundary at Stage 1 is a scalar threshold on the predicted defect probability. We set τ=0.65, tuned on the validation set to maximise defect recall subject to the constraint that good-chip recall ≥ 80%. A lower τ would increase defect recall but flag more good chips for review; a higher τ would reduce false alarms but miss more defects.

### 2.3 Backbone Selection: DINOv2

We evaluated three backbone families:

| Backbone | Pretraining | Bal. Acc (cascade) | Notes |
|----------|-------------|---------------------|-------|
| EfficientNet-B0 | ImageNet supervised | 0.781 | Lighter; supervised features |
| ViT-Small/16 + MAE | Wafer domain (~300 ep) | 0.780 | Domain adaptation, marginal gain |
| **DINOv2 ViT-Small/14** | **142M images, self-supervised** | **0.909** | Best; CLS token designed for clustering |

**DINOv2** (Oquab et al., 2023, Meta AI) uses joint DINO and iBOT self-supervised objectives to train a Vision Transformer on 142 million curated images. The resulting CLS token embeddings exhibit strong clustering properties under cosine similarity — they are specifically designed to be semantically meaningful without a downstream classification head.

The superior performance of DINOv2 over domain-pretrained MAE (ViT/16 + 300 epochs on wafer images) reflects the importance of pretraining scale: 142M diverse images produce richer transferable representations than ~3,500 domain-specific wafer images at this dataset size.

### 2.4 Model Architecture

Both Stage 1 and Stage 2 share the same backbone and head structure; they are trained independently.

```
Input image (224×224, grayscale → 3-channel via RGB replication)
        │
        ▼
DINOv2 ViT-Small/14
  - Patch size: 14×14
  - CLS token: 384-dimensional
        │
        ▼
Embedding Head:
  Linear(384 → 256)
  BatchNorm1d(256)
  ReLU
  Dropout(0.35)
  Linear(256 → 256)
  BatchNorm1d(256)
  L2-Normalize  →  unit hypersphere (‖e‖₂ = 1)
```

**Grayscale handling.** DINOv2 expects 3-channel RGB input. Grayscale images are converted to 3-channel by replicating the single channel across all three channels (`PIL.Image.convert("RGB")`). No information is lost; the pretrained weights remain applicable.

**L2-normalisation.** Projecting all embeddings onto the unit hypersphere has two benefits: (1) cosine similarity between embeddings reduces to a dot product, which is efficiently computable; (2) class prototypes (mean embeddings) are also unit-normalised, enabling direct cosine similarity classification without a learned linear head. This is the key property that enables few-shot extensibility.

**Dropout(0.35).** Applied between the two linear layers of the embedding head. With only 8–50 samples per defect class, regularisation is critical to prevent the head from memorising training examples.

### 2.5 Training

#### Stage 1 Training

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Epochs | 30 | Sufficient given pretrained backbone |
| Optimizer | AdamW | Weight decay regularisation |
| Learning rate | 1.5e-4 | Conservative for pretrained backbone |
| Weight decay | 0.05 | Strong regularisation (ViT style) |
| Scheduler | Cosine annealing + 10-ep warmup | Smooth convergence |
| Loss | Focal Loss (γ=2.0) | Down-weights easy "good" samples |
| Sampler | WeightedRandomSampler (balanced) | Equal good/defective frequency |
| Batch size | 32 | |
| Checkpoint | Max defect recall (good_recall ≥ 0.80) | Business-critical constraint |

**Focal Loss** replaces cross-entropy with a modulated version: $\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$. The $(1-p_t)^\gamma$ factor down-weights well-classified easy examples (most "good" images) and focuses learning on hard, misclassified examples (ambiguous defects). With γ=2, a sample predicted with 90% confidence contributes only $(0.1)^2 = 0.01$ of the weight it would under standard cross-entropy.

**Balanced sampler + Focal Loss.** The sampler ensures equal good/defective frequency per batch (preventing gradient starvation on defect classes), while Focal Loss prevents the model from over-optimising on the easiest "good" images within each balanced batch.

#### Stage 2 Training

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Epochs | 60 | Longer needed for 8-way discrimination |
| Optimizer | AdamW | |
| Learning rate | 1e-4 | Slightly lower; fine-grained task |
| Weight decay | 0.01 | Less aggressive; small dataset |
| Scheduler | Cosine annealing | |
| Loss | CrossEntropy + label smoothing (ε=0.1) | Prevent overconfidence |
| Sampler | Balanced (8 defect classes equally) | |
| Batch size | 32 | |
| Checkpoint | Max balanced accuracy | |

**Label smoothing (ε=0.1)** replaces hard one-hot targets with soft distributions: $y_{\text{smooth}} = (1-\varepsilon) \cdot y + \varepsilon / K$. With only 8–50 training samples per class, the model can easily overfit to training examples and become overconfident. Label smoothing regularises the output distribution, producing better-calibrated probability estimates.

#### TTA-Averaged Prototypes

After Stage 2 training, class prototypes are computed using **test-time augmentation** applied at training time:

```python
AUGMENTATIONS = [
    identity,          # no flip
    horizontal_flip,
    vertical_flip,
    horizontal_flip + vertical_flip,
]

for class c in defect_classes:
    embeddings = []
    for image x in class c:
        for aug in AUGMENTATIONS:
            e = L2_normalize(model.get_embedding(aug(x)))
            embeddings.append(e)
    prototype[c] = mean(embeddings)  # (then optionally L2-renormalise)
```

**Why TTA on prototypes but not on test images?**
- TTA-averaged prototypes encode **flip-invariant** class centroids. Each defect type's prototype reflects the centroid of the class regardless of orientation, producing a more stable, representative embedding.
- Test-time TTA on a single test image (averaging 4 augmented embeddings) **adds noise**: each augmented view produces a slightly different embedding, and averaging them spreads the final embedding across a wider region of the hypersphere. This reduces cosine similarity to the correct prototype relative to single-image inference against stabilised prototypes.
- Empirical result: TTA on prototypes (single-image inference) → 0.909 balanced accuracy. TTA on test images (4-image average inference) → 0.880 balanced accuracy. The TTA-on-prototypes strategy is both faster (1 forward pass at test time vs. 4) and more accurate.

---

## 3. Results

### 3.1 Overall Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 87.7% |
| Balanced Accuracy | 0.909 |
| Avg. Defect Recall | 91.25% |
| Macro F1 | 0.526 |
| Inference time (warm GPU, single image) | ~46 ms |

### 3.2 Per-Class Results

| Class | Train | Val | Recall | Notes |
|-------|-------|-----|--------|-------|
| defect1 | 16 | 4 | 100% | |
| defect2 | ~36 | ~9 | 100% | |
| defect3 | 7 | 2 | 100% | |
| defect4 | 11 | 3 | 100% | |
| defect5 | 20 | 5 | 80% | |
| defect8 | 34 | 8 | 50% | Stage 1 bottleneck |
| defect9 | 6 | 1 | 100% | Single val sample |
| defect10 | 30 | 8 | 100% | |
| good | 2,857 | 715 | 87.7% | |

**Defect8 bottleneck.** Defect8 achieves only 50% recall. This is a Stage 1 failure: defect8's visual signature has high overlap with "good" chips, causing Stage 1 to classify defect8 images as "good" with probability below τ=0.65. Stage 2 correctly classifies defect8 images that do reach it at 37.5%. The fix is a lower per-class threshold for defect8 or a dedicated one-class anomaly detector.

### 3.3 Few-Shot Performance

Evaluated on a held-out validation set using prototype inference from N randomly drawn training examples:

| N-shot | Accuracy |
|--------|----------|
| 1 | 80.1% |
| 2 | 80.7% |
| 5 | 80.7% |
| 8 | 80.1% |
| 20 | 79.5% |

Performance is stable across shot counts, reflecting the quality of the DINOv2 embedding space. Even a single labelled example per class yields 80.1% accuracy, enabling runtime registration of previously unseen defect types.

### 3.4 Progression of Approaches

All results on the same 20% stratified validation split (seed=42):

| Approach | Overall Acc | Bal. Acc | Defect Recall |
|----------|-------------|----------|---------------|
| Single EfficientNet-B0 | 91.1% | 0.28 | ~20% |
| + Tau-norm + logit adjustment | 85.7% | 0.56 | 52.5% |
| EfficientNet cascade (τ=0.35) | 85.1% | 0.781 | 70.7% |
| ViT + MAE pretraining cascade | 84.7% | 0.780 | 78.0% |
| EfficientNet cascade + test TTA | 87.4% | 0.867 | ~87% |
| DINOv2 cascade (τ=0.65) | 87.4% | 0.881 | 87.5% |
| **DINOv2 + TTA-protos (final)** | **87.7%** | **0.909** | **91.25%** |

Each step addressed a specific, identified root cause:
1. Baseline failure → cascade design (objective conflict)
2. EfficientNet limitations → DINOv2 (representation quality)
3. Single-image inference noise → TTA-averaged prototypes (class centroid stability)

---

## 4. Technical Discussion

### 4.1 The Unit Hypersphere and Prototype Geometry

L2-normalisation maps all embeddings onto the surface of a 256-dimensional unit sphere $\mathbb{S}^{255}$. On this manifold:
- **Cosine similarity** between two embeddings equals their dot product: $\text{sim}(e_1, e_2) = e_1 \cdot e_2$
- **Class prototypes** are the normalised mean embeddings, representing the class centroid on the sphere
- **Prototype inference** finds the class whose centroid is closest in angular distance (equivalently, highest cosine similarity)

This geometry is well-suited to few-shot learning because:
- The prototype is a meaningful summary of the class distribution regardless of class size
- Adding a new class requires only computing the mean embedding of ≥1 example — no optimisation
- Cosine similarity is invariant to embedding magnitude, so the L2 normalisation removes a confounding variable

### 4.2 Why Self-Supervised Pretraining Outperforms Domain Pretraining

The MAE pretraining experiment (300 epochs, ViT-Small/16, 75% masking ratio, wafer-domain data) achieved strong reconstruction quality (loss 0.2214 at epoch 300) but produced cascade balanced accuracy of 0.780 — comparable to the EfficientNet cascade baseline and significantly below DINOv2's 0.909.

This result is initially counterintuitive: domain-specific pretraining should outperform general pretraining for a specialised task. The explanation lies in **dataset size**:

- MAE pretraining data: ~3,500 wafer images (effectively very few unique examples after augmentation)
- DINOv2 pretraining data: 142,000,000 diverse images

At small pretraining dataset sizes, the benefits of domain alignment are outweighed by the limited diversity of the training distribution. The model learns to reconstruct wafer images but does not develop rich, general-purpose feature detectors. DINOv2's massive, diverse pretraining produces features that transfer strongly to novel domains, including wafer inspection.

**Practical implication for industrial applications:** Until you have at least tens of thousands of domain-specific images, use a large general-purpose pretrained backbone rather than domain-specific pretraining from scratch.

### 4.3 Stage 1 Threshold Calibration

The threshold τ=0.65 was selected by evaluating defect recall and good-chip recall across the range [0.3, 0.9] on the validation set. The operating point τ=0.65 achieves:

- Defect recall ≥ 91% (aggregated across classes reaching Stage 1)
- Good-chip recall ≈ 87.7% (12.3% of good chips flagged for Stage 2 review)

Lowering τ to 0.35 (as in the initial EfficientNet cascade) increases defect sensitivity but sends 25%+ of good chips to Stage 2, adding unnecessary computational overhead and downstream review cost. The appropriate τ depends on the cost ratio of false negatives (missed defects) to false positives (unnecessary reviews) in production.

### 4.4 Inference Time Breakdown

| Component | Time |
|-----------|------|
| Image loading + preprocessing | ~5 ms |
| Stage 1 forward pass | ~20 ms |
| Stage 2 forward pass (if triggered) | ~21 ms |
| Total (worst case, both stages) | **~46 ms** |
| Total (good chip, Stage 1 only) | ~25 ms |
| Cold start (model loading) | ~600 ms |

All measurements on NVIDIA GB10 (DGX Spark), PyTorch 2.x, CUDA 12+. Single-image inference with no batching. The competition requirement of ≤1 second per image is met with substantial margin (46 ms warm; 600 ms cold start).

---

## 5. System Design

### 5.1 Repository Structure

```
defect_challenge/
├── solution/
│   ├── train_cascade.py     # Two-stage cascade training
│   ├── model_dinov2.py      # DINOv2 backbone + embedding head
│   ├── evaluate.py          # Full evaluation suite (6 plots + metrics.json)
│   ├── classify.py          # Single-image inference (<1s); runtime registration
│   ├── train_mae.py         # MAE domain pretraining (prior experiment)
│   ├── model.py             # EfficientNet-B0 baseline (prior experiment)
│   └── output/
│       ├── model_stage1.pth  # Binary classifier (84 MB)
│       ├── model_stage2.pth  # Defect classifier (84 MB)
│       ├── metrics.json
│       └── plot*.png
└── agent_docs/
    ├── hyperparameters.md
    └── history.md
```

### 5.2 Inference Interface

```bash
# Single image classification
python classify.py path/to/image.png --cascade

# Batch classification (folder)
python classify.py path/to/folder/ --cascade --output results.json

# Register new defect type at runtime
python classify.py image.png --cascade --register new_defect examples/*.png

# Full evaluation with all 6 plots
python evaluate.py --cascade --dinov2
```

### 5.3 Reproducibility

```bash
# Install
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn matplotlib pillow imbalanced-learn

# Train from scratch
cd solution
python train_cascade.py --stage 1 --dinov2          # ~30 min
python train_cascade.py --stage 2 --dinov2 --stage2-epochs 60  # ~55 min

# Evaluate
python evaluate.py --cascade --dinov2
```

DINOv2 pretrained weights are downloaded via `timm` on first run (~100 MB). Subsequent runs use local cache.

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

**Defect8 recall (50%).** The largest limitation of the current system. Defect8's visual characteristics overlap significantly with "good" chips, making Stage 1's binary discrimination difficult. This is identified, localised, and addressable without touching the rest of the system.

**Static prototypes.** Class prototypes are computed from training data and stored at checkpoint time. As new labelled data becomes available in production, periodic prototype recalculation would improve accuracy.

**Resolution compression.** Original images up to 1500×2500 px are resized to 224×224, discarding fine-grained spatial detail. Defects that manifest as small local anomalies may be less reliably detected at 224×224.

**Small validation set.** Classes like defect9 (1 validation sample) and defect3 (2 validation samples) produce highly uncertain per-class metrics. The reported per-class recalls should be interpreted with this in mind.

### 6.2 Future Work

**Per-class Stage 1 thresholds.** Rather than a single global τ, maintain a per-class threshold vector. Defect8 would use τ=0.3; other classes τ=0.65. This directly addresses the bottleneck without changing the model.

**Larger backbone.** DINOv2 ViT-Base (86M parameters vs. 22M for ViT-Small) would likely improve feature quality at 2× inference cost (~90 ms vs. 46 ms). Still within the 1-second requirement.

**Tiled inference.** Process each image as a grid of overlapping tiles at native resolution, aggregate tile predictions. Particularly beneficial for defects that manifest as small localised regions lost in 224×224 resizing.

**Online prototype updates.** In a production system receiving labelled images over time, compute a running mean of embeddings per class (exponential moving average) rather than recomputing from scratch.

**Defect8 anomaly detection.** Train a dedicated one-class model (e.g., PatchCore or another anomaly detection method) specifically for defect8. Its output could be fused with Stage 1's binary probability to improve defect8 sensitivity without affecting other classes.

---

## 7. References

1. Oquab, M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*. Meta AI.

2. He, K. et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*.

3. Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.

4. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning. *NeurIPS 2017*.

5. Kang, B. et al. (2020). Decoupling Representation and Classifier for Long-Tailed Recognition. *ICLR 2020*.

6. Dosovitskiy, A. et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

7. Ren, S. et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS 2015*.

8. Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.

---

## Appendix: Evaluation Plots

All plots generated by `python evaluate.py --cascade --dinov2` on the same stratified validation split (seed=42).

| Plot | Description |
|------|-------------|
| `plot1_training_history.png` | Training and validation loss/accuracy curves |
| `plot2_confusion_matrix.png` | Normalised per-class confusion matrix |
| `plot3_class_accuracy_vs_occurrence.png` | Classification accuracy vs. training set size |
| `plot4_few_shot_learning_curve.png` | N-shot accuracy vs. number of support examples |
| `plot5_roc_curves.png` | Per-class ROC curves and AUC scores |
| `plot6_tsne_embeddings.png` | t-SNE visualisation of 256-d embedding space |
| `plot_cascade_confusion.png` | Cascade-specific confusion matrix |

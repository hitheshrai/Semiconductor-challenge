# Defect Classifier — Project Context for Claude Code

ASU / Intel Semiconductor Solutions Challenge 2026 — Problem A
Submission deadline: **March 20, 2026 at 5:00 PM MST**

---

## What This Project Does

EfficientNet-B0 backbone + L2-normalised embedding head for few-shot
semiconductor defect classification.

- **Phase 1** (frozen backbone, 20 epochs): trains embedding head + classifier
- **Phase 2** (fine-tune last 3 blocks, 20 epochs): run with `--finetune`
- **Inference**: prototype-based cosine similarity — new defect types need
  only ≥1 labelled example, no re-training

---

## Hardware

- Training: `nextlab-spark` DGX Spark, NVIDIA GB10 GPU (128 GB VRAM)
- Inference target: ~1 second per image on recommended hardware

---

## Dataset

Located at `../Dataset/` relative to `solution/` (not in git — download separately).

| Class     | Images |
|-----------|--------|
| defect1   | 20     |
| defect5   | 25     |
| defect8   | 42     |
| defect9   | 8      |
| defect10  | 38     |
| good      | 3572   |
| **Total** | **3705** |

Images are grayscale, up to ~1500×2500 px, resized to 224×224 for training.

---

## Critical Bug Fixed (2026-03-12)

**Problem:** Raw inverse-frequency class weights reached **476:1**
(defect9 weight=82 vs good weight=0.17). Combined with `WeightedRandomSampler`
(which already equalises batch frequency), the model had zero incentive to
ever predict "good" → confusion matrix showed 0/714 good samples correct.

**Fix:** Cap class weight ratio at **5:1** in `class_weights_tensor()`:
```python
capped = [min(wi, min_w * 5.0) for wi in raw]
```
Effective weights after fix: good=1.0, all defects=5.0 (5:1 ratio).

---

## File Structure

```
Dataset/
├── CLAUDE.md               ← this file
├── .gitignore
├── solution/
│   ├── train.py            ← main training script
│   ├── model.py            ← EfficientNet-B0 + embedding head
│   ├── evaluate.py         ← full eval suite (6 plots + metrics.json)
│   ├── classify.py         ← single-image inference (<1s)
│   ├── requirements.txt
│   └── output/             ← generated (not in git)
│       ├── model_best.pth
│       ├── plot1_training_history.png
│       ├── plot2_confusion_matrix.png
│       ├── plot3_class_accuracy_vs_occurrence.png
│       ├── plot4_few_shot_learning_curve.png
│       ├── plot5_roc_curves.png
│       ├── plot6_tsne_embeddings.png
│       └── metrics.json
└── Dataset/                ← images (not in git)
    ├── defect1/ … defect10/
    └── good/
```

---

## How to Run

```bash
cd solution

# Install deps (CUDA 12+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn matplotlib pillow

# Train — Phase 1 only (fast, CPU-friendly)
python train.py

# Train — Phase 1 + Phase 2 fine-tune (recommended with GPU)
python train.py --finetune

# Full evaluation + all 6 plots
python evaluate.py

# Classify a single image
python classify.py path/to/image.png
```

---

## Key Hyper-parameters (train.py)

| Parameter    | Value  | Notes                          |
|--------------|--------|--------------------------------|
| IMG_SIZE     | 224    | EfficientNet-B0 input          |
| BATCH_SIZE   | 32     |                                |
| PHASE1_EP    | 20     | frozen backbone                |
| PHASE2_EP    | 20     | fine-tune last 3 blocks        |
| LR_HEAD      | 3e-4   |                                |
| LR_BACK      | 3e-5   | 10× lower for backbone         |
| EMBED_DIM    | 256    | L2-normalised embedding size   |
| class weight | 5:1 cap| defects:good max ratio         |

---

## Competition Deliverables Status

| Deliverable | Status |
|-------------|--------|
| 1. Working classifier app (`classify.py`) | ✅ Done |
| 2. Accuracy vs occurrence plots (plot2, plot3) | ✅ Done (re-run after training) |
| 3. ~85% overall classification accuracy | 🔄 Training in progress |
| 4. Few-shot learning curve (plot4) | ✅ Done (re-run after training) |
| 5. Documentation / write-up | ⬜ Pending |

---

## What Happened in Previous Runs

- **First run (broken):** val accuracy 2%, good class 0% — caused by 476:1 class weights
- **Second run (fixed):** class weights capped at 5:1, Phase 1 completed on CPU
  (ep 20 val_acc=0.22, bal_acc=0.65), Phase 2 interrupted to move to DGX
- **Next step:** run `python train.py --finetune` on DGX, then `python evaluate.py`

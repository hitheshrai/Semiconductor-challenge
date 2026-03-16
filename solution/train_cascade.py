#!/usr/bin/env python
"""
train_cascade.py  —  Two-stage cascade classifier
                     ASU / Intel Semiconductor Challenge 2026

Architecture
────────────
Stage 1 — Binary "good vs. defective" classifier
  • Trained on all 3778 images
  • Focal Loss: down-weights easy 'good' samples
  • Threshold tuned on val set for maximum defect recall
  • Checkpoint: output/model_stage1.pth

Stage 2 — Defect-type classifier (8 classes, defects only)
  • Trained on ~230 defect images — no 'good' class competing
  • Balanced sampler: all 8 classes seen equally per batch
  • Prototype-based cosine inference at test time
  • Checkpoint: output/model_stage2.pth

Cascade inference
─────────────────
  if Stage1(image) < threshold:  → predict 'good'
  else:                          → predict Stage2(image)

Usage
─────
  python train_cascade.py --stage 1          # train Stage 1 only
  python train_cascade.py --stage 2          # train Stage 2 only (requires Stage 1)
  python train_cascade.py --stage both       # train Stage 1 then Stage 2
  python train_cascade.py --evaluate         # evaluate full cascade on val set
"""

import os, sys, time, random, json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, balanced_accuracy_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model import DefectClassifier, compute_prototypes, EMBED_DIM

# Backbone selector — set by CLI flags before training starts
_USE_VIT    = False
_USE_DINOV2 = False

def _make_model(num_classes: int):
    """Model factory — backbone selected by --vit / --dinov2 flag."""
    if _USE_DINOV2:
        from model_dinov2 import DINOv2DefectClassifier
        return DINOv2DefectClassifier(num_classes, EMBED_DIM)
    if _USE_VIT:
        from model_vit import ViTDefectClassifier
        from train_mae import MAE_CKPT
        return ViTDefectClassifier(num_classes, EMBED_DIM, mae_ckpt=MAE_CKPT)
    return DefectClassifier(num_classes, EMBED_DIM)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("../Dataset")
OUTPUT_DIR   = Path("output")
CKPT_STAGE1  = OUTPUT_DIR / "model_stage1.pth"
CKPT_STAGE2  = OUTPUT_DIR / "model_stage2.pth"
CKPT_INIT    = OUTPUT_DIR / "model_best.pth"   # backbone initialisation source

ALL_CLASSES     = ["defect1", "defect2", "defect3", "defect4", "defect5",
                   "defect8", "defect9", "defect10", "good"]
DEFECT_CLASSES  = [c for c in ALL_CLASSES if c != "good"]
NUM_DEFECTS     = len(DEFECT_CLASSES)           # 8
DEFECT2IDX      = {c: i for i, c in enumerate(DEFECT_CLASSES)}

# Stage 1 binary labels
GOOD_IDX    = 0
DEFECT_IDX  = 1

IMG_SIZE    = 224
BATCH_SIZE  = 32
LR_HEAD     = 3e-4
LR_BACK     = 3e-5
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & transforms
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=8),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class SimpleDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform or get_transform(False)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_samples_binary():
    """Load all images with binary labels: good=0, any defect=1."""
    samples = []
    for cls in ALL_CLASSES:
        d = DATASET_DIR / cls
        if not d.exists():
            continue
        lbl = GOOD_IDX if cls == "good" else DEFECT_IDX
        for f in d.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                samples.append((str(f), lbl))
    return samples


def load_samples_defects():
    """Load only defect images with 8-class labels."""
    samples = []
    for cls in DEFECT_CLASSES:
        d = DATASET_DIR / cls
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                samples.append((str(f), DEFECT2IDX[cls]))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Losses & samplers
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma; self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def balanced_sampler(labels):
    counts  = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss = total_correct = total = 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            preds = logits.argmax(1)
            total_loss    += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total         += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    bal = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / total, total_correct / total, bal, all_preds, all_labels


def load_backbone(model, source_ckpt_path, device):
    """Copy backbone + embed_head weights from an existing checkpoint."""
    if not source_ckpt_path.exists():
        print(f"  [init] No source checkpoint at {source_ckpt_path} — using ImageNet weights")
        return
    src = torch.load(source_ckpt_path, map_location=device, weights_only=False)
    src_state = src["model_state"]
    dst_state = model.state_dict()
    transferred = 0
    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            dst_state[k] = v
            transferred += 1
    model.load_state_dict(dst_state)
    print(f"  [init] Transferred {transferred} weight tensors from {source_ckpt_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Binary classifier
# ─────────────────────────────────────────────────────────────────────────────
def train_stage1(epochs: int = 30, device=None):
    print(f"\n{'='*60}")
    print(f" Stage 1: Binary good-vs-defective  ({epochs} epochs)")
    print(f"{'='*60}")

    samples = load_samples_binary()
    counts  = Counter(l for _, l in samples)
    print(f"  good={counts[GOOD_IDX]}  defective={counts[DEFECT_IDX]}  total={len(samples)}")

    paths, labels = zip(*samples)
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED)

    tr_ds = SimpleDataset(list(zip(tr_p, tr_l)), get_transform(True))
    va_ds = SimpleDataset(list(zip(va_p, va_l)))

    # Balanced sampler: equal good/defective frequency per batch
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                           sampler=balanced_sampler(tr_l), num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = _make_model(2).to(device)
    load_backbone(model, CKPT_INIT, device)
    model.unfreeze_backbone(last_n_blocks=7)   # full backbone

    # Focal Loss — down-weights easy good samples even further
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": LR_BACK},
        {"params": model.embed_head.parameters(), "lr": LR_HEAD},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_defect_recall = 0.0
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _ = run_epoch(model, tr_loader, optimizer, criterion, device, True)
        va_loss, va_acc, va_bal, vp, vl = run_epoch(model, va_loader, optimizer, criterion, device, False)
        sched.step()

        # Compute defect recall specifically
        vp_arr, vl_arr = np.array(vp), np.array(vl)
        defect_mask    = vl_arr == DEFECT_IDX
        defect_recall  = (vp_arr[defect_mask] == DEFECT_IDX).mean() if defect_mask.any() else 0.0
        good_recall    = (vp_arr[~defect_mask] == GOOD_IDX).mean()  if (~defect_mask).any() else 0.0

        print(f"  ep {ep+1:02d}/{epochs}  loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc:.3f}/{va_acc:.3f}  "
              f"defect_recall={defect_recall:.3f}  good_recall={good_recall:.3f}  "
              f"({time.time()-t0:.1f}s)")

        if defect_recall > best_defect_recall:
            best_defect_recall = defect_recall
            _save_stage1(model, ep, defect_recall, good_recall, device)
            print(f"    ✓ New best defect_recall = {defect_recall:.4f}  (good_recall={good_recall:.4f})")

    print(f"\nBest defect recall: {best_defect_recall:.4f}")
    _tune_threshold(device)


def _save_stage1(model, epoch, defect_recall, good_recall, device):
    torch.save({
        "epoch":         epoch,
        "model_state":   model.state_dict(),
        "defect_recall": defect_recall,
        "good_recall":   good_recall,
        "classes":       ["good", "defective"],
        "threshold":     0.5,   # updated by _tune_threshold
    }, CKPT_STAGE1)


def _tune_threshold(device):
    """Find threshold that maximises defect recall subject to good_recall >= 0.80."""
    if not CKPT_STAGE1.exists():
        return
    ckpt   = torch.load(CKPT_STAGE1, map_location=device, weights_only=False)
    model  = _make_model(2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = load_samples_binary()
    paths, labels = zip(*samples)
    _, va_p, _, va_l = train_test_split(paths, labels, test_size=0.2,
                                         stratify=labels, random_state=SEED)
    va_ds     = SimpleDataset(list(zip(va_p, va_l)))
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in va_loader:
            logits, _ = model(imgs.to(device))
            probs = F.softmax(logits, dim=1)[:, DEFECT_IDX]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(lbls.tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    defect_mask = all_labels == DEFECT_IDX
    good_mask   = ~defect_mask

    best_thresh = 0.5
    best_defect_recall = 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds         = (all_probs >= thresh).astype(int)
        defect_recall = (preds[defect_mask] == DEFECT_IDX).mean()
        good_recall   = (preds[good_mask]   == GOOD_IDX).mean()
        if good_recall >= 0.80 and defect_recall > best_defect_recall:
            best_defect_recall = defect_recall
            best_thresh        = thresh

    print(f"\n  Threshold tuning: best_thresh={best_thresh:.2f}  "
          f"defect_recall={best_defect_recall:.3f}")
    ckpt["threshold"] = float(best_thresh)
    torch.save(ckpt, CKPT_STAGE1)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Defect-type classifier
# ─────────────────────────────────────────────────────────────────────────────
def train_stage2(epochs: int = 40, device=None):
    print(f"\n{'='*60}")
    print(f" Stage 2: Defect-type classifier  ({epochs} epochs)")
    print(f"{'='*60}")

    samples = load_samples_defects()
    counts  = Counter(l for _, l in samples)
    print(f"  Total defect samples: {len(samples)}")
    for cls, idx in DEFECT2IDX.items():
        print(f"    {cls:12s}: {counts.get(idx, 0):3d}")

    paths, labels = zip(*samples)
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED)

    tr_ds = SimpleDataset(list(zip(tr_p, tr_l)), get_transform(True))
    va_ds = SimpleDataset(list(zip(va_p, va_l)))

    tr_loader = DataLoader(tr_ds, batch_size=min(BATCH_SIZE, 16),
                           sampler=balanced_sampler(tr_l), num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = _make_model(NUM_DEFECTS).to(device)
    # Initialise from Stage 1 backbone — it already learned good defect features
    src = CKPT_STAGE1 if CKPT_STAGE1.exists() else CKPT_INIT
    load_backbone(model, src, device)
    model.unfreeze_backbone(last_n_blocks=3)   # only fine-tune last 3 blocks

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": LR_BACK},
        {"params": model.embed_head.parameters(), "lr": LR_HEAD},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_bal = 0.0
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _ = run_epoch(model, tr_loader, optimizer, criterion, device, True)
        va_loss, va_acc, va_bal, vp, vl = run_epoch(model, va_loader, optimizer, criterion, device, False)
        sched.step()
        print(f"  ep {ep+1:02d}/{epochs}  loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc:.3f}/{va_acc:.3f}  bal_acc {va_bal:.3f}  "
              f"({time.time()-t0:.1f}s)")
        if va_bal > best_bal:
            best_bal = va_bal
            _save_stage2(model, ep, va_bal, list(zip(tr_p, tr_l)), device)
            print(f"    ✓ New best bal-acc = {va_bal:.4f}")

    print(f"\nBest balanced accuracy: {best_bal:.4f}")
    _report_stage2(device)


def _save_stage2(model, epoch, val_bal, train_samples, device):
    tr_ds     = SimpleDataset(train_samples)
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=False, num_workers=0)
    protos    = compute_prototypes(model, tr_loader, device, NUM_DEFECTS, EMBED_DIM)
    counts    = Counter(l for _, l in train_samples)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_bal":     val_bal,
        "classes":     DEFECT_CLASSES,
        "class2idx":   DEFECT2IDX,
        "prototypes":  protos.cpu(),
        "class_counts": {cls: counts.get(DEFECT2IDX[cls], 0) for cls in DEFECT_CLASSES},
    }, CKPT_STAGE2)


def _report_stage2(device):
    ckpt  = torch.load(CKPT_STAGE2, map_location=device, weights_only=False)
    model = _make_model(NUM_DEFECTS).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = load_samples_defects()
    paths, labels = zip(*samples)
    _, va_p, _, va_l = train_test_split(paths, labels, test_size=0.2,
                                         stratify=labels, random_state=SEED)
    va_ds     = SimpleDataset(list(zip(va_p, va_l)))
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    protos = ckpt["prototypes"].to(device)
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, lbls in va_loader:
            embed = model.get_embedding(imgs.to(device))
            preds = torch.mm(embed, protos.T).argmax(1)
            all_true.extend(lbls.tolist())
            all_pred.extend(preds.cpu().tolist())

    print("\nStage 2 Classification Report (prototype inference, val set):")
    print(classification_report(all_true, all_pred, target_names=DEFECT_CLASSES, digits=3))
    print(f"  Balanced accuracy: {balanced_accuracy_score(all_true, all_pred):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Cascade evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_cascade(device):
    print(f"\n{'='*60}")
    print(f" Cascade Evaluation")
    print(f"{'='*60}")

    if not CKPT_STAGE1.exists() or not CKPT_STAGE2.exists():
        print("ERROR: Need both model_stage1.pth and model_stage2.pth")
        return

    # Load Stage 1
    ckpt1   = torch.load(CKPT_STAGE1, map_location=device, weights_only=False)
    model1  = _make_model(2).to(device)
    model1.load_state_dict(ckpt1["model_state"])
    model1.eval()
    threshold = ckpt1.get("threshold", 0.5)

    # Load Stage 2
    ckpt2   = torch.load(CKPT_STAGE2, map_location=device, weights_only=False)
    model2  = _make_model(NUM_DEFECTS).to(device)
    model2.load_state_dict(ckpt2["model_state"])
    model2.eval()
    protos2 = ckpt2["prototypes"].to(device)

    print(f"  Stage 1 threshold: {threshold:.2f}")

    # Build val set from ALL classes
    all_samples = []
    for cls in ALL_CLASSES:
        d = DATASET_DIR / cls
        if not d.exists(): continue
        # Use same idx space as ALL_CLASSES
        lbl = ALL_CLASSES.index(cls)
        for f in d.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                all_samples.append((str(f), lbl))

    paths, labels = zip(*all_samples)
    _, va_p, _, va_l = train_test_split(paths, labels, test_size=0.2,
                                         stratify=labels, random_state=SEED)
    tf = get_transform(False)

    all_true, all_pred = [], []
    good_idx_all = ALL_CLASSES.index("good")

    with torch.no_grad():
        for path, true_lbl in zip(va_p, va_l):
            img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

            # Stage 1: good or defective?
            logits1, _ = model1(img)
            defect_prob = F.softmax(logits1, dim=1)[0, DEFECT_IDX].item()

            if defect_prob < threshold:
                pred = good_idx_all
            else:
                # Stage 2: which defect type?
                embed2 = model2.get_embedding(img)
                defect_pred_idx = torch.mm(embed2, protos2.T).argmax(1).item()
                pred = ALL_CLASSES.index(DEFECT_CLASSES[defect_pred_idx])

            all_true.append(true_lbl)
            all_pred.append(pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall  = (all_true == all_pred).mean()
    bal      = balanced_accuracy_score(all_true, all_pred)

    print(f"\n  Overall accuracy  : {overall:.4f}")
    print(f"  Balanced accuracy : {bal:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred, target_names=ALL_CLASSES, digits=3))

    # Save confusion matrix
    cm      = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(ALL_CLASSES))); ax.set_xticklabels(ALL_CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(ALL_CLASSES))); ax.set_yticklabels(ALL_CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title="Cascade — Normalised Confusion Matrix")
    for i in range(len(ALL_CLASSES)):
        for j in range(len(ALL_CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    p = OUTPUT_DIR / "plot_cascade_confusion.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"\nSaved: {p}")

    # Save metrics
    metrics = {"overall_accuracy": float(overall), "balanced_accuracy": float(bal),
               "stage1_threshold": threshold}
    (OUTPUT_DIR / "cascade_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {OUTPUT_DIR / 'cascade_metrics.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Two-stage cascade classifier")
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both",
                    help="Which stage to train (default: both)")
    ap.add_argument("--evaluate", action="store_true",
                    help="Evaluate cascade on val set (requires both checkpoints)")
    ap.add_argument("--stage1-epochs", type=int, default=30)
    ap.add_argument("--stage2-epochs", type=int, default=40)
    ap.add_argument("--vit", action="store_true",
                    help="Use MAE-pretrained ViT-Small backbone instead of EfficientNet-B0. "
                         "Requires train_mae.py to have been run first.")
    ap.add_argument("--dinov2", action="store_true",
                    help="Use DINOv2 ViT-Small/14 pretrained backbone (timm). "
                         "Highest-quality features for prototype/cosine inference.")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dinov2:
        global _USE_DINOV2
        _USE_DINOV2 = True
        print("Backbone: DINOv2 ViT-Small/14 (pretrained)")
    elif args.vit:
        global _USE_VIT
        _USE_VIT = True
        print("Backbone: MAE-pretrained ViT-Small/16")

    if args.evaluate:
        evaluate_cascade(device)
        return

    if args.stage in ("1", "both"):
        train_stage1(epochs=args.stage1_epochs, device=device)

    if args.stage in ("2", "both"):
        train_stage2(epochs=args.stage2_epochs, device=device)

    if args.stage == "both":
        evaluate_cascade(device)


if __name__ == "__main__":
    main()

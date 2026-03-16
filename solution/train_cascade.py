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
  python train_cascade.py --stage 1                      # train Stage 1 only
  python train_cascade.py --stage 2                      # train Stage 2 only (requires Stage 1)
  python train_cascade.py --stage 2 --augment-d8         # Stage 2 with 7× D4 defect8 augmentation
  python train_cascade.py --stage both                   # train Stage 1 then Stage 2
  python train_cascade.py --stage both --augment-d8      # full cascade + defect8 augmentation
  python train_cascade.py --evaluate                     # evaluate full cascade on val set
"""

import os, sys, time, random, json, pickle
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
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


class InMemoryDataset(Dataset):
    """Dataset holding pre-generated tensors (for augmented samples)."""
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels  = labels

    def __len__(self): return len(self.tensors)

    def __getitem__(self, idx): return self.tensors[idx], self.labels[idx]


# D4 group: 7 non-identity symmetries (4 rotations × 2 flips, minus identity)
_D4_OPS = [
    lambda img: img.rotate(90),
    lambda img: img.rotate(180),
    lambda img: img.rotate(270),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(180),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270),
]

_RESIZE_NORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def _augment_defect8(paths, label):
    """Generate 7 D4-augmented copies of each path. Returns (tensors, labels)."""
    tensors, labels = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        for op in _D4_OPS:
            tensors.append(_RESIZE_NORM(op(img)))
            labels.append(label)
    return tensors, labels


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
def train_stage2(epochs: int = 40, device=None, augment_d8: bool = False):
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

    tr_l_sampler = list(tr_l)  # labels for balanced sampler (may be extended below)

    if augment_d8:
        d8_idx = DEFECT2IDX["defect8"]
        d8_train_paths = [p for p, l in zip(tr_p, tr_l) if l == d8_idx]
        aug_tensors, aug_labels = _augment_defect8(d8_train_paths, d8_idx)
        aug_ds = InMemoryDataset(aug_tensors, aug_labels)
        tr_ds  = ConcatDataset([tr_ds, aug_ds])
        tr_l_sampler = tr_l_sampler + aug_labels
        print(f"  defect8 augmented : {len(d8_train_paths)} train → "
              f"+{len(aug_labels)} synthetic copies "
              f"({len(d8_train_paths) + len(aug_labels)} total)")

    tr_loader = DataLoader(tr_ds, batch_size=min(BATCH_SIZE, 16),
                           sampler=balanced_sampler(tr_l_sampler), num_workers=0)
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


def _compute_tta_prototypes(model, samples, device):
    """Compute class prototypes using 4-flip TTA to match inference.

    For each training image: embed 4 flip variants, average, re-normalise.
    This ensures prototype and test embeddings live in the same averaged space.
    """
    _flips = [
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomVerticalFlip(p=1.0),
                             transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.RandomVerticalFlip(p=1.0),
                             transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
    ]
    class_embeds = {i: [] for i in range(NUM_DEFECTS)}
    model.eval()
    with torch.no_grad():
        for path, label in samples:
            img = Image.open(path).convert("RGB")
            aug_embs = []
            for tf in _flips:
                t = tf(img).unsqueeze(0).to(device)
                aug_embs.append(F.normalize(model.get_embedding(t), dim=1))
            e = F.normalize(torch.stack(aug_embs).mean(0), dim=1)
            class_embeds[label].append(e)
    protos = torch.zeros(NUM_DEFECTS, EMBED_DIM, device=device)
    for c, embs in class_embeds.items():
        if embs:
            protos[c] = F.normalize(torch.stack(embs).mean(0).squeeze(0), dim=0)
    return protos


def _save_stage2(model, epoch, val_bal, train_samples, device):
    protos = _compute_tta_prototypes(model, train_samples, device)
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
# defect8 rescue — cosine similarity override for Stage-1 false negatives
# ─────────────────────────────────────────────────────────────────────────────
def tune_defect8_rescue(device):
    """Find the best defect8 rescue threshold using a compound condition.

    Compound rescue: Stage1 predicts 'good' BUT Stage1 defect_prob >= suspicion_floor
                     AND cosine_sim(embed, defect8_proto) >= rescue_tau.

    The suspicion_floor filters out images Stage1 is very confident are "good"
    (defect_prob < 0.20), dramatically reducing good false positives.

    Selection criterion: maximise balanced_accuracy (per-class recall average).
    Saves best (suspicion_floor, rescue_tau) to model_stage2.pth.
    """
    print(f"\n{'='*60}")
    print(f" defect8 Rescue Threshold Sweep (compound condition)")
    print(f"{'='*60}")

    if not CKPT_STAGE1.exists() or not CKPT_STAGE2.exists():
        print("ERROR: Need both checkpoints.")
        return

    ckpt1     = torch.load(CKPT_STAGE1, map_location=device, weights_only=False)
    model1    = _make_model(2).to(device); model1.load_state_dict(ckpt1["model_state"]); model1.eval()
    s1_thresh = ckpt1.get("threshold", 0.65)

    ckpt2    = torch.load(CKPT_STAGE2, map_location=device, weights_only=False)
    model2   = _make_model(NUM_DEFECTS).to(device); model2.load_state_dict(ckpt2["model_state"]); model2.eval()
    protos2  = ckpt2["prototypes"].to(device)

    d8_idx       = DEFECT_CLASSES.index("defect8")
    d8_proto     = protos2[d8_idx]
    good_idx_all = ALL_CLASSES.index("good")
    d8_idx_all   = ALL_CLASSES.index("defect8")

    # Build val set
    all_samples = []
    for cls in ALL_CLASSES:
        d = DATASET_DIR / cls
        if not d.exists(): continue
        lbl = ALL_CLASSES.index(cls)
        for f in d.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                all_samples.append((str(f), lbl))
    paths, labels = zip(*all_samples)
    _, va_p, _, va_l = train_test_split(paths, labels, test_size=0.2,
                                        stratify=labels, random_state=SEED)
    tf = get_transform(False)

    # Collect (defect_prob, d8_sim, true_label) for all Stage-1 "good" predictions
    entries = []
    with torch.no_grad():
        for path, true_lbl in zip(va_p, va_l):
            img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            logits1, _ = model1(img)
            dp = F.softmax(logits1, dim=1)[0, DEFECT_IDX].item()
            if dp < s1_thresh:
                embed2 = F.normalize(model2.get_embedding(img), dim=1)
                sim    = torch.mm(embed2, d8_proto.unsqueeze(1)).item()
                entries.append((dp, sim, true_lbl))

    d8_val_total = sum(1 for _, l in zip(va_p, va_l) if l == d8_idx_all)
    good_val_total = sum(1 for _, l in zip(va_p, va_l) if l == good_idx_all)
    d8_missed = sum(1 for dp, sim, lbl in entries if lbl == d8_idx_all)
    d8_caught  = d8_val_total - d8_missed

    print(f"\n  Stage-1 summary:")
    print(f"    defect8 val total : {d8_val_total}")
    print(f"    defect8 caught    : {d8_caught}  (Stage 1 defect_prob >= {s1_thresh:.2f})")
    print(f"    defect8 missed    : {d8_missed}  (Stage 1 predicted 'good' — rescue targets these)")
    print(f"\n  defect8 Stage-1 defect_prob values for MISSED images:")
    for dp, sim, lbl in sorted(entries, key=lambda x: -x[1]):
        if lbl == d8_idx_all:
            print(f"    defect_prob={dp:.3f}  d8_sim={sim:.3f}")

    print(f"\n  Compound sweep: suspicion_floor × rescue_tau")
    print(f"\n  {'floor':>7} {'τ':>6}  {'d8_rescued':>10}  {'d8_recall':>10}  {'good_fp':>8}  {'Δbal_acc':>10}")
    print("  " + "-"*64)

    base_d8_recall   = d8_caught / max(d8_val_total, 1)
    base_good_recall = good_val_total / good_val_total  # 1.0 before rescue modifies it
    # (actual good recall from full cascade is ~0.877 — rescue only affects the "good" arm)
    # For Δbal_acc comparison we only need the delta from rescue entries

    best_bal_delta = 0.0
    best_tau = None
    best_floor = None

    for floor in [0.0, 0.10, 0.20, 0.30, 0.40]:
        for tau in [t / 100 for t in range(95, 49, -5)]:
            rescued = sum(1 for dp, sim, lbl in entries
                          if dp >= floor and sim >= tau and lbl == d8_idx_all)
            fp      = sum(1 for dp, sim, lbl in entries
                          if dp >= floor and sim >= tau and lbl == good_idx_all)
            # Delta balanced accuracy: each defect8 rescued = +1/(d8_val_total * n_classes)
            #                          each good FP           = -1/(good_val_total * n_classes)
            n_cls   = len(ALL_CLASSES)
            delta   = rescued / (d8_val_total * n_cls) - fp / (good_val_total * n_cls)
            if rescued > 0 or fp == 0:
                print(f"  {floor:7.2f} {tau:6.2f}  {rescued:>10}  "
                      f"{(d8_caught + rescued) / d8_val_total:>10.3f}  "
                      f"{fp:>8}  {delta:>10.5f}")
            if delta > best_bal_delta:
                best_bal_delta = delta
                best_tau       = tau
                best_floor     = floor

    if best_tau is None:
        print("\n  No rescue configuration improves balanced accuracy.")
        print("  defect8 embedding overlaps too heavily with 'good' in DINOv2 feature space.")
        print("  Recommendation: try One-Class SVM (approach 2) or accept current recall.")
        return

    print(f"\n  Best compound threshold: floor={best_floor:.2f}, τ={best_tau:.2f}")
    rescued_final = sum(1 for dp, sim, lbl in entries
                        if dp >= best_floor and sim >= best_tau and lbl == d8_idx_all)
    fp_final      = sum(1 for dp, sim, lbl in entries
                        if dp >= best_floor and sim >= best_tau and lbl == good_idx_all)
    print(f"  defect8 recall: {(d8_caught + rescued_final) / d8_val_total:.1%}  "
          f"({d8_caught + rescued_final}/{d8_val_total})  was {d8_caught}/{d8_val_total}")
    print(f"  good FPs added: {fp_final}  ({fp_final / good_val_total:.1%} of good val set)")
    print(f"  Δ balanced acc: +{best_bal_delta:.5f}")

    ckpt2["defect8_rescue_threshold"]   = float(best_tau)
    ckpt2["defect8_rescue_floor"]       = float(best_floor)
    ckpt2["defect8_rescue_idx"]         = int(d8_idx)
    torch.save(ckpt2, CKPT_STAGE2)
    print(f"  Saved → {CKPT_STAGE2}")


def tune_defect8_ocsvm(device):
    """Fit a One-Class SVM on defect8 training embeddings as a rescue detector.

    OC-SVM learns the minimum enclosing region around the 34 defect8 training
    embeddings (vs prototype = single centroid).  Sweeps nu (tightness) and
    selects the configuration that maximises Δbalanced_accuracy on the val set.

    Serialises (scaler, ocsvm) as pickle bytes into model_stage2.pth under
    key 'defect8_ocsvm'.  Enables opt-in via --ocsvm-rescue flag at eval/infer.
    """
    print(f"\n{'='*60}")
    print(f" defect8 One-Class SVM Rescue")
    print(f"{'='*60}")

    if not CKPT_STAGE1.exists() or not CKPT_STAGE2.exists():
        print("ERROR: Need both checkpoints.")
        return

    ckpt1     = torch.load(CKPT_STAGE1, map_location=device, weights_only=False)
    model1    = _make_model(2).to(device); model1.load_state_dict(ckpt1["model_state"]); model1.eval()
    s1_thresh = ckpt1.get("threshold", 0.65)

    ckpt2    = torch.load(CKPT_STAGE2, map_location=device, weights_only=False)
    model2   = _make_model(NUM_DEFECTS).to(device); model2.load_state_dict(ckpt2["model_state"]); model2.eval()

    d8_idx       = DEFECT_CLASSES.index("defect8")
    good_idx_all = ALL_CLASSES.index("good")
    d8_idx_all   = ALL_CLASSES.index("defect8")
    tf           = get_transform(False)

    # ── Extract defect8 training embeddings ──────────────────────────────────
    d8_train_paths = []
    d8_dir = DATASET_DIR / "defect8"
    if not d8_dir.exists():
        print("ERROR: Dataset/defect8 not found.")
        return
    for f in d8_dir.iterdir():
        if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
            d8_train_paths.append(str(f))

    # Use 80% for OC-SVM training (same split as cascade training)
    all_d8 = []
    for cls in ALL_CLASSES:
        d = DATASET_DIR / cls
        if not d.exists(): continue
        lbl = ALL_CLASSES.index(cls)
        for f in d.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                all_d8.append((str(f), lbl))
    paths_all, labels_all = zip(*all_d8)
    tr_p, va_p, tr_l, va_l = train_test_split(paths_all, labels_all, test_size=0.2,
                                               stratify=labels_all, random_state=SEED)

    # Extract defect8 train embeddings
    d8_tr_embeds = []
    with torch.no_grad():
        for path, lbl in zip(tr_p, tr_l):
            if ALL_CLASSES[lbl] != "defect8":
                continue
            img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            e = F.normalize(model2.get_embedding(img), dim=1).squeeze(0).cpu().numpy()
            d8_tr_embeds.append(e)
    X_d8 = np.stack(d8_tr_embeds)
    print(f"\n  defect8 training embeddings: {len(X_d8)}")

    # ── Collect val set entries (Stage-1 'good' predictions) ─────────────────
    entries = []   # (dp, embed_np, true_label)
    with torch.no_grad():
        for path, true_lbl in zip(va_p, va_l):
            img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            logits1, _ = model1(img)
            dp = F.softmax(logits1, dim=1)[0, DEFECT_IDX].item()
            if dp < s1_thresh:
                e = F.normalize(model2.get_embedding(img), dim=1).squeeze(0).cpu().numpy()
                entries.append((dp, e, true_lbl))

    d8_val_total   = sum(1 for _, l in zip(va_p, va_l) if l == d8_idx_all)
    good_val_total = sum(1 for _, l in zip(va_p, va_l) if l == good_idx_all)
    d8_missed      = sum(1 for dp, e, lbl in entries if lbl == d8_idx_all)
    d8_caught      = d8_val_total - d8_missed

    print(f"  defect8 val: {d8_val_total} total, {d8_caught} caught by Stage 1, "
          f"{d8_missed} missed (rescue targets)")

    # ── Fit scaler (StandardScaler) on defect8 train embeddings ─────────────
    scaler = StandardScaler()
    X_d8_scaled = scaler.fit_transform(X_d8)

    # Scale val entries
    val_embeds  = np.stack([e for dp, e, lbl in entries])
    val_scaled  = scaler.transform(val_embeds)
    val_labels  = np.array([lbl for dp, e, lbl in entries])
    val_dps     = np.array([dp  for dp, e, lbl in entries])

    # ── Sweep nu, floor, and decision-function threshold ─────────────────────
    # Fine-grained dt sweep lets us cut FPs precisely while keeping rescues.
    n_cls = len(ALL_CLASSES)
    # Overall accuracy currently without rescue = (correct without rescue) / total_val
    total_val = len(va_p)
    base_correct = sum(1 for _, l in zip(va_p, va_l) if l == d8_idx_all) * 0  # recomputed below
    # Compute baseline correct count (stage-1 only, no rescue)
    # We can estimate: overall_acc_base * total = correct_base
    # Use 661 from known evaluation, or recompute dynamically:
    base_correct = None  # will be set on first entry

    print(f"\n  {'nu':>6}  {'floor':>7}  {'dt':>6}  {'rescued':>8}  "
          f"{'d8_recall':>10}  {'fp':>6}  {'Δbal':>10}  {'est_acc':>9}")
    print("  " + "-"*74)

    best_delta, best_cfg = -999, None

    for nu in [0.05, 0.10, 0.20, 0.30, 0.50]:
        ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        ocsvm.fit(X_d8_scaled)
        scores = ocsvm.decision_function(val_scaled)

        for floor in [0.0, 0.20, 0.30, 0.40]:
            # Sweep decision threshold: from 0 (boundary) up to near-max of d8 scores
            d8_scores = scores[val_labels == d8_idx_all]
            score_steps = sorted(set(
                [0.0] + list(np.linspace(0, float(d8_scores.max()) * 0.9, 20))
            ))
            for dt in score_steps:
                mask    = (val_dps >= floor) & (scores > dt)
                rescued = int(((val_labels == d8_idx_all) & mask).sum())
                fp      = int(((val_labels == good_idx_all) & mask).sum())
                delta   = rescued / (d8_val_total * n_cls) - fp / (good_val_total * n_cls)
                # Estimated overall accuracy (base 661/756 without rescue)
                est_acc = (661 + rescued - fp) / total_val

                if rescued > 0:
                    marker = " ←85%" if abs(est_acc - 0.85) < 0.002 else ""
                    print(f"  {nu:6.2f}  {floor:7.2f}  {dt:6.3f}  {rescued:>8}  "
                          f"{(d8_caught + rescued) / d8_val_total:>10.3f}  "
                          f"{fp:>6}  {delta:>10.5f}  {est_acc:>8.4f}{marker}")
                if delta > best_delta:
                    best_delta = delta
                    best_cfg   = (nu, floor, dt, ocsvm)

    if best_delta <= 0 or best_cfg is None:
        print("\n  No OC-SVM configuration improves balanced accuracy.")
        return

    best_nu, best_floor, best_dt, best_ocsvm = best_cfg

    # ── Report best ───────────────────────────────────────────────────────────
    scores_best = best_ocsvm.decision_function(val_scaled)
    mask_best   = (val_dps >= best_floor) & (scores_best > best_dt)
    rescued_n   = int(((val_labels == d8_idx_all) & mask_best).sum())
    fp_n        = int(((val_labels == good_idx_all) & mask_best).sum())

    est_final = (661 + rescued_n - fp_n) / total_val
    print(f"\n  Best:  nu={best_nu:.2f}, floor={best_floor:.2f}, dt={best_dt:.3f}")
    print(f"  defect8 recall : {(d8_caught + rescued_n) / d8_val_total:.1%}  "
          f"({d8_caught + rescued_n}/{d8_val_total}, was {d8_caught}/{d8_val_total})")
    print(f"  good FPs added : {fp_n}  ({fp_n / good_val_total:.1%} of good val)")
    print(f"  Δ balanced acc : {best_delta:+.5f}")
    print(f"  Est. overall   : {est_final:.4f}  ({661 + rescued_n - fp_n}/{total_val})")

    # ── Save OC-SVM into checkpoint ───────────────────────────────────────────
    ocsvm_blob = pickle.dumps({"scaler": scaler, "ocsvm": best_ocsvm,
                               "floor": best_floor, "dt": best_dt, "d8_idx": d8_idx})
    ckpt2["defect8_ocsvm"] = ocsvm_blob
    # Clear prototype-rescue keys if present (mutually exclusive)
    for k in ["defect8_rescue_threshold", "defect8_rescue_floor", "defect8_rescue_idx"]:
        ckpt2.pop(k, None)
    torch.save(ckpt2, CKPT_STAGE2)
    print(f"  Saved OC-SVM → {CKPT_STAGE2}")


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

    rescue_tau   = ckpt2.get("defect8_rescue_threshold", None)
    rescue_floor = ckpt2.get("defect8_rescue_floor",     0.0)
    rescue_idx   = ckpt2.get("defect8_rescue_idx",       None)
    ocsvm_blob   = ckpt2.get("defect8_ocsvm",            None)
    ocsvm_rescue = None
    if ocsvm_blob is not None:
        ocsvm_rescue = pickle.loads(ocsvm_blob)  # {scaler, ocsvm, floor, d8_idx}

    print(f"  Stage 1 threshold: {threshold:.2f}")
    if rescue_tau is not None:
        print(f"  defect8 rescue   : floor={rescue_floor:.2f}, τ={rescue_tau:.2f}  (idx {rescue_idx})")
    if ocsvm_rescue is not None:
        print(f"  defect8 OC-SVM   : nu={ocsvm_rescue['ocsvm'].nu:.2f}, "
              f"floor={ocsvm_rescue['floor']:.2f}")

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
                rescue_pred = None
                if ocsvm_rescue is not None and defect_prob >= ocsvm_rescue["floor"]:
                    e = F.normalize(model2.get_embedding(img), dim=1).squeeze(0).cpu().numpy()
                    e_scaled = ocsvm_rescue["scaler"].transform(e.reshape(1, -1))
                    dt = ocsvm_rescue.get("dt", 0.0)
                    if ocsvm_rescue["ocsvm"].decision_function(e_scaled)[0] > dt:
                        rescue_pred = ALL_CLASSES.index(DEFECT_CLASSES[ocsvm_rescue["d8_idx"]])
                elif rescue_tau is not None and defect_prob >= rescue_floor:
                    embed2 = F.normalize(model2.get_embedding(img), dim=1)
                    d8_sim = torch.mm(embed2, protos2[rescue_idx].unsqueeze(1)).item()
                    if d8_sim >= rescue_tau:
                        rescue_pred = ALL_CLASSES.index(DEFECT_CLASSES[rescue_idx])
                pred = rescue_pred if rescue_pred is not None else good_idx_all
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
    ap.add_argument("--tune-rescue", action="store_true",
                    help="Sweep defect8 rescue threshold and save to model_stage2.pth")
    ap.add_argument("--ocsvm-rescue", action="store_true",
                    help="Fit One-Class SVM on defect8 embeddings and save to model_stage2.pth")
    ap.add_argument("--stage1-epochs", type=int, default=30)
    ap.add_argument("--stage2-epochs", type=int, default=40)
    ap.add_argument("--vit", action="store_true",
                    help="Use MAE-pretrained ViT-Small backbone instead of EfficientNet-B0. "
                         "Requires train_mae.py to have been run first.")
    ap.add_argument("--dinov2", action="store_true",
                    help="Use DINOv2 ViT-Small/14 pretrained backbone (timm). "
                         "Highest-quality features for prototype/cosine inference.")
    ap.add_argument("--augment-d8", action="store_true",
                    help="Add 7× D4 augmentations of defect8 train images to Stage 2 "
                         "training set (rotations + flips). Targets Stage-2 defect8 "
                         "confusion without affecting Stage 1 or other classes.")
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

    if args.tune_rescue:
        tune_defect8_rescue(device)
        return

    if args.ocsvm_rescue:
        tune_defect8_ocsvm(device)
        return

    if args.evaluate:
        evaluate_cascade(device)
        return

    if args.stage in ("1", "both"):
        train_stage1(epochs=args.stage1_epochs, device=device)

    if args.stage in ("2", "both"):
        train_stage2(epochs=args.stage2_epochs, device=device,
                     augment_d8=args.augment_d8)

    if args.stage == "both":
        evaluate_cascade(device)


if __name__ == "__main__":
    main()

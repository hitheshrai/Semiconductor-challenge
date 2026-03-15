#!/usr/bin/env python
"""
train.py  –  Small-Sample Defect Classifier Training
              ASU / Intel Semiconductor Solutions Challenge 2026

Strategy
────────
1. Phase 1 (warm-up, ~5 epochs):
   Backbone frozen; only train embedding head + classifier.
   Fast convergence even on CPU.

2. Phase 2 (fine-tune, ~20 epochs):
   Unfreeze last 3 EfficientNet-B0 blocks.
   Lower LR on backbone, higher on head.

Imbalance handling
──────────────────
• WeightedRandomSampler  – each batch sees all classes roughly equally
• Class-weighted CE loss  – penalises misses on rare defect classes more
• Heavy augmentation      – virtually expands the tiny defect sets
• Label smoothing ε=0.1  – prevents over-fitting on small classes

Few-shot support
────────────────
After training, class prototypes (mean L2-normalised embeddings) are saved
in the checkpoint.  New unseen defect types can be registered at inference
time by supplying ≥1 labelled examples — no re-training required.
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
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── add parent dir so model.py is importable when run from solution/ ──────────
sys.path.insert(0, str(Path(__file__).parent))
from model import DefectClassifier, compute_prototypes, EMBED_DIM

# ─────────────────────────────────────────────────────────────────────────────
# Paths & hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("../Dataset")       # relative to solution/
OUTPUT_DIR  = Path("output")
CHECKPOINT  = OUTPUT_DIR / "model_best.pth"

IMG_SIZE   = 224
BATCH_SIZE = 32   # larger batch = faster on CPU
PHASE1_EP  = 20   # frozen backbone — fast on CPU (~20–30 min total)
PHASE2_EP  = 40   # unfreeze last 3 blocks — only used with --finetune
LR_HEAD    = 3e-4
LR_BACK    = 3e-5  # 10× lower for backbone
SEED       = 42

CLASSES    = ["defect1", "defect2", "defect3", "defect4", "defect5", "defect8", "defect9", "defect10", "good"]
NUM_CLASSES = len(CLASSES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class DefectDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples   # list of (path_str, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_samples(dataset_dir: Path):
    samples = []
    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF"}:
                samples.append((str(f), CLASS2IDX[cls]))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
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
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_weighted_sampler(labels):
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def class_weights_tensor(labels, num_classes, device, max_ratio: float = 15.0):
    """Inverse-frequency class weights, capped so no class is penalised
    more than ``max_ratio`` × the majority class.

    WHY: WeightedRandomSampler already equalises batch frequency across
    classes, so the loss weight only needs a *mild* additional nudge.
    Previous runs showed:
      - 476:1 raw weights → model never predicted "good" (0% recall)
      - 5:1 cap + sampler → model over-predicted defects (88% of good wrong)
    Sampler removed: model now sees true distribution (~95% good). A 15:1 cap
    ensures defect losses dominate enough to learn rare classes without
    causing the model to over-predict them.
    """
    counts = Counter(labels)
    total  = len(labels)
    raw    = [total / (num_classes * max(counts.get(i, 1), 1)) for i in range(num_classes)]
    min_w  = min(raw)
    capped = [min(wi, min_w * max_ratio) for wi in raw]
    return torch.tensor(capped, dtype=torch.float32, device=device)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss = total_correct = total = 0
    all_preds, all_labels = [], []

    ctx = torch.no_grad() if not train else torch.enable_grad()
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

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / total, total_correct / total, bal_acc, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train defect classifier")
    ap.add_argument("--finetune", action="store_true",
                    help="Also run Phase 2 (unfreeze backbone). Recommended only with GPU.")
    ap.add_argument("--phase2-only", action="store_true",
                    help="Skip Phase 1, load existing checkpoint, run Phase 2 only.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override Phase 1 epochs (default: 20)")
    ap.add_argument("--phase2-epochs", type=int, default=None,
                    help="Override Phase 2 epochs (default: 20)")
    args = ap.parse_args()

    if args.epochs:
        global PHASE1_EP
        PHASE1_EP = args.epochs
    if args.phase2_epochs:
        global PHASE2_EP
        PHASE2_EP = args.phase2_epochs

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.phase2_only:
        mode_str = "Phase 2 only (load checkpoint, fine-tune backbone)"
    elif args.finetune:
        mode_str = "Phase 1 + Phase 2 (full fine-tune)"
    else:
        mode_str = "Phase 1 only (frozen backbone)"
    print(f"Mode:   {mode_str}")

    # ── Load samples ────────────────────────────────────────────────────────
    ds_dir  = DATASET_DIR
    samples = load_samples(ds_dir)
    if len(samples) == 0:
        # Try absolute path fallback
        ds_dir  = Path("C:/Users/hithe/Downloads/Dataset/Dataset")
        samples = load_samples(ds_dir)

    print(f"\nTotal samples: {len(samples)}")
    counts = Counter(lbl for _, lbl in samples)
    for cls, idx in CLASS2IDX.items():
        print(f"  {cls:12s}: {counts.get(idx, 0):4d}")

    # ── Stratified split ─────────────────────────────────────────────────────
    paths, labels = zip(*samples)
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    train_samples = list(zip(tr_p, tr_l))
    val_samples   = list(zip(va_p, va_l))
    print(f"\nTrain: {len(train_samples)}  Val: {len(val_samples)}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    tr_ds = DefectDataset(train_samples, get_transform(True))
    va_ds = DefectDataset(val_samples,   get_transform(False))

    tr_loader = DataLoader(
        tr_ds, batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, pin_memory=False,
    )
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DefectClassifier(NUM_CLASSES, EMBED_DIM).to(device)
    cw    = class_weights_tensor(tr_l, NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    best_metric = 0.0
    history = {k: [] for k in ("tr_loss", "va_loss", "tr_acc", "va_acc", "va_bal")}
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4,
    )

    if args.phase2_only:
        # ── Load existing Phase 1 checkpoint, skip to Phase 2 ────────────
        if not CHECKPOINT.exists():
            print(f"ERROR: No checkpoint at {CHECKPOINT}. Run Phase 1 first.")
            return
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_metric = ckpt.get("val_acc", ckpt.get("val_bal_acc", 0.0))
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, val_acc={best_metric:.4f})")
    else:
        # ════════════════════════════════════════════════════════════════════
        # Phase 1 — frozen backbone, train head only  (fast)
        # ════════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f" Phase 1: frozen backbone  ({PHASE1_EP} epochs)")
        print(f"{'='*60}")
        model.freeze_backbone()

        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_EP, eta_min=1e-6)
        for ep in range(PHASE1_EP):
            t0 = time.time()
            tr_loss, tr_acc, _, _, _        = run_epoch(model, tr_loader, opt1, criterion, device, True)
            va_loss, va_acc, va_bal, vp, vl = run_epoch(model, va_loader, opt1, criterion, device, False)
            sched1.step()

            _log(history, tr_loss, va_loss, tr_acc, va_acc, va_bal)
            print(f"  ep {ep+1:02d}/{PHASE1_EP}  loss {tr_loss:.4f}/{va_loss:.4f}  "
                  f"acc {tr_acc:.3f}/{va_acc:.3f}  bal_acc {va_bal:.3f}  "
                  f"({time.time()-t0:.1f}s)")

            if va_acc > best_metric:
                best_metric = va_acc
                _save(model, ep, va_acc, CHECKPOINT, train_samples, device)
                print(f"    ✓ New best val-acc = {va_acc:.4f}")

        if not args.finetune:
            print("\nSkipping Phase 2 (backbone remains frozen).")
            print("Run with --finetune to unlock Phase 2 (recommended: GPU only).")
            _finish(model, history, best_metric, CHECKPOINT, train_samples, device,
                    va_loader, opt1, criterion, all_samples=samples)
            return

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2 — unfreeze last 3 blocks, fine-tune
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f" Phase 2: fine-tune last 3 blocks  ({PHASE2_EP} epochs)")
    print(f"{'='*60}")
    model.unfreeze_backbone(last_n_blocks=3)

    opt2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACK},
        {"params": model.embed_head.parameters(), "lr": LR_HEAD},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=1e-6)

    for ep in range(PHASE2_EP):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _        = run_epoch(model, tr_loader, opt2, criterion, device, True)
        va_loss, va_acc, va_bal, vp, vl = run_epoch(model, va_loader, opt2, criterion, device, False)
        sched2.step()

        _log(history, tr_loss, va_loss, tr_acc, va_acc, va_bal)
        print(f"  ep {ep+1:02d}/{PHASE2_EP}  loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc:.3f}/{va_acc:.3f}  bal_acc {va_bal:.3f}  "
              f"({time.time()-t0:.1f}s)")

        if va_acc > best_metric:
            best_metric = va_acc
            _save(model, PHASE1_EP + ep, va_acc, CHECKPOINT, train_samples, device)
            print(f"    ✓ New best val-acc = {va_acc:.4f}")

    print(f"\nBest val accuracy: {best_metric:.4f}")
    _finish(model, history, best_metric, CHECKPOINT, train_samples, device,
            va_loader, opt2, criterion, all_samples=samples)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _finish(model, history, best_metric, checkpoint, train_samples, device,
            va_loader, optimizer, criterion, all_samples):
    """Load best checkpoint, print report, save plots."""
    print(f"\nBest val accuracy: {best_metric:.4f}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, _, _, final_preds, final_labels = run_epoch(
        model, va_loader, optimizer, criterion, device, False
    )
    print("\nClassification Report (validation set):")
    print(classification_report(final_labels, final_preds, target_names=CLASSES, digits=3))

    ckpt["history"] = history
    torch.save(ckpt, checkpoint)

    plot_training_history(history, OUTPUT_DIR)
    plot_confusion_matrix(final_labels, final_preds, CLASSES, OUTPUT_DIR)
    plot_class_accuracy(final_labels, final_preds, CLASSES, all_samples, OUTPUT_DIR)
    plot_learning_curve(model, all_samples, device, OUTPUT_DIR)
    print(f"\nAll outputs saved to {OUTPUT_DIR.resolve()}")


def _log(h, tl, vl, ta, va, vb):
    h["tr_loss"].append(tl); h["va_loss"].append(vl)
    h["tr_acc"].append(ta);  h["va_acc"].append(va); h["va_bal"].append(vb)


def _save(model, epoch, val_acc, path, train_samples, device):
    """Save model + prototypes computed from training data."""
    tr_ds     = DefectDataset(train_samples, get_transform(False))
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=False, num_workers=0)
    protos    = compute_prototypes(model, tr_loader, device, NUM_CLASSES, EMBED_DIM)

    counts = Counter(l for _, l in train_samples)
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "val_acc":      val_acc,
        "classes":      CLASSES,
        "class2idx":    CLASS2IDX,
        "prototypes":   protos.cpu(),
        "class_counts": {cls: counts.get(CLASS2IDX[cls], 0) for cls in CLASSES},
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Plots (Deliverables 2 & 4)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    eps = range(1, len(history["tr_loss"]) + 1)

    a1.plot(eps, history["tr_loss"], label="Train"); a1.plot(eps, history["va_loss"], label="Val")
    a1.set(xlabel="Epoch", ylabel="Loss", title="Training / Validation Loss"); a1.legend(); a1.grid(alpha=.3)

    a2.plot(eps, history["tr_acc"],  label="Train acc")
    a2.plot(eps, history["va_acc"],  label="Val acc")
    a2.plot(eps, history["va_bal"],  label="Val balanced acc", linestyle="--")
    a2.axhline(0.85, color="red", linestyle=":", linewidth=1.5, label="85 % target")
    a2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Curves"); a2.legend(); a2.grid(alpha=.3)

    plt.tight_layout()
    p = out / "plot1_training_history.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")


def plot_confusion_matrix(labels, preds, classes, out):
    cm      = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set(xlabel="Predicted", ylabel="True", title="Normalised Confusion Matrix")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    p = out / "plot2_confusion_matrix.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")


def plot_class_accuracy(labels, preds, classes, all_samples, out):
    """Deliverable 2: accuracy vs class occurrence (both charts on one figure)."""
    lbl_list  = list(labels)
    pred_list = list(preds)
    counts_all = Counter(l for _, l in all_samples)

    cls_accs, cls_occ = [], []
    for i in range(len(classes)):
        mask = [j for j, l in enumerate(lbl_list) if l == i]
        acc  = sum(1 for j in mask if pred_list[j] == i) / max(len(mask), 1)
        cls_accs.append(acc)
        cls_occ.append(counts_all.get(i, 0))

    colors = ["#d32f2f" if a < 0.70 else "#f57c00" if a < 0.85 else "#388e3c"
              for a in cls_accs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = ax1.bar(classes, cls_accs, color=colors, edgecolor="black", linewidth=0.6)
    ax1.axhline(0.85, color="navy", linestyle="--", linewidth=1.5, label="85 % target")
    ax1.set_ylim(0, 1.12); ax1.set_ylabel("Accuracy"); ax1.legend()
    ax1.set_title("Per-Class Classification Accuracy")
    for bar, acc in zip(bars, cls_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{acc:.2f}", ha="center", fontsize=10, fontweight="bold")

    # Scatter: accuracy vs occurrence
    ax2.scatter(cls_occ, cls_accs, s=120, c=colors, edgecolors="black", zorder=5)
    for i, cls in enumerate(classes):
        ax2.annotate(cls, (cls_occ[i], cls_accs[i]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax2.axhline(0.85, color="navy", linestyle="--", linewidth=1.5, label="85 % target")
    ax2.set_xscale("log"); ax2.set_xlabel("Class Occurrence (log scale)")
    ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy vs. Class Occurrence")
    ax2.legend(); ax2.grid(alpha=.3)

    plt.tight_layout()
    p = out / "plot3_class_accuracy_vs_occurrence.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")


def plot_learning_curve(model, all_samples, device, out):
    """
    Deliverable 4: demonstrate how quickly the model learns.
    We hold out 40 % of defect images as a query set, then build
    prototypes from n = 1,2,3,5,8,… support samples per class.
    """
    val_tf = get_transform(False)

    defect_samples = [(p, l) for p, l in all_samples if CLASSES[l] != "good"]
    random.shuffle(defect_samples)
    split   = max(int(0.6 * len(defect_samples)), NUM_CLASSES)
    support = defect_samples[:split]
    query   = defect_samples[split:]

    good_samp = [(p, l) for p, l in all_samples if CLASSES[l] == "good"]
    random.shuffle(good_samp)
    query += good_samp[:len(query)]   # balance query with good samples

    by_class = {}
    for p, l in support:
        by_class.setdefault(l, []).append(p)

    max_shot = min(max(len(v) for v in by_class.values()), 20)
    shot_counts = [n for n in [1, 2, 3, 5, 8, 12, 20] if n <= max_shot]

    model.eval()
    accs = []

    for n_shot in shot_counts:
        proto_sum = torch.zeros(NUM_CLASSES, EMBED_DIM, device=device)
        proto_cnt = torch.zeros(NUM_CLASSES, device=device)

        # "Good" prototype from up to 200 good images
        for p, l in good_samp[:200]:
            img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                e = model.get_embedding(img).squeeze(0)
            proto_sum[l] += e; proto_cnt[l] += 1

        # Defect prototypes with n_shot per class
        for lbl, paths in by_class.items():
            for p in paths[:n_shot]:
                img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = model.get_embedding(img).squeeze(0)
                proto_sum[lbl] += e; proto_cnt[lbl] += 1

        proto_cnt = proto_cnt.clamp(min=1e-6)
        protos = F.normalize(proto_sum / proto_cnt.unsqueeze(1), dim=1)

        correct = 0
        for p, true_l in query:
            img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                e    = model.get_embedding(img)
                pred = torch.mm(e, protos.T).argmax(1).item()
            correct += (pred == true_l)
        acc = correct / len(query)
        accs.append(acc)
        print(f"  {n_shot:2d}-shot  acc={acc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shot_counts, [a * 100 for a in accs], "bo-", linewidth=2, markersize=8)
    ax.axhline(85, color="red", linestyle="--", linewidth=1.5, label="85 % target")
    ax.fill_between(shot_counts, [a * 100 for a in accs], 0, alpha=0.1, color="blue")
    ax.set(xlabel="Support examples per defect class  (N-shot)",
           ylabel="Accuracy  (%)",
           title="Few-Shot Learning Curve  –  Accuracy vs. Examples per Class",
           ylim=(0, 105))
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    p = out / "plot4_few_shot_learning_curve.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()

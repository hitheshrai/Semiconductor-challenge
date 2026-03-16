#!/usr/bin/env python
"""
evaluate.py  –  Full evaluation suite
                ASU / Intel Semiconductor Solutions Challenge 2026

Produces all competition deliverable plots:
  plot1_training_history.png          (training curves — single model only)
  plot2_confusion_matrix.png          (normalised confusion matrix)
  plot3_class_accuracy_vs_occurrence.png (deliverable 2)
  plot4_few_shot_learning_curve.png   (deliverable 4)
  plot5_roc_curves.png                (per-class ROC / AUC)
  plot6_tsne_embeddings.png           (2D t-SNE of learned feature space)

Usage
─────
  python evaluate.py                             # single EfficientNet model
  python evaluate.py --checkpoint output/model_best.pth
  python evaluate.py --cascade                   # DINOv2 two-stage cascade (best)
  python evaluate.py --cascade --no-tta          # cascade without TTA (faster)
"""

import sys, random, json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, roc_curve, auc,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from model import DefectClassifier, compute_prototypes, EMBED_DIM

# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path(__file__).parent.parent / "Dataset"
CHECKPOINT   = Path(__file__).parent / "output" / "model_best.pth"
CKPT_STAGE1  = Path(__file__).parent / "output" / "model_stage1.pth"
CKPT_STAGE2  = Path(__file__).parent / "output" / "model_stage2.pth"
OUTPUT_DIR   = Path(__file__).parent / "output"
IMG_SIZE     = 224
SEED         = 42
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Class ordering used by the cascade (matches train_cascade.py)
_CASCADE_CLASSES  = ["defect1", "defect2", "defect3", "defect4", "defect5",
                     "defect8", "defect9", "defect10", "good"]
_DEFECT_CLASSES   = [c for c in _CASCADE_CLASSES if c != "good"]
_GOOD_IDX_ALL     = _CASCADE_CLASSES.index("good")   # 8
_DEFECT_IDX_BIN   = 1                                  # Stage 1: good=0, defective=1

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


class SimpleDataset(Dataset):
    def __init__(self, samples, tf=None):
        self.samples = samples
        self.tf = tf or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), l


def _make_tta_transforms():
    """4 deterministic TTA variants: identity + h-flip + v-flip + h+v-flip.

    Flip-only TTA matches the training augmentation (RandomHorizontalFlip +
    RandomVerticalFlip). Full 90° rotation TTA is out-of-distribution for
    Stage 2 (trained with RandomRotation(30) only) and degrades embeddings.
    """
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    norm = [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    hf = transforms.RandomHorizontalFlip(p=1.0)
    vf = transforms.RandomVerticalFlip(p=1.0)
    return [
        transforms.Compose(base + norm),                   # identity
        transforms.Compose(base + [hf] + norm),            # h-flip
        transforms.Compose(base + [vf] + norm),            # v-flip
        transforms.Compose(base + [hf, vf] + norm),        # h+v-flip
    ]


def _auto_load_model(num_classes, state_dict, device):
    """Detect backbone from checkpoint key names and load the correct model class."""
    keys = "".join(state_dict.keys())
    if "backbone.cls_token" in keys:
        if "backbone.blocks.0.ls1" in keys:
            from model_dinov2 import DINOv2DefectClassifier
            m = DINOv2DefectClassifier(num_classes, EMBED_DIM).to(device)
        else:
            from model_vit import ViTDefectClassifier
            m = ViTDefectClassifier(num_classes, EMBED_DIM, mae_ckpt=None).to(device)
    else:
        m = DefectClassifier(num_classes, EMBED_DIM).to(device)
    m.load_state_dict(state_dict)
    m.eval()
    return m


def load_cascade_and_data(stage1_path=CKPT_STAGE1, stage2_path=CKPT_STAGE2):
    """Load Stage 1 + Stage 2 cascade models and reproduce the training val split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt1 = torch.load(stage1_path, map_location=device, weights_only=False)
    model1 = _auto_load_model(2, ckpt1["model_state"], device)
    threshold = ckpt1.get("threshold", 0.65)

    ckpt2 = torch.load(stage2_path, map_location=device, weights_only=False)
    model2 = _auto_load_model(len(_DEFECT_CLASSES), ckpt2["model_state"], device)
    prototypes = ckpt2["prototypes"].to(device)
    rescue_tau   = ckpt2.get("defect8_rescue_threshold", None)
    rescue_floor = ckpt2.get("defect8_rescue_floor",     0.0)
    rescue_idx   = ckpt2.get("defect8_rescue_idx",       None)

    # Rebuild dataset with ALL_CLASSES ordering from train_cascade.py
    all_samples = []
    for cls in _CASCADE_CLASSES:
        d = DATASET_DIR / cls
        if not d.exists():
            continue
        lbl = _CASCADE_CLASSES.index(cls)
        for f in d.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                all_samples.append((str(f), lbl))

    paths, labels = zip(*all_samples)
    _, va_p, _, va_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    val_samples = list(zip(va_p, va_l))

    print(f"Device: {device}  |  Stage 1 threshold: {threshold:.2f}")
    if rescue_tau is not None:
        print(f"defect8 rescue   : floor={rescue_floor:.2f}, τ={rescue_tau:.2f}")
    print(f"All samples: {len(all_samples)}  |  Val set: {len(val_samples)}")
    return model1, model2, threshold, prototypes, rescue_tau, rescue_floor, rescue_idx, device, val_samples, all_samples


@torch.no_grad()
def predict_all_cascade(model1, model2, threshold, prototypes, rescue_tau, rescue_floor,
                        rescue_idx, val_samples, device, use_tta=True):
    """Two-stage cascade prediction.  Returns (true_l, pred_l, probs [N x 9])."""
    tta_transforms = _make_tta_transforms() if use_tta else None
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    n_classes = len(_CASCADE_CLASSES)
    all_true, all_pred, all_probs = [], [], []

    for img_path, true_lbl in val_samples:
        img_pil = Image.open(img_path).convert("RGB")

        # Stage 1 — TTA-averaged defect probability
        if use_tta:
            dp_list = []
            for tf in tta_transforms:
                img_t = tf(img_pil).unsqueeze(0).to(device)
                logits1, _ = model1(img_t)
                dp_list.append(F.softmax(logits1, dim=1)[0, _DEFECT_IDX_BIN].item())
            defect_prob = float(np.mean(dp_list))
        else:
            img_t = val_tf(img_pil).unsqueeze(0).to(device)
            logits1, _ = model1(img_t)
            defect_prob = F.softmax(logits1, dim=1)[0, _DEFECT_IDX_BIN].item()

        # Compute Stage 2 embedding once (needed for rescue and for defect typing)
        if use_tta:
            emb_list = []
            for tf in tta_transforms:
                img_t = tf(img_pil).unsqueeze(0).to(device)
                emb_list.append(model2.get_embedding(img_t))
            embed2 = F.normalize(torch.stack(emb_list).mean(0), dim=1)
        else:
            img_t = val_tf(img_pil).unsqueeze(0).to(device)
            embed2 = model2.get_embedding(img_t)

        # Build full probability vector (for ROC curves)
        probs = np.zeros(n_classes)

        if defect_prob < threshold:
            # defect8 rescue: compound condition — suspicion floor + proto similarity
            if rescue_tau is not None and defect_prob >= rescue_floor:
                d8_sim = torch.mm(embed2, prototypes[rescue_idx].unsqueeze(1)).item()
                if d8_sim >= rescue_tau:
                    pred = _CASCADE_CLASSES.index(_DEFECT_CLASSES[rescue_idx])
                    probs[_GOOD_IDX_ALL] = 1.0 - defect_prob
                    for j, dc in enumerate(_DEFECT_CLASSES):
                        probs[_CASCADE_CLASSES.index(dc)] = max(0.0,
                            torch.mm(embed2, prototypes[j].unsqueeze(1)).item())
                    all_true.append(true_lbl); all_pred.append(pred); all_probs.append(probs)
                    continue
            pred = _GOOD_IDX_ALL
            probs[_GOOD_IDX_ALL] = 1.0 - defect_prob
            for j in range(n_classes):
                if j != _GOOD_IDX_ALL:
                    probs[j] = defect_prob / len(_DEFECT_CLASSES)
        else:
            # Stage 2 — use already-computed embed2
            sims = torch.mm(embed2, prototypes.T).squeeze(0)   # (8,)
            stage2_probs = F.softmax(sims * 10, dim=0).cpu().numpy()
            defect_pred_idx = int(sims.argmax().item())
            pred = _CASCADE_CLASSES.index(_DEFECT_CLASSES[defect_pred_idx])

            probs[_GOOD_IDX_ALL] = 1.0 - defect_prob
            for i, dc in enumerate(_DEFECT_CLASSES):
                probs[_CASCADE_CLASSES.index(dc)] = defect_prob * float(stage2_probs[i])

        all_true.append(true_lbl)
        all_pred.append(pred)
        all_probs.append(probs)

    return np.array(all_true), np.array(all_pred), np.vstack(all_probs)


def tau_normalize(model, tau: float = 0.5):
    """Normalize classifier weight norms per class.

    After biased training (95% good), minority-class weight vectors end up
    with smaller norms. Dividing each row by ||w_c||^tau corrects this
    without any retraining.  tau=0.5 (sqrt) is the standard setting.
    """
    W = model.classifier.weight.data          # (num_classes, embed_dim)
    norms = W.norm(dim=1, keepdim=True)       # (num_classes, 1)
    model.classifier.weight.data = W / (norms.clamp(min=1e-8) ** tau)
    return model


def load_model_and_data(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes   = ckpt["classes"]
    class2idx = ckpt["class2idx"]
    history   = ckpt.get("history", {})

    model = DefectClassifier(len(classes), EMBED_DIM).to(device)
    model.load_state_dict(ckpt["model_state"])
    tau_normalize(model, tau=0.3)   # correct minority-class weight norm bias
    model.eval()

    prototypes = ckpt["prototypes"].to(device)

    # Load all samples and reproduce the same val split
    samples = []
    for cls in classes:
        d = DATASET_DIR / cls
        if d.exists():
            for f in d.iterdir():
                if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                    samples.append((str(f), class2idx[cls]))

    paths, labels = zip(*samples)
    _, va_p, _, va_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    val_samples = list(zip(va_p, va_l))

    # Log-prior for logit adjustment: rare classes get a boost at inference
    total_counts = Counter(l for _, l in samples)
    total = len(samples)
    log_prior = torch.log(torch.tensor(
        [total_counts.get(i, 1) / total for i in range(len(classes))],
        dtype=torch.float32, device=device,
    ))

    return model, classes, class2idx, prototypes, device, val_samples, samples, history, log_prior


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_all(model, val_samples, prototypes, device, classes, log_prior):
    """Return (true_labels, pred_labels, probs_matrix).
    log_prior corrects for the sampling bias introduced by WeightedRandomSampler:
    the model is trained on a balanced distribution but tested on the true prior.
    """
    ds     = SimpleDataset(val_samples)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    TAU_LA = 0.1   # logit adjustment strength; tuned on val set (sweep: 0.0–0.5)
    all_true, all_pred, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits, _ = model(imgs)
        # Logit adjustment: subtract tau * log(class_prior) so rare defect
        # classes are not penalised by the model's learned frequency bias
        adj_logits = logits - TAU_LA * log_prior.unsqueeze(0)
        probs  = F.softmax(adj_logits, dim=1)
        preds  = probs.argmax(1)
        all_true.extend(labels.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    return np.array(all_true), np.array(all_pred), np.vstack(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history, out):
    if not history:
        print("No history found in checkpoint — skipping training history plot.")
        return
    eps = range(1, len(history.get("tr_loss", [])) + 1)
    if not eps:
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    a1.plot(eps, history["tr_loss"], label="Train"); a1.plot(eps, history["va_loss"], label="Val")
    a1.set(xlabel="Epoch", ylabel="Loss", title="Training / Validation Loss"); a1.legend(); a1.grid(alpha=.3)

    a2.plot(eps, history["tr_acc"],  label="Train acc")
    a2.plot(eps, history["va_acc"],  label="Val acc")
    a2.plot(eps, history["va_bal"],  label="Balanced acc", linestyle="--")
    a2.axhline(0.85, color="red", linestyle=":", linewidth=1.5, label="85 % target")
    a2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Curves"); a2.legend(); a2.grid(alpha=.3)

    plt.tight_layout()
    p = out / "plot1_training_history.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"Saved: {p}")


def plot_confusion_matrix(true_l, pred_l, classes, out):
    cm      = confusion_matrix(true_l, pred_l, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set(xlabel="Predicted", ylabel="True", title="Normalised Confusion Matrix")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    p = out / "plot2_confusion_matrix.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"Saved: {p}")


def plot_class_accuracy_vs_occurrence(true_l, pred_l, classes, all_samples, out):
    counts_all = Counter(l for _, l in all_samples)
    cls_accs, cls_occ = [], []
    for i in range(len(classes)):
        mask = true_l == i
        acc  = (pred_l[mask] == i).mean() if mask.any() else 0.0
        cls_accs.append(acc)
        cls_occ.append(counts_all.get(i, 0))

    colors = ["#d32f2f" if a < 0.70 else "#f57c00" if a < 0.85 else "#388e3c" for a in cls_accs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(classes, cls_accs, color=colors, edgecolor="black", linewidth=0.6)
    ax1.axhline(0.85, color="navy", linestyle="--", linewidth=1.5, label="85 % target")
    ax1.set_ylim(0, 1.12); ax1.set_ylabel("Accuracy"); ax1.legend()
    ax1.set_title("Per-Class Classification Accuracy")
    for bar, acc in zip(bars, cls_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{acc:.2f}", ha="center", fontsize=10, fontweight="bold")

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
    plt.savefig(p, dpi=150); plt.close(); print(f"Saved: {p}")


def plot_roc_curves(true_l, probs, classes, out):
    """Per-class one-vs-rest ROC curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        binary = (true_l == i).astype(int)
        if binary.sum() < 2:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        auc_score   = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{cls}  (AUC={auc_score:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Per-Class ROC Curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout()
    p = out / "plot5_roc_curves.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"Saved: {p}")


def plot_tsne(model, val_samples, classes, device, out, max_per_class=80, title_suffix=""):
    """t-SNE of learned embeddings — shows how well classes are separated."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed — skipping t-SNE plot")
        return

    # Subsample
    by_class = {}
    for p, l in val_samples:
        by_class.setdefault(l, []).append(p)
    sub = []
    for l, paths in by_class.items():
        for p in paths[:max_per_class]:
            sub.append((p, l))
    random.shuffle(sub)

    ds     = SimpleDataset(sub)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    embeds_list, lbls = [], []

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            e = model.get_embedding(imgs.to(device)).cpu().numpy()
            embeds_list.append(e)
            lbls.extend(labels.tolist())

    embeds = np.vstack(embeds_list)
    tsne   = TSNE(n_components=2, perplexity=min(30, len(embeds) // 5),
                  random_state=SEED, max_iter=1000)
    coords = tsne.fit_transform(embeds)

    cmap   = plt.cm.get_cmap("tab10", len(classes))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(classes):
        mask = np.array(lbls) == i
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=cmap(i), label=cls, s=25, alpha=0.75, edgecolors="none")
    ax.set(title=f"t-SNE of Learned Embeddings{title_suffix}",
           xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
    ax.legend(markerscale=2, fontsize=9); ax.grid(alpha=.2)
    plt.tight_layout()
    p = out / "plot6_tsne_embeddings.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"Saved: {p}")


def plot_few_shot_learning_curve(model, all_samples, classes, class2idx, device, out):
    """Deliverable 4: accuracy vs N-shot support per class."""
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    num_classes = len(classes)

    defect_samp = [(p, l) for p, l in all_samples if classes[l] != "good"]
    good_samp   = [(p, l) for p, l in all_samples if classes[l] == "good"]
    random.shuffle(defect_samp); random.shuffle(good_samp)

    split   = max(int(0.6 * len(defect_samp)), num_classes)
    support = defect_samp[:split]
    query   = defect_samp[split:] + good_samp[:len(defect_samp[split:])]

    by_class = {}
    for p, l in support:
        by_class.setdefault(l, []).append(p)

    max_shot   = min(max((len(v) for v in by_class.values()), default=1), 20)
    shot_counts = [n for n in [1, 2, 3, 5, 8, 12, 20] if n <= max_shot]

    model.eval()
    accs = []
    for n_shot in shot_counts:
        proto_sum = torch.zeros(num_classes, EMBED_DIM, device=device)
        proto_cnt = torch.zeros(num_classes, device=device)

        for p, l in good_samp[:200]:
            img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                e = model.get_embedding(img).squeeze(0)
            proto_sum[l] += e; proto_cnt[l] += 1

        for lbl, paths in by_class.items():
            for p in paths[:n_shot]:
                img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = model.get_embedding(img).squeeze(0)
                proto_sum[lbl] += e; proto_cnt[lbl] += 1

        proto_cnt = proto_cnt.clamp(min=1e-6)
        protos    = F.normalize(proto_sum / proto_cnt.unsqueeze(1), dim=1)

        correct = 0
        for p, tl in query:
            img = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.mm(model.get_embedding(img), protos.T).argmax(1).item()
            correct += (pred == tl)
        accs.append(correct / len(query))
        print(f"  {n_shot:2d}-shot  acc={accs[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shot_counts, [a * 100 for a in accs], "bo-", linewidth=2, markersize=8)
    ax.fill_between(shot_counts, [a * 100 for a in accs], 0, alpha=0.1)
    ax.axhline(85, color="red", linestyle="--", linewidth=1.5, label="85 % target")
    ax.set(xlabel="Support Examples per Defect Class (N-shot)", ylabel="Accuracy (%)",
           title="Few-Shot Learning Curve  –  How Quickly the Model Learns", ylim=(0, 105))
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    p = out / "plot4_few_shot_learning_curve.png"
    plt.savefig(p, dpi=150); plt.close(); print(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT),
                    help="Single-model checkpoint (ignored when --cascade is set)")
    ap.add_argument("--cascade", action="store_true",
                    help="Use two-stage cascade (model_stage1.pth + model_stage2.pth)")
    ap.add_argument("--no-tta", action="store_true",
                    help="Disable test-time augmentation for cascade mode (faster)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.cascade:
        # ── Cascade path (DINOv2 two-stage) ─────────────────────────────────
        print("Loading DINOv2 two-stage cascade …")
        model1, model2, threshold, prototypes, rescue_tau, rescue_floor, rescue_idx, \
            device, val_samples, all_samples = load_cascade_and_data()

        use_tta = not args.no_tta
        print(f"\nRunning cascade predictions (TTA={'4× flip' if use_tta else 'off'}) …")
        true_l, pred_l, probs = predict_all_cascade(
            model1, model2, threshold, prototypes, rescue_tau, rescue_floor,
            rescue_idx, val_samples, device, use_tta=use_tta
        )
        classes = _CASCADE_CLASSES
        class2idx = {c: i for i, c in enumerate(classes)}
        history = {}   # no combined history for cascade
        # t-SNE and few-shot use model1 (trained on all classes incl. "good")
        tsne_model = model1
        tsne_title_suffix = " (DINOv2 ViT-Small/14 — Stage 1)"

    else:
        # ── Single-model path (EfficientNet baseline) ────────────────────────
        print("Loading model and data …")
        model, classes, class2idx, protos, device, val_samples, all_samples, history, log_prior = \
            load_model_and_data(Path(args.checkpoint))
        print(f"Val set size: {len(val_samples)}")
        print("\nRunning predictions …")
        true_l, pred_l, probs = predict_all(model, val_samples, protos, device, classes, log_prior)
        tsne_model = model
        tsne_title_suffix = " (EfficientNet-B0 + Head)"

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(true_l, pred_l, target_names=classes, digits=3))
    ba = balanced_accuracy_score(true_l, pred_l)
    f1 = f1_score(true_l, pred_l, average="macro")
    overall_acc = (true_l == pred_l).mean()
    print(f"  Overall accuracy  : {overall_acc:.4f}")
    print(f"  Balanced accuracy : {ba:.4f}")
    print(f"  Macro F1          : {f1:.4f}")

    # ── Save metrics JSON ────────────────────────────────────────────────────
    p_rec_f1_sup = precision_recall_fscore_support(true_l, pred_l, labels=list(range(len(classes))))
    metrics = {
        "overall_accuracy":  round(float(overall_acc), 4),
        "balanced_accuracy": round(ba, 4),
        "macro_f1":          round(f1, 4),
        "per_class": {
            cls: {
                "precision": round(float(p_rec_f1_sup[0][i]), 4),
                "recall":    round(float(p_rec_f1_sup[1][i]), 4),
                "f1":        round(float(p_rec_f1_sup[2][i]), 4),
                "support":   int(p_rec_f1_sup[3][i]),
            }
            for i, cls in enumerate(classes)
        },
    }
    mpath = OUTPUT_DIR / "metrics.json"
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {mpath}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_training_history(history, OUTPUT_DIR)
    plot_confusion_matrix(true_l, pred_l, classes, OUTPUT_DIR)
    plot_class_accuracy_vs_occurrence(true_l, pred_l, classes, all_samples, OUTPUT_DIR)
    plot_roc_curves(true_l, probs, classes, OUTPUT_DIR)
    print("\nGenerating t-SNE (may take 1–2 min on CPU) …")
    plot_tsne(tsne_model, val_samples, classes, device, OUTPUT_DIR,
              title_suffix=tsne_title_suffix)
    print("\nGenerating few-shot learning curve …")
    plot_few_shot_learning_curve(tsne_model, all_samples, classes, class2idx, device, OUTPUT_DIR)

    print(f"\nAll evaluation outputs saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

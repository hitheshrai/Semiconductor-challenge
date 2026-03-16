#!/usr/bin/env python
"""
smote_stage2.py  —  Feature-space SMOTE for Stage 2 classifier head retraining.

Extracts embeddings from the trained Stage 2 backbone (DINOv2 or EfficientNet),
applies SMOTE in embedding space to balance defect classes, then retrains only
the classifier head + recomputes prototypes. Backbone stays frozen.

Usage:
  python smote_stage2.py            # uses current model_stage2.pth (DINOv2 if trained)
  python smote_stage2.py --dinov2   # explicitly use DINOv2 backbone
"""

import sys, argparse, numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent))
from train_cascade import (
    SimpleDataset, load_samples_defects,
    CKPT_STAGE2, SEED, NUM_DEFECTS, OUTPUT_DIR
)
from model import EMBED_DIM


def run_smote(use_dinov2: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load Stage 2 checkpoint ───────────────────────────────────────────────
    ckpt = torch.load(CKPT_STAGE2, map_location=device, weights_only=False)
    classes   = ckpt["classes"]
    class2idx = ckpt["class2idx"]
    n_classes = len(classes)

    if use_dinov2:
        from model_dinov2 import DINOv2DefectClassifier
        model = DINOv2DefectClassifier(n_classes, EMBED_DIM).to(device)
    else:
        from model import DefectClassifier
        model = DefectClassifier(n_classes, EMBED_DIM).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded Stage 2 checkpoint  ({n_classes} classes)")

    # ── Extract embeddings ────────────────────────────────────────────────────
    samples = load_samples_defects()
    paths, labels = zip(*samples)
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    def embed_split(path_list, label_list):
        ds     = SimpleDataset(list(zip(path_list, label_list)))
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
        embeds, labs = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                embeds.append(model.get_embedding(imgs.to(device)).cpu())
                labs.extend(lbls.tolist())
        return torch.cat(embeds).numpy(), np.array(labs)

    print("Extracting train embeddings...")
    X_tr, y_tr = embed_split(tr_p, tr_l)
    print("Extracting val embeddings...")
    X_va, y_va = embed_split(va_p, va_l)

    print(f"\nBefore SMOTE: {dict(Counter(y_tr))}")

    # ── SMOTE in embedding space ──────────────────────────────────────────────
    k = max(1, min(Counter(y_tr).values()) - 1)
    k = min(k, 5)
    smote = SMOTE(k_neighbors=k, random_state=SEED)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    print(f"After  SMOTE: {dict(Counter(y_res))}")

    # ── Retrain classifier head only ──────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    aug_loader = DataLoader(
        TensorDataset(torch.tensor(X_res, dtype=torch.float32),
                      torch.tensor(y_res, dtype=torch.long)),
        batch_size=32, shuffle=True
    )
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    best_bal, best_state = 0.0, None

    print("\nRetraining classifier head (40 epochs)...")
    for ep in range(40):
        model.train()
        for xb, yb in aug_loader:
            logits = model.classifier(xb.to(device))
            loss   = F.cross_entropy(logits, yb.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model.classifier(X_va_t).argmax(1).cpu()
        bal = balanced_accuracy_score(y_va_t.numpy(), preds.numpy())
        if bal > best_bal:
            best_bal   = bal
            best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}
            print(f"  ep {ep+1:02d}  bal_acc={bal:.4f}  ✓ new best")

    print(f"\nBest classifier bal_acc: {best_bal:.4f}")
    model.classifier.load_state_dict(best_state)

    # ── Recompute prototypes (include SMOTE synthetic embeddings) ────────────
    # Using SMOTE-augmented embeddings gives minority classes richer prototypes
    model.eval()
    proto_sums   = torch.zeros(n_classes, EMBED_DIM)
    proto_counts = torch.zeros(n_classes)
    for emb_vec, lbl in zip(X_res, y_res):
        proto_sums[lbl]   += torch.tensor(emb_vec)
        proto_counts[lbl] += 1
    protos = F.normalize(proto_sums / proto_counts.unsqueeze(1).clamp(min=1), dim=1)
    print(f"Prototypes recomputed from SMOTE-augmented embeddings: {dict(Counter(y_res.tolist()))}")

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt["model_state"] = model.state_dict()
    ckpt["prototypes"]  = protos
    out_path = OUTPUT_DIR / "model_stage2_smote.pth"
    torch.save(ckpt, out_path)
    print(f"Saved → {out_path}")

    # ── Eval report ───────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds_val = model.classifier(X_va_t).argmax(1).cpu().numpy()
    print("\nStage 2 val (SMOTE head):")
    print(classification_report(y_va, preds_val, target_names=classes, digits=3))
    print(f"Balanced accuracy: {balanced_accuracy_score(y_va, preds_val):.4f}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov2", action="store_true")
    args = ap.parse_args()
    run_smote(use_dinov2=args.dinov2)

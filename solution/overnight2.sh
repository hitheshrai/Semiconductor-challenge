#!/bin/bash
set -e
cd /home/hithesh/defect_challenge/solution
PYTHON=/home/hithesh/defect_challenge/semi/bin/python
LOG=output/overnight.log

echo -e "\n=====================================" | tee -a $LOG
echo "OVERNIGHT2 — $(date)" | tee -a $LOG
echo "=====================================" | tee -a $LOG

# ── [3] DINOv2 threshold sweep ────────────────────────────────────────────────
echo -e "\n[3/5] DINOv2 Threshold Sweep — $(date)" | tee -a $LOG
$PYTHON - << 'PYEOF' 2>&1 | tee -a $LOG
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from train_cascade import (
    _make_model, load_samples_binary, SimpleDataset,
    CKPT_STAGE1, BATCH_SIZE, SEED, DEFECT_IDX, GOOD_IDX
)
import train_cascade as _tc
_tc._USE_DINOV2 = True   # use DINOv2 stage1 checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load(CKPT_STAGE1, map_location=device, weights_only=False)
model  = _make_model(2).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

samples = load_samples_binary()
paths, labels = zip(*samples)
_, va_p, _, va_l = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
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

print(f"{'Thresh':>7} {'Overall':>8} {'DefRec':>8} {'GoodRec':>9}")
print("-" * 40)
best_thresh, best_overall = 0.35, 0.0
for thresh in np.arange(0.20, 0.75, 0.05):
    preds = (all_probs >= thresh).astype(int)
    overall       = (preds == all_labels).mean()
    defect_recall = (preds[defect_mask] == DEFECT_IDX).mean()
    good_recall   = (preds[good_mask]   == GOOD_IDX).mean()
    marker = " <-- best" if overall > best_overall and good_recall >= 0.80 else ""
    if overall > best_overall and good_recall >= 0.80:
        best_overall = overall
        best_thresh  = float(thresh)
    print(f"  {thresh:.2f}   {overall:.3f}    {defect_recall:.3f}     {good_recall:.3f}{marker}")

# Save best threshold
ckpt["threshold"] = best_thresh
torch.save(ckpt, CKPT_STAGE1)
print(f"\nBest threshold {best_thresh:.2f} saved to {CKPT_STAGE1}")
PYEOF

# ── [4] DINOv2 cascade eval with best threshold ───────────────────────────────
echo -e "\n[4/5] DINOv2 Cascade Eval (best threshold) — $(date)" | tee -a $LOG
$PYTHON -u train_cascade.py --evaluate --dinov2 2>&1 | tee -a $LOG

# ── [5] EfficientNet Stage 2 retrain with Feature-space SMOTE ────────────────
echo -e "\n[5/5] Feature-space SMOTE → EfficientNet Stage 2 retrain — $(date)" | tee -a $LOG
$PYTHON - << 'PYEOF' 2>&1 | tee -a $LOG
"""
Extract EfficientNet Stage 2 embeddings for all defect images,
apply SMOTE in embedding space to oversample rare classes to 50 samples each,
then retrain Stage 2 classifier head only (backbone frozen) on augmented embeddings.
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from train_cascade import (
    _make_model, load_samples_defects, SimpleDataset,
    CKPT_STAGE2, CKPT_STAGE1, BATCH_SIZE, SEED, EMBED_DIM, NUM_DEFECTS,
    OUTPUT_DIR
)
from model import DefectClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load EfficientNet Stage 2 checkpoint
ckpt2  = torch.load(CKPT_STAGE2, map_location=device, weights_only=False)
classes   = ckpt2["classes"]
class2idx = ckpt2["class2idx"]
model = DefectClassifier(len(classes), EMBED_DIM).to(device)
model.load_state_dict(ckpt2["model_state"])
model.eval()

# Extract embeddings for all defect training images
samples = load_samples_defects()
paths, labels = zip(*samples)
tr_p, va_p, tr_l, va_l = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=SEED
)
tr_ds = SimpleDataset(list(zip(tr_p, tr_l)))
tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=False, num_workers=0)

all_embeds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in tr_loader:
        embeds = model.get_embedding(imgs.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.extend(lbls.tolist())

X = torch.cat(all_embeds).numpy()
y = np.array(all_labels)

from collections import Counter
print(f"Before SMOTE: {Counter(y)}")

# SMOTE: oversample each class to at least 20 samples
k = min(Counter(y).values()) - 1
k = max(1, min(k, 5))
smote = SMOTE(k_neighbors=k, random_state=SEED)
X_res, y_res = smote.fit_resample(X, y)
print(f"After  SMOTE: {Counter(y_res)}")

# Build val embeddings
va_ds = SimpleDataset(list(zip(va_p, va_l)))
va_loader = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0)
val_embeds, val_labels = [], []
with torch.no_grad():
    for imgs, lbls in va_loader:
        val_embeds.append(model.get_embedding(imgs.to(device)).cpu())
        val_labels.extend(lbls.tolist())
X_val = torch.cat(val_embeds)
y_val = torch.tensor(val_labels)

# Retrain only the classifier head on augmented embeddings
X_res_t = torch.tensor(X_res, dtype=torch.float32)
y_res_t  = torch.tensor(y_res, dtype=torch.long)
tr_aug = TensorDataset(X_res_t, y_res_t)
aug_loader = DataLoader(tr_aug, batch_size=32, shuffle=True)

# Freeze backbone, retrain head
for p in model.backbone.parameters():
    p.requires_grad = False
for p in model.embed_head.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
best_bal, best_state = 0.0, None

print("\nRetraining classifier head on SMOTE-augmented embeddings (30 epochs)...")
for ep in range(30):
    model.train()
    for xb, yb in aug_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model.classifier(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Val
    model.eval()
    with torch.no_grad():
        logits_val = model.classifier(X_val.to(device))
        preds = logits_val.argmax(dim=1).cpu()
    from sklearn.metrics import balanced_accuracy_score
    bal = balanced_accuracy_score(y_val.numpy(), preds.numpy())
    if bal > best_bal:
        best_bal = bal
        best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        print(f"  ep {ep+1:02d}  bal_acc={bal:.4f}  ✓ new best")

# Patch best classifier state into checkpoint and save
print(f"\nBest classifier bal_acc: {best_bal:.4f}")
model.classifier.load_state_dict(best_state)
# Recompute prototypes from training embeddings
model.eval()
protos = torch.zeros(len(classes), EMBED_DIM)
with torch.no_grad():
    for imgs, lbls in DataLoader(SimpleDataset(list(zip(tr_p, tr_l))), batch_size=32, num_workers=0):
        emb = model.get_embedding(imgs.to(device)).cpu()
        for e, l in zip(emb, lbls):
            protos[l] += e
counts = Counter(tr_l)
for i in range(len(classes)):
    protos[i] = F.normalize(protos[i].unsqueeze(0), dim=1).squeeze(0) if counts.get(i,0) > 0 else protos[i]

ckpt2["model_state"] = model.state_dict()
ckpt2["prototypes"]  = protos
smote_ckpt = OUTPUT_DIR / "model_stage2_smote.pth"
torch.save(ckpt2, smote_ckpt)
print(f"Saved SMOTE-enhanced Stage 2 to {smote_ckpt}")

# Quick eval
print("\nEval with SMOTE Stage 2:")
from sklearn.metrics import classification_report
model.eval()
with torch.no_grad():
    logits_val = model.classifier(X_val.to(device))
    preds_val = logits_val.argmax(1).cpu().numpy()
y_val_np = y_val.numpy()
print(classification_report(y_val_np, preds_val,
      target_names=classes, digits=3))
print(f"Balanced accuracy: {balanced_accuracy_score(y_val_np, preds_val):.4f}")
PYEOF

echo -e "\nAll overnight jobs complete — $(date)" | tee -a $LOG

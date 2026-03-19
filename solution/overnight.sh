#!/bin/bash
set -e
cd /home/hithesh/defect_challenge/solution
PYTHON=/home/hithesh/defect_challenge/semi/bin/python
LOG=output/overnight.log

echo "=====================================" | tee -a $LOG
echo "OVERNIGHT RUN — $(date)" | tee -a $LOG
echo "=====================================" | tee -a $LOG

# 1. TTA cascade eval
echo -e "\n[1/2] TTA Cascade Evaluation — $(date)" | tee -a $LOG
$PYTHON - << 'PYEOF' 2>&1 | tee -a $LOG
import sys; sys.path.insert(0, ".")
from pathlib import Path
from sklearn.metrics import classification_report, balanced_accuracy_score
from classify import CascadeClassificationApp

app = CascadeClassificationApp()
root = Path("../Dataset")
y_true, y_pred = [], []
for cls_dir in sorted(root.iterdir()):
    if not cls_dir.is_dir(): continue
    for img in cls_dir.iterdir():
        if img.suffix.upper() not in {".PNG", ".JPG", ".JPEG"}: continue
        r = app.predict(str(img))
        y_true.append(cls_dir.name)
        y_pred.append(r["class"])

known = list(dict.fromkeys(y_true + y_pred))
print(classification_report(y_true, y_pred, labels=known, digits=3))
print(f"Overall accuracy  : {sum(t==p for t,p in zip(y_true,y_pred))/len(y_true):.4f}")
print(f"Balanced accuracy : {balanced_accuracy_score(y_true, y_pred):.4f}")
PYEOF

# 2. DINOv2 cascade training
echo -e "\n[2/2] DINOv2 Cascade Training — $(date)" | tee -a $LOG
$PYTHON -u train_cascade.py --stage both --dinov2 \
    --stage1-epochs 30 --stage2-epochs 40 2>&1 | tee -a $LOG

echo -e "\nDone — $(date)" | tee -a $LOG

# Active Tasks

## Competition Deliverables

- [x] 1. Working classifier app (`classify.py`)
- [x] 2. Accuracy vs occurrence plots (plot2, plot3)
- [ ] 3. ~85% overall classification accuracy — **modelling problem, not a training run problem**
- [x] 4. Few-shot learning curve (plot4)
- [ ] 5. Documentation / write-up

## Blockers to 85% Accuracy

Training (Phase 1 + Phase 2) has been completed multiple times. The gap is a model issue:

1. **Wrong checkpoint criterion** — saving on `balanced_accuracy` rewards defect recall,
   ignores false alarms on the good class. Should use weighted F1 or overall accuracy.
2. **Overfitting** — val loss diverges (train ~0.55, val ~1.7–3.0) while train acc climbs
3. **Good class imbalance** — 715/756 val samples are "good"; tiny weight changes cascade

## Next Step

Investigate `train.py` checkpoint selection and regularization before next run.

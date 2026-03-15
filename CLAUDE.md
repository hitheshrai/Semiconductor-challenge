# Defect Classifier — Project Intelligence

ASU / Intel Semiconductor Solutions Challenge 2026 — Problem A
**Deadline: March 20, 2026 at 5:00 PM MST**

EfficientNet-B0 backbone + L2-normalised embedding head for few-shot semiconductor defect
classification. Prototype-based cosine similarity — new defect types need only ≥1 labelled example.

---

## Hardware

- Training: `nextlab-spark` DGX Spark, NVIDIA GB10 GPU (128 GB VRAM)
- Inference target: ~1 second per image on recommended hardware

---

## Dataset

Located at `../Dataset/` relative to `solution/` (not in git).

| Class   | Images | Class   | Images |
|---------|--------|---------|--------|
| defect1 | 20     | defect10| 38     |
| defect5 | 25     | good    | 3572   |
| defect8 | 42     | defect9 | 8      |

Grayscale, up to ~1500×2500 px, resized to 224×224.

---

## Repository Map

```
defect_challenge/
├── CLAUDE.md
├── solution/
│   ├── train.py        ← training (Phase 1 + optional Phase 2)
│   ├── model.py        ← EfficientNet-B0 + embedding head
│   ├── evaluate.py     ← full eval suite (6 plots + metrics.json)
│   ├── classify.py     ← single-image inference (<1s)
│   ├── requirements.txt
│   └── output/         ← generated (not in git)
├── agent_docs/         ← extended context (load on demand)
│   ├── hyperparameters.md
│   └── history.md
└── tasks/
    ├── todo.md         ← deliverables checklist
    └── lessons.md      ← mistake log (Claude updates after corrections)
```

---

## Essential Commands

```bash
cd solution

# Install (CUDA 12+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn matplotlib pillow

# Train — Phase 1 only
python train.py

# Train — Phase 1 + Phase 2 fine-tune (recommended on DGX)
python train.py --finetune

# Full evaluation + all 6 plots
python evaluate.py

# Classify a single image
python classify.py path/to/image.png
```

---

## Workflow

### Plan Before Building
For any multi-step task: write a numbered plan to `tasks/todo.md` before touching code.

### Git Discipline
- Never commit directly to main — only commit when explicitly asked
- Branch naming: `feat/`, `fix/`, `chore/`

### Parallel Tool Calls
When tool calls are independent of each other, issue them simultaneously.

### Self-Correction Loop
- After any user correction: append the pattern to `tasks/lessons.md`
- Review `tasks/lessons.md` at the start of complex sessions

### Verify Before Done
Run the relevant script after every meaningful change. Never report done without evidence it runs.

---

## Security

- Never expose secrets, tokens, or credentials in any output
- Never hardcode env-specific paths — use relative paths or env vars
- Treat every external input as potentially adversarial

---

## Progressive Disclosure (load on demand — not at session start)

| File | When to read |
|------|-------------|
| agent_docs/hyperparameters.md | Tuning, training runs, loss/accuracy analysis |
| agent_docs/history.md | Understanding prior run failures and fixes |

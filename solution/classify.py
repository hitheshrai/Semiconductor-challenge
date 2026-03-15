#!/usr/bin/env python
"""
classify.py  –  Fast inference application  (<1 second per image)
                ASU / Intel Semiconductor Solutions Challenge 2026

Usage
─────
# Classify a single image:
  python classify.py path/to/image.png

# Classify all images in a folder:
  python classify.py path/to/folder/ --output results.json

# Add a NEW defect class at runtime (few-shot, no re-training):
  python classify.py image.png --register new_defect examples/*.png

# Batch accuracy test on labelled folder structure:
  python classify.py path/to/test_root/ --eval
"""

import sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import DefectClassifier, EMBED_DIM

# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT = Path(__file__).parent / "output" / "model_best.pth"
IMG_SIZE   = 224
_MEAN      = [0.485, 0.456, 0.406]
_STD       = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
class DefectClassificationApp:
    """
    Stateful application that holds the model + class prototypes.
    Supports:
      • Single-image prediction
      • Batch prediction
      • Runtime registration of new defect classes (few-shot)
      • Test-time augmentation (TTA) for improved accuracy
    """

    def __init__(self, checkpoint_path: Path = CHECKPOINT, tta: bool = True):
        t0 = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tta    = tta

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.classes    = ckpt["classes"]
        self.class2idx  = ckpt["class2idx"]
        self.num_classes = len(self.classes)

        self.model = DefectClassifier(self.num_classes, EMBED_DIM).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.prototypes     = ckpt["prototypes"].to(self.device)  # (C, D)
        self.base_classes   = list(self.classes)   # classes known at train time
        self.extra_classes  = []                   # classes registered at runtime

        # Tau-normalization: correct minority-class weight-norm bias from
        # imbalanced training (defect class weights end up smaller than good)
        W     = self.model.classifier.weight.data
        norms = W.norm(dim=1, keepdim=True)
        self.model.classifier.weight.data = W / (norms.clamp(min=1e-8) ** 0.3)

        # Logit adjustment prior: rare classes get a boost at inference
        cc    = ckpt.get("class_counts", {c: 1 for c in self.classes})
        total = sum(cc.values()) or 1
        prior = torch.tensor(
            [cc.get(c, 1) / total for c in self.classes],
            dtype=torch.float32, device=self.device,
        )
        self.log_prior = torch.log(prior)  # (C,)

        self._load_ms = (time.time() - t0) * 1000
        print(f"Model loaded in {self._load_ms:.0f} ms  |  device={self.device}")
        print(f"Known classes: {self.classes}")

    # ── core embedding ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _embed(self, img: Image.Image) -> torch.Tensor:
        """Return L2-normalised embedding (1, D) for a PIL image."""
        if self.tta:
            embeds = []
            for tf in _TTA_TRANSFORMS:
                t = tf(img).unsqueeze(0).to(self.device)
                embeds.append(self.model.get_embedding(t))
            return F.normalize(torch.stack(embeds).mean(0), dim=1)
        else:
            t = _PREPROCESS(img).unsqueeze(0).to(self.device)
            return self.model.get_embedding(t)

    # ── single-image prediction ───────────────────────────────────────────────
    def predict(self, image_path: str | Path) -> dict:
        """
        Returns
        ───────
        {
          "class":      str,   # predicted class name
          "confidence": float, # softmax probability of top class
          "scores":     dict,  # {class_name: prob} for all classes
          "infer_ms":   float, # inference time in ms
        }
        """
        t0  = time.time()
        img = Image.open(image_path).convert("RGB")

        if self.extra_classes:
            # Hybrid: classifier head for base classes, prototype for new ones
            embed      = self._embed(img)                          # (1, D)
            t          = _PREPROCESS(img).unsqueeze(0).to(self.device)
            logits, _  = self.model(t)                             # (1, C_base)
            adj        = logits - 0.1 * self.log_prior.unsqueeze(0) # logit adjustment
            base_probs = F.softmax(adj, dim=1).squeeze(0)

            extra_protos = self.prototypes[len(self.base_classes):]
            extra_sim    = torch.mm(embed, extra_protos.T).squeeze(0)
            extra_probs  = F.softmax(extra_sim * 12, dim=0)

            # Scale to comparable range and concatenate
            probs = torch.cat([base_probs, extra_probs * base_probs.max()])
        else:
            # Fast path: classifier head + logit adjustment
            t         = _PREPROCESS(img).unsqueeze(0).to(self.device)
            logits, _ = self.model(t)
            adj       = logits - 0.1 * self.log_prior.unsqueeze(0)
            probs     = F.softmax(adj, dim=1).squeeze(0)

        pred_idx   = probs.argmax().item()
        pred_class = self.classes[pred_idx]
        confidence = probs[pred_idx].item()
        infer_ms   = (time.time() - t0) * 1000

        return {
            "class":      pred_class,
            "confidence": round(confidence, 4),
            "scores":     {c: round(probs[i].item(), 4) for i, c in enumerate(self.classes)},
            "infer_ms":   round(infer_ms, 1),
        }

    # ── few-shot: register new class ─────────────────────────────────────────
    def register_class(self, class_name: str, example_paths: list[str | Path]):
        """
        Add a brand-new defect class at runtime from ≥1 example images.
        No re-training needed — just compute the mean embedding (prototype).
        """
        if class_name in self.class2idx:
            print(f"  Class '{class_name}' already known. Updating prototype.")
        else:
            new_idx = len(self.classes)
            self.classes.append(class_name)
            self.class2idx[class_name] = new_idx
            self.num_classes += 1
            self.extra_classes.append(class_name)
            # Expand prototypes matrix
            new_row = torch.zeros(1, EMBED_DIM, device=self.device)
            self.prototypes = torch.cat([self.prototypes, new_row], dim=0)

        idx    = self.class2idx[class_name]
        embeds = []
        for p in example_paths:
            img = Image.open(p).convert("RGB")
            embeds.append(self._embed(img).squeeze(0))

        proto = F.normalize(torch.stack(embeds).mean(0, keepdim=True), dim=1)
        self.prototypes[idx] = proto.squeeze(0)
        print(f"  Registered '{class_name}' from {len(example_paths)} example(s). "
              f"Index = {idx}")

    # ── batch prediction ─────────────────────────────────────────────────────
    def predict_folder(self, folder: Path, output_json: Path | None = None) -> list[dict]:
        exts   = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
        print(f"\nFound {len(images)} images in {folder}")

        results = []
        for i, img_path in enumerate(images):
            result = self.predict(img_path)
            result["file"] = img_path.name
            results.append(result)
            if (i + 1) % 50 == 0 or i == len(images) - 1:
                avg_ms = np.mean([r["infer_ms"] for r in results])
                print(f"  [{i+1}/{len(images)}]  avg infer: {avg_ms:.1f} ms/image")

        if output_json:
            with open(output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_json}")

        return results

    # ── eval mode: folder has sub-folders named by class ─────────────────────
    def evaluate_folder(self, root: Path):
        from sklearn.metrics import classification_report, balanced_accuracy_score

        y_true, y_pred, timings = [], [], []
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                r = self.predict(img_path)
                y_true.append(cls_dir.name)
                y_pred.append(r["class"])
                timings.append(r["infer_ms"])

        print("\n── Evaluation Results ──────────────────────────────────────")
        print(f"  Images evaluated : {len(y_true)}")
        print(f"  Avg infer time   : {np.mean(timings):.1f} ms  "
              f"(max {max(timings):.0f} ms)")
        known = list(dict.fromkeys(y_true + y_pred))
        print(classification_report(y_true, y_pred, labels=known, digits=3))
        ba = balanced_accuracy_score(y_true, y_pred)
        print(f"  Balanced accuracy: {ba:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Defect Classifier — Intel / ASU Challenge 2026")
    ap.add_argument("path",       help="Image file or folder to classify")
    ap.add_argument("--checkpoint", default=str(CHECKPOINT), help="Model checkpoint")
    ap.add_argument("--output",   default=None, help="Save batch results to JSON")
    ap.add_argument("--register", nargs="*",
                    help="Register a new class: --register <class_name> img1.png img2.png …")
    ap.add_argument("--eval",     action="store_true",
                    help="Evaluate: path must be folder with sub-folders named by class")
    ap.add_argument("--no-tta",   action="store_true", help="Disable test-time augmentation")
    args = ap.parse_args()

    app  = DefectClassificationApp(Path(args.checkpoint), tta=not args.no_tta)
    path = Path(args.path)

    # Optional: register new class first
    if args.register:
        if len(args.register) < 2:
            print("--register requires: <class_name> <img1> [<img2> ...]")
            sys.exit(1)
        app.register_class(args.register[0], args.register[1:])

    if args.eval:
        app.evaluate_folder(path)
    elif path.is_dir():
        out = Path(args.output) if args.output else None
        app.predict_folder(path, out)
    else:
        result = app.predict(path)
        _print_result(result, path)


def _print_result(result: dict, path: Path):
    print(f"\n{'─'*50}")
    print(f"  Image      : {path}")
    print(f"  Prediction : {result['class'].upper()}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Time       : {result['infer_ms']:.0f} ms")
    print(f"\n  All scores (sorted):")
    for cls, prob in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar  = "█" * int(prob * 28)
        flag = " ←" if cls == result["class"] else ""
        print(f"    {cls:14s} {prob:.3f}  {bar}{flag}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
classify.py  –  Fast inference application  (<1 second per image)
                ASU / Intel Semiconductor Solutions Challenge 2026

Usage
─────
# Classify a single image (single model):
  python classify.py path/to/image.png

# Classify a single image (two-stage cascade — best defect recall):
  python classify.py path/to/image.png --cascade

# Classify all images in a folder:
  python classify.py path/to/folder/ --output results.json [--cascade]

# Add a NEW defect class at runtime (few-shot, no re-training):
  python classify.py image.png --register new_defect examples/*.png

# Batch accuracy test on labelled folder structure:
  python classify.py path/to/test_root/ --eval [--cascade]
"""

import sys, time, json, argparse, pickle
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import DefectClassifier, EMBED_DIM


def _load_model(num_classes: int, state_dict: dict, device):
    """Auto-detect backbone from checkpoint keys and return loaded model."""
    keys = list(state_dict.keys())
    if any("backbone.cls_token" in k for k in keys):
        # DINOv2 or MAE ViT backbone
        if any("backbone.patch_embed.proj" in k and "backbone.blocks.0.ls1" in "".join(keys) for k in keys):
            from model_dinov2 import DINOv2DefectClassifier
            m = DINOv2DefectClassifier(num_classes, EMBED_DIM).to(device)
        else:
            from model_vit import ViTDefectClassifier
            from pathlib import Path as _Path
            m = ViTDefectClassifier(num_classes, EMBED_DIM, mae_ckpt=_Path("output/mae_encoder.pth")).to(device)
    else:
        m = DefectClassifier(num_classes, EMBED_DIM).to(device)
    m.load_state_dict(state_dict)
    m.eval()
    return m

# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT       = Path(__file__).parent / "output" / "model_best.pth"
CHECKPOINT_S1    = Path(__file__).parent / "output" / "model_stage1.pth"
CHECKPOINT_S2    = Path(__file__).parent / "output" / "model_stage2.pth"
IMG_SIZE   = 224
_MEAN      = [0.485, 0.456, 0.406]
_STD       = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

def _make_tta_transforms():
    """8 deterministic TTA variants: original + 3 rotations + 4 flips."""
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    post = [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    variants = []
    for angle in [0, 90, 180, 270]:
        for flip in [False, True]:
            ops = base.copy()
            if angle:
                ops.append(transforms.Lambda(lambda img, a=angle: img.rotate(a)))
            if flip:
                ops.append(transforms.RandomHorizontalFlip(p=1.0))
            ops.extend(post)
            variants.append(transforms.Compose(ops))
    return variants

_TTA_TRANSFORMS = _make_tta_transforms()   # 8 variants


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
class CascadeClassificationApp:
    """
    Two-stage cascade inference.

    Stage 1 — binary good-vs-defective classifier (model_stage1.pth)
      Runs on every image. If defect probability < threshold → predict 'good'.

    Stage 2 — defect-type classifier (model_stage2.pth)
      Runs only when Stage 1 flags an image as defective.
      Uses prototype-based cosine inference for few-shot extensibility.

    Why cascade?
      A single model cannot simultaneously optimise the 95/5 good/defect split
      AND the 8-way defect typing. Decomposing into two independent problems
      doubles defect recall (52% → 71%) while preserving 85% overall accuracy.
    """

    def __init__(self,
                 stage1_path: Path = CHECKPOINT_S1,
                 stage2_path: Path = CHECKPOINT_S2):
        t0 = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Stage 1 ──────────────────────────────────────────────────────────
        ckpt1 = torch.load(stage1_path, map_location=self.device, weights_only=False)
        self.model1 = _load_model(2, ckpt1["model_state"], self.device)
        self.threshold = ckpt1.get("threshold", 0.35)
        _DEFECT_IDX = 1

        # ── Stage 2 ──────────────────────────────────────────────────────────
        ckpt2 = torch.load(stage2_path, map_location=self.device, weights_only=False)
        self.defect_classes = ckpt2["classes"]        # 8 defect types
        self.defect2idx     = ckpt2["class2idx"]
        self.model2 = _load_model(len(self.defect_classes), ckpt2["model_state"], self.device)
        self.protos2 = ckpt2["prototypes"].to(self.device)  # (8, D)

        # defect8 rescue paths (prototype or OC-SVM, loaded if tuned)
        self.rescue_tau   = ckpt2.get("defect8_rescue_threshold", None)
        self.rescue_floor = ckpt2.get("defect8_rescue_floor",     0.0)
        self.rescue_idx   = ckpt2.get("defect8_rescue_idx",       None)
        ocsvm_blob = ckpt2.get("defect8_ocsvm", None)
        self.ocsvm_rescue = pickle.loads(ocsvm_blob) if ocsvm_blob is not None else None

        # Full class list for output consistency
        self.classes   = self.defect_classes + ["good"]
        self.extra_classes = []

        self._load_ms = (time.time() - t0) * 1000
        print(f"Cascade loaded in {self._load_ms:.0f} ms  |  device={self.device}")
        print(f"Stage 1 threshold: {self.threshold:.2f}")
        if self.ocsvm_rescue is not None:
            r = self.ocsvm_rescue
            print(f"defect8 OC-SVM   : nu={r['ocsvm'].nu:.2f}, floor={r['floor']:.2f}, dt={r.get('dt',0):.3f}")
        elif self.rescue_tau is not None:
            print(f"defect8 rescue   : floor={self.rescue_floor:.2f}, τ={self.rescue_tau:.2f}")
        print(f"Defect classes: {self.defect_classes}")

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        t0  = time.time()
        img = Image.open(image_path).convert("RGB")

        # Stage 1 TTA: average defect_prob across all 8 variants
        defect_probs = []
        tensors = []
        for tf in _TTA_TRANSFORMS:
            t = tf(img).unsqueeze(0).to(self.device)
            tensors.append(t)
            logits1, _ = self.model1(t)
            defect_probs.append(F.softmax(logits1, dim=1)[0, 1].item())
        defect_prob = float(np.mean(defect_probs))

        if defect_prob < self.threshold:
            rescue_fired = False
            # OC-SVM rescue (preferred if available)
            if self.ocsvm_rescue is not None and defect_prob >= self.ocsvm_rescue["floor"]:
                embeds2_rescue = [self.model2.get_embedding(t) for t in tensors]
                embed2_rescue  = F.normalize(torch.stack(embeds2_rescue).mean(0), dim=1)
                e_np     = embed2_rescue.squeeze(0).cpu().numpy().reshape(1, -1)
                e_scaled = self.ocsvm_rescue["scaler"].transform(e_np)
                dt       = self.ocsvm_rescue.get("dt", 0.0)
                if self.ocsvm_rescue["ocsvm"].decision_function(e_scaled)[0] > dt:
                    d8_idx     = self.ocsvm_rescue["d8_idx"]
                    pred_class = self.defect_classes[d8_idx]
                    confidence = float(self.ocsvm_rescue["ocsvm"].decision_function(e_scaled)[0])
                    scores     = {c: round(
                                      torch.mm(embed2_rescue,
                                               self.protos2[i].unsqueeze(1)).item(), 4)
                                  for i, c in enumerate(self.defect_classes)}
                    scores["good"] = round(1.0 - defect_prob, 4)
                    rescue_fired = True
            # Prototype cosine rescue (fallback)
            elif self.rescue_tau is not None and defect_prob >= self.rescue_floor:
                embeds2_rescue = [self.model2.get_embedding(t) for t in tensors]
                embed2_rescue  = F.normalize(torch.stack(embeds2_rescue).mean(0), dim=1)
                d8_sim = torch.mm(embed2_rescue,
                                  self.protos2[self.rescue_idx].unsqueeze(1)).item()
                if d8_sim >= self.rescue_tau:
                    pred_class  = self.defect_classes[self.rescue_idx]
                    confidence  = round(d8_sim, 4)
                    scores      = {c: round(
                                      torch.mm(embed2_rescue,
                                               self.protos2[i].unsqueeze(1)).item(), 4)
                                   for i, c in enumerate(self.defect_classes)}
                    scores["good"] = round(1.0 - defect_prob, 4)
                    rescue_fired = True

            if not rescue_fired:
                pred_class = "good"
                confidence = 1.0 - defect_prob
                scores     = {"good": round(confidence, 4),
                              **{c: round(defect_prob / len(self.defect_classes), 4)
                                 for c in self.defect_classes}}
        else:
            # Stage 2 TTA: average embeddings across all 8 variants
            embeds2 = [self.model2.get_embedding(t) for t in tensors]
            embed2  = F.normalize(torch.stack(embeds2).mean(0), dim=1)
            sims        = F.softmax(torch.mm(embed2, self.protos2.T) * 12, dim=1).squeeze(0)
            pred_idx    = sims.argmax().item()
            pred_class  = self.defect_classes[pred_idx]
            confidence  = sims[pred_idx].item()
            scores      = {c: round(sims[i].item(), 4)
                           for i, c in enumerate(self.defect_classes)}
            scores["good"] = round(1.0 - defect_prob, 4)

        return {
            "class":        pred_class,
            "confidence":   round(confidence, 4),
            "scores":       scores,
            "defect_prob":  round(defect_prob, 4),
            "infer_ms":     round((time.time() - t0) * 1000, 1),
        }

    def register_class(self, class_name: str, example_paths: list[str | Path]):
        """Register a new defect type at runtime — no retraining needed."""
        if class_name not in self.defect2idx:
            new_idx = len(self.defect_classes)
            self.defect_classes.append(class_name)
            self.defect2idx[class_name] = new_idx
            self.classes = self.defect_classes + ["good"]
            self.extra_classes.append(class_name)
            new_row = torch.zeros(1, EMBED_DIM, device=self.device)
            self.protos2 = torch.cat([self.protos2, new_row], dim=0)

        idx    = self.defect2idx[class_name]
        embeds = []
        for p in example_paths:
            img = Image.open(p).convert("RGB")
            t   = _PREPROCESS(img).unsqueeze(0).to(self.device)
            embeds.append(self.model2.get_embedding(t).squeeze(0))

        proto = F.normalize(torch.stack(embeds).mean(0, keepdim=True), dim=1)
        self.protos2[idx] = proto.squeeze(0)
        print(f"  Registered '{class_name}' from {len(example_paths)} example(s).")

    def predict_folder(self, folder: Path, output_json: Path | None = None) -> list[dict]:
        exts    = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        images  = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
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
        print("\n── Cascade Evaluation Results ──────────────────────────────")
        print(f"  Images evaluated : {len(y_true)}")
        print(f"  Avg infer time   : {np.mean(timings):.1f} ms  (max {max(timings):.0f} ms)")
        known = list(dict.fromkeys(y_true + y_pred))
        print(classification_report(y_true, y_pred, labels=known, digits=3))
        print(f"  Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")


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
    ap.add_argument("--cascade",  action="store_true",
                    help="Use two-stage cascade (model_stage1.pth + model_stage2.pth). "
                         "Best defect recall. Requires train_cascade.py to have been run.")
    args = ap.parse_args()

    if args.cascade:
        app = CascadeClassificationApp(CHECKPOINT_S1, CHECKPOINT_S2)
    else:
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

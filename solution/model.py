"""
model.py  –  EfficientNet-B0 backbone + L2-normalized embedding head
             for prototypical few-shot defect classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

EMBED_DIM = 256


class DefectClassifier(nn.Module):
    """
    Architecture
    ────────────
    backbone  : EfficientNet-B0 (ImageNet pretrained) → 1280-d global average pool
    embed_head: FC(1280 → 256) → BN → ReLU → Dropout → FC(256 → 256) → BN → L2-norm
    classifier: cosine linear  (256 → num_classes)

    The L2-normalized embedding enables:
      • Standard softmax classification during training
      • Prototype-based few-shot classification at inference
        (new classes = just compute mean embedding of ≥1 example)
    """

    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features  # 1280 for B0

        self.embed_head = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        # Cosine classifier: weight rows are L2-normalized class vectors
        self.classifier = nn.utils.parametrize if False else nn.Linear(embed_dim, num_classes, bias=False)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x):
        feat  = self.backbone(x)           # (B, 1280)
        embed = self.embed_head(feat)      # (B, 256)
        embed = F.normalize(embed, dim=1)  # unit sphere → cosine similarity
        logits = self.classifier(embed)    # (B, C)  – dot product = cosine sim
        return logits, embed

    def get_embedding(self, x):
        feat  = self.backbone(x)
        embed = self.embed_head(feat)
        return F.normalize(embed, dim=1)

    # ── freeze / unfreeze helpers ─────────────────────────────────────────────
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 3):
        """Unfreeze the last N EfficientNet blocks for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        # blocks are in self.backbone.blocks
        blocks = list(self.backbone.blocks)
        for blk in blocks[-last_n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        # always unfreeze the head conv
        for p in self.backbone.conv_head.parameters():
            p.requires_grad = True
        for p in self.backbone.bn2.parameters():
            p.requires_grad = True


# ── Prototype utilities ───────────────────────────────────────────────────────

def compute_prototypes(
    model: DefectClassifier,
    dataloader,
    device: torch.device,
    num_classes: int,
    embed_dim: int = EMBED_DIM,
) -> torch.Tensor:
    """
    Return L2-normalized class prototypes (mean embeddings) of shape
    (num_classes, embed_dim).  Classes with no samples keep a zero vector.
    """
    model.eval()
    proto_sum = torch.zeros(num_classes, embed_dim, device=device)
    proto_cnt = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            embeds = model.get_embedding(imgs)
            for i, lbl in enumerate(labels):
                proto_sum[lbl] += embeds[i]
                proto_cnt[lbl] += 1

    proto_cnt = proto_cnt.clamp(min=1e-6)
    prototypes = proto_sum / proto_cnt.unsqueeze(1)
    return F.normalize(prototypes, dim=1)


def proto_predict(embed: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Cosine similarity scores (B, C) — higher = more likely."""
    return torch.mm(embed, prototypes.T)

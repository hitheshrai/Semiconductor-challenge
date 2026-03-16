"""
model_dinov2.py  —  DINOv2 ViT-Small/14 backbone for defect classification
                    Drop-in replacement for DefectClassifier (same interface).

Uses Meta's DINOv2 pretrained weights (vit_small_patch14_dinov2) via timm.
DINOv2 CLS token features excel at cosine/prototype similarity — directly
aligned with the cascade Stage 2 inference strategy.

Input: 224x224 (14-pixel patches → 16×16 grid = 256 patches)
CLS token dim: 384 (ViT-Small)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from model import EMBED_DIM

DINOV2_DIM = 384   # ViT-Small CLS token dimension


class DINOv2DefectClassifier(nn.Module):
    """
    Architecture
    ────────────
    backbone  : DINOv2 ViT-Small/14 (pretrained, timm) — CLS token (384-d)
    embed_head: FC(384→256) → BN → ReLU → Dropout → FC(256→256) → BN → L2-Norm
    classifier: cosine linear (256 → num_classes)

    Same forward/get_embedding/freeze_backbone/unfreeze_backbone interface
    as DefectClassifier and ViTDefectClassifier.
    """

    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=True,
            num_classes=0,       # remove classification head
            global_pool="token", # return CLS token
            img_size=224,
        )
        print(f"  [DINOv2] Loaded vit_small_patch14_dinov2 pretrained weights")

        self.embed_head = nn.Sequential(
            nn.Linear(DINOV2_DIM, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        feat   = self.backbone(x)          # (B, 384) CLS token
        embed  = self.embed_head(feat)     # (B, 256)
        embed  = F.normalize(embed, dim=1)
        logits = self.classifier(embed)    # (B, C)
        return logits, embed

    def get_embedding(self, x):
        feat  = self.backbone(x)
        embed = self.embed_head(feat)
        return F.normalize(embed, dim=1)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 3):
        """Unfreeze the last N transformer blocks + norm."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = list(self.backbone.blocks)
        for blk in blocks[-last_n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

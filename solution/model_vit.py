"""
model_vit.py  —  ViT-Small + MAE encoder backbone for defect classification
                 Drop-in replacement for DefectClassifier (same interface).

After MAE pretraining (train_mae.py), this model fine-tunes the encoder
for the cascade classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mae import MAEEncoder, MAE_CKPT, VIT_DIM
from model import EMBED_DIM    # reuse same 256-d embedding space

OUTPUT_DIR = MAE_CKPT.parent


class ViTDefectClassifier(nn.Module):
    """
    Architecture
    ────────────
    backbone  : MAE-pretrained ViT-Small/16 — CLS token output (384-d)
    embed_head: FC(384→256) → BN → ReLU → Dropout → FC(256→256) → BN → L2-Norm
    classifier: cosine linear (256 → num_classes)

    Same forward/get_embedding/freeze/unfreeze interface as DefectClassifier,
    so it can be dropped into train_cascade.py without other changes.
    """

    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM,
                 mae_ckpt=None):
        super().__init__()
        self.backbone = MAEEncoder()

        if mae_ckpt is not None and mae_ckpt.exists():
            ckpt = torch.load(mae_ckpt, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(ckpt["encoder_state"])
            print(f"  [ViT] Loaded MAE encoder from {mae_ckpt.name} "
                  f"(loss={ckpt.get('loss', '?'):.4f})")
        else:
            print("  [ViT] No MAE checkpoint — using random ViT-Small weights")

        self.embed_head = nn.Sequential(
            nn.Linear(VIT_DIM, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        feat  = self._cls_token(x)                 # (B, 384)
        embed = self.embed_head(feat)              # (B, 256)
        embed = F.normalize(embed, dim=1)
        logits = self.classifier(embed)            # (B, C)
        return logits, embed

    def get_embedding(self, x):
        feat  = self._cls_token(x)
        embed = self.embed_head(feat)
        return F.normalize(embed, dim=1)

    @torch.no_grad()
    def _cls_token(self, x):
        """Full ViT forward (no masking), returns CLS token (B, 384)."""
        B = x.shape[0]
        patches = self.backbone.patch_embed(x) + self.backbone.pos_embed[:, 1:, :]
        cls     = self.backbone.cls_token.expand(B, -1, -1) + self.backbone.pos_embed[:, :1, :]
        tokens  = torch.cat([cls, patches], dim=1)
        for blk in self.backbone.blocks:
            tokens = blk(tokens)
        tokens = self.backbone.norm(tokens)
        return tokens[:, 0]    # CLS token

    def _cls_token_grad(self, x):
        """Same as _cls_token but with gradients (for training)."""
        B = x.shape[0]
        patches = self.backbone.patch_embed(x) + self.backbone.pos_embed[:, 1:, :]
        cls     = self.backbone.cls_token.expand(B, -1, -1) + self.backbone.pos_embed[:, :1, :]
        tokens  = torch.cat([cls, patches], dim=1)
        for blk in self.backbone.blocks:
            tokens = blk(tokens)
        tokens = self.backbone.norm(tokens)
        return tokens[:, 0]

    def forward(self, x):
        feat  = self._cls_token_grad(x)
        embed = self.embed_head(feat)
        embed = F.normalize(embed, dim=1)
        logits = self.classifier(embed)
        return logits, embed

    def get_embedding(self, x):
        feat  = self._cls_token_grad(x)
        embed = self.embed_head(feat)
        return F.normalize(embed, dim=1)

    # ── freeze / unfreeze helpers ──────────────────────────────────────────
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

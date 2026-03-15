#!/usr/bin/env python
"""
train_mae.py  —  Masked Autoencoder (MAE) domain pretraining
               ASU / Intel Semiconductor Challenge 2026

Pretrains a ViT-Small encoder on ALL 3,778 wafer images with no labels.
The encoder learns wafer-specific visual patterns (scratch lines, ring defects,
edge patterns) that ImageNet pretraining never sees.

Architecture (He et al., "Masked Autoencoders Are Scalable Vision Learners", 2021)
────────────────────────────────────────────────────────────────────────────────────
  Encoder : ViT-Small/16 (384-d, 12 blocks, 6 heads)  — processes visible patches only
  Decoder : lightweight transformer (256-d, 4 blocks)  — reconstructs all patches
  Masking : 75% of patches masked (random uniform)
  Loss    : MSE on normalized pixel values of masked patches only

After pretraining, the encoder is saved to output/mae_encoder.pth.
Use train_cascade.py --vit to fine-tune the cascade with this backbone.

Usage
─────
  python train_mae.py                      # 300 epochs (default)
  python train_mae.py --epochs 400         # longer pretraining
"""

import sys, time, math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("../Dataset")
OUTPUT_DIR  = Path("output")
MAE_CKPT    = OUTPUT_DIR / "mae_encoder.pth"

ALL_CLASSES = ["defect1", "defect2", "defect3", "defect4", "defect5",
               "defect8", "defect9", "defect10", "good"]

IMG_SIZE    = 224
PATCH_SIZE  = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
MASK_RATIO  = 0.75

# ViT-Small dims
VIT_DIM     = 384
VIT_DEPTH   = 12

# Decoder (lightweight — does not need to be as powerful as encoder)
DEC_DIM     = 256
DEC_DEPTH   = 4
DEC_HEADS   = 8

BATCH_SIZE  = 64
LR          = 1.5e-4    # scaled: base_lr * batch_size / 256
WEIGHT_DECAY = 0.05
WARMUP_EP   = 40
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — all images, no labels
# ─────────────────────────────────────────────────────────────────────────────
def get_mae_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class UnlabeledDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths; self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def load_all_paths():
    paths = []
    for cls in ALL_CLASSES:
        d = DATASET_DIR / cls
        if not d.exists(): continue
        for f in d.iterdir():
            if f.suffix.upper() in {".PNG", ".JPG", ".JPEG"}:
                paths.append(str(f))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# MAE model
# ─────────────────────────────────────────────────────────────────────────────
class MAEEncoder(nn.Module):
    """ViT-Small encoder that processes only visible (unmasked) patches."""

    def __init__(self):
        super().__init__()
        vit = timm.create_model(
            "vit_small_patch16_224", pretrained=False, num_classes=0, global_pool=""
        )
        # Expose ViT internals
        self.patch_embed = vit.patch_embed  # Conv2d(3, 384, 16, 16)
        self.cls_token   = vit.cls_token    # (1, 1, 384)
        self.pos_embed   = vit.pos_embed    # (1, 197, 384) — idx 0 = CLS
        self.blocks      = vit.blocks
        self.norm        = vit.norm
        self.embed_dim   = vit.embed_dim    # 384

    def forward(self, x, mask: torch.Tensor):
        """
        x    : (B, 3, H, W)
        mask : (B, N) bool, True = masked (not processed by encoder)
        Returns encoder output (B, 1+n_vis, D) and ids_restore for decoder.
        """
        B = x.shape[0]

        # Patchify + add positional embeddings (skip CLS pos at index 0)
        patches = self.patch_embed(x)                         # (B, 196, 384)
        patches = patches + self.pos_embed[:, 1:, :]          # add patch pos

        # Select visible patches
        n_vis = (~mask).sum(1)[0].item()
        vis_patches = patches[~mask.unsqueeze(-1).expand_as(patches)].view(B, -1, self.embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tokens = torch.cat([cls, vis_patches], dim=1)          # (B, 1+n_vis, 384)

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens   # (B, 1+n_vis, 384)

    def get_cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward (no masking) — returns L2-normalised CLS token."""
        B = x.shape[0]
        patches = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls     = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tokens  = torch.cat([cls, patches], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return F.normalize(tokens[:, 0], dim=1)   # CLS token, L2 normalised


class MAEDecoder(nn.Module):
    """Lightweight transformer decoder that reconstructs masked patches."""

    def __init__(self, encoder_dim: int = VIT_DIM, decoder_dim: int = DEC_DIM,
                 depth: int = DEC_DEPTH, num_heads: int = DEC_HEADS,
                 num_patches: int = NUM_PATCHES, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.proj       = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, 1 + num_patches, decoder_dim), requires_grad=False
        )
        self._init_pos_embed(num_patches)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm   = nn.LayerNorm(decoder_dim)
        self.head   = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        nn.init.normal_(self.mask_token, std=0.02)

    def _init_pos_embed(self, num_patches: int):
        """2D sinusoidal positional embeddings."""
        grid = int(num_patches ** 0.5)
        pe   = torch.zeros(num_patches, self.pos_embed.shape[-1])
        pos  = torch.arange(num_patches, dtype=torch.float)
        dim  = self.pos_embed.shape[-1]
        for i in range(0, dim, 2):
            pe[:, i]     = torch.sin(pos / 10000 ** (i / dim))
            if i + 1 < dim:
                pe[:, i+1] = torch.cos(pos / 10000 ** (i / dim))
        # index 0 = CLS placeholder (zeros), 1: = patch positions
        with torch.no_grad():
            self.pos_embed[0, 1:] = pe

    def forward(self, encoder_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        encoder_out : (B, 1+n_vis, enc_dim)  — CLS + visible patch tokens
        mask        : (B, N) bool, True = masked
        Returns reconstructed patches (B, N, patch_size^2 * 3) for all patches.
        """
        B, N = mask.shape
        tokens = self.proj(encoder_out)                       # (B, 1+n_vis, dec_dim)

        # Expand mask tokens to fill masked positions
        mask_tokens = self.mask_token.expand(B, N, -1)        # (B, N, dec_dim)

        # Build full sequence: CLS + all N patch slots
        full = torch.zeros(B, 1 + N, tokens.shape[-1], device=tokens.device)
        full[:, 0]       = tokens[:, 0]                       # CLS
        full[:, 1:][~mask] = tokens[:, 1:].reshape(-1, tokens.shape[-1])
        full[:, 1:][mask]  = mask_tokens[mask]

        full = full + self.pos_embed
        full = self.blocks(full)
        full = self.norm(full)
        return self.head(full[:, 1:])                         # (B, N, p*p*3)


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MAEEncoder()
        self.decoder = MAEDecoder()
        self.patch_size = PATCH_SIZE

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B, N, p*p*3)  where N = (H/p)*(W/p)"""
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x

    def random_mask(self, N: int, mask_ratio: float, B: int, device):
        """Returns bool mask (B, N) — True = masked."""
        n_mask = int(N * mask_ratio)
        noise  = torch.rand(B, N, device=device)
        ids    = noise.argsort(dim=1)
        mask   = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :n_mask], True)
        return mask

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        N = (H // self.patch_size) ** 2

        mask          = self.random_mask(N, MASK_RATIO, B, x.device)
        encoder_out   = self.encoder(x, mask)
        pred          = self.decoder(encoder_out, mask)       # (B, N, p*p*3)

        # Target: per-patch normalised pixels (as in original MAE paper)
        target = self.patchify(x)                             # (B, N, p*p*3)
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        # Loss on masked patches only
        loss = ((pred - target) ** 2)[mask].mean()
        return loss, pred, mask


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────
def lr_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(epochs: int = 300):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths  = load_all_paths()
    print(f"Pretraining on {len(paths)} images (no labels)")

    loader = DataLoader(
        UnlabeledDataset(paths, get_mae_transform()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
    )

    model     = MaskedAutoencoder().to(device)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MAE parameters: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: lr_lambda(ep, WARMUP_EP, epochs)
    )

    losses = []
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f" MAE Pretraining — {epochs} epochs, mask_ratio={MASK_RATIO}")
    print(f"{'='*60}")

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for imgs in loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  ep {ep+1:03d}/{epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}  "
              f"({time.time()-t0:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":        ep,
                "encoder_state": model.encoder.state_dict(),
                "loss":         avg_loss,
                "vit_dim":      VIT_DIM,
                "patch_size":   PATCH_SIZE,
                "img_size":     IMG_SIZE,
            }, MAE_CKPT)
            print(f"    ✓ Saved encoder (loss={avg_loss:.4f})")

        # Visualise reconstruction every 50 epochs
        if (ep + 1) % 50 == 0:
            _visualise(model, paths[:4], device, ep + 1)

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")
    print(f"Encoder saved to {MAE_CKPT}")
    _plot_loss(losses, epochs)


def _visualise(model, paths, device, epoch):
    """Save a reconstruction sample to output/mae_recon_epXXX.png."""
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    model.eval()
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(device)
    with torch.no_grad():
        _, pred, mask = model(imgs)

    # Unpatchify prediction
    p    = PATCH_SIZE
    B, N = mask.shape
    G    = int(N ** 0.5)
    pred_patches = pred.reshape(B, G, G, 3, p, p)
    pred_imgs    = pred_patches.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, IMG_SIZE, IMG_SIZE)

    # Unnormalise for display
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_STD,  device=device).view(1, 3, 1, 1)
    orig = (imgs * std + mean).clamp(0, 1).cpu()
    recon = (pred_imgs * std + mean).clamp(0, 1).cpu()

    fig, axes = plt.subplots(2, B, figsize=(3 * B, 6))
    for i in range(B):
        axes[0, i].imshow(orig[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title("Original"); axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title("Reconstructed"); axes[1, i].axis("off")
    plt.suptitle(f"MAE Reconstruction — epoch {epoch}")
    plt.tight_layout()
    p_out = OUTPUT_DIR / f"mae_recon_ep{epoch:03d}.png"
    plt.savefig(p_out, dpi=100); plt.close()
    print(f"  Saved: {p_out}")
    model.train()


def _plot_loss(losses, epochs):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
    ax.set(xlabel="Epoch", ylabel="MAE Loss (MSE on masked patches)",
           title=f"MAE Pretraining — {epochs} epochs on wafer images")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = OUTPUT_DIR / "mae_loss.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    train(args.epochs)

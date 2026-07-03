"""
SensorLM sensor encoder reimplemented in PyTorch for MESA PSG sleep staging.

Reference architecture (Zhang et al., NeurIPS 2025, arXiv 2506.09108):
  - Sensor encoder = ViT-B with rectangular patch embedding and MAP pooling
  - JAX config: variant='B/10/2', pool_type='map', head_zeroinit=False
  - Original input: (1440, 26, 1) — 24h×26 per-minute wearable features

MESA adaptation decisions (see Task notes for full rationale):
  - Input: (B, n_channels, 3840) raw PSG signals @ 128Hz × 30s
  - Reshape to (B, 1, 3840, n_channels) — treat as single-channel 2D image
  - patch_h=64 (0.5s @ 128Hz), patch_w=1 (per-channel)
    → 60 time patches × n_channels channel patches = 60*n_channels tokens
  - Factored time+channel positional embedding handles variable n_channels
  - ViT-B: depth=12, width=768, mlp_dim=3072, num_heads=12 (~86M params)
  - MAP pooling (Multihead Attention Pooling, faithful to original)
  - head_zeroinit=False as in the SensorLM config
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ViT-B config (matches SensorLM paper)
VIT_B = dict(depth=12, width=768, mlp_dim=3072, num_heads=12)

PATCH_H = 64   # time samples per patch (0.5s @ 128Hz)
PATCH_W = 1    # channels per patch (process each channel independently)

MESA_HZ    = 128
EPOCH_SEC  = 30
EPOCH_SAMP = MESA_HZ * EPOCH_SEC   # 3840
T_PATCHES  = EPOCH_SAMP // PATCH_H  # 60 time patches per channel


class PatchEmbedding(nn.Module):
    """Conv2d-based patch embedding faithful to the JAX ViT stem."""

    def __init__(self, width):
        super().__init__()
        # Input: (B, 1, T, C) — 1-channel 2D image
        # Output: (B, width, T_patches, C_patches)
        self.proj = nn.Conv2d(
            1, width,
            kernel_size=(PATCH_H, PATCH_W),
            stride=(PATCH_H, PATCH_W),
        )
        # Initialise with xavier_uniform to match JAX default
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, 1, T, C)
        x = self.proj(x)        # (B, width, T_patches, C_patches)
        B, W, Tp, Cp = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, Tp, Cp, W)
        x = x.reshape(B, Tp * Cp, W)  # (B, n_tokens, width)
        return x, Tp, Cp


class MLP(nn.Module):
    def __init__(self, width, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(width, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, width)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, width, mlp_dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2  = nn.LayerNorm(width)
        self.mlp  = MLP(width, mlp_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ln1(x)
        y, _ = self.attn(y, y, y)
        x = x + self.drop(y)
        y = self.ln2(x)
        x = x + self.drop(self.mlp(y))
        return x


class MAPHead(nn.Module):
    """Multihead Attention Pooling (faithful to JAX MAPHead in vit.py)."""

    def __init__(self, width, mlp_dim, num_heads):
        super().__init__()
        self.probe = nn.Parameter(torch.empty(1, 1, width))
        nn.init.xavier_uniform_(self.probe)
        self.attn  = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.ln    = nn.LayerNorm(width)
        self.mlp   = MLP(width, mlp_dim)

    def forward(self, x):
        # x: (B, n_tokens, width)
        B = x.size(0)
        probe = self.probe.expand(B, -1, -1)   # (B, 1, width)
        y, _  = self.attn(probe, x, x)         # (B, 1, width)
        y     = y + self.mlp(self.ln(y))
        return y.squeeze(1)                     # (B, width)


class SensorLMEncoder(nn.Module):
    """
    SensorLM sensor encoder (ViT-B) adapted for MESA PSG.

    Args:
        n_channels: number of signal channels for this modality (1, 3, or 4)
        n_classes:  number of output classes (5 for sleep staging)
        dropout:    dropout rate (default 0.0 — no dropout during scratch training)
    """

    def __init__(self, n_channels, n_classes=5, dropout=0.0,
                 depth=VIT_B["depth"], width=VIT_B["width"],
                 mlp_dim=VIT_B["mlp_dim"], num_heads=VIT_B["num_heads"]):
        super().__init__()
        self.n_channels = n_channels
        self.width      = width

        # Patch embedding
        self.patch_embed = PatchEmbedding(width)

        # Factored 2D positional embedding: time_pe + channel_pe
        # Time axis: T_PATCHES positions, channel axis: n_channels positions
        # Summing (not concatenating) matches the JAX add-posemb convention.
        self.time_pe    = nn.Parameter(torch.empty(1, T_PATCHES, 1, width))
        self.channel_pe = nn.Parameter(torch.empty(1, 1, n_channels, width))
        nn.init.normal_(self.time_pe,    std=1.0 / math.sqrt(width))
        nn.init.normal_(self.channel_pe, std=1.0 / math.sqrt(width))

        # Transformer
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(width, mlp_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.ln_post = nn.LayerNorm(width)

        # MAP pooling
        self.map_head = MAPHead(width, mlp_dim, num_heads)

        # Classification head — head_zeroinit=False as in SensorLM config
        self.head = nn.Linear(width, n_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, n_channels, 3840)
        # Reshape to (B, 1, T, C) — single-channel 2D "image"
        x = x.transpose(1, 2)  # (B, 3840, n_channels)
        x = x.unsqueeze(1)     # (B, 1, 3840, n_channels) = (B, 1, T, C)

        # Patch embedding → (B, Tp*Cp, width)
        x, Tp, Cp = self.patch_embed(x)  # Tp=60, Cp=n_channels

        # Add factored positional embedding
        # time_pe:    (1, Tp, 1,  width) → broadcast over Cp
        # channel_pe: (1, 1,  Cp, width) → broadcast over Tp
        pos = (self.time_pe + self.channel_pe)  # (1, Tp, Cp, width)
        pos = pos.reshape(1, Tp * Cp, self.width)
        x   = self.drop(x + pos)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)

        # MAP pooling
        x = self.map_head(x)  # (B, width)

        return self.head(x)   # (B, n_classes)

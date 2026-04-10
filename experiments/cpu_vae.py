#!/usr/bin/env python3
"""CPU VAE experiment -- unified pipeline with cascaded stages.

Pipeline stages:
  s1:       Train UnrolledPatchVAE (fresh | resume | extend)
  refiner:  Train LatentRefiner on frozen pipeline
  s2:       Train FlattenDeflatten on frozen pipeline
  infer:    Unified inference with timing

No spatial convolutions in the encoder/decoder -- entire pipeline is
CPU-friendly (patch-based unfold/fold + linear layers).

Usage:
    # Train first stage from scratch
    python -m experiments.cpu_vae s1 --mode fresh

    # Resume training
    python -m experiments.cpu_vae s1 --mode resume --input-ckpt pipeline.pt

    # Extend with second spatial compression stage
    python -m experiments.cpu_vae s1 --mode extend --input-ckpt pipeline.pt

    # Train refiner
    python -m experiments.cpu_vae refiner --input-ckpt pipeline.pt

    # Train flatten bottleneck
    python -m experiments.cpu_vae s2 --input-ckpt pipeline.pt

    # Unified inference
    python -m experiments.cpu_vae infer --ckpt pipeline.pt
"""

import argparse
import math
import os
import pathlib
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generator import VAEpp0rGenerator
from experiments.flatten import FlattenDeflatten


# =============================================================================
# Model
# =============================================================================

class PatchVAE(nn.Module):
    """Patch-based VAE with zero spatial convolutions.

    Encoder: Unfold patches -> Linear projection to latent channels.
    Decoder: Linear projection back to patch pixels -> Fold to image.

    Supports overlapping patches: when overlap > 0, patches are extracted
    with stride = patch_size - overlap. Overlapping regions are averaged
    on decode, eliminating patch boundary artifacts.

    The only operations are reshape + matrix multiply. Fully CPU-friendly.

    Args:
        patch_size: spatial patch size (default 8 for 8x compression)
        overlap: pixel overlap between adjacent patches (default 0)
        image_channels: input channels (3 for RGB)
        latent_channels: channels per patch in latent space
        hidden_dim: optional hidden layer width (0 = direct projection)
    """

    def __init__(self, patch_size=8, overlap=0, image_channels=3,
                 latent_channels=32, hidden_dim=0):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        self.patch_dim = image_channels * patch_size * patch_size  # 3*8*8 = 192

        # Encoder: patch pixels -> latent
        if hidden_dim > 0:
            self.encoder = nn.Sequential(
                nn.Linear(self.patch_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_channels),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.patch_dim),
            )
        else:
            self.encoder = nn.Linear(self.patch_dim, latent_channels)
            self.decoder = nn.Linear(latent_channels, self.patch_dim)

    def _pad_for_patches(self, H, W):
        """Compute padding so patches tile the full image."""
        if self.overlap == 0:
            pad_h = (self.stride - H % self.stride) % self.stride
            pad_w = (self.stride - W % self.stride) % self.stride
        else:
            usable_h = H - self.patch_size
            pad_h = (self.stride - usable_h % self.stride) % self.stride if usable_h % self.stride != 0 else 0
            usable_w = W - self.patch_size
            pad_w = (self.stride - usable_w % self.stride) % self.stride if usable_w % self.stride != 0 else 0
        return pad_h, pad_w

    def _patch_grid_size(self, H, W):
        """Number of patches along each axis (after padding)."""
        pad_h, pad_w = self._pad_for_patches(H, W)
        Hp, Wp = H + pad_h, W + pad_w
        pH = (Hp - self.patch_size) // self.stride + 1
        pW = (Wp - self.patch_size) // self.stride + 1
        return pH, pW

    def encode(self, x):
        """Encode image to spatial latent grid.

        Args:
            x: (B, C, H, W) image tensor in [0, 1]

        Returns:
            latent: (B, latent_channels, pH, pW) spatial latent
        """
        B, C, H, W = x.shape
        ps = self.patch_size

        # Pad so patches tile exactly
        pad_h, pad_w = self._pad_for_patches(H, W)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        pH, pW = self._patch_grid_size(H, W)

        # Unfold to patches with stride
        patches = F.unfold(x, kernel_size=ps, stride=self.stride)

        # Transpose for linear: (B, n_patches, patch_dim)
        patches = patches.transpose(1, 2)

        # Project: (B, n_patches, patch_dim) -> (B, n_patches, latent_channels)
        latent = self.encoder(patches)

        # Reshape to spatial grid
        latent = latent.transpose(1, 2).reshape(B, self.latent_channels, pH, pW)

        return latent

    def decode(self, latent, original_size=None):
        """Decode spatial latent grid to image.

        When overlap > 0, patches overlap and F.fold sums them.
        We divide by the overlap count to average, then crop.

        Args:
            latent: (B, latent_channels, pH, pW) spatial latent
            original_size: (H, W) to crop output to. If None, returns padded.

        Returns:
            recon: (B, C, H, W) reconstructed image
        """
        B, C_lat, pH, pW = latent.shape
        ps = self.patch_size
        n_patches = pH * pW

        # Padded output size
        H_pad = (pH - 1) * self.stride + ps
        W_pad = (pW - 1) * self.stride + ps

        # Reshape to sequence: (B, n_patches, C_lat)
        seq = latent.reshape(B, C_lat, n_patches).transpose(1, 2)

        # Project: (B, n_patches, C_lat) -> (B, n_patches, patch_dim)
        patches = self.decoder(seq)

        # Transpose: (B, patch_dim, n_patches)
        patches = patches.transpose(1, 2)

        # Fold to image (sums overlapping regions)
        recon = F.fold(patches, output_size=(H_pad, W_pad),
                       kernel_size=ps, stride=self.stride)

        # Normalize by overlap count
        if self.overlap > 0:
            ones = torch.ones_like(patches)
            divisor = F.fold(ones, output_size=(H_pad, W_pad),
                             kernel_size=ps, stride=self.stride)
            recon = recon / divisor.clamp(min=1)

        # Crop to original size
        if original_size is not None:
            recon = recon[:, :, :original_size[0], :original_size[1]]

        return recon

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        latent = self.encode(x)
        recon = self.decode(latent, original_size=(H, W))
        return recon, latent

    def param_count(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


class UnrolledPatchVAE(nn.Module):
    """Unrolled patch VAE -- sub-patch pixel structure with positional encoding.

    Instead of treating each patch as an opaque 192-dim vector, this model:
      1. Unrolls each 8x8 patch into a line of 64 pixels
      2. Adds learned positional embeddings (row, col within patch)
      3. Splits into separate channel lines (R, G, B)
      4. Appends into a single sequence of 192 positions, each tagged
         with (spatial_position, channel_id) via embeddings
      5. Compresses per-patch via linear layers to latent_channels

    The encoder knows sub-patch spatial structure explicitly rather than
    learning it implicitly through a single linear projection.

    All operations are batched across patches -- no per-patch loops.
    Fully CPU-friendly -- no Conv2d anywhere.

    Args:
        patch_size: spatial patch size (default 8 for 8x compression)
        image_channels: input channels (3 for RGB)
        latent_channels: channels per patch in latent space
        inner_dim: channel width for the position-aware encoding
    """

    def __init__(self, patch_size=8, overlap=0, image_channels=3,
                 latent_channels=32, inner_dim=64, post_kernel=0,
                 hidden_dim=0):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        self.inner_dim = inner_dim
        self.post_kernel = post_kernel
        self.n_pixels = patch_size * patch_size      # 64
        self.n_positions = image_channels * self.n_pixels  # 192

        # Precompute and cache the full (1, inner_dim, 192) position embedding
        # from spatial (64 pixels) + channel (3 channels) components
        # Spatial position: which pixel in the patch (0..63)
        self.spatial_embed = nn.Parameter(
            torch.randn(1, inner_dim, self.n_pixels) * 0.02)
        # Channel identity: which channel (R=0, G=1, B=2)
        self.channel_embed = nn.Parameter(
            torch.randn(1, inner_dim, image_channels) * 0.02)

        self.hidden_dim = hidden_dim

        # Encoder: scalar values -> position-aware features -> latent
        # Step 1: project each scalar pixel value to inner_dim
        self.value_proj = nn.Linear(1, inner_dim, bias=False)
        # Step 2: mix position-aware features across the n_positions
        if hidden_dim > 0:
            self.enc_mix = nn.Sequential(
                nn.Linear(self.n_positions, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_channels),
            )
        else:
            self.enc_mix = nn.Sequential(
                nn.Linear(self.n_positions, self.n_positions),
                nn.GELU(),
                nn.Linear(self.n_positions, latent_channels),
            )

        # Optional post-encode cross-patch mixing (Conv1d over flattened grid)
        if post_kernel > 0:
            self.post_enc_mix = nn.Sequential(
                nn.Conv1d(latent_channels, latent_channels,
                          post_kernel, padding="same"),
                nn.GELU(),
                nn.Conv1d(latent_channels, latent_channels,
                          post_kernel, padding="same"),
            )
            self.pre_dec_mix = nn.Sequential(
                nn.Conv1d(latent_channels, latent_channels,
                          post_kernel, padding="same"),
                nn.GELU(),
                nn.Conv1d(latent_channels, latent_channels,
                          post_kernel, padding="same"),
            )
        else:
            self.post_enc_mix = None
            self.pre_dec_mix = None

        # Decoder: latent -> position features -> scalar pixel values
        if hidden_dim > 0:
            self.dec_mix = nn.Sequential(
                nn.Linear(latent_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_positions),
            )
        else:
            self.dec_mix = nn.Sequential(
                nn.Linear(latent_channels, self.n_positions),
                nn.GELU(),
                nn.Linear(self.n_positions, self.n_positions),
            )
        # Decoder position embeddings
        self.dec_spatial_embed = nn.Parameter(
            torch.randn(1, inner_dim, self.n_pixels) * 0.02)
        self.dec_channel_embed = nn.Parameter(
            torch.randn(1, inner_dim, image_channels) * 0.02)
        # Output: inner_dim -> 1 scalar per position
        self.value_out = nn.Linear(inner_dim, 1, bias=False)

    def _get_pos_embed(self, spatial_emb, channel_emb):
        """Build (1, inner_dim, n_positions) position encoding.

        Combines spatial (per-pixel) and channel identity embeddings.
        Layout: [R_pix0..R_pix63, G_pix0..G_pix63, B_pix0..B_pix63]
        """
        C = self.image_channels
        # spatial_emb: (1, D, 64), channel_emb: (1, D, 3)
        # For each channel c: spatial_emb + channel_emb[:,:,c:c+1] -> (1, D, 64)
        # Concatenate all -> (1, D, 192)
        return torch.cat(
            [spatial_emb + channel_emb[:, :, c:c+1] for c in range(C)],
            dim=2)

    def _pad_for_patches(self, H, W):
        """Compute padding so patches tile the full image."""
        if self.overlap == 0:
            pad_h = (self.stride - H % self.stride) % self.stride
            pad_w = (self.stride - W % self.stride) % self.stride
        else:
            usable_h = H - self.patch_size
            pad_h = (self.stride - usable_h % self.stride) % self.stride if usable_h % self.stride != 0 else 0
            usable_w = W - self.patch_size
            pad_w = (self.stride - usable_w % self.stride) % self.stride if usable_w % self.stride != 0 else 0
        return pad_h, pad_w

    def _patch_grid_size(self, H, W):
        """Number of patches along each axis (after padding)."""
        pad_h, pad_w = self._pad_for_patches(H, W)
        Hp, Wp = H + pad_h, W + pad_w
        pH = (Hp - self.patch_size) // self.stride + 1
        pW = (Wp - self.patch_size) // self.stride + 1
        return pH, pW

    def encode(self, x):
        """Encode image to spatial latent grid.

        Args:
            x: (B, C, H, W) image tensor in [0, 1]

        Returns:
            latent: (B, latent_channels, pH, pW) spatial latent
        """
        B, C, H, W = x.shape
        ps = self.patch_size

        # Pad so patches tile exactly
        pad_h, pad_w = self._pad_for_patches(H, W)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        pH, pW = self._patch_grid_size(H, W)
        n_patches = pH * pW

        # Unfold: (B, 192, n_patches)
        patches = F.unfold(x, kernel_size=ps, stride=self.stride)

        # Reshape to (B * n_patches, 192, 1) for value projection
        patches = patches.permute(0, 2, 1).reshape(B * n_patches,
                                                     self.n_positions, 1)

        # Project values: (B*P, 192, 1) -> (B*P, 192, D)
        vals = self.value_proj(patches)
        # -> (B*P, D, 192)
        vals = vals.transpose(1, 2)

        # Add position embeddings: (1, D, 192) broadcast to (B*P, D, 192)
        pos = self._get_pos_embed(self.spatial_embed, self.channel_embed)
        vals = vals + pos

        # Mix across positions and compress: (B*P, D, 192) -> (B*P, D, lat_ch)
        latent = self.enc_mix(vals)

        # Average over inner_dim: (B*P, D, lat_ch) -> (B*P, lat_ch)
        latent = latent.mean(dim=1)

        # Reshape to spatial grid: (B, lat_ch, pH, pW)
        latent = latent.reshape(B, n_patches, self.latent_channels)
        latent = latent.transpose(1, 2).reshape(B, self.latent_channels, pH, pW)

        # Optional cross-patch mixing
        if self.post_enc_mix is not None:
            B, C, pH, pW = latent.shape
            flat = latent.reshape(B, C, pH * pW)
            flat = self.post_enc_mix(flat) + flat  # residual
            latent = flat.reshape(B, C, pH, pW)

        return latent

    def decode(self, latent, original_size=None):
        """Decode spatial latent grid to image.

        When overlap > 0, patches overlap and F.fold sums them.
        We divide by the overlap count to average, then crop.

        Args:
            latent: (B, latent_channels, pH, pW)
            original_size: (H, W) to crop output to. If None, returns padded.

        Returns:
            recon: (B, C, H, W)
        """
        B, C_lat, pH, pW = latent.shape
        ps = self.patch_size
        n_patches = pH * pW

        # Padded output size
        H_pad = (pH - 1) * self.stride + ps
        W_pad = (pW - 1) * self.stride + ps

        # Optional pre-decode cross-patch mixing
        if self.pre_dec_mix is not None:
            flat = latent.reshape(B, C_lat, n_patches)
            latent = (self.pre_dec_mix(flat) + flat).reshape(B, C_lat, pH, pW)

        # (B*P, lat_ch)
        lat_seq = latent.reshape(B, C_lat, n_patches).permute(0, 2, 1)
        lat_seq = lat_seq.reshape(B * n_patches, C_lat)

        # Expand and mix: (B*P, lat_ch) -> (B*P, D, 192) via broadcast
        unpooled = self.dec_mix(lat_seq.unsqueeze(1).expand(-1, self.inner_dim, -1))

        # Add decoder position embeddings
        pos = self._get_pos_embed(self.dec_spatial_embed, self.dec_channel_embed)
        decoded = unpooled + pos

        # Project to scalar: (B*P, D, 192) -> (B*P, 192, D) -> (B*P, 192)
        decoded = decoded.transpose(1, 2)
        pixel_vals = self.value_out(decoded).squeeze(-1)

        # Reshape and fold (sums overlapping regions)
        pixel_vals = pixel_vals.reshape(B, n_patches, self.n_positions)
        pixel_vals = pixel_vals.permute(0, 2, 1)
        recon = F.fold(pixel_vals, output_size=(H_pad, W_pad),
                       kernel_size=ps, stride=self.stride)

        # Normalize by overlap count
        if self.overlap > 0:
            ones = torch.ones_like(pixel_vals)
            divisor = F.fold(ones, output_size=(H_pad, W_pad),
                             kernel_size=ps, stride=self.stride)
            recon = recon / divisor.clamp(min=1)

        # Crop to original size
        if original_size is not None:
            recon = recon[:, :, :original_size[0], :original_size[1]]

        return recon

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        latent = self.encode(x)
        recon = self.decode(latent, original_size=(H, W))
        return recon, latent

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        enc = (sum(p.numel() for p in self.value_proj.parameters())
               + sum(p.numel() for p in self.enc_mix.parameters())
               + self.spatial_embed.numel()
               + self.channel_embed.numel())
        dec = total - enc
        return {"encoder": enc, "decoder": dec, "total": total}


class LatentRefiner(nn.Module):
    """Conv1d refinement module for latent grids.

    Takes a spatial latent (B, C, H, W), serializes via walk order,
    runs residual Conv1d blocks for cross-position mixing, reshapes back.
    Same input/output shape -- no dimensionality change.

    Smooths patch boundary artifacts by mixing neighboring latent positions.

    Args:
        latent_channels: channel count of the latent grid
        spatial_h, spatial_w: spatial dims of the latent grid
        n_blocks: number of residual Conv1d blocks
        hidden_channels: internal channel width (0 = same as latent_channels)
        kernel_size: Conv1d kernel size per block
        walk_order: serialization order for 1D mixing
        dropout: dropout rate between blocks (0 = off)
    """

    def __init__(self, latent_channels=3, spatial_h=36, spatial_w=64,
                 n_blocks=4, hidden_channels=0, kernel_size=5,
                 walk_order="hilbert", dropout=0.0):
        super().__init__()
        self.C = latent_channels
        self.H = spatial_h
        self.W = spatial_w
        self.n_positions = spatial_h * spatial_w
        self.n_blocks = n_blocks
        self.walk_order = walk_order
        hc = hidden_channels if hidden_channels > 0 else latent_channels
        self.hidden_channels = hc

        # Input projection (if hidden != latent)
        if hc != latent_channels:
            self.proj_in = nn.Conv1d(latent_channels, hc, 1)
            self.proj_out = nn.Conv1d(hc, latent_channels, 1)
        else:
            self.proj_in = None
            self.proj_out = None

        # Residual Conv1d blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            layers = [
                nn.Conv1d(hc, hc, kernel_size, padding="same"),
                nn.GELU(),
                nn.Conv1d(hc, hc, kernel_size, padding="same"),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*layers))

        # Walk order indices
        self.register_buffer("walk_idx", self._build_walk_index())
        self.register_buffer("unwalk_idx", self._build_unwalk_index())

    def _build_walk_index(self):
        if self.walk_order == "raster":
            return torch.arange(self.n_positions)
        elif self.walk_order == "hilbert":
            return torch.tensor(self._hilbert_curve(self.H, self.W),
                                dtype=torch.long)
        elif self.walk_order == "morton":
            return torch.tensor(self._morton_curve(self.H, self.W),
                                dtype=torch.long)
        return torch.arange(self.n_positions)

    def _build_unwalk_index(self):
        idx = self.walk_idx
        inv = torch.zeros_like(idx)
        inv[idx] = torch.arange(len(idx))
        return inv

    def _hilbert_curve(self, h, w):
        order = max(h, w).bit_length()
        n = 1 << order
        def d2xy(n, d):
            x = y = 0
            s = 1
            while s < n:
                rx = 1 if (d & 2) else 0
                ry = 1 if ((d & 1) ^ rx) else 0
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                d >>= 2
                s <<= 1
            return x, y
        coords = []
        for d in range(n * n):
            x, y = d2xy(n, d)
            if y < h and x < w:
                coords.append(y * w + x)
        return coords

    def _morton_curve(self, h, w):
        coords = []
        for i in range(h):
            for j in range(w):
                z = 0
                for bit in range(16):
                    z |= ((i >> bit) & 1) << (2 * bit + 1)
                    z |= ((j >> bit) & 1) << (2 * bit)
                coords.append((z, i * w + j))
        coords.sort()
        return [c[1] for c in coords]

    def forward(self, latent):
        """Refine latent grid via residual Conv1d mixing.

        Args:
            latent: (B, C, H, W)

        Returns:
            refined: (B, C, H, W) -- same shape, smoothed
        """
        B, C, H, W = latent.shape
        # Flatten to 1D sequence
        seq = latent.reshape(B, C, H * W)
        # Walk reorder
        seq = seq[:, :, self.walk_idx]

        # Project to hidden channels
        if self.proj_in is not None:
            seq = self.proj_in(seq)

        # Residual blocks
        for block in self.blocks:
            seq = block(seq) + seq

        # Project back to latent channels
        if self.proj_out is not None:
            seq = self.proj_out(seq)

        # Unwalk and reshape
        seq = seq[:, :, self.unwalk_idx]
        return seq.reshape(B, C, H, W)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def _make_model(model_type, patch_size=8, overlap=0, image_channels=3,
                latent_channels=32, hidden_dim=0, inner_dim=64,
                post_kernel=0):
    """Factory to create PatchVAE or UnrolledPatchVAE from config."""
    if model_type == "unrolled":
        return UnrolledPatchVAE(
            patch_size=patch_size,
            overlap=overlap,
            image_channels=image_channels,
            latent_channels=latent_channels,
            inner_dim=inner_dim,
            post_kernel=post_kernel,
            hidden_dim=hidden_dim,
        )
    else:
        return PatchVAE(
            patch_size=patch_size,
            overlap=overlap,
            image_channels=image_channels,
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
        )


# =============================================================================
# Image loading helper
# =============================================================================

def load_real_image(path, H, W, device="cpu"):
    """Load a real image, resize to (H, W), return (1, 3, H, W) tensor in [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device)


def load_real_images(paths, H, W, device="cpu"):
    """Load multiple images, return (B, 3, H, W) tensor."""
    tensors = [load_real_image(p, H, W, device) for p in paths]
    return torch.cat(tensors, dim=0)


# =============================================================================
# Signal handling
# =============================================================================

_stop_requested = False

def _handle_stop(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[Stop requested]", flush=True)

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, _handle_stop)


# =============================================================================
# Unified pipeline: load / save / convert
# =============================================================================

def _convert_legacy(ckpt):
    """Convert old checkpoint formats to unified pipeline format.

    Handles:
    - Stage 1 checkpoints (just a single model with 'model' key)
    - Stage 1.5 fused checkpoints (s1_model + model keys)
    - Refiner checkpoints (s1_model + optional s1_5_model + refiner)

    Returns a unified pipeline checkpoint dict.
    """
    # Already in pipeline format
    if "pipeline" in ckpt:
        return ckpt

    pipeline_stages = []
    image_size = (ckpt.get("config", {}).get("H", 360),
                  ckpt.get("config", {}).get("W", 640))

    # Check if this is a refiner checkpoint
    if "refiner" in ckpt:
        # Has upstream S1
        if "s1_model" in ckpt:
            s1_cfg = ckpt["s1_config"]
            pipeline_stages.append({
                "type": s1_cfg.get("model_type", "unrolled"),
                "config": s1_cfg,
                "state_dict": ckpt["s1_model"],
            })
        # Has upstream S1.5
        if "s1_5_model" in ckpt:
            s1_5_cfg = ckpt["s1_5_config"]
            pipeline_stages.append({
                "type": s1_5_cfg.get("model_type", "unrolled"),
                "config": s1_5_cfg,
                "state_dict": ckpt["s1_5_model"],
            })
        # Refiner itself
        rcfg = ckpt.get("config", {})
        pipeline_stages.append({
            "type": "refiner",
            "config": rcfg,
            "state_dict": ckpt["refiner"],
        })
        active_stage = len(pipeline_stages) - 1

    # Check if this is an S1.5 fused checkpoint
    elif "s1_model" in ckpt and "model" in ckpt:
        s1_cfg = ckpt["s1_config"]
        pipeline_stages.append({
            "type": s1_cfg.get("model_type", "unrolled"),
            "config": s1_cfg,
            "state_dict": ckpt["s1_model"],
        })
        s1_5_cfg = ckpt.get("config", {})
        pipeline_stages.append({
            "type": s1_5_cfg.get("model_type", "unrolled"),
            "config": s1_5_cfg,
            "state_dict": ckpt["model"],
        })
        active_stage = 1

    # Check if this is a stage 2 (flatten) checkpoint
    elif "bottleneck" in ckpt:
        # Need to know patch_ckpt path -- can't load it, just store config
        fcfg = ckpt.get("config", {})
        pipeline_stages.append({
            "type": "flatten",
            "config": fcfg,
            "state_dict": ckpt["bottleneck"],
        })
        active_stage = 0

    # Plain S1 checkpoint
    elif "model" in ckpt:
        cfg = ckpt.get("config", {})
        pipeline_stages.append({
            "type": cfg.get("model_type", "unrolled"),
            "config": cfg,
            "state_dict": ckpt["model"],
        })
        active_stage = 0

    else:
        raise ValueError("Unrecognized checkpoint format")

    return {
        "format_version": 2,
        "pipeline": pipeline_stages,
        "active_stage": active_stage,
        "image_size": image_size,
        "global_step": ckpt.get("global_step", ckpt.get("step", 0)),
        "optimizer": ckpt.get("optimizer"),
        "scheduler": ckpt.get("scheduler"),
        "scaler": ckpt.get("scaler"),
    }


def _load_pipeline(ckpt_path, device):
    """Load unified pipeline from checkpoint.

    Returns (models_list, encode_fn, decode_fn, spatial_sizes, ckpt)
    where:
        models_list: list of (type_str, model, config) tuples
        encode_fn: chains all stages' encode
        decode_fn: chains all stages' decode in reverse
        spatial_sizes: list of (H, W) at each level (index 0 = image size)
        ckpt: the raw (possibly converted) checkpoint dict
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt = _convert_legacy(raw)

    pipeline_stages = ckpt["pipeline"]
    image_size = ckpt.get("image_size", (360, 640))

    models_list = []
    for stage in pipeline_stages:
        stype = stage["type"]
        cfg = stage["config"]
        sd = stage["state_dict"]

        if stype in ("unrolled", "patch"):
            model = _make_model(
                model_type=stype,
                patch_size=cfg.get("patch_size", 8),
                overlap=cfg.get("overlap", 0),
                image_channels=cfg.get("image_channels", 3),
                latent_channels=cfg.get("latent_channels", 32),
                hidden_dim=cfg.get("hidden_dim", 0),
                inner_dim=cfg.get("inner_dim", 64),
                post_kernel=cfg.get("post_kernel", 0),
            ).to(device)
            model.load_state_dict(sd)
            models_list.append((stype, model, cfg))

        elif stype == "refiner":
            model = LatentRefiner(
                latent_channels=cfg.get("latent_channels", 3),
                spatial_h=cfg.get("spatial_h", 36),
                spatial_w=cfg.get("spatial_w", 64),
                n_blocks=cfg.get("n_blocks", 4),
                hidden_channels=cfg.get("hidden_channels", 0),
                kernel_size=cfg.get("kernel_size", 5),
                walk_order=cfg.get("walk_order", "hilbert"),
                dropout=cfg.get("dropout", 0.0),
            ).to(device)
            model.load_state_dict(sd)
            models_list.append(("refiner", model, cfg))

        elif stype == "flatten":
            model = FlattenDeflatten(
                latent_channels=cfg.get("latent_channels", 32),
                bottleneck_channels=cfg.get("bottleneck_channels", 6),
                spatial_h=cfg.get("spatial_h", 45),
                spatial_w=cfg.get("spatial_w", 80),
                walk_order=cfg.get("walk_order", "raster"),
                kernel_size=cfg.get("kernel_size", 1),
                deflatten_hidden=cfg.get("deflatten_hidden", 0),
            ).to(device)
            model.load_state_dict(sd)
            models_list.append(("flatten", model, cfg))

    # Compute spatial sizes at each level
    sizes = [image_size]
    for model_type, model, cfg in models_list:
        if model_type in ("unrolled", "patch"):
            pH, pW = model._patch_grid_size(sizes[-1][0], sizes[-1][1])
            sizes.append((pH, pW))
        elif model_type == "refiner":
            sizes.append(sizes[-1])
        elif model_type == "flatten":
            sizes.append(sizes[-1])

    # Build encode_fn
    def encode_fn(x):
        for model_type, model, cfg in models_list:
            if model_type in ("unrolled", "patch"):
                x = model.encode(x)
            elif model_type == "refiner":
                x = model(x)
            elif model_type == "flatten":
                x = model.flatten(x)
        return x

    # Build decode_fn
    def decode_fn(x):
        for i in range(len(models_list) - 1, -1, -1):
            model_type, model, cfg = models_list[i]
            if model_type in ("unrolled", "patch"):
                x = model.decode(x, original_size=sizes[i])
            elif model_type == "refiner":
                x = model(x)
            elif model_type == "flatten":
                x = model.deflatten(x)
        return x

    return models_list, encode_fn, decode_fn, sizes, ckpt


def _make_pipeline_checkpoint(models_list, active_stage, image_size,
                              opt, sched, scaler, global_step):
    """Serialize full pipeline to checkpoint dict."""
    pipeline_stages = []
    for model_type, model, cfg in models_list:
        pipeline_stages.append({
            "type": model_type,
            "config": cfg,
            "state_dict": model.state_dict(),
        })

    ckpt = {
        "format_version": 2,
        "pipeline": pipeline_stages,
        "active_stage": active_stage,
        "image_size": image_size,
        "global_step": global_step,
        "optimizer": opt.state_dict() if opt is not None else None,
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    return ckpt


def _pipeline_name(models_list, global_step):
    """Generate a descriptive checkpoint filename from pipeline config.

    Format: {stage_descriptions}-{steps}k.pt
    Examples:
        ur-ps8-lc3-id4-hd32-o2-pk5-10k.pt
        ur-ps8-lc3-id4-hd32-o2-pk5_ur-ps3-lc3-id4-hd32-o1-20k.pt
        ur-ps8-lc3-id4-hd32-o2-pk5_ref-b4-k5-h16_10k.pt
        ur-ps8-lc3-id4-hd32-o2-pk5_flat-b1-k10-hil_5k.pt
    """
    parts = []
    for mt, m, cfg in models_list:
        if mt in ("unrolled", "patch"):
            tag = "ur" if mt == "unrolled" else "pv"
            ps = cfg.get("patch_size", 8)
            lc = cfg.get("latent_channels", 3)
            id_ = cfg.get("inner_dim", 4)
            hd = cfg.get("hidden_dim", 0)
            o = cfg.get("overlap", 0)
            pk = cfg.get("post_kernel", 0)
            s = f"{tag}-ps{ps}-lc{lc}-id{id_}"
            if hd > 0:
                s += f"-hd{hd}"
            if o > 0:
                s += f"-o{o}"
            if pk > 0:
                s += f"-pk{pk}"
            parts.append(s)
        elif mt == "refiner":
            nb = cfg.get("n_blocks", 4)
            ks = cfg.get("kernel_size", 5)
            hc = cfg.get("hidden_channels", 0)
            s = f"ref-b{nb}-k{ks}"
            if hc > 0:
                s += f"-h{hc}"
            parts.append(s)
        elif mt == "flatten":
            bc = cfg.get("bottleneck_channels", 6)
            ks = cfg.get("kernel_size", 1)
            wo = cfg.get("walk_order", "raster")
            wo_short = {"raster": "ras", "hilbert": "hil", "morton": "mor"}.get(wo, wo[:3])
            s = f"flat-b{bc}-k{ks}-{wo_short}"
            parts.append(s)

    steps_k = f"{global_step // 1000}k" if global_step >= 1000 else str(global_step)
    return "_".join(parts) + f"-{steps_k}.pt"


# =============================================================================
# Preview helpers
# =============================================================================

@torch.no_grad()
def save_preview_pipeline(encode_fn, decode_fn, gen, logdir, step, device,
                          amp_dtype, preview_image=None):
    """Save GT | Full Decode preview.

    If preview_image is set, render a large reference GT|Decode row above
    a smaller synthetic sample row.
    """
    try:
        from PIL import Image

        H, W = gen.H, gen.W
        sections = []

        # -- Reference image (large, top) --
        if preview_image and os.path.exists(preview_image):
            ref = load_real_image(preview_image, H, W, device)
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                lat = encode_fn(ref)
                ref_recon = decode_fn(lat)
            ref_gt = ref[0].cpu().numpy().transpose(1, 2, 0)
            ref_rc = ref_recon[0, :3].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            ref_gt = (ref_gt * 255).clip(0, 255).astype(np.uint8)
            ref_rc = (ref_rc * 255).clip(0, 255).astype(np.uint8)
            sep_v = np.full((H, 4, 3), 14, dtype=np.uint8)
            ref_row = np.concatenate([ref_gt, sep_v, ref_rc], axis=1)
            sections.append(ref_row)
            del ref, lat, ref_recon

        # -- Synthetic strip (small, bottom) --
        images = gen.generate(8)
        x = images.to(device)

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            lat = encode_fn(x)
            recon = decode_fn(lat)

        rc = recon[:, :3].clamp(0, 1).float().cpu().numpy()
        gt = images.cpu().numpy()
        del recon, x, lat

        cols, rows = 4, 2
        sep = 4
        grid_w = cols * (W * 2 + sep) + (cols - 1) * 2
        grid_h = rows * H + (rows - 1) * 2

        synth_grid = np.full((grid_h, grid_w, 3), 14, dtype=np.uint8)
        for i in range(min(8, len(gt))):
            r, c = i // cols, i % cols
            gy = r * (H + 2)
            gx = c * (W * 2 + sep + 2)
            g_img = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            r_img = (rc[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            synth_grid[gy:gy+H, gx:gx+W] = g_img
            synth_grid[gy:gy+H, gx+W+2:gx+W*2+2] = r_img
        sections.append(synth_grid)

        # -- Combine --
        if len(sections) > 1:
            syn_w = sections[1].shape[1]
            ref_h, ref_w = sections[0].shape[:2]
            from PIL import Image as _PILImg
            ref_pil = _PILImg.fromarray(sections[0])
            scale = syn_w / ref_w
            new_h = int(ref_h * scale)
            ref_pil = ref_pil.resize((syn_w, new_h), _PILImg.BILINEAR)
            sections[0] = np.array(ref_pil)
            gap = np.full((6, syn_w, 3), 14, dtype=np.uint8)
            grid = np.concatenate([sections[0], gap, sections[1]], axis=0)
        else:
            grid = sections[0]

        stepped = os.path.join(logdir, f"preview_{step:06d}.png")
        latest = os.path.join(logdir, "preview_latest.png")
        Image.fromarray(grid).save(stepped)
        Image.fromarray(grid).save(latest)
        print(f"  preview: {stepped}", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()


# =============================================================================
# train_s1: fresh | resume | extend
# =============================================================================

def train_s1(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    mode = args.mode  # fresh | resume | extend
    image_size = (args.H, args.W)

    # ---- Build pipeline ----
    models_list = []
    upstream_encode_fn = None
    upstream_decode_fn = None
    spatial_sizes = [image_size]
    global_step = 0
    loaded_opt = None
    loaded_sched = None
    loaded_scaler = None

    if mode in ("resume", "extend"):
        if not args.input_ckpt:
            raise ValueError(f"--input-ckpt required for mode={mode}")
        print(f"Loading pipeline from {args.input_ckpt}...")
        models_list, upstream_encode_fn, upstream_decode_fn, spatial_sizes, ckpt = \
            _load_pipeline(args.input_ckpt, device)
        global_step = ckpt.get("global_step", 0)
        loaded_opt = ckpt.get("optimizer")
        loaded_sched = ckpt.get("scheduler")
        loaded_scaler = ckpt.get("scaler")

        # Print loaded pipeline
        for i, (mt, m, c) in enumerate(models_list):
            if mt in ("unrolled", "patch"):
                pc = m.param_count()["total"]
            elif mt == "refiner":
                pc = m.param_count()
            else:
                pc = m.param_count()
            print(f"  Stage {i}: {mt}, {pc:,} params, "
                  f"spatial {spatial_sizes[i]} -> {spatial_sizes[i+1]}")

    if mode == "fresh":
        # Create new model operating on images (image_channels=3)
        active_model = _make_model(
            model_type=args.model_type,
            patch_size=args.patch_size,
            overlap=args.overlap,
            image_channels=3,
            latent_channels=args.latent_ch,
            hidden_dim=args.hidden_dim,
            inner_dim=args.inner_dim,
            post_kernel=args.post_kernel,
        ).to(device)

        active_cfg = {
            "model_type": args.model_type,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "image_channels": 3,
            "latent_channels": args.latent_ch,
            "hidden_dim": args.hidden_dim,
            "inner_dim": args.inner_dim,
            "post_kernel": args.post_kernel,
            "H": args.H,
            "W": args.W,
        }
        models_list = [(args.model_type, active_model, active_cfg)]
        active_stage = 0
        pH, pW = active_model._patch_grid_size(args.H, args.W)
        spatial_sizes = [image_size, (pH, pW)]

    elif mode == "resume":
        # Unfreeze the active stage and continue training
        active_stage = ckpt.get("active_stage", len(models_list) - 1)
        _mt, active_model, active_cfg = models_list[active_stage]
        active_model.requires_grad_(True)
        active_model.train()
        # Freeze everything else
        for i, (mt, m, c) in enumerate(models_list):
            if i != active_stage:
                m.eval()
                m.requires_grad_(False)

    elif mode == "extend":
        # Freeze all existing, create NEW stage
        for mt, m, c in models_list:
            m.eval()
            m.requires_grad_(False)

        # The new stage's input channels = last stage's latent channels
        last_mt, last_model, last_cfg = models_list[-1]
        if last_mt in ("unrolled", "patch"):
            input_ch = last_model.latent_channels
        elif last_mt == "refiner":
            input_ch = last_cfg.get("latent_channels", 3)
        elif last_mt == "flatten":
            input_ch = last_cfg.get("bottleneck_channels", 6)
        else:
            input_ch = 3

        active_model = _make_model(
            model_type="unrolled",
            patch_size=args.patch_size,
            overlap=args.overlap,
            image_channels=input_ch,
            latent_channels=args.latent_ch,
            hidden_dim=args.hidden_dim,
            inner_dim=args.inner_dim,
            post_kernel=args.post_kernel,
        ).to(device)

        active_cfg = {
            "model_type": "unrolled",
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "image_channels": input_ch,
            "latent_channels": args.latent_ch,
            "hidden_dim": args.hidden_dim,
            "inner_dim": args.inner_dim,
            "post_kernel": args.post_kernel,
            "H": args.H,
            "W": args.W,
        }

        active_stage = len(models_list)
        models_list.append(("unrolled", active_model, active_cfg))
        pH, pW = active_model._patch_grid_size(
            spatial_sizes[-1][0], spatial_sizes[-1][1])
        spatial_sizes.append((pH, pW))

        # Build upstream encode/decode (everything before active stage)
        upstream_models = models_list[:active_stage]
        upstream_sizes = spatial_sizes[:active_stage + 1]

        def _upstream_encode(x):
            for mt, m, c in upstream_models:
                if mt in ("unrolled", "patch"):
                    x = m.encode(x)
                elif mt == "refiner":
                    x = m(x)
                elif mt == "flatten":
                    x = m.flatten(x)
            return x

        def _upstream_decode(x):
            for i in range(len(upstream_models) - 1, -1, -1):
                mt, m, c = upstream_models[i]
                if mt in ("unrolled", "patch"):
                    x = m.decode(x, original_size=upstream_sizes[i])
                elif mt == "refiner":
                    x = m(x)
                elif mt == "flatten":
                    x = m.deflatten(x)
            return x

        upstream_encode_fn = _upstream_encode
        upstream_decode_fn = _upstream_decode

        # Reset step for new stage
        global_step = 0
        loaded_opt = None
        loaded_sched = None
        loaded_scaler = None

    # Print active model info
    pc = active_model.param_count()
    if isinstance(pc, dict):
        total_params = pc["total"]
    else:
        total_params = pc
    mb = sum(p.numel() * 4 for p in active_model.parameters()) / 1024 / 1024
    print(f"Active stage {active_stage}: {total_params:,} params, {mb:.1f}MB")
    print(f"  Pipeline: {len(models_list)} stages, "
          f"spatial {spatial_sizes[0]} -> {spatial_sizes[-1]}")
    total_dims = 1
    for d in spatial_sizes[-1]:
        total_dims *= d
    if len(models_list) > 0:
        mt, m, c = models_list[-1]
        if mt in ("unrolled", "patch"):
            lat_ch = m.latent_channels
        elif mt == "refiner":
            lat_ch = c.get("latent_channels", 3)
        elif mt == "flatten":
            lat_ch = c.get("bottleneck_channels", 6)
        else:
            lat_ch = 3
        print(f"  Final latent: {lat_ch}x{spatial_sizes[-1][0]}x{spatial_sizes[-1][1]} "
              f"= {lat_ch * spatial_sizes[-1][0] * spatial_sizes[-1][1]} dims")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True
    print(f"Generator: bank=5000, layers=128, disco=True")

    # -- LPIPS --
    lpips_fn = None
    if args.w_lpips > 0:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="squeeze").to(device)
            lpips_fn.eval()
            lpips_fn.requires_grad_(False)
            print(f"LPIPS (squeeze) on {device}")
        except ImportError:
            print("WARNING: pip install lpips for perceptual loss")

    # -- Optimizer --
    opt = torch.optim.AdamW(active_model.parameters(), lr=float(args.lr),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    # Restore optimizer state for resume
    if mode == "resume" and not args.fresh_opt:
        if loaded_opt:
            try:
                opt.load_state_dict(loaded_opt)
            except Exception:
                print("  Fresh optimizer (mismatch)")
        if loaded_sched:
            try:
                sched.load_state_dict(loaded_sched)
            except Exception:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.total_steps,
                    eta_min=float(args.lr) * 0.01, last_epoch=global_step)

    if args.fresh_opt and global_step > 0:
        opt = torch.optim.AdamW(active_model.parameters(), lr=float(args.lr),
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps - global_step,
            eta_min=float(args.lr) * 0.01)
        print(f"  Fresh optimizer from step {global_step}")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))
    if mode == "resume" and loaded_scaler and not args.fresh_opt:
        try:
            scaler.load_state_dict(loaded_scaler)
        except Exception:
            pass

    accum = args.grad_accum

    print(f"Mode: {mode}, Steps: {args.total_steps}, LR: {args.lr}, "
          f"Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}")
    print(f"Weights: mse={args.w_mse} lpips={args.w_lpips}"
          f"{f' latent={args.w_latent} pixel={args.w_pixel}' if mode == 'extend' else ''}")
    print(f"Precision: {args.precision}, Device: {device}")
    print(flush=True)

    # Build full pipeline encode/decode for previews
    all_sizes = spatial_sizes

    def full_encode(x):
        for mt, m, c in models_list:
            if mt in ("unrolled", "patch"):
                x = m.encode(x)
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.flatten(x)
        return x

    def full_decode(x):
        for i in range(len(models_list) - 1, -1, -1):
            mt, m, c = models_list[i]
            if mt in ("unrolled", "patch"):
                x = m.decode(x, original_size=all_sizes[i])
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.deflatten(x)
        return x

    def _make_ckpt():
        return _make_pipeline_checkpoint(
            models_list, active_stage, image_size,
            opt, sched, scaler, global_step)

    # Initial preview
    preview_image = getattr(args, 'preview_image', None)
    active_model.eval()
    save_preview_pipeline(full_encode, full_decode, gen, str(logdir),
                          global_step, device, amp_dtype,
                          preview_image=preview_image)

    # -- Training loop --
    t0 = time.time()
    start_step = global_step
    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    while global_step < args.total_steps:
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            print("[Stop detected, saving...]", flush=True)
            break

        active_model.train()
        opt.zero_grad(set_to_none=True)
        losses = {}

        for _ai in range(accum):
            images = gen.generate(args.batch_size)
            x = images.to(device)

            with torch.amp.autocast(device.type, dtype=amp_dtype):
                if mode == "fresh" or mode == "resume":
                    if mode == "resume" and active_stage > 0:
                        # Encode through frozen upstream stages
                        with torch.no_grad():
                            for i in range(active_stage):
                                mt, m, c = models_list[i]
                                if mt in ("unrolled", "patch"):
                                    x = m.encode(x)
                                elif mt == "refiner":
                                    x = m(x)
                                elif mt == "flatten":
                                    x = m.flatten(x)
                            target = x.clone()

                        # Forward through active model
                        recon, latent = active_model(x)
                        mse = F.mse_loss(recon, target)
                        total = args.w_mse * mse
                        losses["mse"] = losses.get("mse", 0) + mse.item() / accum
                    else:
                        # Fresh: train directly on images
                        recon, latent = active_model(x)
                        mse = F.mse_loss(recon, x)
                        total = args.w_mse * mse
                        losses["mse"] = losses.get("mse", 0) + mse.item() / accum

                    if lpips_fn is not None and mode in ("fresh",):
                        rc_lp = recon[:, :3] * 2 - 1
                        gt_lp = x[:, :3] * 2 - 1
                        lp = lpips_fn(rc_lp, gt_lp).mean()
                        total = total + args.w_lpips * lp
                        losses["lpips"] = losses.get("lpips", 0) + lp.item() / accum

                elif mode == "extend":
                    # Encode through frozen upstream
                    with torch.no_grad():
                        input_latent = upstream_encode_fn(x)

                    # Forward through active model
                    recon_lat, lat = active_model(input_latent)

                    # Latent loss: reconstructed latent vs input latent
                    lat_loss = F.mse_loss(recon_lat, input_latent)
                    losses["lat"] = losses.get("lat", 0) + lat_loss.item() / accum

                    # Pixel loss: full pipeline decode vs original image
                    with torch.no_grad():
                        # Decode through upstream
                        pass
                    recon_pixels = upstream_decode_fn(recon_lat)
                    pixel_loss = F.mse_loss(recon_pixels, x)
                    losses["pix"] = losses.get("pix", 0) + pixel_loss.item() / accum

                    total = args.w_latent * lat_loss + args.w_pixel * pixel_loss

            scaler.scale(total / accum).backward()
            del images

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        global_step += 1

        if global_step % args.log_every == 0:
            el = time.time() - t0
            steps_run = global_step - start_step
            sps = steps_run / max(el, 1)
            eta = (args.total_steps - global_step) / max(sps, 1e-6)
            lr = opt.param_groups[0]["lr"]
            ls = " ".join(f"{k}={v:.6f}" for k, v in losses.items())
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
            print(f"[{global_step}/{args.total_steps}] {ls} "
                  f"lr={lr:.1e} ({sps:.1f} step/s, {eta_str} left)", flush=True)

        if global_step % args.preview_every == 0:
            active_model.eval()
            save_preview_pipeline(full_encode, full_decode, gen, str(logdir),
                                  global_step, device, amp_dtype,
                                  preview_image=preview_image)

        if global_step % args.save_every == 0:
            d = _make_ckpt()
            named = _pipeline_name(models_list, global_step)
            torch.save(d, logdir / named)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {named}", flush=True)

            # Keep last 10 named checkpoints
            ckpts = sorted([f for f in logdir.glob("*.pt")
                           if f.name not in ("latest.pt",)
                           and not f.name.startswith(".")],
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if global_step > start_step:
        d = _make_ckpt()
        named = _pipeline_name(models_list, global_step)
        torch.save(d, logdir / named)
        torch.save(d, logdir / "latest.pt")
        print(f"  saved {named}", flush=True)

    print(f"\nDone. {global_step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min", flush=True)


# =============================================================================
# train_refiner
# =============================================================================

def train_refiner(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Load pipeline: from --resume (continue) or --input-ckpt (fresh refiner) --
    start_step = 0
    resume_ckpt = None

    if args.resume:
        # Resume: load full pipeline (includes refiner as last stage)
        print(f"Resuming from {args.resume}...")
        models_list, encode_fn, decode_fn, spatial_sizes, resume_ckpt = \
            _load_pipeline(args.resume, device)
        active_stage = resume_ckpt.get("active_stage", len(models_list) - 1)
        start_step = resume_ckpt.get("global_step", 0)
        # The refiner is already in the pipeline as the last stage
        # Freeze everything except the refiner
        for i, (mt, m, c) in enumerate(models_list):
            if i == active_stage:
                m.train()
                m.requires_grad_(True)
            else:
                m.eval()
                m.requires_grad_(False)
        _, refiner, refiner_cfg = models_list[active_stage]
        lat_H, lat_W = spatial_sizes[active_stage]
        lat_C = refiner_cfg.get("latent_channels", 3)
        print(f"  Resumed refiner at step {start_step}")

    elif args.input_ckpt:
        # Fresh refiner: load upstream pipeline, create new refiner
        print(f"Loading pipeline from {args.input_ckpt}...")
        models_list, encode_fn, decode_fn, spatial_sizes, _ = \
            _load_pipeline(args.input_ckpt, device)

        # Freeze all existing stages
        for mt, m, c in models_list:
            m.eval()
            m.requires_grad_(False)

        # Determine latent dims at end of pipeline
        last_mt, last_model, last_cfg = models_list[-1]
        if last_mt in ("unrolled", "patch"):
            lat_C = last_model.latent_channels
        elif last_mt == "refiner":
            lat_C = last_cfg.get("latent_channels", 3)
        elif last_mt == "flatten":
            lat_C = last_cfg.get("bottleneck_channels", 6)
        else:
            lat_C = 3
        lat_H, lat_W = spatial_sizes[-1]

        # Create refiner
        refiner = LatentRefiner(
            latent_channels=lat_C,
            spatial_h=lat_H, spatial_w=lat_W,
            n_blocks=args.n_blocks,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            walk_order=args.walk_order,
            dropout=args.dropout,
        ).to(device)

        refiner_cfg = {
            "latent_channels": lat_C,
            "spatial_h": lat_H,
            "spatial_w": lat_W,
            "n_blocks": args.n_blocks,
            "hidden_channels": args.hidden_channels,
            "kernel_size": args.kernel_size,
            "walk_order": args.walk_order,
            "dropout": args.dropout,
        }

        # Append refiner to pipeline
        active_stage = len(models_list)
        models_list.append(("refiner", refiner, refiner_cfg))
        spatial_sizes.append(spatial_sizes[-1])
    else:
        raise ValueError("Need --input-ckpt (fresh) or --resume (continue)")

    # Print pipeline info
    for i, (mt, m, c) in enumerate(models_list):
        pc = m.param_count()["total"] if hasattr(m.param_count(), '__getitem__') else m.param_count()
        frozen = "(frozen)" if i != active_stage else "(training)"
        print(f"  Stage {i}: {mt}, {pc:,} params {frozen}")

    hc_str = f", hidden={refiner_cfg.get('hidden_channels', 0)}" if refiner_cfg.get('hidden_channels', 0) > 0 else ""
    print(f"  Refiner: {refiner_cfg.get('n_blocks', 4)} blocks, "
          f"kernel={refiner_cfg.get('kernel_size', 5)}{hc_str}, "
          f"{refiner.param_count():,} params")

    # Finetune decoder: unfreeze pipeline[-2] (stage before refiner)
    finetune_decoder = getattr(args, 'finetune_decoder', False)
    decoder_model = None
    if finetune_decoder and active_stage >= 1:
        _, decoder_model, _ = models_list[active_stage - 1]
        decoder_model.requires_grad_(True)
        decoder_model.train()
        dec_params = sum(p.numel() for p in decoder_model.parameters()
                        if p.requires_grad)
        print(f"  Finetuning decoder (stage {active_stage - 1}): "
              f"{dec_params:,} trainable params")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True

    # -- Optimizer --
    train_params = list(refiner.parameters())
    if finetune_decoder and decoder_model is not None:
        train_params += [p for p in decoder_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=float(args.lr),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    # Load optimizer state if resuming
    if resume_ckpt and not args.fresh_opt:
        if resume_ckpt.get("optimizer"):
            try:
                opt.load_state_dict(resume_ckpt["optimizer"])
            except Exception:
                print("  Fresh optimizer (mismatch)")
        if resume_ckpt.get("scheduler"):
            sched.load_state_dict(resume_ckpt["scheduler"])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01,
                last_epoch=start_step)

    if args.fresh_opt and start_step > 0:
        opt = torch.optim.AdamW(train_params, lr=float(args.lr),
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps - start_step,
            eta_min=float(args.lr) * 0.01)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    accum = args.grad_accum
    image_size = (args.H, args.W)

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}")
    print(flush=True)

    # Build full pipeline encode/decode with refiner
    all_sizes = spatial_sizes

    def full_encode(x):
        for mt, m, c in models_list:
            if mt in ("unrolled", "patch"):
                x = m.encode(x)
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.flatten(x)
        return x

    def full_decode(x):
        for i in range(len(models_list) - 1, -1, -1):
            mt, m, c = models_list[i]
            if mt in ("unrolled", "patch"):
                x = m.decode(x, original_size=all_sizes[i])
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.deflatten(x)
        return x

    # Build upstream-only encode (without refiner) for getting clean latent
    upstream_models = models_list[:active_stage]

    def upstream_encode(x):
        for mt, m, c in upstream_models:
            if mt in ("unrolled", "patch"):
                x = m.encode(x)
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.flatten(x)
        return x

    def _make_ckpt():
        return _make_pipeline_checkpoint(
            models_list, active_stage, image_size,
            opt, sched, scaler, step)

    preview_image = getattr(args, 'preview_image', None)

    # Initial preview
    refiner.eval()
    save_preview_pipeline(full_encode, full_decode, gen, str(logdir),
                          0, device, amp_dtype, preview_image=preview_image)

    # -- Loop --
    t0 = time.time()
    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    step = 0
    for step in range(1, args.total_steps + 1):
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            break

        refiner.train()
        if finetune_decoder and decoder_model is not None:
            decoder_model.train()
        opt.zero_grad(set_to_none=True)

        for _ai in range(accum):
            images = gen.generate(args.batch_size)
            x = images.to(device)

            with torch.amp.autocast(device.type, dtype=amp_dtype):
                with torch.no_grad():
                    lat = upstream_encode(x)

                # Optional blur to give refiner a denoising signal
                if args.blur_sigma > 0:
                    noise = torch.randn_like(lat) * args.blur_sigma
                    lat_input = lat + noise
                else:
                    lat_input = lat

                lat_refined = refiner(lat_input)

                # Decode through full pipeline (refiner output -> upstream decode)
                pixel_refined = full_decode(lat_refined)

                # Pixel loss: refined decode should match original image
                pixel_loss = F.mse_loss(pixel_refined, x)
                # Latent reg: keep refined close to clean (not blurred) latent
                lat_reg = F.mse_loss(lat_refined, lat)

                total = args.w_pixel * pixel_loss + args.w_reg * lat_reg

            scaler.scale(total / accum).backward()
            del lat, lat_refined, pixel_refined, images, x

        scaler.unscale_(opt)
        clip_params = list(refiner.parameters())
        if finetune_decoder and decoder_model is not None:
            clip_params += list(decoder_model.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_every == 0:
            el = time.time() - t0
            sps = step / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"[{step}/{args.total_steps}] pix={pixel_loss.item():.6f} "
                  f"reg={lat_reg.item():.6f} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            refiner.eval()
            save_preview_pipeline(full_encode, full_decode, gen, str(logdir),
                                  step, device, amp_dtype,
                                  preview_image=preview_image)

        if step % args.save_every == 0:
            d = _make_ckpt()
            named = _pipeline_name(models_list, step)
            torch.save(d, logdir / named)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {named}", flush=True)

            ckpts = sorted([f for f in logdir.glob("*.pt")
                           if f.name not in ("latest.pt",)
                           and not f.name.startswith(".")],
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    if step > 0:
        d = _make_ckpt()
        named = _pipeline_name(models_list, step)
        torch.save(d, logdir / named)
        torch.save(d, logdir / "latest.pt")
        print(f"  saved {named}", flush=True)

    print(f"\nDone. {step} steps in {(time.time() - t0) / 60:.1f}min")


# =============================================================================
# train_s2: flatten bottleneck
# =============================================================================

def train_s2(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Load pipeline: from --resume (continue) or --input-ckpt (fresh S2) --
    start_step = 0
    resume_ckpt = None

    if args.resume:
        print(f"Resuming from {args.resume}...")
        models_list, encode_fn, decode_fn, spatial_sizes, resume_ckpt = \
            _load_pipeline(args.resume, device)
        active_stage = resume_ckpt.get("active_stage", len(models_list) - 1)
        start_step = resume_ckpt.get("global_step", 0)
        for i, (mt, m, c) in enumerate(models_list):
            if i == active_stage:
                m.train()
                m.requires_grad_(True)
            else:
                m.eval()
                m.requires_grad_(False)
        _, bottleneck, flatten_cfg = models_list[active_stage]
        lat_ch = flatten_cfg.get("latent_channels", 3)
        lat_H, lat_W = spatial_sizes[active_stage]
        print(f"  Resumed S2 at step {start_step}")

    elif args.input_ckpt:
        print(f"Loading pipeline from {args.input_ckpt}...")
        models_list, encode_fn, decode_fn, spatial_sizes, _ = \
            _load_pipeline(args.input_ckpt, device)

        for mt, m, c in models_list:
            m.eval()
            m.requires_grad_(False)

        last_mt, last_model, last_cfg = models_list[-1]
        if last_mt in ("unrolled", "patch"):
            lat_ch = last_model.latent_channels
        elif last_mt == "refiner":
            lat_ch = last_cfg.get("latent_channels", 3)
        elif last_mt == "flatten":
            lat_ch = last_cfg.get("bottleneck_channels", 6)
        else:
            lat_ch = 3
        lat_H, lat_W = spatial_sizes[-1]

        print(f"  Upstream latent: ({lat_ch}, {lat_H}, {lat_W}) = "
              f"{lat_ch * lat_H * lat_W} values")
        print(f"  Bottleneck: {args.bottleneck_ch}ch x {lat_H * lat_W} positions "
              f"= {args.bottleneck_ch * lat_H * lat_W} flat values")

        bottleneck = FlattenDeflatten(
            latent_channels=lat_ch,
            bottleneck_channels=args.bottleneck_ch,
            spatial_h=lat_H, spatial_w=lat_W,
            walk_order=args.walk_order,
            kernel_size=args.kernel_size,
            deflatten_hidden=args.deflatten_hidden,
        ).to(device)

        flatten_cfg = {
            "latent_channels": lat_ch,
            "bottleneck_channels": args.bottleneck_ch,
            "spatial_h": lat_H,
            "spatial_w": lat_W,
            "walk_order": args.walk_order,
            "kernel_size": args.kernel_size,
            "deflatten_hidden": args.deflatten_hidden,
        }

        active_stage = len(models_list)
        models_list.append(("flatten", bottleneck, flatten_cfg))
        spatial_sizes.append(spatial_sizes[-1])
    else:
        raise ValueError("Need --input-ckpt (fresh) or --resume (continue)")

    # Print pipeline
    for i, (mt, m, c) in enumerate(models_list):
        pc = m.param_count()["total"] if hasattr(m.param_count(), '__getitem__') else m.param_count()
        frozen = "(frozen)" if i != active_stage else "(training)"
        print(f"  Stage {i}: {mt}, {pc:,} params {frozen}")

    print(f"  Bottleneck: {bottleneck.param_count():,} params, "
          f"walk={flatten_cfg.get('walk_order', 'raster')}")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True
    print(f"  Generator: bank=5000, layers=128, disco=True")

    # -- Optimizer --
    opt = torch.optim.AdamW(bottleneck.parameters(), lr=float(args.lr),
                            weight_decay=0.01)

    # Load optimizer state if resuming
    if resume_ckpt and not args.fresh_opt:
        if resume_ckpt.get("optimizer"):
            try:
                opt.load_state_dict(resume_ckpt["optimizer"])
            except Exception:
                print("  Fresh optimizer (mismatch)")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    accum = args.grad_accum
    image_size = (args.H, args.W)

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}")
    print(flush=True)

    # Build full pipeline encode/decode with flatten
    all_sizes = spatial_sizes

    def full_encode(x):
        for mt, m, c in models_list:
            if mt in ("unrolled", "patch"):
                x = m.encode(x)
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.flatten(x)
        return x

    def full_decode(x):
        for i in range(len(models_list) - 1, -1, -1):
            mt, m, c = models_list[i]
            if mt in ("unrolled", "patch"):
                x = m.decode(x, original_size=all_sizes[i])
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.deflatten(x)
        return x

    # Upstream-only encode/decode (without flatten)
    upstream_models = models_list[:active_stage]
    upstream_sizes = spatial_sizes[:active_stage + 1]

    def upstream_encode(x):
        for mt, m, c in upstream_models:
            if mt in ("unrolled", "patch"):
                x = m.encode(x)
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.flatten(x)
        return x

    def upstream_decode(x):
        for i in range(len(upstream_models) - 1, -1, -1):
            mt, m, c = upstream_models[i]
            if mt in ("unrolled", "patch"):
                x = m.decode(x, original_size=upstream_sizes[i])
            elif mt == "refiner":
                x = m(x)
            elif mt == "flatten":
                x = m.deflatten(x)
        return x

    def _make_ckpt():
        return _make_pipeline_checkpoint(
            models_list, active_stage, image_size,
            opt, None, scaler, step)

    preview_image = getattr(args, 'preview_image', None)

    # Initial preview
    bottleneck.eval()
    save_preview_pipeline(full_encode, full_decode, gen, str(logdir),
                          0, device, amp_dtype, preview_image=preview_image)

    # -- Loop --
    t0 = time.time()
    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    step = 0
    for step in range(1, args.total_steps + 1):
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            break

        bottleneck.train()
        opt.zero_grad(set_to_none=True)

        for _ai in range(accum):
            images = gen.generate(args.batch_size)
            x = images.to(device)

            with torch.amp.autocast(device.type, dtype=amp_dtype):
                # Encode through frozen upstream
                with torch.no_grad():
                    latent = upstream_encode(x)

                # Flatten + deflatten
                lat_recon, flat = bottleneck(latent)

                # Latent reconstruction loss
                lat_loss = F.mse_loss(lat_recon, latent)

                # Pixel reconstruction loss through frozen upstream decoder
                _orig = image_size
                with torch.no_grad():
                    gt_recon = upstream_decode(latent)
                flat_recon = upstream_decode(lat_recon)
                pixel_loss = F.mse_loss(flat_recon, gt_recon)

                total = args.w_latent * lat_loss + args.w_pixel * pixel_loss

            scaler.scale(total / accum).backward()
            del latent, lat_recon, flat, gt_recon, flat_recon, images, x

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            el = time.time() - t0
            sps = step / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"[{step}/{args.total_steps}] lat={lat_loss.item():.6f} "
                  f"pix={pixel_loss.item():.6f} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            bottleneck.eval()
            save_preview_pipeline(full_encode, full_decode, gen, str(logdir),
                                  step, device, amp_dtype,
                                  preview_image=preview_image)

        if step % args.save_every == 0:
            d = _make_ckpt()
            named = _pipeline_name(models_list, step)
            torch.save(d, logdir / named)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {named}", flush=True)

            ckpts = sorted([f for f in logdir.glob("*.pt")
                           if f.name not in ("latest.pt",)
                           and not f.name.startswith(".")],
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    if step > 0:
        d = _make_ckpt()
        named = _pipeline_name(models_list, step)
        torch.save(d, logdir / named)
        torch.save(d, logdir / "latest.pt")
        print(f"  saved {named}", flush=True)

    print(f"\nDone. {step} steps in {(time.time() - t0) / 60:.1f}min")


# =============================================================================
# infer_pipeline: unified inference with timing
# =============================================================================

@torch.no_grad()
def infer_pipeline(args):
    """Unified inference: log pipeline info, per-stage timing, generate preview."""
    device = torch.device(args.device)

    print(f"Loading pipeline from {args.ckpt}...")
    models_list, encode_fn, decode_fn, spatial_sizes, ckpt = \
        _load_pipeline(args.ckpt, device)

    # Set all models to eval
    for mt, m, c in models_list:
        m.eval()

    print(f"\nPipeline: {len(models_list)} stages")
    print(f"  Image size: {spatial_sizes[0]}")

    total_params = 0
    for i, (mt, m, c) in enumerate(models_list):
        if mt in ("unrolled", "patch"):
            pc = m.param_count()["total"]
        elif mt == "refiner":
            pc = m.param_count()
        else:
            pc = m.param_count()
        total_params += pc

        # Latent dims at this stage's output
        out_size = spatial_sizes[i + 1]
        if mt in ("unrolled", "patch"):
            lat_ch = m.latent_channels
        elif mt == "refiner":
            lat_ch = c.get("latent_channels", 3)
        elif mt == "flatten":
            lat_ch = c.get("bottleneck_channels", 6)
        else:
            lat_ch = 3
        lat_dims = lat_ch * out_size[0] * out_size[1]

        print(f"  Stage {i}: {mt}, {pc:,} params, "
              f"latent ({lat_ch}, {out_size[0]}, {out_size[1]}) = {lat_dims} dims")

    print(f"  Total params: {total_params:,}")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]

    # -- Per-stage timing --
    print(f"\nTiming (single image, {args.precision}):")

    images = gen.generate(1)
    x = images.to(device)

    # Encode timing per stage
    encode_times = []
    current = x
    for i, (mt, m, c) in enumerate(models_list):
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            if mt in ("unrolled", "patch"):
                current = m.encode(current)
            elif mt == "refiner":
                current = m(current)
            elif mt == "flatten":
                current = m.flatten(current)

        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = (time.perf_counter() - t0) * 1000
        encode_times.append(dt)
        print(f"  Encode stage {i} ({mt}): {dt:.1f}ms")

    latent = current

    # Decode timing per stage (reverse)
    decode_times = []
    current = latent
    for i in range(len(models_list) - 1, -1, -1):
        mt, m, c = models_list[i]
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            if mt in ("unrolled", "patch"):
                current = m.decode(current, original_size=spatial_sizes[i])
            elif mt == "refiner":
                current = m(current)
            elif mt == "flatten":
                current = m.deflatten(current)

        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = (time.perf_counter() - t0) * 1000
        decode_times.append(dt)
        print(f"  Decode stage {i} ({mt}): {dt:.1f}ms")

    total_encode = sum(encode_times)
    total_decode = sum(decode_times)
    print(f"\n  Total encode: {total_encode:.1f}ms")
    print(f"  Total decode: {total_decode:.1f}ms")
    print(f"  Round-trip:   {total_encode + total_decode:.1f}ms")

    # -- Preview --
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    preview_image = getattr(args, 'preview_image', None)
    save_preview_pipeline(encode_fn, decode_fn, gen, str(logdir), 0,
                          device, amp_dtype, preview_image=preview_image)
    print(f"\nSaved to {logdir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="CPU VAE experiment (unified pipeline)")
    sub = p.add_subparsers(dest="command", required=True)

    # -- s1: train spatial compression --
    s1 = sub.add_parser("s1", help="Train spatial compression stage "
                        "(fresh | resume | extend)")
    s1.add_argument("--mode", default="fresh",
                    choices=["fresh", "resume", "extend"],
                    help="fresh=new model, resume=continue training, "
                         "extend=add new cascade stage")
    s1.add_argument("--input-ckpt", default=None,
                    help="Pipeline checkpoint (required for resume/extend)")
    s1.add_argument("--model-type", default="unrolled",
                    choices=["patch", "unrolled"],
                    help="'patch' = PatchVAE, 'unrolled' = UnrolledPatchVAE")
    s1.add_argument("--H", type=int, default=360)
    s1.add_argument("--W", type=int, default=640)
    s1.add_argument("--patch-size", type=int, default=8)
    s1.add_argument("--overlap", type=int, default=0,
                    help="Pixel overlap between adjacent patches (0=none)")
    s1.add_argument("--latent-ch", type=int, default=32)
    s1.add_argument("--hidden-dim", type=int, default=0,
                    help="Hidden layer width for PatchVAE (0 = direct)")
    s1.add_argument("--inner-dim", type=int, default=8,
                    help="Inner channel width for UnrolledPatchVAE")
    s1.add_argument("--post-kernel", type=int, default=0,
                    help="Cross-patch Conv1d kernel (0=off)")
    s1.add_argument("--batch-size", type=int, default=4)
    s1.add_argument("--lr", default="2e-4")
    s1.add_argument("--total-steps", type=int, default=30000)
    s1.add_argument("--w-mse", type=float, default=1.0)
    s1.add_argument("--w-lpips", type=float, default=0.5)
    s1.add_argument("--w-latent", type=float, default=1.0,
                    help="Latent MSE weight (extend mode)")
    s1.add_argument("--w-pixel", type=float, default=0.5,
                    help="Pixel loss weight (extend mode)")
    s1.add_argument("--precision", default="bf16",
                    choices=["fp16", "bf16", "fp32"])
    s1.add_argument("--grad-accum", type=int, default=1)
    s1.add_argument("--seed", type=int, default=42)
    s1.add_argument("--device", default="cuda:0")
    s1.add_argument("--fresh-opt", action="store_true")
    s1.add_argument("--logdir", default="cpu_vae_logs")
    s1.add_argument("--log-every", type=int, default=1)
    s1.add_argument("--save-every", type=int, default=5000)
    s1.add_argument("--preview-every", type=int, default=100)
    s1.add_argument("--preview-image", default=None,
                    help="Path to a reference image for tracking progress")

    # -- refiner: train latent refiner --
    sr = sub.add_parser("refiner", help="Train latent refiner on frozen pipeline")
    sr.add_argument("--input-ckpt", required=True,
                    help="Pipeline checkpoint to refine")
    sr.add_argument("--H", type=int, default=360)
    sr.add_argument("--W", type=int, default=640)
    sr.add_argument("--n-blocks", type=int, default=4)
    sr.add_argument("--hidden-channels", type=int, default=0,
                    help="Internal channel width (0=same as latent channels)")
    sr.add_argument("--kernel-size", type=int, default=5)
    sr.add_argument("--walk-order", default="hilbert",
                    choices=["raster", "hilbert", "morton"])
    sr.add_argument("--dropout", type=float, default=0.0)
    sr.add_argument("--blur-sigma", type=float, default=0.0,
                    help="Gaussian noise sigma added to input latent (0=off)")
    sr.add_argument("--finetune-decoder", action="store_true",
                    help="Jointly finetune the upstream decoder with the refiner")
    sr.add_argument("--batch-size", type=int, default=4)
    sr.add_argument("--lr", default="1e-3")
    sr.add_argument("--total-steps", type=int, default=10000)
    sr.add_argument("--w-pixel", type=float, default=1.0)
    sr.add_argument("--w-reg", type=float, default=0.01,
                    help="Latent regularization weight")
    sr.add_argument("--precision", default="bf16",
                    choices=["fp16", "bf16", "fp32"])
    sr.add_argument("--grad-accum", type=int, default=1)
    sr.add_argument("--seed", type=int, default=42)
    sr.add_argument("--device", default="cuda:0")
    sr.add_argument("--resume", default=None,
                    help="Resume refiner training from checkpoint")
    sr.add_argument("--fresh-opt", action="store_true")
    sr.add_argument("--logdir", default="cpu_vae_refiner_logs")
    sr.add_argument("--log-every", type=int, default=1)
    sr.add_argument("--save-every", type=int, default=2000)
    sr.add_argument("--preview-every", type=int, default=100)
    sr.add_argument("--preview-image", default=None)

    # -- s2: train flatten bottleneck --
    s2 = sub.add_parser("s2", help="Train FlattenDeflatten on frozen pipeline")
    s2.add_argument("--input-ckpt", required=True,
                    help="Pipeline checkpoint")
    s2.add_argument("--H", type=int, default=360)
    s2.add_argument("--W", type=int, default=640)
    s2.add_argument("--bottleneck-ch", type=int, default=6)
    s2.add_argument("--walk-order", default="raster",
                    choices=["raster", "hilbert", "morton"])
    s2.add_argument("--kernel-size", type=int, default=1,
                    help="Conv1d kernel size (1=per-position, 3+=cross-position mixing)")
    s2.add_argument("--deflatten-hidden", type=int, default=0,
                    help="Hidden dim in deflatten path (0=direct)")
    s2.add_argument("--batch-size", type=int, default=4)
    s2.add_argument("--lr", default="1e-3")
    s2.add_argument("--total-steps", type=int, default=10000)
    s2.add_argument("--w-latent", type=float, default=1.0)
    s2.add_argument("--w-pixel", type=float, default=0.5)
    s2.add_argument("--precision", default="bf16",
                    choices=["fp16", "bf16", "fp32"])
    s2.add_argument("--grad-accum", type=int, default=1)
    s2.add_argument("--seed", type=int, default=42)
    s2.add_argument("--device", default="cuda:0")
    s2.add_argument("--resume", default=None,
                    help="Resume S2 training from checkpoint")
    s2.add_argument("--fresh-opt", action="store_true")
    s2.add_argument("--logdir", default="cpu_vae_flatten_logs")
    s2.add_argument("--log-every", type=int, default=1)
    s2.add_argument("--save-every", type=int, default=2000)
    s2.add_argument("--preview-every", type=int, default=200)
    s2.add_argument("--preview-image", default=None)

    # -- infer: unified inference --
    inf = sub.add_parser("infer", help="Unified pipeline inference with timing")
    inf.add_argument("--ckpt", required=True,
                     help="Pipeline checkpoint")
    inf.add_argument("--H", type=int, default=360)
    inf.add_argument("--W", type=int, default=640)
    inf.add_argument("--precision", default="bf16",
                     choices=["fp16", "bf16", "fp32"])
    inf.add_argument("--device", default="cuda:0")
    inf.add_argument("--logdir", default="cpu_vae_logs")
    inf.add_argument("--preview-image", default=None)

    args = p.parse_args()

    if args.command == "s1":
        train_s1(args)
    elif args.command == "refiner":
        train_refiner(args)
    elif args.command == "s2":
        train_s2(args)
    elif args.command == "infer":
        infer_pipeline(args)


if __name__ == "__main__":
    main()

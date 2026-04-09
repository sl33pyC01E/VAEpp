#!/usr/bin/env python3
"""CPU VAE experiment — patch-based encoder/decoder with no Conv2d.

Stage 1: Train PatchVAE end-to-end (Unfold+Linear encoder, Linear+Fold decoder).
Stage 2: Freeze PatchVAE, train FlattenDeflatten bottleneck in latent space.

No spatial convolutions anywhere — entire pipeline is CPU-friendly.

Usage:
    # Stage 1: train patch encoder/decoder
    python -m experiments.cpu_vae stage1

    # Stage 2: train flatten bottleneck on frozen patch VAE
    python -m experiments.cpu_vae stage2 --patch-ckpt cpu_vae_logs/latest.pt

    # Stage 1 inference
    python -m experiments.cpu_vae infer1 --patch-ckpt cpu_vae_logs/latest.pt

    # Stage 2 inference
    python -m experiments.cpu_vae infer2 --patch-ckpt cpu_vae_logs/latest.pt \
        --flatten-ckpt cpu_vae_flatten_logs/latest.pt
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
    """Unrolled patch VAE — sub-patch pixel structure with positional encoding.

    Instead of treating each patch as an opaque 192-dim vector, this model:
      1. Unrolls each 8x8 patch into a line of 64 pixels
      2. Adds learned positional embeddings (row, col within patch)
      3. Splits into separate channel lines (R, G, B)
      4. Appends into a single sequence of 192 positions, each tagged
         with (spatial_position, channel_id) via embeddings
      5. Compresses per-patch via linear layers to latent_channels

    The encoder knows sub-patch spatial structure explicitly rather than
    learning it implicitly through a single linear projection.

    All operations are batched across patches — no per-patch loops.
    Fully CPU-friendly — no Conv2d anywhere.

    Args:
        patch_size: spatial patch size (default 8 for 8x compression)
        image_channels: input channels (3 for RGB)
        latent_channels: channels per patch in latent space
        inner_dim: channel width for the position-aware encoding
    """

    def __init__(self, patch_size=8, overlap=0, image_channels=3,
                 latent_channels=32, inner_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        self.inner_dim = inner_dim
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

        # Encoder: scalar values -> position-aware features -> latent
        # Step 1: project each scalar pixel value to inner_dim
        self.value_proj = nn.Linear(1, inner_dim, bias=False)
        # Step 2: mix position-aware features across the 192 positions
        # Using Linear (equivalent to Conv1d(k=1)) but on the position dim
        self.enc_mix = nn.Sequential(
            nn.Linear(self.n_positions, self.n_positions),
            nn.GELU(),
            nn.Linear(self.n_positions, latent_channels),
        )

        # Decoder: latent -> position features -> scalar pixel values
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


def _make_model(model_type, patch_size=8, overlap=0, image_channels=3,
                latent_channels=32, hidden_dim=0, inner_dim=64):
    """Factory to create PatchVAE or UnrolledPatchVAE from config."""
    if model_type == "unrolled":
        return UnrolledPatchVAE(
            patch_size=patch_size,
            overlap=overlap,
            image_channels=image_channels,
            latent_channels=latent_channels,
            inner_dim=inner_dim,
        )
    else:
        return PatchVAE(
            patch_size=patch_size,
            overlap=overlap,
            image_channels=image_channels,
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
        )


def _load_model(ckpt_path, device="cpu"):
    """Load PatchVAE or UnrolledPatchVAE from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model_type", "patch")

    model = _make_model(
        model_type=model_type,
        patch_size=cfg.get("patch_size", 8),
        overlap=cfg.get("overlap", 0),
        image_channels=cfg.get("image_channels", 3),
        latent_channels=cfg.get("latent_channels", 32),
        hidden_dim=cfg.get("hidden_dim", 0),
        inner_dim=cfg.get("inner_dim", 64),
    ).to(device)

    src_sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(src_sd)
    return model, ckpt, cfg


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
# Preview helpers
# =============================================================================

@torch.no_grad()
def save_preview_stage1(model, gen, logdir, step, device, amp_dtype,
                        preview_image=None):
    """Save GT | Recon grid as PNG. If preview_image path is set, render
    a large reference GT|Recon row above the synthetic strip."""
    try:
        model.eval()
        from PIL import Image

        H, W = gen.H, gen.W
        sections = []

        # -- Reference image (large, top) --
        if preview_image and os.path.exists(preview_image):
            ref = load_real_image(preview_image, H, W, device)
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                ref_recon, _ = model(ref)
            ref_gt = ref[0].cpu().numpy().transpose(1, 2, 0)
            ref_rc = ref_recon[0, :3].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            ref_gt = (ref_gt * 255).clip(0, 255).astype(np.uint8)
            ref_rc = (ref_rc * 255).clip(0, 255).astype(np.uint8)
            sep_v = np.full((H, 4, 3), 14, dtype=np.uint8)
            ref_row = np.concatenate([ref_gt, sep_v, ref_rc], axis=1)
            sections.append(ref_row)
            del ref, ref_recon

        # -- Synthetic strip (small, bottom) --
        images = gen.generate(8)
        x = images.to(device)

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            recon, _ = model(x)

        rc = recon[:, :3].clamp(0, 1).float().cpu().numpy()
        gt = images.cpu().numpy()
        del recon, x

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
            # Scale reference row to match synthetic grid width (centered)
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

        model.train()

        stepped = os.path.join(logdir, f"preview_{step:06d}.png")
        latest = os.path.join(logdir, "preview_latest.png")
        Image.fromarray(grid).save(stepped)
        Image.fromarray(grid).save(latest)
        print(f"  preview: {stepped}", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        model.train()


@torch.no_grad()
def save_preview_stage2(patch_vae, bottleneck, gen, logdir, step, device,
                        amp_dtype, preview_image=None):
    """Save GT | PatchVAE | PatchVAE+Flatten comparison. If preview_image
    is set, render a reference row above the synthetic strip."""
    try:
        patch_vae.eval()
        bottleneck.eval()
        from PIL import Image

        H, W = gen.H, gen.W
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        sections = []

        # -- Reference image (large, top) --
        if preview_image and os.path.exists(preview_image):
            ref = load_real_image(preview_image, H, W, device)
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                ref_lat = patch_vae.encode(ref)
                ref_vae = patch_vae.decode(ref_lat, original_size=(H, W))
                ref_lat_r, _ = bottleneck(ref_lat)
                ref_flat = patch_vae.decode(ref_lat_r, original_size=(H, W))
            rg = (ref[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            rv = (ref_vae[0, :3].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            rf = (ref_flat[0, :3].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            ref_row = np.concatenate([rg, sep, rv, sep, rf], axis=1)
            sections.append(ref_row)
            del ref, ref_lat, ref_vae, ref_lat_r, ref_flat

        # -- Synthetic strip --
        images = gen.generate(8)
        x = images.to(device)

        H_orig, W_orig = x.shape[2], x.shape[3]
        with torch.amp.autocast(device.type, dtype=amp_dtype):
            latent = patch_vae.encode(x)
            recon_vae = patch_vae.decode(latent, original_size=(H_orig, W_orig))
            lat_recon, _ = bottleneck(latent)
            recon_flat = patch_vae.decode(lat_recon, original_size=(H_orig, W_orig))

        gt = images.cpu().numpy()
        rc_vae = recon_vae[:, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[:, :3].clamp(0, 1).float().cpu().numpy()
        del recon_vae, recon_flat, latent, lat_recon

        rows = []
        for i in range(min(4, len(gt))):
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            v = (rc_vae[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            f = (rc_flat[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            rows.append(np.concatenate([g, sep, v, sep, f], axis=1))

        gap = np.full((4, rows[0].shape[1], 3), 14, dtype=np.uint8)
        synth_grid = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], axis=0)
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

        bottleneck.train()

        stepped = os.path.join(logdir, f"preview_{step:06d}.png")
        latest = os.path.join(logdir, "preview_latest.png")
        Image.fromarray(grid).save(stepped)
        Image.fromarray(grid).save(latest)
        print(f"  preview: {stepped} (GT | Encoder | Flatten)", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        bottleneck.train()


@torch.no_grad()
def save_real_preview_stage1(model, image_paths, H, W, logdir, device,
                             amp_dtype):
    """Load real images, encode/decode, save GT | Recon grid."""
    try:
        model.eval()
        x = load_real_images(image_paths, H, W, device)

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            recon, _ = model(x)

        rc = recon[:, :3].clamp(0, 1).float().cpu().numpy()
        gt = x.cpu().numpy()

        from PIL import Image as PILImage
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        rows = []
        for i in range(len(gt)):
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            r = (rc[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            rows.append(np.concatenate([g, sep, r], axis=1))

        gap = np.full((4, rows[0].shape[1], 3), 14, dtype=np.uint8)
        grid = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], axis=0)

        out = os.path.join(logdir, "real_preview_latest.png")
        PILImage.fromarray(grid).save(out)
        print(f"  real preview: {out} (GT | Recon)", flush=True)
        return out
    except Exception as e:
        import traceback
        print(f"  real preview failed: {e}", flush=True)
        traceback.print_exc()
        return None


@torch.no_grad()
def save_real_preview_stage2(patch_vae, bottleneck, image_paths, H, W,
                             logdir, device, amp_dtype):
    """Load real images, encode through full pipeline, save comparison."""
    try:
        patch_vae.eval()
        bottleneck.eval()
        x = load_real_images(image_paths, H, W, device)

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            latent = patch_vae.encode(x)
            recon_vae = patch_vae.decode(latent, original_size=(H, W))
            lat_recon, _ = bottleneck(latent)
            recon_flat = patch_vae.decode(lat_recon, original_size=(H, W))

        gt = x.cpu().numpy()
        rc_vae = recon_vae[:, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[:, :3].clamp(0, 1).float().cpu().numpy()

        from PIL import Image as PILImage
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        rows = []
        for i in range(len(gt)):
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            v = (rc_vae[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            f = (rc_flat[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            rows.append(np.concatenate([g, sep, v, sep, f], axis=1))

        gap = np.full((4, rows[0].shape[1], 3), 14, dtype=np.uint8)
        grid = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], axis=0)

        out = os.path.join(logdir, "real_preview_latest.png")
        PILImage.fromarray(grid).save(out)
        print(f"  real preview: {out} (GT | Encoder | Flatten)", flush=True)
        return out
    except Exception as e:
        import traceback
        print(f"  real preview failed: {e}", flush=True)
        traceback.print_exc()
        return None


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
# Stage 1: Train PatchVAE
# =============================================================================

def train_stage1(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Model --
    latent_ch = args.latent_ch
    hidden_dim = args.hidden_dim
    inner_dim = args.inner_dim
    overlap = args.overlap
    model_type = args.model_type
    if args.resume:
        _peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        _cfg = _peek.get("config", {})
        latent_ch = _cfg.get("latent_channels", latent_ch)
        hidden_dim = _cfg.get("hidden_dim", hidden_dim)
        inner_dim = _cfg.get("inner_dim", inner_dim)
        overlap = _cfg.get("overlap", overlap)
        model_type = _cfg.get("model_type", model_type)
        del _peek

    model = _make_model(
        model_type=model_type,
        patch_size=args.patch_size,
        overlap=overlap,
        image_channels=3,
        latent_channels=latent_ch,
        hidden_dim=hidden_dim,
        inner_dim=inner_dim,
    ).to(device)

    pc = model.param_count()
    mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    type_label = "UnrolledPatchVAE" if model_type == "unrolled" else "PatchVAE"
    extra = f"inner={inner_dim}" if model_type == "unrolled" else f"hidden={hidden_dim}"
    ovl = f", overlap={overlap}" if overlap > 0 else ""
    print(f"{type_label}: patch={args.patch_size}, latent={latent_ch}, "
          f"{extra}{ovl}, {pc['total']:,} params, {mb:.1f}MB")

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
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    # -- Resume --
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            global_step = ckpt.get("global_step", 0)
            if not args.fresh_opt and ckpt.get("optimizer"):
                try:
                    opt.load_state_dict(ckpt["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
            if ckpt.get("scheduler") and not args.fresh_opt:
                sched.load_state_dict(ckpt["scheduler"])
            else:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01,
                    last_epoch=global_step)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed from {args.resume} at step {global_step}")

    if args.fresh_opt and global_step > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps - global_step,
            eta_min=float(args.lr) * 0.01)
        print(f"  Fresh optimizer from step {global_step}")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    accum = args.grad_accum

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}")
    print(f"Weights: mse={args.w_mse} lpips={args.w_lpips}")
    print(f"Precision: {args.precision}, Device: {device}")
    print(flush=True)

    def _make_checkpoint():
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "config": {
                "model_type": model_type,
                "patch_size": args.patch_size,
                "overlap": overlap,
                "latent_channels": latent_ch,
                "hidden_dim": hidden_dim,
                "inner_dim": inner_dim,
                "image_channels": 3,
                "H": args.H,
                "W": args.W,
            },
        }

    # Initial preview
    preview_image = getattr(args, 'preview_image', None)
    save_preview_stage1(model, gen, str(logdir), global_step, device, amp_dtype,
                        preview_image=preview_image)

    # -- Loop --
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

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = {}

        for _ai in range(accum):
            images = gen.generate(args.batch_size)  # (B, 3, H, W)
            x = images.to(device)

            with torch.amp.autocast(device.type, dtype=amp_dtype):
                recon, latent = model(x)

                mse = F.mse_loss(recon, x)
                total = args.w_mse * mse
                losses["mse"] = losses.get("mse", 0) + mse.item() / accum

                if lpips_fn is not None:
                    rc_lp = recon[:, :3] * 2 - 1
                    gt_lp = x[:, :3] * 2 - 1
                    lp = lpips_fn(rc_lp, gt_lp).mean()
                    total = total + args.w_lpips * lp
                    losses["lpips"] = losses.get("lpips", 0) + lp.item() / accum

            scaler.scale(total / accum).backward()
            del recon, latent, images, x

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            ls = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
            print(f"[{global_step}/{args.total_steps}] {ls} "
                  f"lr={lr:.1e} ({sps:.1f} step/s, {eta_str} left)", flush=True)

        if global_step % args.preview_every == 0:
            save_preview_stage1(model, gen, str(logdir), global_step,
                                device, amp_dtype,
                                preview_image=preview_image)

        if global_step % args.save_every == 0:
            d = _make_checkpoint()
            p = logdir / f"step_{global_step:06d}.pt"
            torch.save(d, p)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {p}", flush=True)

            ckpts = sorted(logdir.glob("step_*.pt"),
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if global_step > start_step:
        d = _make_checkpoint()
        torch.save(d, logdir / f"step_{global_step:06d}.pt")
        torch.save(d, logdir / "latest.pt")
        print(f"  saved step {global_step}", flush=True)

    print(f"\nDone. {global_step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min", flush=True)


# =============================================================================
# Stage 2: Train FlattenDeflatten on frozen PatchVAE
# =============================================================================

def train_stage2(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Load frozen PatchVAE --
    print(f"Loading encoder from {args.patch_ckpt}...")
    patch_vae, ckpt, patch_cfg = _load_model(args.patch_ckpt, device)
    patch_size = patch_cfg.get("patch_size", 8)
    lat_ch = patch_cfg.get("latent_channels", 32)
    model_type = patch_cfg.get("model_type", "patch")
    patch_vae.eval()
    patch_vae.requires_grad_(False)
    type_label = "UnrolledPatchVAE" if model_type == "unrolled" else "PatchVAE"
    print(f"  {type_label}: patch={patch_size}, latent={lat_ch}, frozen")

    # Probe latent spatial dims
    lat_H = args.H // patch_size
    lat_W = args.W // patch_size
    print(f"  Latent: ({lat_ch}, {lat_H}, {lat_W}) = "
          f"{lat_ch * lat_H * lat_W} values")
    print(f"  Bottleneck: {args.bottleneck_ch}ch x {lat_H * lat_W} positions "
          f"= {args.bottleneck_ch * lat_H * lat_W} flat values")
    print(f"  Compression: {lat_ch / args.bottleneck_ch:.1f}:1 channel")

    # -- Flatten/Deflatten bottleneck --
    bottleneck = FlattenDeflatten(
        latent_channels=lat_ch,
        bottleneck_channels=args.bottleneck_ch,
        spatial_h=lat_H, spatial_w=lat_W,
        walk_order=args.walk_order,
        kernel_size=args.kernel_size,
    ).to(device)
    print(f"  Bottleneck: {bottleneck.param_count():,} params, "
          f"walk={args.walk_order}")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True
    print(f"  Generator: bank=5000, layers=128, disco=True")

    # -- Optimizer (bottleneck only) --
    opt = torch.optim.AdamW(bottleneck.parameters(), lr=float(args.lr),
                            weight_decay=0.01)

    # -- Resume --
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "bottleneck" in rk:
            bottleneck.load_state_dict(rk["bottleneck"])
            start_step = rk.get("step", 0)
            if rk.get("optimizer") and not args.fresh_opt:
                try:
                    opt.load_state_dict(rk["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
            print(f"  Resumed bottleneck from {args.resume} at step {start_step}")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    accum = args.grad_accum

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}")
    print(flush=True)

    def _make_checkpoint():
        return {
            "bottleneck": bottleneck.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": {
                "latent_channels": lat_ch,
                "bottleneck_channels": args.bottleneck_ch,
                "spatial_h": lat_H,
                "spatial_w": lat_W,
                "walk_order": args.walk_order,
                "kernel_size": args.kernel_size,
                "patch_ckpt": args.patch_ckpt,
                "patch_size": patch_size,
                "model_type": model_type,
            },
        }

    # Initial preview
    preview_image = getattr(args, 'preview_image', None)
    save_preview_stage2(patch_vae, bottleneck, gen, str(logdir), start_step,
                        device, amp_dtype, preview_image=preview_image)

    # -- Loop --
    t0 = time.time()
    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    step = start_step
    for step in range(start_step + 1, args.total_steps + 1):
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
                # Encode through frozen PatchVAE
                with torch.no_grad():
                    latent = patch_vae.encode(x)  # (B, C, pH, pW)

                # Flatten + deflatten
                lat_recon, flat = bottleneck(latent)

                # Latent reconstruction loss
                lat_loss = F.mse_loss(lat_recon, latent)

                # Pixel reconstruction loss through frozen decoder
                _orig = (args.H, args.W)
                with torch.no_grad():
                    gt_recon = patch_vae.decode(latent, original_size=_orig)
                flat_recon = patch_vae.decode(lat_recon, original_size=_orig)
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
            sps = (step - start_step) / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"[{step}/{args.total_steps}] lat={lat_loss.item():.6f} "
                  f"pix={pixel_loss.item():.6f} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            save_preview_stage2(patch_vae, bottleneck, gen, str(logdir), step,
                                device, amp_dtype,
                                preview_image=preview_image)

        if step % args.save_every == 0:
            d = _make_checkpoint()
            torch.save(d, logdir / f"step_{step:06d}.pt")
            torch.save(d, logdir / "latest.pt")
            print(f"  saved step {step}", flush=True)

            ckpts = sorted(logdir.glob("step_*.pt"),
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if step > start_step:
        d = _make_checkpoint()
        torch.save(d, logdir / f"step_{step:06d}.pt")
        torch.save(d, logdir / "latest.pt")
        print(f"  saved step {step}", flush=True)

    print(f"\nDone. {step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min")


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def infer_stage1(args):
    """Stage 1 inference: load PatchVAE/UnrolledPatchVAE, show GT | Recon."""
    device = torch.device(args.device)

    print(f"Loading model from {args.patch_ckpt}...")
    model, _, cfg = _load_model(args.patch_ckpt, device)
    model.eval()

    pc = model.param_count()
    print(f"  {cfg.get('model_type', 'patch')}: {pc['total']:,} params")

    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    save_preview_stage1(model, gen, str(logdir), 0, device, amp_dtype)
    print(f"Saved to {logdir}")


@torch.no_grad()
def infer_stage2(args):
    """Stage 2 inference: encoder + FlattenDeflatten, show GT | Patch | Flat."""
    device = torch.device(args.device)

    # Load encoder (PatchVAE or UnrolledPatchVAE)
    print(f"Loading encoder from {args.patch_ckpt}...")
    patch_vae, _, cfg = _load_model(args.patch_ckpt, device)
    patch_vae.eval()

    # Load FlattenDeflatten
    print(f"Loading FlattenDeflatten from {args.flatten_ckpt}...")
    fk = torch.load(args.flatten_ckpt, map_location="cpu", weights_only=False)
    fcfg = fk.get("config", {})

    bottleneck = FlattenDeflatten(
        latent_channels=fcfg.get("latent_channels", 32),
        bottleneck_channels=fcfg.get("bottleneck_channels", 6),
        spatial_h=fcfg.get("spatial_h", args.H // 8),
        spatial_w=fcfg.get("spatial_w", args.W // 8),
        walk_order=fcfg.get("walk_order", "raster"),
        kernel_size=fcfg.get("kernel_size", 1),
    ).to(device)
    bottleneck.load_state_dict(fk["bottleneck"])
    bottleneck.eval()

    print(f"  Bottleneck: {bottleneck.param_count():,} params")

    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    save_preview_stage2(patch_vae, bottleneck, gen, str(logdir), 0,
                        device, amp_dtype)
    print(f"Saved to {logdir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="CPU VAE experiment")
    sub = p.add_subparsers(dest="command", required=True)

    # -- Stage 1: train --
    s1 = sub.add_parser("stage1", help="Train PatchVAE or UnrolledPatchVAE")
    s1.add_argument("--model-type", default="patch",
                    choices=["patch", "unrolled"],
                    help="'patch' = PatchVAE (linear projection), "
                         "'unrolled' = UnrolledPatchVAE (sub-patch structure)")
    s1.add_argument("--H", type=int, default=360)
    s1.add_argument("--W", type=int, default=640)
    s1.add_argument("--patch-size", type=int, default=8)
    s1.add_argument("--overlap", type=int, default=0,
                    help="Pixel overlap between adjacent patches (0=none, "
                         "1-3 recommended). Blends patch boundaries on decode.")
    s1.add_argument("--latent-ch", type=int, default=32)
    s1.add_argument("--hidden-dim", type=int, default=0,
                    help="Hidden layer width for PatchVAE (0 = direct)")
    s1.add_argument("--inner-dim", type=int, default=8,
                    help="Inner channel width for UnrolledPatchVAE "
                         "(4=fast ~9ms, 8=default ~21ms, 64=slow ~250ms)")
    s1.add_argument("--batch-size", type=int, default=4)
    s1.add_argument("--lr", default="2e-4")
    s1.add_argument("--total-steps", type=int, default=30000)
    s1.add_argument("--w-mse", type=float, default=1.0)
    s1.add_argument("--w-lpips", type=float, default=0.5)
    s1.add_argument("--precision", default="bf16",
                    choices=["fp16", "bf16", "fp32"])
    s1.add_argument("--grad-accum", type=int, default=1)
    s1.add_argument("--seed", type=int, default=42)
    s1.add_argument("--device", default="cuda:0")
    s1.add_argument("--resume", default=None)
    s1.add_argument("--fresh-opt", action="store_true")
    s1.add_argument("--logdir", default="cpu_vae_logs")
    s1.add_argument("--log-every", type=int, default=1)
    s1.add_argument("--save-every", type=int, default=5000)
    s1.add_argument("--preview-every", type=int, default=100)
    s1.add_argument("--preview-image", default=None,
                    help="Path to a reference image for tracking progress")

    # -- Stage 2: train flatten --
    s2 = sub.add_parser("stage2", help="Train FlattenDeflatten on frozen PatchVAE")
    s2.add_argument("--patch-ckpt", required=True,
                    help="Path to trained PatchVAE checkpoint")
    s2.add_argument("--H", type=int, default=360)
    s2.add_argument("--W", type=int, default=640)
    s2.add_argument("--bottleneck-ch", type=int, default=6)
    s2.add_argument("--walk-order", default="raster",
                    choices=["raster", "hilbert", "morton"])
    s2.add_argument("--kernel-size", type=int, default=1,
                    help="Conv1d kernel size (1=per-position, 3+=cross-position mixing)")
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
    s2.add_argument("--resume", default=None)
    s2.add_argument("--fresh-opt", action="store_true")
    s2.add_argument("--logdir", default="cpu_vae_flatten_logs")
    s2.add_argument("--log-every", type=int, default=1)
    s2.add_argument("--save-every", type=int, default=2000)
    s2.add_argument("--preview-every", type=int, default=200)
    s2.add_argument("--preview-image", default=None,
                    help="Path to a reference image for tracking progress")

    # -- Stage 1 inference --
    i1 = sub.add_parser("infer1", help="PatchVAE inference")
    i1.add_argument("--patch-ckpt", required=True)
    i1.add_argument("--H", type=int, default=360)
    i1.add_argument("--W", type=int, default=640)
    i1.add_argument("--precision", default="bf16")
    i1.add_argument("--device", default="cuda:0")
    i1.add_argument("--logdir", default="cpu_vae_logs")

    # -- Stage 2 inference --
    i2 = sub.add_parser("infer2", help="PatchVAE + Flatten inference")
    i2.add_argument("--patch-ckpt", required=True)
    i2.add_argument("--flatten-ckpt", required=True)
    i2.add_argument("--H", type=int, default=360)
    i2.add_argument("--W", type=int, default=640)
    i2.add_argument("--precision", default="bf16")
    i2.add_argument("--device", default="cuda:0")
    i2.add_argument("--logdir", default="cpu_vae_flatten_logs")

    args = p.parse_args()

    if args.command == "stage1":
        train_stage1(args)
    elif args.command == "stage2":
        train_stage2(args)
    elif args.command == "infer1":
        infer_stage1(args)
    elif args.command == "infer2":
        infer_stage2(args)


if __name__ == "__main__":
    main()

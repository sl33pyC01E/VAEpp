#!/usr/bin/env python3
"""Pattern collage operations — combine, splice, tear, and blend patterns.

All functions take (B, 3, H, W) tensors and return (B, 3, H, W) on device.
"""

import math
import torch
import torch.nn.functional as F


def _perlin_mask(B, H, W, device, beta=1.5):
    """Generate (B, H, W) Perlin noise mask in [0, 1]."""
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.rfftfreq(W, device=device)
    fy_g, fx_g = torch.meshgrid(fy, fx, indexing="ij")
    freq_r = torch.sqrt(fy_g ** 2 + fx_g ** 2)
    phase = torch.rand(B, H, W // 2 + 1, device=device) * 2 * math.pi
    amp = 1.0 / (freq_r.unsqueeze(0) + 1e-6) ** beta
    amp[:, 0, 0] = 0
    spectrum = amp * torch.exp(1j * phase)
    noise = torch.fft.irfft2(spectrum, s=(H, W))
    n_flat = noise.reshape(B, -1)
    lo = n_flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
    hi = n_flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
    return (noise - lo) / (hi - lo + 1e-8)


def rip_collage(pat_a, pat_b):
    """Jagged tear — composite pat_a through a ragged Perlin tear onto pat_b."""
    B, _, H, W = pat_a.shape
    device = pat_a.device
    noise = _perlin_mask(B, H, W, device, beta=1.2)
    # Random threshold per image for tear position
    threshold = torch.rand(B, 1, 1, device=device) * 0.4 + 0.3
    # Soft edge tear mask
    tear = torch.sigmoid((noise - threshold) * 30)  # sharp-ish edge
    tear = tear.unsqueeze(1)  # (B, 1, H, W)
    return pat_a * tear + pat_b * (1 - tear)


def splice_regions(patterns, device):
    """Voronoi partition — each region filled by a different pattern.

    Args:
        patterns: list of (B, 3, H, W) tensors (2-6 patterns)
        device: torch device

    Returns: (B, 3, H, W) composite
    """
    n = len(patterns)
    B, _, H, W = patterns[0].shape

    # Random Voronoi centers
    centers_x = torch.rand(n, device=device) * 2 - 1
    centers_y = torch.rand(n, device=device) * 2 - 1

    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(y, x, indexing="ij")

    # Distance to each center
    dists = torch.stack([
        torch.sqrt((gx - centers_x[i]) ** 2 + (gy - centers_y[i]) ** 2)
        for i in range(n)
    ], dim=0)  # (n, H, W)
    region_map = dists.argmin(dim=0)  # (H, W)

    out = torch.zeros(B, 3, H, W, device=device)
    for i in range(n):
        mask = (region_map == i).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        out += patterns[i] * mask
    return out


def alpha_blend(pat_a, pat_b):
    """Spatially varying blend using Perlin alpha map."""
    B, _, H, W = pat_a.shape
    device = pat_a.device
    alpha = _perlin_mask(B, H, W, device, beta=1.5).unsqueeze(1)
    return pat_a * alpha + pat_b * (1 - alpha)


def merge_halves(pat_a, pat_b):
    """Split canvas in half (random direction), each half a different pattern."""
    B, _, H, W = pat_a.shape
    device = pat_a.device

    # Random split direction per image
    angle = torch.rand(B, device=device) * 2 * math.pi
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(y, x, indexing="ij")

    cos_a = angle.cos().view(B, 1, 1)
    sin_a = angle.sin().view(B, 1, 1)
    # Signed distance from split line through center
    d = gx.unsqueeze(0) * cos_a + gy.unsqueeze(0) * sin_a
    # Offset the split randomly
    offset = (torch.rand(B, device=device) * 0.6 - 0.3).view(B, 1, 1)
    mask = torch.sigmoid((d - offset) * 40).unsqueeze(1)  # soft edge
    return pat_a * mask + pat_b * (1 - mask)

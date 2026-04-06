#!/usr/bin/env python3
"""Finite Scalar Quantization (FSQ) for Tiny Game VAE.

Quantizes continuous latents into discrete tokens using per-channel
rounding with straight-through estimator. No codebook lookup needed —
codes are implicit from the quantized scalar values.

Default config: 2 groups x 6ch x 12 levels = 12^6 ~ 2.99M codes/group.
"""

import torch
import torch.nn as nn
import math


class FSQ(nn.Module):
    """Finite Scalar Quantization layer.

    Splits latent channels into groups, quantizes each channel to a
    fixed number of levels using round + straight-through estimator.

    Args:
        levels: number of quantization levels per channel (default 12).
        n_groups: number of independent quantization groups (default 2).
        channels_per_group: channels in each group (default 6).
            Total channels = n_groups * channels_per_group.
    """

    def __init__(self, levels=12, n_groups=2, channels_per_group=6):
        super().__init__()
        self.levels = levels
        self.n_groups = n_groups
        self.channels_per_group = channels_per_group
        self.total_channels = n_groups * channels_per_group
        self.codebook_size = levels ** channels_per_group

        # Precompute basis for index computation
        basis = torch.tensor(
            [levels ** i for i in range(channels_per_group)],
            dtype=torch.long)
        self.register_buffer("basis", basis)

    @property
    def num_codes(self):
        """Total number of unique codes per group."""
        return self.codebook_size

    def _bound(self, z):
        """Bound values to [-1, 1] using tanh."""
        return torch.tanh(z)

    def _quantize(self, z_bounded):
        """Quantize bounded [-1, 1] values to discrete levels."""
        half_levels = (self.levels - 1) / 2
        # Map [-1, 1] -> [0, levels-1], round, map back to [-1, 1]
        quantized = torch.round(z_bounded * half_levels) / half_levels
        return quantized

    def forward(self, z):
        """Quantize latent tensor with straight-through estimator.

        Args:
            z: (B, C, H, W) continuous latent tensor.
                C must equal n_groups * channels_per_group.

        Returns:
            z_quant: (B, C, H, W) quantized tensor (gradients flow through).
            indices: (B, n_groups, H, W) integer indices per group.
        """
        B, C, H, W = z.shape
        assert C == self.total_channels, (
            f"Expected {self.total_channels} channels, got {C}")

        # Bound to [-1, 1]
        z_bounded = self._bound(z)

        # Quantize
        z_quantized = self._quantize(z_bounded)

        # Straight-through estimator
        z_quant = z_bounded + (z_quantized - z_bounded).detach()

        # Compute indices per group
        indices = self.codes_to_indices(z_quantized)

        return z_quant, indices

    def codes_to_indices(self, z_quant):
        """Convert quantized values to integer indices.

        Args:
            z_quant: (B, C, H, W) quantized tensor with values in [-1, 1].

        Returns:
            (B, n_groups, H, W) integer indices in [0, codebook_size).
        """
        B, C, H, W = z_quant.shape
        half_levels = (self.levels - 1) / 2

        # Reshape to groups: (B, n_groups, cpg, H, W)
        z_groups = z_quant.reshape(B, self.n_groups, self.channels_per_group,
                                   H, W)

        # Map [-1, 1] -> [0, levels-1] integer
        int_codes = torch.round(
            z_groups * half_levels + half_levels).long().clamp(
                0, self.levels - 1)

        # Flatten to single index per group using mixed-radix
        # basis: [1, L, L^2, L^3, ...]
        indices = (int_codes * self.basis.view(1, 1, -1, 1, 1)).sum(dim=2)

        return indices  # (B, n_groups, H, W)

    def indices_to_codes(self, indices):
        """Convert integer indices back to quantized values.

        Args:
            indices: (B, n_groups, H, W) integer indices.

        Returns:
            (B, C, H, W) quantized tensor with values in [-1, 1].
        """
        B, G, H, W = indices.shape
        half_levels = (self.levels - 1) / 2

        # Decompose mixed-radix index into per-channel integers
        channels = []
        remaining = indices
        for i in range(self.channels_per_group):
            ch = remaining % self.levels
            remaining = remaining // self.levels
            channels.append(ch)

        # (B, G, cpg, H, W) integer codes
        int_codes = torch.stack(channels, dim=2)

        # Map [0, levels-1] -> [-1, 1]
        z_quant = (int_codes.float() - half_levels) / half_levels

        # Reshape to (B, C, H, W)
        return z_quant.reshape(B, self.total_channels, H, W)

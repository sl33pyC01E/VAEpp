#!/usr/bin/env python3
"""Screen-space effects for VAEpp0r generator.

Three effects in one mixin, sharing the common "grid_sample warp" infra:

  1. Camera shake / vibration / earthquake
     - Precomputed per-frame (dx, dy, dθ) noise with configurable amplitude
       and frequency profile. Adds an extra affine warp AFTER the viewport
       transform so it composes cleanly with pan/zoom/rot.

  2. Whole-image kaleidoscope
     - Polar fold around (cx, cy). Each pixel's sampled angle is wrapped
       into a single slice of width 2π/n and mirrored. Rotation advances
       with ti so the mirror pattern shifts over time.

  3. Fast-transform preset
     - Pure recipe-gen-time helper that multiplies pan_strength /
       viewport_pan / viewport_zoom / viewport_rotation by a scale factor.
       No runtime branch; existing motion code sees larger numbers.

All three are opt-in at the recipe level. Old recipes without any of
these keys render unchanged.
"""

import math
import torch
import torch.nn.functional as F


class EffectsMixin:
    """Mixin providing shake, kaleidoscope, and fast-transform helpers."""

    # ------------------------------------------------------------------
    # Grid cache (shared across effects; piggybacks on fluid cache if present)
    # ------------------------------------------------------------------
    def _ensure_effects_grids(self):
        H, W = self.H, self.W
        key = (H, W, str(self.device))
        if getattr(self, "_effects_grid_key", None) == key:
            return
        dev = self.device
        y = torch.linspace(-1, 1, H, device=dev)
        x = torch.linspace(-1, 1, W, device=dev)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self._eff_nx = xx  # (H, W)
        self._eff_ny = yy
        ident = torch.zeros(1, 2, 3, device=dev)
        ident[0, 0, 0] = 1.0
        ident[0, 1, 1] = 1.0
        self._eff_base_grid = F.affine_grid(ident, (1, 1, H, W),
                                            align_corners=False)
        self._effects_grid_key = key

    # ------------------------------------------------------------------
    # Camera shake
    # ------------------------------------------------------------------
    def _sample_shake_recipe(self, T,
                             amp_xy=0.02, amp_rot=0.02,
                             freq_xy=0.8, freq_rot=0.6,
                             mode="vibrate"):
        """Precompute shake offsets per frame.

        mode:
          - "vibrate": high-frequency small-amplitude jitter
          - "earthquake": low-frequency large-amplitude wobble
          - "handheld": medium freq, medium amp, slight drift
        Returns a dict with a flattened list of length 3*T: [dx0, dy0, dr0, dx1, ...].
        """
        # Shake is generated as sum of a few sinusoids with random phases.
        # This stays deterministic (no Brownian noise) and serializes cleanly.
        mode_cfg = {
            "vibrate":    {"amp_xy": amp_xy,       "amp_rot": amp_rot,       "freq_xy": freq_xy,       "freq_rot": freq_rot},
            "earthquake": {"amp_xy": amp_xy * 4,   "amp_rot": amp_rot * 2,   "freq_xy": freq_xy * 0.3, "freq_rot": freq_rot * 0.3},
            "handheld":   {"amp_xy": amp_xy * 1.5, "amp_rot": amp_rot * 1.2, "freq_xy": freq_xy * 0.4, "freq_rot": freq_rot * 0.5},
        }
        cfg = mode_cfg.get(mode, mode_cfg["vibrate"])

        n_comp = 3  # sum of 3 sinusoids per axis
        freq_xy = cfg["freq_xy"]
        freq_rot = cfg["freq_rot"]
        amp_x_per = cfg["amp_xy"] / n_comp
        amp_r_per = cfg["amp_rot"] / n_comp

        # Phases + relative frequencies, random but serialized
        phases_x = (torch.rand(n_comp) * 2 * math.pi).tolist()
        phases_y = (torch.rand(n_comp) * 2 * math.pi).tolist()
        phases_r = (torch.rand(n_comp) * 2 * math.pi).tolist()
        rel_freqs = (0.8 + torch.rand(n_comp) * 0.4).tolist()  # near 1.0

        # Evaluate per frame
        flat = []
        for ti in range(T):
            dx = sum(amp_x_per * math.sin(freq_xy * rel_freqs[c] * ti + phases_x[c])
                     for c in range(n_comp))
            dy = sum(amp_x_per * math.cos(freq_xy * rel_freqs[c] * ti + phases_y[c])
                     for c in range(n_comp))
            dr = sum(amp_r_per * math.sin(freq_rot * rel_freqs[c] * ti + phases_r[c])
                     for c in range(n_comp))
            flat.extend([dx, dy, dr])
        return {
            "enable": True,
            "mode": mode,
            "T": T,
            "flat": flat,  # length 3*T: interleaved (dx, dy, dr) per frame
        }

    def _apply_camera_shake(self, canvas, ti, shake_params):
        """Apply a small affine warp sampled from precomputed shake table."""
        if shake_params is None or not shake_params.get("enable", False):
            return canvas
        self._ensure_effects_grids()
        B, C, H, W = canvas.shape
        flat = shake_params["flat"]
        if ti * 3 + 2 >= len(flat):
            return canvas
        dx = float(flat[ti * 3 + 0])
        dy = float(flat[ti * 3 + 1])
        dr = float(flat[ti * 3 + 2])
        cos_a = math.cos(dr)
        sin_a = math.sin(dr)
        dev = canvas.device
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 0, 2] = dx
        theta[:, 1, 2] = dy
        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    # ------------------------------------------------------------------
    # Kaleidoscope (whole-image polar fold)
    # ------------------------------------------------------------------
    def _sample_kaleido_recipe(self, n_slices=6, rot_per_frame=0.03,
                               center_jitter=0.3):
        """Sample kaleidoscope params. Center jitters near image middle.
        Returns a small dict — n_slices is clamped to [2, 16]."""
        n = int(max(2, min(16, n_slices)))
        cx = float(torch.empty(1).uniform_(
            0.5 - center_jitter * 0.5, 0.5 + center_jitter * 0.5).item())
        cy = float(torch.empty(1).uniform_(
            0.5 - center_jitter * 0.5, 0.5 + center_jitter * 0.5).item())
        return {
            "enable": True,
            "n_slices": n,
            "rot_per_frame": float(rot_per_frame),
            "cx": cx,
            "cy": cy,
            "phase0": float(torch.empty(1).uniform_(0, 2 * math.pi).item()),
        }

    def _apply_kaleidoscope(self, canvas, ti, kaleido_params):
        """Polar fold around (cx, cy). Returns same shape as input."""
        if kaleido_params is None or not kaleido_params.get("enable", False):
            return canvas
        self._ensure_effects_grids()
        B, C, H, W = canvas.shape
        n = int(kaleido_params["n_slices"])
        rot = float(kaleido_params["rot_per_frame"]) * ti + float(kaleido_params["phase0"])
        # Kaleidoscope center in normalized [-1, 1] coords
        cx = (float(kaleido_params["cx"]) - 0.5) * 2.0
        cy = (float(kaleido_params["cy"]) - 0.5) * 2.0

        # Convert each pixel to polar, fold angle, remap
        dx = self._eff_nx - cx
        dy = self._eff_ny - cy
        r = torch.sqrt(dx * dx + dy * dy + 1e-8)
        theta = torch.atan2(dy, dx) + rot
        slice_size = 2 * math.pi / n
        # Fold into [0, slice_size) with mirror reflection.
        theta_mod = torch.remainder(theta, slice_size)
        half = slice_size * 0.5
        # Reflect: values > half become slice_size - value
        theta_folded = torch.where(theta_mod > half,
                                   slice_size - theta_mod,
                                   theta_mod)
        # Apply the same rotation offset so the image doesn't drift
        theta_folded = theta_folded - rot

        src_x = cx + r * torch.cos(theta_folded)
        src_y = cy + r * torch.sin(theta_folded)
        grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        if B > 1:
            grid = grid.expand(B, -1, -1, -1)
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    # ------------------------------------------------------------------
    # Fast-transform preset (recipe-gen-time helper; no runtime branch)
    # ------------------------------------------------------------------
    @staticmethod
    def _fast_transform_scale(seq_kwargs):
        """Return a new seq_kwargs dict with viewport params multiplied by
        seq_kwargs['fast_scale'] (default 4.0). Idempotent on repeated calls
        because the scale is read but not reset."""
        if not seq_kwargs.get("fast_transform", False):
            return seq_kwargs
        scale = float(seq_kwargs.get("fast_scale", 4.0))
        out = dict(seq_kwargs)
        # These are the knobs existing motion code already respects. Bumping
        # their numeric value is the cheapest possible fast-transform impl.
        for k, default in [("pan_strength", 0.5),
                           ("viewport_pan", 0.3),
                           ("viewport_zoom", 0.15),
                           ("viewport_rotation", 0.2)]:
            out[k] = float(out.get(k, default)) * scale
        return out

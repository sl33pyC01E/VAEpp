#!/usr/bin/env python3
"""Fluid-surface warp effects for VAEpp0r generator.

Height-field based screen-space ripple warping:
- Gerstner-ish background waves (sum of directional sinusoids)
- Radial raindrop impacts (expanding wavefronts with damping)

Each frame's height field is a closed-form function of (ti, T, params);
no cross-frame state. The height field is converted to a normalized-UV
displacement via finite-difference gradient, then the canvas is warped
with F.grid_sample.

All new motion is serializable into the recipe format so the recipe
pool round-trips through JSON.
"""

import math
import torch
import torch.nn.functional as F


class FluidMixin:
    """Mixin providing fluid-surface ripple warping for VAEpp0rGenerator."""

    # ------------------------------------------------------------------
    # Lazy grid init
    # ------------------------------------------------------------------
    def _ensure_fluid_grids(self):
        """Build and cache coordinate grids used by fluid math. Called lazily
        so the grids match whatever (H, W) the generator was configured with."""
        H, W = self.H, self.W
        key = (H, W, str(self.device))
        if getattr(self, "_fluid_grid_key", None) == key:
            return
        dev = self.device
        # Normalized grids in [-1, 1] for Gerstner phase / ring math
        y = torch.linspace(-1, 1, H, device=dev)
        x = torch.linspace(-1, 1, W, device=dev)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H, W) each
        self._fluid_nx = xx  # (H, W)
        self._fluid_ny = yy
        # Base affine grid for warp: identity grid_sample coords
        ident = torch.zeros(1, 2, 3, device=dev)
        ident[0, 0, 0] = 1.0
        ident[0, 1, 1] = 1.0
        self._fluid_base_grid = F.affine_grid(
            ident, (1, 1, H, W), align_corners=False)  # (1, H, W, 2)
        # Border attenuation mask (precomputed). Keeps warped UVs from reaching
        # the reflection-padded border where doubling artifacts would show.
        m = (1 - xx * xx) * (1 - yy * yy)
        self._fluid_border_mask = m.clamp(0, 1)  # (H, W)
        self._fluid_grid_key = key

    # ------------------------------------------------------------------
    # Recipe sampling helpers (serializable)
    # ------------------------------------------------------------------
    def _sample_fluid_recipe(self, T, n_drops=3,
                             amp_range=(0.01, 0.04),
                             wavelength_range=(0.04, 0.16),
                             damp_range=(0.05, 0.2),
                             speed_range=(0.5, 2.0),
                             gerstner_amp_range=(0.003, 0.018),
                             gerstner_wavelength_range=(0.1, 0.35),
                             gerstner_omega_range=(0.1, 0.6),
                             warp_strength=8.0,
                             border_atten=0.15,
                             n_gerstner=4):
        """Build a serializable fluid-effect dict for a recipe.

        Returns:
            dict with "gerstner": list, "impacts": list, "warp_strength": float,
            "border_atten": float, "enable": bool.
        """
        gerstner = []
        for _ in range(n_gerstner):
            gerstner.append({
                "amp": float(torch.empty(1).uniform_(*gerstner_amp_range).item()),
                "lambda": float(torch.empty(1).uniform_(*gerstner_wavelength_range).item()),
                "direction": float(torch.empty(1).uniform_(0, 2 * math.pi).item()),
                "omega": float(torch.empty(1).uniform_(*gerstner_omega_range).item()),
                "phase": float(torch.empty(1).uniform_(0, 2 * math.pi).item()),
            })
        impacts = []
        n_drops = max(0, int(n_drops))
        for _ in range(n_drops):
            impacts.append({
                "t": int(torch.randint(0, max(T - 1, 1), (1,)).item()),
                "x": float(torch.empty(1).uniform_(-0.9, 0.9).item()),
                "y": float(torch.empty(1).uniform_(-0.9, 0.9).item()),
                "amp": float(torch.empty(1).uniform_(*amp_range).item()),
                "lambda": float(torch.empty(1).uniform_(*wavelength_range).item()),
                "damp": float(torch.empty(1).uniform_(*damp_range).item()),
                "speed": float(torch.empty(1).uniform_(*speed_range).item()),
            })
        return {
            "enable": True,
            "gerstner": gerstner,
            "impacts": impacts,
            "warp_strength": float(warp_strength),
            "border_atten": float(border_atten),
        }

    # ------------------------------------------------------------------
    # Height field + displacement
    # ------------------------------------------------------------------
    def _compute_height_field(self, ti, T, fluid_params):
        """Compute scalar height h(x, y) at frame ti.

        Args:
            ti: frame index (int)
            T: total frames
            fluid_params: dict with "gerstner" and "impacts" lists.
        Returns:
            (H, W) float tensor on self.device.
        """
        self._ensure_fluid_grids()
        H, W = self.H, self.W
        dev = self.device
        xx, yy = self._fluid_nx, self._fluid_ny  # (H, W) each, normalized

        h = torch.zeros(H, W, device=dev)
        # Gerstner-ish: sum of directional sinusoids, phase increments with ti
        for g in fluid_params.get("gerstner", []):
            amp = g["amp"]
            wl = max(g["lambda"], 1e-3)
            dirn = g["direction"]
            omega = g["omega"]
            phase = g["phase"]
            kx = math.cos(dirn) * (2.0 * math.pi / wl)
            ky = math.sin(dirn) * (2.0 * math.pi / wl)
            h = h + amp * torch.sin(kx * xx + ky * yy + omega * ti + phase)

        # Radial impacts with Gaussian-pulse wavefront + exponential damping
        for imp in fluid_params.get("impacts", []):
            t_i = imp["t"]
            if ti < t_i:
                continue
            dt = float(ti - t_i)
            damp = imp["damp"]
            amp = imp["amp"] * math.exp(-damp * dt)
            if amp < 1e-5:  # fully decayed — cull for perf
                continue
            cx = imp["x"]
            cy = imp["y"]
            wl = max(imp["lambda"], 1e-3)
            c = imp["speed"]  # wavefront speed in normalized units per frame
            dx = xx - cx
            dy = yy - cy
            r = torch.sqrt(dx * dx + dy * dy + 1e-8)
            ring = r - c * dt
            # Gaussian pulse envelope centered on wavefront
            sigma = max(wl, 0.02)
            envelope = torch.exp(-(ring * ring) / (sigma * sigma))
            ripple = torch.cos(2.0 * math.pi * ring / wl)
            h = h + amp * ripple * envelope
        return h

    def _height_to_displacement(self, h):
        """Finite-difference gradient of h → (dx, dy) in normalized-UV units.
        Returns two (H, W) tensors."""
        # central-diff with torch.roll; boundary wraps (harmless, border-masked later)
        dx = (torch.roll(h, -1, dims=-1) - torch.roll(h, 1, dims=-1)) * 0.5
        dy = (torch.roll(h, -1, dims=-2) - torch.roll(h, 1, dims=-2)) * 0.5
        return dx, dy

    def _apply_ripples(self, canvas, ti, T, fluid_params):
        """Warp canvas via fluid-surface gradient.

        Args:
            canvas: (B, 3, H, W) in [0, 1]
            ti, T: current frame / total frames
            fluid_params: output of _sample_fluid_recipe (or None/disabled → passthrough)
        Returns:
            Warped (B, 3, H, W).
        """
        if fluid_params is None or not fluid_params.get("enable", False):
            return canvas
        self._ensure_fluid_grids()
        B, C, H, W = canvas.shape

        h = self._compute_height_field(ti, T, fluid_params)  # (H, W)
        dx, dy = self._height_to_displacement(h)  # (H, W), (H, W)

        warp_strength = float(fluid_params.get("warp_strength", 8.0))
        border_atten = float(fluid_params.get("border_atten", 0.15))

        # Convert pixel displacement to normalized-UV: u_norm = u_px / (W/2), etc.
        u = dx * (warp_strength / max(W * 0.5, 1.0))
        v = dy * (warp_strength / max(H * 0.5, 1.0))

        # Border attenuation to prevent reflection-mode seams.
        # border_mask ∈ [0, 1]; shape (H, W). We blend toward zero displacement
        # as we approach the canvas edge, with `border_atten` controlling how
        # wide the attenuation falloff is.
        atten = self._fluid_border_mask.pow(max(border_atten, 1e-3))  # (H, W)
        u = u * atten
        v = v * atten

        # Add displacement to base grid (shape (1, H, W, 2), last dim is (x, y))
        base = self._fluid_base_grid  # (1, H, W, 2)
        delta = torch.stack([u, v], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        grid = base + delta
        if B > 1:
            grid = grid.expand(B, -1, -1, -1)
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

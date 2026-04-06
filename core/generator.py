#!/usr/bin/env python3
"""SynthyperVAE procedural image generator.

Three-tier architecture for fast generation:
  Tier 1: Shape Bank — 1000+ pre-rendered RGBA shape textures
  Tier 2: Base Layers — composited scenes from shape bank
  Tier 3: Final Output — fast layer compositing with transforms

Stage 1: 2D static (single frames, RGB only).
"""

import math
import os
import torch
import torch.nn.functional as F

from core.patterns import PatternBank
from core import pattern_collage


class SynthyperGenerator:
    """GPU-accelerated procedural image generator.

    Usage:
        gen = SynthyperGenerator(360, 640, device="cuda")
        gen.build_banks()  # one-time setup
        batch = gen.generate(4)  # (4, 3, 360, 640) on GPU, [0,1]
    """

    def __init__(
        self,
        height=360,
        width=640,
        device="cuda",
        # Shape bank config
        bank_size=1000,
        shape_res=128,
        # circle, rect, triangle, ellipse, blob, line, stroke, hatch, stipple, fractal
        shape_weights=(0.12, 0.12, 0.09, 0.09, 0.10, 0.11, 0.09, 0.09, 0.07, 0.12),
        # Texture config
        texture_weights=(0.30, 0.35, 0.15, 0.20),
        perlin_bank_size=64,
        perlin_beta_range=(0.8, 2.0),
        voronoi_cells=16,
        # Edge config
        edge_weights=(0.45, 0.35, 0.20),
        soft_range=(1.0, 6.0),
        # Size distribution
        alpha=3.0,
        min_radius=0.1,   # fraction of shape_res
        max_radius=0.95,
        # Base layer config
        n_base_layers=256,
        shapes_per_layer=30,
        # Final compositing config
        layers_per_image=(3, 8),
        stamps_per_image=(3, 10),
        # Color
        saturation_range=(0.3, 0.95),
        value_range=(0.25, 1.0),
    ):
        self.H = height
        self.W = width
        self.device = torch.device(device)
        self.shape_res = shape_res

        self.bank_size = bank_size
        self.n_base_layers = n_base_layers
        self.shapes_per_layer = shapes_per_layer
        self.layers_per_image = layers_per_image
        self.stamps_per_image = stamps_per_image

        self.alpha = alpha
        self.min_r_frac = min_radius
        self.max_r_frac = max_radius
        self.voronoi_cells = voronoi_cells
        self.soft_range = soft_range
        self.sat_range = saturation_range
        self.val_range = value_range

        self.shape_probs = torch.tensor(shape_weights, device=self.device)
        self.texture_probs = torch.tensor(texture_weights, device=self.device)
        self.edge_probs = torch.tensor(edge_weights, device=self.device)

        # Scene template weights
        self.template_names = [
            "random", "horizon", "v_stripes", "h_stripes", "d_stripes",
            "grid", "radial", "perspective", "depth_layers", "symmetry",
            "border", "clusters", "gradient",
            "block_city", "landscape", "interior", "road", "water", "forest",
        ]
        default_tw = [0.20, 0.06, 0.04, 0.04, 0.04,
                       0.05, 0.03, 0.05, 0.05, 0.03,
                       0.03, 0.03, 0.03,
                       0.05, 0.05, 0.04, 0.04, 0.04, 0.04]
        self.template_probs = torch.tensor(default_tw, device=self.device)

        # Coordinate grids for shape rendering (shape_res x shape_res)
        y = torch.linspace(-1, 1, shape_res, device=self.device)
        x = torch.linspace(-1, 1, shape_res, device=self.device)
        self.sy_grid, self.sx_grid = torch.meshgrid(y, x, indexing="ij")

        # FFT freq grid for Perlin at shape resolution
        fy = torch.fft.fftfreq(shape_res, device=self.device)
        fx = torch.fft.rfftfreq(shape_res, device=self.device)
        fy_g, fx_g = torch.meshgrid(fy, fx, indexing="ij")
        self.freq_r = torch.sqrt(fy_g ** 2 + fx_g ** 2)

        # Perlin bank at shape resolution
        self.perlin_beta_range = perlin_beta_range
        self._build_perlin_bank(perlin_bank_size)

        # Banks (populated by build_banks())
        self.shape_bank = None   # (bank_size, 4, shape_res, shape_res) RGBA
        self.base_layers = None  # (n_base_layers, 3, H, W)
        self._recipe_pool = []    # list of recipe dicts (lightweight)
        self._motion_pool_T = 0
        self._motion_pool_call_count = 0

        # Pattern bank for structured pattern generation
        self.pattern_bank = PatternBank(height, width, device, self._sample_colors)
        # Disco quadrant mode: 0=pure pattern, 1=pattern+shapes, 2=dense random, 3=structured
        self.disco_quadrant = False
        self._disco_weights = torch.tensor([0.25, 0.25, 0.25, 0.25],
                                            device=device)

    # ------------------------------------------------------------------
    # Perlin noise bank
    # ------------------------------------------------------------------

    def _build_perlin_bank(self, K):
        betas = torch.rand(K, device=self.device) * (
            self.perlin_beta_range[1] - self.perlin_beta_range[0]
        ) + self.perlin_beta_range[0]
        S = self.shape_res
        bank = []
        for i in range(K):
            phase = torch.rand(S, S // 2 + 1, device=self.device) * 2 * math.pi
            amp = 1.0 / (self.freq_r + 1e-6) ** betas[i]
            amp[0, 0] = 0
            spectrum = amp * torch.exp(1j * phase)
            noise = torch.fft.irfft2(spectrum, s=(S, S))
            n_min, n_max = noise.min(), noise.max()
            noise = (noise - n_min) / (n_max - n_min + 1e-8)
            bank.append(noise)
        self.perlin_bank = torch.stack(bank)  # (K, S, S)

    # ------------------------------------------------------------------
    # Color
    # ------------------------------------------------------------------

    def _hsv_to_rgb(self, h, s, v):
        """HSV to RGB. Inputs: (N,) tensors. Returns (N, 3)."""
        h6 = (h * 6.0).unsqueeze(1)  # (N, 1)
        s = s.unsqueeze(1)
        v = v.unsqueeze(1)
        i = h6.floor().long() % 6
        f = h6 - h6.floor()
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(
            i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(
            i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(
            i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
        return torch.cat([r, g, b], dim=1)  # (N, 3)

    def _sample_colors(self, n):
        h = torch.rand(n, device=self.device)
        s_lo, s_hi = self.sat_range
        v_lo, v_hi = self.val_range
        s = torch.rand(n, device=self.device) * (s_hi - s_lo) + s_lo
        v = torch.rand(n, device=self.device) * (v_hi - v_lo) + v_lo
        return self._hsv_to_rgb(h, s, v)  # (n, 3)

    # ------------------------------------------------------------------
    # Shape SDFs (on normalized [-1,1] grid, single shape each)
    # ------------------------------------------------------------------

    def _sdf_circle(self, r_frac):
        return torch.sqrt(self.sx_grid ** 2 + self.sy_grid ** 2) - r_frac

    def _sdf_rect(self, r_frac, angle):
        hw, hh = r_frac * 1.2, r_frac * 0.6
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        lx = self.sx_grid * cos_a + self.sy_grid * sin_a
        ly = -self.sx_grid * sin_a + self.sy_grid * cos_a
        qx = lx.abs() - hw
        qy = ly.abs() - hh
        return torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2) + \
            torch.max(qx, qy).clamp(max=0)

    def _sdf_triangle(self, r_frac, angle):
        sdf = torch.full_like(self.sx_grid, -1e6)
        for k in range(3):
            a = angle + k * 2 * math.pi / 3
            nx, ny = math.cos(a), math.sin(a)
            d = self.sx_grid * nx + self.sy_grid * ny - r_frac * 0.5
            sdf = torch.max(sdf, d)
        return sdf

    def _sdf_ellipse(self, r_frac, angle):
        a, b = r_frac * 1.3, r_frac * 0.5
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        lx = self.sx_grid * cos_a + self.sy_grid * sin_a
        ly = -self.sx_grid * sin_a + self.sy_grid * cos_a
        return torch.sqrt((lx / max(a, 0.01)) ** 2 + (ly / max(b, 0.01)) ** 2) - 1.0

    def _sdf_blob(self, r_frac):
        theta = torch.atan2(self.sy_grid, self.sx_grid)
        dist = torch.sqrt(self.sx_grid ** 2 + self.sy_grid ** 2)
        r_var = r_frac
        for k in range(1, 5):
            amp = torch.rand(1, device=self.device).item() * 0.2
            phase = torch.rand(1, device=self.device).item() * 2 * math.pi
            r_var = r_var + r_frac * amp * torch.cos(k * theta + phase)
        return dist - r_var

    def _sdf_line(self, r_frac, angle):
        """Straight line with variable thickness."""
        thickness = r_frac * 0.15  # thin relative to size
        length = r_frac * 1.5
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Project onto line direction
        along = self.sx_grid * cos_a + self.sy_grid * sin_a
        perp = -self.sx_grid * sin_a + self.sy_grid * cos_a
        # Clamp along to line length
        along_clamped = along.clamp(-length, length)
        # Distance to nearest point on line segment
        dx = along - along_clamped
        return torch.sqrt(dx ** 2 + perp ** 2) - thickness

    def _sdf_stroke(self, r_frac, angle):
        """Curved brushstroke — bezier-like thick curve."""
        thickness = r_frac * 0.12
        # Quadratic bezier: start, control, end
        # Map along the curve parametrically
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        along = self.sx_grid * cos_a + self.sy_grid * sin_a
        perp = -self.sx_grid * sin_a + self.sy_grid * cos_a
        # Bend: offset perpendicular by a curve based on along position
        bend = torch.rand(1, device=self.device).item() * 0.6 - 0.3
        curve_offset = bend * (1.0 - (along / max(r_frac, 0.01)) ** 2).clamp(0)
        perp_adjusted = perp - curve_offset * r_frac
        # Taper: thickness varies along the stroke
        taper = 1.0 - 0.5 * (along / max(r_frac, 0.01)).abs()
        taper = taper.clamp(0.3, 1.0)
        along_clamped = along.clamp(-r_frac, r_frac)
        dx = along - along_clamped
        return torch.sqrt(dx ** 2 + perp_adjusted ** 2) - thickness * taper

    def _sdf_hatch(self, r_frac, angle):
        """Group of parallel lines — hatching pattern."""
        n_lines = torch.randint(3, 8, (1,)).item()
        spacing = r_frac * 2 / max(n_lines, 1)
        thickness = r_frac * 0.04
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        perp = -self.sx_grid * sin_a + self.sy_grid * cos_a
        along = self.sx_grid * cos_a + self.sy_grid * sin_a
        # Repeat perpendicular distance to create parallel lines
        offset = perp + r_frac  # shift so lines are centered
        # Modular distance to nearest line
        line_dist = (offset % spacing) - spacing / 2
        line_dist = line_dist.abs()
        # Clip to a bounding region
        bound = torch.max(along.abs() - r_frac, perp.abs() - r_frac)
        return torch.max(line_dist - thickness, bound)

    def _sdf_stipple(self, r_frac):
        """Cluster of random dots."""
        n_dots = torch.randint(8, 25, (1,)).item()
        dot_r = r_frac * 0.08
        sdf = torch.full_like(self.sx_grid, 1e6)
        for _ in range(n_dots):
            dx = (torch.rand(1, device=self.device).item() - 0.5) * r_frac * 1.6
            dy = (torch.rand(1, device=self.device).item() - 0.5) * r_frac * 1.6
            dr = dot_r * (0.5 + torch.rand(1, device=self.device).item())
            d = torch.sqrt((self.sx_grid - dx) ** 2 +
                           (self.sy_grid - dy) ** 2) - dr
            sdf = torch.min(sdf, d)
        return sdf

    def _sdf_fractal(self, r_frac):
        """IFS fractal rendered as density -> alpha mask.

        Runs chaos game with 2-6 random affine transforms,
        accumulates point density into a grid, converts to SDF-like field.
        """
        S = self.shape_res
        n_transforms = torch.randint(2, 7, (1,)).item()

        # Random affine transforms: (n, 2, 3) — each is [a b tx; c d ty]
        affines = torch.randn(n_transforms, 2, 3, device=self.device) * 0.4
        # Contract toward origin (ensure convergence)
        for i in range(n_transforms):
            affines[i, 0, 0] = affines[i, 0, 0] * 0.5 + 0.3
            affines[i, 1, 1] = affines[i, 1, 1] * 0.5 + 0.3
            affines[i, 0, 2] *= 0.5  # translation
            affines[i, 1, 2] *= 0.5

        # Chaos game on CPU (Python loop is faster without CUDA kernel overhead)
        n_points = 5000  # 128x128 grid doesn't need more
        affines_cpu = affines.cpu().numpy()

        import numpy as _np
        p = _np.zeros(2, dtype=_np.float32)
        # Burn-in
        for _ in range(30):
            idx = _np.random.randint(0, n_transforms)
            p = affines_cpu[idx, :, :2] @ p + affines_cpu[idx, :, 2]

        pts = _np.zeros((n_points, 2), dtype=_np.float32)
        for i in range(n_points):
            idx = _np.random.randint(0, n_transforms)
            p = affines_cpu[idx, :, :2] @ p + affines_cpu[idx, :, 2]
            pts[i] = p
        points = torch.from_numpy(pts).to(self.device)

        # Map points to grid coords
        px = points[:, 0]
        py = points[:, 1]
        if px.max() > px.min():
            px = (px - px.min()) / (px.max() - px.min()) * (S - 1)
            py = (py - py.min()) / (py.max() - py.min()) * (S - 1)
        px = px.long().clamp(0, S - 1)
        py = py.long().clamp(0, S - 1)

        # Scatter into density grid
        density = torch.zeros(S, S, device=self.device)
        idx_flat = py * S + px
        density.view(-1).scatter_add_(0, idx_flat,
                                       torch.ones(n_points, device=self.device))

        # Gaussian blur for smooth SDF-like field
        k = 5
        g = torch.exp(-torch.arange(k, device=self.device, dtype=torch.float32).sub(k // 2).pow(2) / 2)
        g = g / g.sum()
        density = density.unsqueeze(0).unsqueeze(0)
        density = F.conv2d(density, g.view(1, 1, k, 1), padding=(k // 2, 0))
        density = F.conv2d(density, g.view(1, 1, 1, k), padding=(0, k // 2))
        density = density.squeeze()

        # Normalize and convert to SDF: high density = inside (negative)
        d_max = density.max()
        if d_max > 0:
            density = density / d_max
        threshold = 0.05
        # Pseudo-SDF: negative inside, positive outside
        return threshold - density

    def _render_fractal_layout(self, canvas, n_shapes=15):
        """Place shapes from bank along an IFS fractal structure.

        Generates fractal point positions, places bank shapes at those positions.
        Returns modified canvas.
        """
        if self.shape_bank is None:
            return canvas

        B, C, H, W = canvas.shape
        n_transforms = torch.randint(2, 5, (1,)).item()

        # Per-image fractal layout
        for bi in range(B):
            # Generate fractal positions
            affines = torch.randn(n_transforms, 2, 3, device=self.device) * 0.3
            for i in range(n_transforms):
                affines[i, 0, 0] = affines[i, 0, 0] * 0.4 + 0.4
                affines[i, 1, 1] = affines[i, 1, 1] * 0.4 + 0.4
                affines[i, 0, 2] *= 0.6
                affines[i, 1, 2] *= 0.6

            p = torch.zeros(2, device=self.device)
            for _ in range(50):
                idx = torch.randint(0, n_transforms, (1,)).item()
                p = affines[idx, :, :2] @ p + affines[idx, :, 2]

            positions = []
            for _ in range(n_shapes):
                idx = torch.randint(0, n_transforms, (1,)).item()
                p = affines[idx, :, :2] @ p + affines[idx, :, 2]
                positions.append(p.clone())

            # Map positions to pixel coords
            pts = torch.stack(positions)  # (n, 2)
            if pts[:, 0].max() > pts[:, 0].min():
                pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / \
                    (pts[:, 0].max() - pts[:, 0].min()) * W * 0.8 + W * 0.1
                pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / \
                    (pts[:, 1].max() - pts[:, 1].min()) * H * 0.8 + H * 0.1

            # Place shapes at fractal positions
            for pi in range(len(positions)):
                sidx = torch.randint(0, self.shape_bank.shape[0], (1,)).item()
                rgba = self.shape_bank[sidx]
                rgb = rgba[:3]
                alpha = rgba[3:4]

                sc = torch.rand(1, device=self.device).item() * 0.8 + 0.3
                th = max(4, int(self.shape_res * sc * H / 360))
                tw = max(4, int(self.shape_res * sc * W / 640))
                th, tw = min(th, H), min(tw, W)

                rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                      mode="bilinear", align_corners=False)[0]
                a_r = F.interpolate(alpha.unsqueeze(0), (th, tw),
                                    mode="bilinear", align_corners=False)[0]

                px = int(pts[pi, 0].item()) - tw // 2
                py = int(pts[pi, 1].item()) - th // 2

                sx = max(0, -px)
                sy = max(0, -py)
                ex = min(tw, W - px)
                ey = min(th, H - py)
                if ex <= sx or ey <= sy:
                    continue
                cx, cy = max(0, px), max(0, py)
                rh, rw = ey - sy, ex - sx

                a = a_r[:, sy:ey, sx:ex]
                canvas[bi, :, cy:cy+rh, cx:cx+rw] = \
                    canvas[bi, :, cy:cy+rh, cx:cx+rw] * (1 - a) + \
                    rgb_r[:, sy:ey, sx:ex] * a

        return canvas

    # ------------------------------------------------------------------
    # Tier 1: Shape Bank Generation
    # ------------------------------------------------------------------

    _SHAPE_NAMES = ["circle", "rect", "triangle", "ellipse", "blob",
                     "line", "stroke", "hatch", "stipple", "fractal"]
    _TEX_NAMES = ["flat", "perlin", "gradient", "voronoi"]
    _EDGE_NAMES = ["hard", "soft", "textured"]

    @torch.no_grad()
    def _render_one_shape(self):
        """Render a single RGBA shape texture at shape_res x shape_res."""
        S = self.shape_res

        # Sample shape type
        shape_type = torch.multinomial(self.shape_probs, 1).item()
        # Sample size
        u = torch.rand(1, device=self.device).item()
        r_frac = self.min_r_frac * (self.max_r_frac / max(self.min_r_frac, 0.01)) ** u
        angle = torch.rand(1, device=self.device).item() * 2 * math.pi

        # Compute SDF
        if shape_type == 0:
            sdf = self._sdf_circle(r_frac)
        elif shape_type == 1:
            sdf = self._sdf_rect(r_frac, angle)
        elif shape_type == 2:
            sdf = self._sdf_triangle(r_frac, angle)
        elif shape_type == 3:
            sdf = self._sdf_ellipse(r_frac, angle)
        elif shape_type == 4:
            sdf = self._sdf_blob(r_frac)
        elif shape_type == 5:
            sdf = self._sdf_line(r_frac, angle)
        elif shape_type == 6:
            sdf = self._sdf_stroke(r_frac, angle)
        elif shape_type == 7:
            sdf = self._sdf_hatch(r_frac, angle)
        elif shape_type == 8:
            sdf = self._sdf_stipple(r_frac)
        elif shape_type == 9:
            sdf = self._sdf_fractal(r_frac)
        else:
            sdf = self._sdf_circle(r_frac)

        # Edge type
        edge_type = torch.multinomial(self.edge_probs, 1).item()
        if edge_type == 0:
            alpha = (sdf < 0).float()
        elif edge_type == 1:
            softness = torch.rand(1, device=self.device).item() * (
                self.soft_range[1] - self.soft_range[0]) + self.soft_range[0]
            # Scale softness relative to shape_res normalized coords
            alpha = torch.sigmoid(-sdf / (softness / S))
        else:
            idx = torch.randint(0, len(self.perlin_bank), (1,)).item()
            noise = self.perlin_bank[idx]
            softness = torch.rand(1, device=self.device).item() * (
                self.soft_range[1] - self.soft_range[0]) + self.soft_range[0]
            sdf_mod = sdf - (noise - 0.5) * (softness / S) * 3
            alpha = torch.sigmoid(-sdf_mod / (softness / S * 0.5))

        # Texture type
        tex_type = torch.multinomial(self.texture_probs, 1).item()
        color1 = self._sample_colors(1)[0]  # (3,)
        color2 = self._sample_colors(1)[0]

        if tex_type == 0:  # flat
            tex = color1.view(3, 1, 1).expand(3, S, S)
        elif tex_type == 1:  # perlin
            idx = torch.randint(0, len(self.perlin_bank), (1,)).item()
            noise = self.perlin_bank[idx].unsqueeze(0)  # (1, S, S)
            c1 = color1.view(3, 1, 1)
            c2 = color2.view(3, 1, 1)
            tex = c1 * (1 - noise) + c2 * noise
        elif tex_type == 2:  # gradient
            t = (self.sx_grid * math.cos(angle) +
                 self.sy_grid * math.sin(angle))
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            t = t.unsqueeze(0)
            c1 = color1.view(3, 1, 1)
            c2 = color2.view(3, 1, 1)
            tex = c1 * (1 - t) + c2 * t
        else:  # voronoi
            n = self.voronoi_cells
            sx = torch.rand(n, device=self.device) * 2 - 1
            sy = torch.rand(n, device=self.device) * 2 - 1
            dx = self.sx_grid.unsqueeze(0) - sx.view(n, 1, 1)
            dy = self.sy_grid.unsqueeze(0) - sy.view(n, 1, 1)
            dist = dx ** 2 + dy ** 2
            min_idx = dist.argmin(dim=0)  # (S, S)
            cell_colors = torch.rand(n, 3, device=self.device)
            tex = cell_colors[min_idx].permute(2, 0, 1)  # (3, S, S)

        # Combine: RGBA
        rgba = torch.cat([tex, alpha.unsqueeze(0)], dim=0)  # (4, S, S)

        # Log
        sn = self._SHAPE_NAMES[shape_type] if shape_type < len(self._SHAPE_NAMES) else "?"
        tn = self._TEX_NAMES[tex_type] if tex_type < len(self._TEX_NAMES) else "?"
        en = self._EDGE_NAMES[edge_type] if edge_type < len(self._EDGE_NAMES) else "?"
        self._last_shape_log = f"{sn}/{tn}/{en} r={r_frac:.2f}"

        return rgba

    @torch.no_grad()
    def build_shape_bank(self):
        """Generate the shape bank. Call once at startup."""
        print(f"Building shape bank ({self.bank_size} shapes at "
              f"{self.shape_res}x{self.shape_res})...", flush=True)
        shapes = []
        self._last_shape_log = ""
        for i in range(self.bank_size):
            shapes.append(self._render_one_shape())
            print(f"  [{i+1}/{self.bank_size}] {self._last_shape_log}", flush=True)
        self.shape_bank = torch.stack(shapes)  # (N, 4, S, S)
        mb = self.shape_bank.element_size() * self.shape_bank.nelement() / 1e6
        print(f"Shape bank done: {self.bank_size} shapes ({mb:.0f} MB)", flush=True)

    # ------------------------------------------------------------------
    # Tier 2: Base Layer Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_base_layers(self):
        """Generate base layers by compositing shapes from the bank."""
        if self.shape_bank is None:
            self.build_shape_bank()

        N = self.n_base_layers
        print(f"Building {N} base layers ({self.shapes_per_layer} shapes each)...",
              flush=True)

        layers = []
        import time as _time
        t0 = _time.time()
        for li in range(N):
            # Random background
            bg = self._sample_colors(1)[0].view(3, 1, 1)
            canvas = bg.expand(3, self.H, self.W).clone()

            for _ in range(self.shapes_per_layer):
                # Pick random shape from bank + transform
                idx = torch.randint(0, self.bank_size, (1,)).item()
                rgba = self._transform_bank_shape(self.shape_bank[idx].clone())
                rgb = rgba[:3]
                alpha = rgba[3:4]

                # Random transform: position, scale
                scale = torch.rand(1, device=self.device).item() * 1.5 + 0.2
                # Target size in pixels
                tgt_h = int(self.shape_res * scale * self.H / 360)
                tgt_w = int(self.shape_res * scale * self.W / 640)
                tgt_h = max(4, min(tgt_h, self.H * 2))
                tgt_w = max(4, min(tgt_w, self.W * 2))

                # Resize shape
                rgb_r = F.interpolate(rgb.unsqueeze(0), (tgt_h, tgt_w),
                                      mode="bilinear", align_corners=False)[0]
                alpha_r = F.interpolate(alpha.unsqueeze(0), (tgt_h, tgt_w),
                                        mode="bilinear", align_corners=False)[0]

                # Random position (can be partially off-canvas)
                px = int(torch.randint(-tgt_w // 2, self.W, (1,)).item())
                py = int(torch.randint(-tgt_h // 2, self.H, (1,)).item())

                # Compute overlap region
                sx = max(0, -px)
                sy = max(0, -py)
                ex = min(tgt_w, self.W - px)
                ey = min(tgt_h, self.H - py)
                if ex <= sx or ey <= sy:
                    continue

                cx = max(0, px)
                cy = max(0, py)

                # Composite
                a = alpha_r[:, sy:ey, sx:ex]
                canvas[:, cy:cy + (ey - sy), cx:cx + (ex - sx)] = \
                    canvas[:, cy:cy + (ey - sy), cx:cx + (ex - sx)] * (1 - a) + \
                    rgb_r[:, sy:ey, sx:ex] * a

            layers.append(canvas)
            elapsed = _time.time() - t0
            lps = (li + 1) / max(elapsed, 0.01)
            eta = (N - li - 1) / max(lps, 0.01)
            print(f"  [{li+1}/{N}] {lps:.1f} layers/s  ETA {eta:.0f}s", flush=True)

        self.base_layers = torch.stack(layers)  # (N, 3, H, W)
        mb = self.base_layers.element_size() * self.base_layers.nelement() / 1e6
        print(f"Base layers done: {N} layers ({mb:.0f} MB) in "
              f"{_time.time() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Tier 1+2: Build all banks
    # ------------------------------------------------------------------

    def build_banks(self):
        """Build shape bank and base layers. Call once before generate()."""
        self.build_shape_bank()
        self.build_base_layers()

    # ------------------------------------------------------------------
    # Scene Templates
    # ------------------------------------------------------------------

    def _pick_template(self):
        """Pick a random scene template based on weights."""
        idx = torch.multinomial(self.template_probs, 1).item()
        return self.template_names[idx]

    def _apply_scene_template(self, canvas, template, B):
        """Dispatch to the appropriate template renderer."""
        fn = {
            "horizon": self._tmpl_horizon,
            "v_stripes": self._tmpl_v_stripes,
            "h_stripes": self._tmpl_h_stripes,
            "d_stripes": self._tmpl_d_stripes,
            "grid": self._tmpl_grid,
            "radial": self._tmpl_radial,
            "perspective": self._tmpl_perspective,
            "depth_layers": self._tmpl_depth_layers,
            "symmetry": self._tmpl_symmetry,
            "border": self._tmpl_border,
            "clusters": self._tmpl_clusters,
            "gradient": self._tmpl_gradient,
            "block_city": self._tmpl_block_city,
            "landscape": self._tmpl_landscape,
            "interior": self._tmpl_interior,
            "road": self._tmpl_road,
            "water": self._tmpl_water,
            "forest": self._tmpl_forest,
        }.get(template)
        if fn:
            return fn(canvas, B)
        return canvas

    def _sample_layer(self, B):
        """Sample a base layer with random color shift. Returns (B, 3, H, W)."""
        idx = torch.randint(0, self.n_base_layers, (B,), device=self.device)
        layer = self.base_layers[idx].clone()
        shift = (torch.rand(B, 3, 1, 1, device=self.device) - 0.5) * 0.4
        return (layer + shift).clamp(0, 1)

    def _tmpl_horizon(self, canvas, B):
        H, W = self.H, self.W
        # Random horizon height per image
        h_line = torch.randint(int(H * 0.25), int(H * 0.7), (B,))
        sky = self._sample_layer(B)
        ground = self._sample_layer(B)
        for bi in range(B):
            hl = h_line[bi].item()
            canvas[bi, :, :hl] = sky[bi, :, :hl] * 0.7 + canvas[bi, :, :hl] * 0.3
            canvas[bi, :, hl:] = ground[bi, :, hl:] * 0.7 + canvas[bi, :, hl:] * 0.3
            # Horizon line
            lw = torch.randint(1, 4, (1,)).item()
            y0 = max(0, hl - lw)
            y1 = min(H, hl + lw)
            if y1 > y0:
                line_color = torch.rand(3, device=self.device) * 0.5
                canvas[bi, :, y0:y1, :] = line_color.view(3, 1, 1)
        return canvas

    def _tmpl_v_stripes(self, canvas, B):
        H, W = self.H, self.W
        n = torch.randint(3, 12, (1,)).item()
        for bi in range(B):
            positions = sorted(torch.randint(0, W, (n-1,)).tolist())
            positions = [0] + positions + [W]
            for i in range(len(positions) - 1):
                x0, x1 = positions[i], positions[i+1]
                if x1 <= x0:
                    continue
                lidx = torch.randint(0, self.n_base_layers, (1,)).item()
                stripe = self.base_layers[lidx, :, :, x0:x1].clone()
                cs = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.4
                canvas[bi, :, :, x0:x1] = (stripe + cs).clamp(0, 1)
        return canvas

    def _tmpl_h_stripes(self, canvas, B):
        H, W = self.H, self.W
        n = torch.randint(3, 10, (1,)).item()
        for bi in range(B):
            positions = sorted(torch.randint(0, H, (n-1,)).tolist())
            positions = [0] + positions + [H]
            for i in range(len(positions) - 1):
                y0, y1 = positions[i], positions[i+1]
                if y1 <= y0:
                    continue
                lidx = torch.randint(0, self.n_base_layers, (1,)).item()
                stripe = self.base_layers[lidx, :, y0:y1, :].clone()
                cs = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.4
                canvas[bi, :, y0:y1, :] = (stripe + cs).clamp(0, 1)
        return canvas

    def _tmpl_d_stripes(self, canvas, B):
        H, W = self.H, self.W
        n = torch.randint(4, 10, (1,)).item()
        angle = torch.rand(1, device=self.device).item() * math.pi
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Project each pixel along the stripe direction
        y_grid = torch.arange(H, device=self.device).float().view(H, 1)
        x_grid = torch.arange(W, device=self.device).float().view(1, W)
        proj = y_grid * cos_a + x_grid * sin_a
        p_min, p_max = proj.min(), proj.max()
        proj_norm = (proj - p_min) / (p_max - p_min + 1e-6)
        stripe_idx = (proj_norm * n).long().clamp(0, n - 1)
        for bi in range(B):
            colors = self._sample_colors(n)  # (n, 3)
            for si in range(n):
                mask = (stripe_idx == si).float().unsqueeze(0)  # (1, H, W)
                lidx = torch.randint(0, self.n_base_layers, (1,)).item()
                fill = self.base_layers[lidx].clone()
                cs = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
                fill = (fill + cs).clamp(0, 1)
                canvas[bi] = canvas[bi] * (1 - mask) + fill * mask
        return canvas

    def _tmpl_grid(self, canvas, B):
        H, W = self.H, self.W
        rows = torch.randint(2, 5, (1,)).item()
        cols = torch.randint(2, 6, (1,)).item()
        ch, cw = H // rows, W // cols
        for bi in range(B):
            for r in range(rows):
                for c in range(cols):
                    y0, x0 = r * ch, c * cw
                    y1, x1 = min(y0 + ch, H), min(x0 + cw, W)
                    lidx = torch.randint(0, self.n_base_layers, (1,)).item()
                    cell = F.interpolate(
                        self.base_layers[lidx:lidx+1], (y1 - y0, x1 - x0),
                        mode="bilinear", align_corners=False)[0]
                    cs = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
                    canvas[bi, :, y0:y1, x0:x1] = (cell + cs).clamp(0, 1)
        return canvas

    def _tmpl_radial(self, canvas, B):
        H, W = self.H, self.W
        cy, cx = H // 2, W // 2
        n_rings = torch.randint(3, 8, (1,)).item()
        max_r = min(H, W) // 2
        y_grid = torch.arange(H, device=self.device).float().view(H, 1) - cy
        x_grid = torch.arange(W, device=self.device).float().view(1, W) - cx
        dist = torch.sqrt(y_grid ** 2 + x_grid ** 2)
        ring_idx = (dist / max_r * n_rings).long().clamp(0, n_rings - 1)
        for bi in range(B):
            for ri in range(n_rings):
                mask = (ring_idx == ri).float().unsqueeze(0)
                lidx = torch.randint(0, self.n_base_layers, (1,)).item()
                fill = self.base_layers[lidx].clone()
                cs = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
                canvas[bi] = canvas[bi] * (1 - mask) + (fill + cs).clamp(0, 1) * mask
        return canvas

    def _tmpl_perspective(self, canvas, B):
        H, W = self.H, self.W
        # Vanishing point near top center
        vp_y = torch.randint(int(H * 0.1), int(H * 0.35), (1,)).item()
        vp_x = W // 2 + torch.randint(-W // 6, W // 6, (1,)).item()
        # Draw converging lines
        n_lines = torch.randint(5, 12, (1,)).item()
        ground = self._sample_layer(B)
        sky = self._sample_layer(B)
        for bi in range(B):
            canvas[bi, :, :vp_y] = sky[bi, :, :vp_y]
            canvas[bi, :, vp_y:] = ground[bi, :, vp_y:]
            # Converging lines from bottom edge to vanishing point
            for li in range(n_lines):
                bx = int(li * W / n_lines)
                color = torch.rand(3, device=self.device) * 0.4 + 0.3
                # Draw line from (bx, H) to (vp_x, vp_y)
                n_pts = H - vp_y
                for t in range(0, n_pts, 2):
                    frac = t / max(n_pts - 1, 1)
                    px = int(bx + (vp_x - bx) * frac)
                    py = int(H - t)
                    lw = max(1, int(3 * (1 - frac)))
                    if 0 <= px < W and 0 <= py < H:
                        canvas[bi, :, max(0,py-lw):py+lw,
                               max(0,px-lw):px+lw] = color.view(3, 1, 1)
        return canvas

    def _tmpl_depth_layers(self, canvas, B):
        H, W = self.H, self.W
        bg = self._sample_layer(B)
        mid = self._sample_layer(B)
        fg = self._sample_layer(B)
        for bi in range(B):
            # Background: full canvas, muted
            canvas[bi] = bg[bi] * 0.6 + canvas[bi] * 0.4
            # Midground: middle band
            m_top = torch.randint(int(H * 0.2), int(H * 0.4), (1,)).item()
            m_bot = torch.randint(int(H * 0.6), int(H * 0.85), (1,)).item()
            canvas[bi, :, m_top:m_bot] = mid[bi, :, m_top:m_bot] * 0.8 + \
                canvas[bi, :, m_top:m_bot] * 0.2
            # Foreground: bottom, vivid
            f_top = torch.randint(int(H * 0.6), int(H * 0.8), (1,)).item()
            canvas[bi, :, f_top:] = fg[bi, :, f_top:]
        return canvas

    def _tmpl_symmetry(self, canvas, B):
        H, W = self.H, self.W
        half = self._sample_layer(B)
        for bi in range(B):
            if torch.rand(1).item() < 0.5:
                # Vertical mirror
                canvas[bi, :, :, :W//2] = half[bi, :, :, :W//2]
                canvas[bi, :, :, W//2:] = half[bi, :, :, :W//2].flip(-1)[:, :, :W-W//2]
            else:
                # Horizontal mirror
                canvas[bi, :, :H//2] = half[bi, :, :H//2]
                canvas[bi, :, H//2:] = half[bi, :, :H//2].flip(-2)[:, :H-H//2]
        return canvas

    def _tmpl_border(self, canvas, B):
        H, W = self.H, self.W
        bw = torch.randint(int(min(H,W) * 0.05), int(min(H,W) * 0.15), (1,)).item()
        border_layer = self._sample_layer(B)
        inner_layer = self._sample_layer(B)
        for bi in range(B):
            canvas[bi] = border_layer[bi]
            canvas[bi, :, bw:H-bw, bw:W-bw] = inner_layer[bi, :, bw:H-bw, bw:W-bw]
        return canvas

    def _tmpl_clusters(self, canvas, B):
        H, W = self.H, self.W
        n_clusters = torch.randint(2, 6, (1,)).item()
        for bi in range(B):
            for _ in range(n_clusters):
                cx = torch.randint(W // 6, W * 5 // 6, (1,)).item()
                cy = torch.randint(H // 6, H * 5 // 6, (1,)).item()
                cr = torch.randint(min(H,W) // 8, min(H,W) // 3, (1,)).item()
                n_shapes = torch.randint(5, 15, (1,)).item()
                for _ in range(n_shapes):
                    if self.shape_bank is None:
                        break
                    sidx = torch.randint(0, self.shape_bank.shape[0], (1,)).item()
                    rgba = self._transform_bank_shape(self.shape_bank[sidx].clone())
                    rgb, alpha = rgba[:3], rgba[3:4]
                    sc = torch.rand(1).item() * 0.4 + 0.1
                    th = max(3, int(self.shape_res * sc * H / 360))
                    tw = max(3, int(self.shape_res * sc * W / 640))
                    rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                          mode="bilinear", align_corners=False)[0]
                    a_r = F.interpolate(alpha.unsqueeze(0), (th, tw),
                                        mode="bilinear", align_corners=False)[0]
                    # Position near cluster center
                    px = cx + int((torch.rand(1).item() - 0.5) * cr * 2) - tw // 2
                    py = cy + int((torch.rand(1).item() - 0.5) * cr * 2) - th // 2
                    if 0 <= px < W - tw and 0 <= py < H - th:
                        a = a_r
                        canvas[bi, :, py:py+th, px:px+tw] = \
                            canvas[bi, :, py:py+th, px:px+tw] * (1 - a) + rgb_r * a
        return canvas

    def _tmpl_gradient(self, canvas, B):
        H, W = self.H, self.W
        angle = torch.rand(1, device=self.device).item() * math.pi * 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        y_grid = torch.linspace(0, 1, H, device=self.device).view(H, 1)
        x_grid = torch.linspace(0, 1, W, device=self.device).view(1, W)
        t = (y_grid * cos_a + x_grid * sin_a)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        t = t.unsqueeze(0)  # (1, H, W)
        for bi in range(B):
            c1 = self._sample_colors(1)[0].view(3, 1, 1)
            c2 = self._sample_colors(1)[0].view(3, 1, 1)
            grad = c1 * (1 - t) + c2 * t
            canvas[bi] = grad * 0.6 + canvas[bi] * 0.4
        return canvas

    # -- Primitive scene templates --

    def _tmpl_block_city(self, canvas, B):
        """Simplified city skyline: sky gradient + rectangular buildings."""
        H, W = self.H, self.W
        for bi in range(B):
            # Sky gradient (top)
            sky_c1 = self._sample_colors(1)[0].view(3, 1, 1) * 0.7 + 0.3
            sky_c2 = self._sample_colors(1)[0].view(3, 1, 1) * 0.5
            t = torch.linspace(0, 1, H, device=self.device).view(1, H, 1)
            canvas[bi] = sky_c1 * (1 - t) + sky_c2 * t

            # Ground
            ground_y = int(H * (0.6 + torch.rand(1).item() * 0.2))
            ground_c = self._sample_colors(1)[0].view(3, 1, 1) * 0.4
            canvas[bi, :, ground_y:] = ground_c

            # Buildings: random rectangles rising from ground
            n_buildings = torch.randint(5, 15, (1,)).item()
            for _ in range(n_buildings):
                bw = torch.randint(W // 20, W // 6, (1,)).item()
                bh = torch.randint(H // 8, int(H * 0.6), (1,)).item()
                bx = torch.randint(0, W - bw, (1,)).item()
                by = ground_y - bh
                color = self._sample_colors(1)[0].view(3, 1, 1) * 0.6 + 0.2
                if by >= 0:
                    canvas[bi, :, by:ground_y, bx:bx+bw] = color
                    # Window dots
                    for wy in range(by + 4, ground_y - 4, max(bh // 6, 4)):
                        for wx in range(bx + 3, bx + bw - 3, max(bw // 4, 4)):
                            wc = torch.rand(1, device=self.device).item()
                            if wc > 0.4:
                                ws = min(3, bw // 6, bh // 10)
                                if ws > 0 and wy + ws < H and wx + ws < W:
                                    canvas[bi, :, wy:wy+ws, wx:wx+ws] = \
                                        torch.tensor([0.9, 0.85, 0.5],
                                                     device=self.device).view(3, 1, 1)
        return canvas

    def _tmpl_landscape(self, canvas, B):
        """Simple landscape: sky, mountains, ground, trees."""
        H, W = self.H, self.W
        for bi in range(B):
            # Sky
            sky_top = self._sample_colors(1)[0].view(3, 1, 1)
            sky_bot = sky_top * 0.6 + 0.2
            t = torch.linspace(0, 1, H, device=self.device).view(1, H, 1)
            canvas[bi] = sky_top * (1 - t) + sky_bot * t

            # Mountains (2-4 triangles)
            horizon = int(H * (0.4 + torch.rand(1).item() * 0.2))
            n_mtn = torch.randint(2, 5, (1,)).item()
            mtn_color = self._sample_colors(1)[0] * 0.5 + 0.2
            for _ in range(n_mtn):
                peak_x = torch.randint(0, W, (1,)).item()
                peak_y = torch.randint(int(H * 0.15), horizon, (1,)).item()
                base_w = torch.randint(W // 6, W // 2, (1,)).item()
                # Draw triangle
                for row in range(peak_y, horizon):
                    frac = (row - peak_y) / max(horizon - peak_y, 1)
                    half_w = int(base_w * frac / 2)
                    x0 = max(0, peak_x - half_w)
                    x1 = min(W, peak_x + half_w)
                    shade = mtn_color * (1 - frac * 0.3)
                    canvas[bi, :, row, x0:x1] = shade.view(3, 1).to(self.device)

            # Ground
            ground_c = torch.tensor([0.3, 0.5, 0.2], device=self.device).view(3, 1, 1)
            ground_c = ground_c + (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.2
            canvas[bi, :, horizon:] = ground_c.clamp(0, 1)

            # Simple trees (circles on sticks)
            n_trees = torch.randint(3, 10, (1,)).item()
            for _ in range(n_trees):
                tx = torch.randint(0, W, (1,)).item()
                ty = torch.randint(horizon, int(H * 0.85), (1,)).item()
                trunk_h = torch.randint(10, 30, (1,)).item()
                canopy_r = torch.randint(8, 20, (1,)).item()
                # Trunk
                tw = max(2, canopy_r // 3)
                canvas[bi, :, ty-trunk_h:ty, max(0,tx-tw):tx+tw] = \
                    torch.tensor([0.4, 0.25, 0.1], device=self.device).view(3, 1, 1)
                # Canopy (approximate circle via square)
                cy = ty - trunk_h - canopy_r
                cx = tx
                y0, y1 = max(0, cy - canopy_r), max(0, cy + canopy_r)
                x0, x1 = max(0, cx - canopy_r), min(W, cx + canopy_r)
                if y1 > y0 and x1 > x0:
                    green = torch.tensor([0.15, 0.5, 0.15],
                                         device=self.device).view(3, 1, 1)
                    green = green + torch.rand(3, 1, 1, device=self.device) * 0.15
                    canvas[bi, :, y0:y1, x0:x1] = green.clamp(0, 1)
        return canvas

    def _tmpl_interior(self, canvas, B):
        """Simple room interior: floor, walls, ceiling, furniture."""
        H, W = self.H, self.W
        for bi in range(B):
            # Ceiling
            ceil_h = int(H * (0.15 + torch.rand(1).item() * 0.15))
            ceil_c = self._sample_colors(1)[0].view(3, 1, 1) * 0.3 + 0.6
            canvas[bi, :, :ceil_h] = ceil_c

            # Back wall
            wall_c = self._sample_colors(1)[0].view(3, 1, 1) * 0.4 + 0.4
            floor_y = int(H * (0.6 + torch.rand(1).item() * 0.15))
            canvas[bi, :, ceil_h:floor_y] = wall_c

            # Floor
            floor_c = self._sample_colors(1)[0].view(3, 1, 1) * 0.4 + 0.1
            canvas[bi, :, floor_y:] = floor_c

            # Window (rectangle on wall)
            if torch.rand(1).item() < 0.7:
                ww = torch.randint(W // 8, W // 3, (1,)).item()
                wh = torch.randint(H // 8, (floor_y - ceil_h) * 2 // 3, (1,)).item()
                wx = torch.randint(W // 6, W - ww - W // 6, (1,)).item()
                wy = ceil_h + (floor_y - ceil_h - wh) // 3
                canvas[bi, :, wy:wy+wh, wx:wx+ww] = \
                    torch.tensor([0.6, 0.75, 0.9], device=self.device).view(3, 1, 1)

            # Furniture: rectangles on the floor
            n_furn = torch.randint(1, 4, (1,)).item()
            for _ in range(n_furn):
                fw = torch.randint(W // 10, W // 4, (1,)).item()
                fh = torch.randint(H // 10, (H - floor_y) + H // 6, (1,)).item()
                fx = torch.randint(0, W - fw, (1,)).item()
                fy = floor_y - fh // 3
                fc = self._sample_colors(1)[0].view(3, 1, 1) * 0.5 + 0.2
                fy = max(0, fy)
                canvas[bi, :, fy:min(H, fy+fh), fx:fx+fw] = fc
        return canvas

    def _tmpl_road(self, canvas, B):
        """Road/path with perspective lines."""
        H, W = self.H, self.W
        for bi in range(B):
            # Sky
            sky_c = self._sample_colors(1)[0].view(3, 1, 1) * 0.5 + 0.4
            canvas[bi] = sky_c

            # Ground on both sides
            horizon = int(H * (0.35 + torch.rand(1).item() * 0.15))
            ground_c = torch.tensor([0.35, 0.45, 0.2], device=self.device).view(3, 1, 1)
            ground_c = ground_c + (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.15
            canvas[bi, :, horizon:] = ground_c.clamp(0, 1)

            # Road: trapezoid from bottom to horizon
            road_c = torch.tensor([0.35, 0.35, 0.35], device=self.device).view(3, 1)
            vp_x = W // 2 + torch.randint(-W // 8, W // 8, (1,)).item()
            road_w_top = torch.randint(5, W // 8, (1,)).item()
            road_w_bot = torch.randint(W // 4, W // 2, (1,)).item()
            for row in range(horizon, H):
                frac = (row - horizon) / max(H - horizon - 1, 1)
                half_w = int(road_w_top + (road_w_bot - road_w_top) * frac) // 2
                x0 = max(0, vp_x - half_w)
                x1 = min(W, vp_x + half_w)
                canvas[bi, :, row, x0:x1] = road_c

            # Center line (dashed)
            line_c = torch.tensor([0.9, 0.9, 0.3], device=self.device).view(3, 1, 1)
            for row in range(horizon, H, 8):
                if (row // 8) % 2 == 0:
                    frac = (row - horizon) / max(H - horizon - 1, 1)
                    lw = max(1, int(1 + frac * 3))
                    canvas[bi, :, row:min(H, row+4),
                           max(0, vp_x-lw):vp_x+lw] = line_c
        return canvas

    def _tmpl_water(self, canvas, B):
        """Scene with water reflection: top half scene, bottom half mirror."""
        H, W = self.H, self.W
        mid = H // 2
        for bi in range(B):
            # Generate top half as a simple scene
            sky_c = self._sample_colors(1)[0].view(3, 1, 1) * 0.5 + 0.4
            canvas[bi, :, :mid] = sky_c

            # Some shapes in top half
            if self.shape_bank is not None:
                for _ in range(torch.randint(3, 8, (1,)).item()):
                    sidx = torch.randint(0, self.shape_bank.shape[0], (1,)).item()
                    rgba = self._transform_bank_shape(self.shape_bank[sidx].clone())
                    rgb, alpha = rgba[:3], rgba[3:4]
                    sc = torch.rand(1).item() * 0.5 + 0.2
                    th = max(3, int(self.shape_res * sc * mid / 360))
                    tw = max(3, int(self.shape_res * sc * W / 640))
                    rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                          mode="bilinear", align_corners=False)[0]
                    a_r = F.interpolate(alpha.unsqueeze(0), (th, tw),
                                        mode="bilinear", align_corners=False)[0]
                    px = torch.randint(0, max(1, W - tw), (1,)).item()
                    py = torch.randint(0, max(1, mid - th), (1,)).item()
                    if py + th <= mid:
                        canvas[bi, :, py:py+th, px:px+tw] = \
                            canvas[bi, :, py:py+th, px:px+tw] * (1 - a_r) + rgb_r * a_r

            # Bottom half: flip top half, darken and blue-shift for water
            reflection = canvas[bi, :, :mid].flip(-2)
            water_tint = torch.tensor([0.7, 0.8, 1.0], device=self.device).view(3, 1, 1)
            canvas[bi, :, mid:mid + reflection.shape[1]] = reflection * water_tint * 0.6

            # Water line
            canvas[bi, :, mid-1:mid+1] = torch.tensor(
                [0.5, 0.6, 0.7], device=self.device).view(3, 1, 1)
        return canvas

    def _tmpl_forest(self, canvas, B):
        """Dense forest: vertical trunks with canopy, ground layer."""
        H, W = self.H, self.W
        for bi in range(B):
            # Sky peeking through
            sky_c = self._sample_colors(1)[0].view(3, 1, 1) * 0.4 + 0.5
            canvas[bi] = sky_c

            # Ground
            ground_y = int(H * 0.75)
            ground_c = torch.tensor([0.2, 0.3, 0.1], device=self.device).view(3, 1, 1)
            canvas[bi, :, ground_y:] = ground_c + torch.rand(3, 1, 1, device=self.device) * 0.1

            # Trees: many vertical trunks
            n_trees = torch.randint(8, 25, (1,)).item()
            for _ in range(n_trees):
                tx = torch.randint(0, W, (1,)).item()
                trunk_w = torch.randint(3, 12, (1,)).item()
                trunk_top = torch.randint(int(H * 0.1), int(H * 0.4), (1,)).item()
                trunk_c = torch.tensor([0.3, 0.2, 0.1], device=self.device
                    ).view(3, 1, 1) + torch.rand(3, 1, 1, device=self.device) * 0.1
                x0 = max(0, tx - trunk_w // 2)
                x1 = min(W, tx + trunk_w // 2)
                canvas[bi, :, trunk_top:ground_y, x0:x1] = trunk_c.clamp(0, 1)

            # Canopy layer (dense green band)
            canopy_top = int(H * 0.05)
            canopy_bot = int(H * 0.45)
            for cy in range(canopy_top, canopy_bot):
                density = 0.5 + 0.3 * math.sin(cy * 0.1)
                green = torch.tensor([0.1, 0.35, 0.08], device=self.device
                    ).view(3, 1) + torch.rand(3, 1, device=self.device) * 0.15
                mask = torch.rand(W, device=self.device) < density
                canvas[bi, :, cy, mask] = green.clamp(0, 1)
        return canvas

    # ------------------------------------------------------------------
    # Tier 3: Fast Final Compositing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _overlay_shapes_on_canvas(self, canvas, B, n_stamps_range=(3, 10),
                                    n_micro_range=(15, 50), light=False):
        """Composite shape stamps + micro-stamps onto an existing canvas.

        Args:
            canvas: (B, 3, H, W) tensor to overlay onto (modified in-place)
            B: batch size
            n_stamps_range: (min, max) stamps per image
            n_micro_range: (min, max) micro-stamps (50% chance)
            light: if True, reduce counts and opacity
        """
        if self.shape_bank is None:
            return canvas
        H, W = self.H, self.W
        if light:
            n_stamps_range = (1, 4)
            n_micro_range = (5, 15)

        n_stamps = torch.randint(n_stamps_range[0], n_stamps_range[1] + 1,
                                  (B,), device=self.device)
        max_stamps = n_stamps.max().item()

        for si in range(max_stamps):
            active = (si < n_stamps).float().view(B, 1, 1, 1)
            idx = torch.randint(0, self.bank_size, (B,), device=self.device)
            shapes = self.shape_bank[idx].clone()
            for bi in range(B):
                if active[bi, 0, 0, 0] > 0.5:
                    shapes[bi] = self._transform_bank_shape(shapes[bi])
            s_rgb = shapes[:, :3]
            s_alpha = shapes[:, 3:4]
            if light:
                s_alpha = s_alpha * 0.5
            scales = torch.rand(B, device=self.device) * 1.5 + 0.3
            for bi in range(B):
                if active[bi, 0, 0, 0] < 0.5:
                    continue
                sc = scales[bi].item()
                th = max(4, int(self.shape_res * sc * H / 360))
                tw = max(4, int(self.shape_res * sc * W / 640))
                th, tw = min(th, H), min(tw, W)
                rgb_r = F.interpolate(s_rgb[bi:bi+1], (th, tw),
                                       mode="bilinear", align_corners=False)[0]
                a_r = F.interpolate(s_alpha[bi:bi+1], (th, tw),
                                     mode="bilinear", align_corners=False)[0]
                px = torch.randint(-tw // 4, W - tw // 4, (1,)).item()
                py = torch.randint(-th // 4, H - th // 4, (1,)).item()
                sx = max(0, -px)
                sy = max(0, -py)
                ex = min(tw, W - px)
                ey = min(th, H - py)
                if ex <= sx or ey <= sy:
                    continue
                cx, cy_s = max(0, px), max(0, py)
                rh, rw = ey - sy, ex - sx
                a = a_r[:, sy:ey, sx:ex]
                canvas[bi, :, cy_s:cy_s+rh, cx:cx+rw] = \
                    canvas[bi, :, cy_s:cy_s+rh, cx:cx+rw] * (1 - a) + \
                    rgb_r[:, sy:ey, sx:ex] * a

        # Micro-stamps
        if torch.rand(1).item() < 0.5:
            n_micro = torch.randint(n_micro_range[0], n_micro_range[1] + 1,
                                     (1,)).item()
            for _ in range(n_micro):
                for bi in range(B):
                    sidx = torch.randint(0, self.shape_bank.shape[0], (1,)).item()
                    rgba = self._transform_bank_shape(self.shape_bank[sidx].clone())
                    rgb = rgba[:3]
                    alpha = rgba[3:4]
                    if light:
                        alpha = alpha * 0.4
                    sc = torch.rand(1, device=self.device).item() * 0.2 + 0.05
                    th = max(3, int(self.shape_res * sc * H / 360))
                    tw = max(3, int(self.shape_res * sc * W / 640))
                    th, tw = min(th, H // 2), min(tw, W // 2)
                    rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                           mode="bilinear", align_corners=False)[0]
                    a_r = F.interpolate(alpha.unsqueeze(0), (th, tw),
                                         mode="bilinear", align_corners=False)[0]
                    px = torch.randint(0, max(1, W - tw + 1), (1,)).item()
                    py = torch.randint(0, max(1, H - th + 1), (1,)).item()
                    canvas[bi, :, py:py+th, px:px+tw] = \
                        canvas[bi, :, py:py+th, px:px+tw] * (1 - a_r) + rgb_r * a_r
        return canvas

    def _generate_disco(self, B):
        """Four-quadrant disco generation.

        Q0 (25%): pure pattern — clean mathematical pattern, no shapes
        Q1 (25%): pattern + light shapes — collaged patterns with sparse overlay
        Q2 (25%): dense random — current pipeline with cranked density
        Q3 (25%): structured scenes — existing pipeline unchanged
        """
        H, W = self.H, self.W
        quadrants = torch.multinomial(self._disco_weights, B, replacement=True)
        canvas = torch.zeros(B, 3, H, W, device=self.device)

        # Q0: Pure pattern
        q0_mask = (quadrants == 0)
        n0 = q0_mask.sum().item()
        if n0 > 0:
            canvas[q0_mask] = self.pattern_bank.generate(n0)

        # Q1: Pattern + collage + light shapes
        q1_mask = (quadrants == 1)
        n1 = q1_mask.sum().item()
        if n1 > 0:
            pat_a = self.pattern_bank.generate(n1)
            pat_b = self.pattern_bank.generate(n1)
            # Random collage operation
            op = torch.randint(0, 4, (1,)).item()
            if op == 0:
                combined = pattern_collage.rip_collage(pat_a, pat_b)
            elif op == 1:
                combined = pattern_collage.alpha_blend(pat_a, pat_b)
            elif op == 2:
                combined = pattern_collage.merge_halves(pat_a, pat_b)
            else:
                combined = pattern_collage.splice_regions(
                    [pat_a, pat_b], self.device)
            combined = self._overlay_shapes_on_canvas(combined, n1, light=True)
            combined = self._post_process(combined)
            canvas[q1_mask] = combined.clamp(0, 1)

        # Q2: Dense random — existing pipeline with cranked micro-stamps
        q2_mask = (quadrants == 2)
        n2 = q2_mask.sum().item()
        if n2 > 0:
            # Run existing compositing pipeline
            bg = self._sample_colors(n2)
            c2 = bg.view(n2, 3, 1, 1).expand(n2, 3, H, W).clone()
            template = self._pick_template()
            if template != "random":
                c2 = self._apply_scene_template(c2, template, n2)
            # Dense layers
            n_layers = torch.randint(5, 10, (n2,), device=self.device)
            for li in range(n_layers.max().item()):
                active = (li < n_layers).float().view(n2, 1, 1, 1)
                layer_idx = torch.randint(0, self.base_layers.shape[0],
                                           (n2,), device=self.device)
                layer = self.base_layers[layer_idx].clone()
                shift = (torch.rand(n2, 3, 1, 1, device=self.device) - 0.5) * 0.3
                layer = (layer + shift).clamp(0, 1)
                opacity = torch.rand(n2, 1, 1, 1, device=self.device) * 0.6 + 0.2
                alpha = active * opacity
                c2 = c2 * (1 - alpha) + layer * alpha
            # Dense overlay: many stamps + many micro-stamps
            c2 = self._overlay_shapes_on_canvas(c2, n2,
                                                 n_stamps_range=(5, 15),
                                                 n_micro_range=(200, 500))
            # Fractal layout
            if torch.rand(1).item() < 0.5 and self.shape_bank is not None:
                c2 = self._render_fractal_layout(c2, n_shapes=15)
            c2 = self._post_process(c2)
            canvas[q2_mask] = c2.clamp(0, 1)

        # Q3: Structured scenes — existing template pipeline
        q3_mask = (quadrants == 3)
        n3 = q3_mask.sum().item()
        if n3 > 0:
            # Force a structured template
            old_probs = self.template_probs.clone()
            # Zero out "random" and boost structured templates
            self.template_probs[0] = 0
            self.template_probs = self.template_probs / (self.template_probs.sum() + 1e-8)
            bg = self._sample_colors(n3)
            c3 = bg.view(n3, 3, 1, 1).expand(n3, 3, H, W).clone()
            template = self._pick_template()
            c3 = self._apply_scene_template(c3, template, n3)
            c3 = self._overlay_shapes_on_canvas(c3, n3,
                                                 n_stamps_range=(2, 6),
                                                 n_micro_range=(10, 30))
            c3 = self._post_process(c3)
            canvas[q3_mask] = c3.clamp(0, 1)
            self.template_probs = old_probs

        return canvas.clamp(0, 1)

    def generate(self, batch_size):
        """Generate a batch of synthetic images.

        Returns: (B, 3, H, W) tensor in [0, 1] on self.device.
        """
        if self.base_layers is None:
            self.build_banks()

        # Dynamic bank refresh if configured
        self._maybe_refresh_dynamic()

        B = batch_size

        # Disco quadrant mode: four balanced quadrants
        if self.disco_quadrant:
            return self._generate_disco(B)

        H, W = self.H, self.W

        # Start with random background
        bg = self._sample_colors(B)  # (B, 3)
        canvas = bg.view(B, 3, 1, 1).expand(B, 3, H, W).clone()

        # Apply scene template (structural composition)
        template = self._pick_template()
        has_template = template != "random"
        if has_template:
            canvas = self._apply_scene_template(canvas, template, B)

        # When template is active, reduce overlay layers so structure shows
        if has_template:
            n_layers = torch.randint(1, 3, (B,), device=self.device)
            n_stamps = torch.randint(2, 6, (B,), device=self.device)
            overlay_opacity_scale = 0.3  # layers much more transparent
        else:
            n_layers = torch.randint(
                self.layers_per_image[0], self.layers_per_image[1] + 1,
                (B,), device=self.device)
            n_stamps = torch.randint(
                self.stamps_per_image[0], self.stamps_per_image[1] + 1,
                (B,), device=self.device)
            overlay_opacity_scale = 1.0
        max_layers = n_layers.max().item()
        max_stamps = n_stamps.max().item()

        # --- Single-frame tessellation (30% chance) ---
        # One base layer shrunk and tiled as repeating texture
        if torch.rand(1).item() < 0.3:
            idx = torch.randint(0, self.n_base_layers, (B,), device=self.device)
            fine_layer = self.base_layers[idx].clone()
            tile_f = torch.randint(2, 6, (1,)).item()
            sh = (H + tile_f - 1) // tile_f
            sw = (W + tile_f - 1) // tile_f
            small = F.interpolate(fine_layer, (sh, sw),
                                  mode="bilinear", align_corners=False)
            tiled = small.repeat(1, 1, tile_f, tile_f)[:, :, :H, :W]
            shift = (torch.rand(B, 3, 1, 1, device=self.device) - 0.5) * 0.4
            tiled = (tiled + shift).clamp(0, 1)
            opacity = torch.rand(B, 1, 1, 1, device=self.device) * 0.5 + 0.2
            canvas = canvas * (1 - opacity) + tiled * opacity

        # --- Multi-frame tile grid (25% chance) ---
        # Shrink multiple different base layers into a grid
        if torch.rand(1).item() < 0.25:
            grid_n = torch.randint(2, 5, (1,)).item()  # 2x2 to 4x4
            cell_h = H // grid_n
            cell_w = W // grid_n
            for bi in range(B):
                for gy in range(grid_n):
                    for gx in range(grid_n):
                        lidx = torch.randint(0, self.n_base_layers, (1,)).item()
                        cell = F.interpolate(
                            self.base_layers[lidx:lidx+1], (cell_h, cell_w),
                            mode="bilinear", align_corners=False)[0]
                        # Random color shift per cell
                        cs = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
                        cell = (cell + cs).clamp(0, 1)
                        y0 = gy * cell_h
                        x0 = gx * cell_w
                        eh = min(cell_h, H - y0)
                        ew = min(cell_w, W - x0)
                        opacity = torch.rand(1, device=self.device).item() * 0.5 + 0.3
                        canvas[bi, :, y0:y0+eh, x0:x0+ew] = \
                            canvas[bi, :, y0:y0+eh, x0:x0+ew] * (1 - opacity) + \
                            cell[:, :eh, :ew] * opacity

        # --- Layer compositing ---
        for li in range(max_layers):
            active = (li < n_layers).float().view(B, 1, 1, 1)

            # Select random base layers
            idx = torch.randint(0, self.n_base_layers, (B,), device=self.device)
            layer = self.base_layers[idx].clone()  # (B, 3, H, W)

            # Random color shift
            shift = (torch.rand(B, 3, 1, 1, device=self.device) - 0.5) * 0.3
            layer = (layer + shift).clamp(0, 1)

            # Random horizontal flip
            flip_mask = torch.rand(B, device=self.device) > 0.5
            for bi in range(B):
                if flip_mask[bi]:
                    layer[bi] = layer[bi].flip(-1)

            # Random tiling (sometimes)
            tile_mask = torch.rand(B, device=self.device) > 0.7
            for bi in range(B):
                if tile_mask[bi]:
                    scale = torch.randint(2, 4, (1,)).item()
                    sh = (H + scale - 1) // scale
                    sw = (W + scale - 1) // scale
                    small = F.interpolate(
                        layer[bi:bi+1], (sh, sw),
                        mode="bilinear", align_corners=False)
                    tiled = small.repeat(1, 1, scale, scale)
                    layer[bi] = tiled[0, :, :H, :W]

            # Random opacity (scaled down when template is active)
            opacity = (torch.rand(B, 1, 1, 1, device=self.device) * 0.6 + 0.3) * overlay_opacity_scale

            # Optional: mask with a shape from bank
            mask_prob = 0.4
            use_mask = torch.rand(B, device=self.device) < mask_prob
            if use_mask.any():
                midx = torch.randint(0, self.bank_size, (B,), device=self.device)
                mask_alpha = self.shape_bank[midx, 3:4]  # (B, 1, S, S)
                mask_alpha = F.interpolate(
                    mask_alpha, (H, W),
                    mode="bilinear", align_corners=False)  # (B, 1, H, W)
                # Apply mask only to images that want it
                mask_full = torch.ones(B, 1, H, W, device=self.device)
                mask_full[use_mask] = mask_alpha[use_mask]
                opacity = opacity * mask_full

            alpha = opacity * active
            canvas = canvas * (1 - alpha) + layer * alpha

        # --- Shape stamps + micro-stamps (via extracted method) ---
        stamp_range = (n_stamps.min().item(), n_stamps.max().item())
        canvas = self._overlay_shapes_on_canvas(canvas, B,
                                                 n_stamps_range=stamp_range)

        # --- Optional: fractal layout pass (30% of batches) ---
        if torch.rand(1).item() < 0.3 and self.shape_bank is not None:
            n_frac = torch.randint(8, 20, (1,)).item()
            canvas = self._render_fractal_layout(canvas, n_shapes=n_frac)

        # --- Post-processing mutations ---
        canvas = self._post_process(canvas)

        return canvas.clamp(0, 1)

    # ------------------------------------------------------------------
    # Temporal sequence generation (Stage 2)
    # ------------------------------------------------------------------

    def _simulate_physics(self, B, M, T):
        """Pre-compute stamp positions for T frames with physics.

        Returns: (B, M, T, 4) tensor — [x, y, scale, rotation] per stamp per frame.
        """
        H, W = self.H, self.W
        dt = 1.0 / 30.0  # simulate at 30fps

        # Initial positions
        x = torch.rand(B, M, device=self.device) * W
        y = torch.rand(B, M, device=self.device) * H * 0.5  # start upper half

        # Initial velocities (pixels per frame)
        vx = (torch.rand(B, M, device=self.device) - 0.5) * 15
        vy = (torch.rand(B, M, device=self.device) - 0.5) * 5

        # Gravity
        gravity = torch.rand(B, M, device=self.device) * 3 + 1  # pixels/frame^2

        # Scale: start + oscillation
        scale_base = torch.rand(B, M, device=self.device) * 1.0 + 0.3
        scale_freq = torch.rand(B, M, device=self.device) * 2  # oscillation speed
        scale_amp = torch.rand(B, M, device=self.device) * 0.3  # oscillation amplitude

        # Rotation
        rot = torch.rand(B, M, device=self.device) * 6.28
        rot_speed = (torch.rand(B, M, device=self.device) - 0.5) * 0.3  # rad/frame

        trajectories = torch.zeros(B, M, T, 4, device=self.device)

        for ti in range(T):
            # Record state
            scale_t = scale_base + scale_amp * torch.sin(
                scale_freq * ti * 0.5)
            trajectories[:, :, ti, 0] = x
            trajectories[:, :, ti, 1] = y
            trajectories[:, :, ti, 2] = scale_t
            trajectories[:, :, ti, 3] = rot

            # Update physics
            vy = vy + gravity  # gravity pulls down
            x = x + vx
            y = y + vy
            rot = rot + rot_speed

            # Bounce off walls
            hit_left = x < 0
            hit_right = x > W
            hit_top = y < 0
            hit_bottom = y > H

            vx = torch.where(hit_left | hit_right, -vx * 0.8, vx)
            vy = torch.where(hit_top | hit_bottom, -vy * 0.7, vy)
            x = x.clamp(0, W)
            y = y.clamp(0, H)

        return trajectories

    def _generate_flow_field(self, B, T):
        """Generate a smooth 2D velocity field for fluid-like advection.

        Returns: (B, T, 2, H, W) displacement field — how much each pixel
        moves per frame. Based on time-varying Perlin noise.
        """
        H, W = self.H, self.W
        # Coarse velocity field (faster than full-res)
        ch, cw = H // 8, W // 8

        # Random phase for temporal variation
        phase = torch.rand(B, 1, 1, 1, device=self.device) * 6.28

        fields = []
        for ti in range(T):
            t_val = ti / max(T - 1, 1)
            # Two Perlin noise fields (one per velocity component)
            vx_noise = torch.randn(B, 1, ch, cw, device=self.device)
            vy_noise = torch.randn(B, 1, ch, cw, device=self.device)
            # Smooth
            k = 3
            g = torch.tensor([0.25, 0.5, 0.25], device=self.device).view(1, 1, k, 1)
            vx_noise = F.conv2d(vx_noise, g, padding=(1, 0))
            vx_noise = F.conv2d(vx_noise, g.permute(0, 1, 3, 2), padding=(0, 1))
            vy_noise = F.conv2d(vy_noise, g, padding=(1, 0))
            vy_noise = F.conv2d(vy_noise, g.permute(0, 1, 3, 2), padding=(0, 1))
            # Upsample to full res
            vx = F.interpolate(vx_noise, (H, W), mode="bilinear",
                               align_corners=False)
            vy = F.interpolate(vy_noise, (H, W), mode="bilinear",
                               align_corners=False)
            # Scale displacement (pixels per frame)
            strength = 3.0
            field = torch.cat([vx * strength, vy * strength], dim=1)  # (B, 2, H, W)
            fields.append(field)

        return torch.stack(fields, dim=1)  # (B, T, 2, H, W)

    def _apply_viewport(self, canvas, ti, T, vp_pan, vp_zoom, vp_rot):
        """Apply viewport-level transform to entire canvas.

        Args:
            canvas: (B, 3, H, W)
            ti: current frame index
            T: total frames
            vp_pan: (B, 2) total pan displacement [dx, dy]
            vp_zoom: (B, 2) start/end zoom [start, end]
            vp_rot: (B, 2) start/end rotation [start, end] in radians
        """
        B = canvas.shape[0]
        H, W = self.H, self.W
        t_frac = ti / max(T - 1, 1)

        dx = vp_pan[:, 0] * t_frac
        dy = vp_pan[:, 1] * t_frac
        zoom = vp_zoom[:, 0] * (1 - t_frac) + vp_zoom[:, 1] * t_frac
        angle = vp_rot[:, 0] * (1 - t_frac) + vp_rot[:, 1] * t_frac

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = cos_a / zoom
        theta[:, 0, 1] = -sin_a / zoom
        theta[:, 1, 0] = sin_a / zoom
        theta[:, 1, 1] = cos_a / zoom
        theta[:, 0, 2] = -dx / (W / 2)
        theta[:, 1, 2] = -dy / (H / 2)

        grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    def _apply_fluid(self, canvas, flow_field):
        """Advect canvas pixels through a velocity field.

        Args:
            canvas: (B, 3, H, W)
            flow_field: (B, 2, H, W) displacement in pixels
        """
        B, _, H, W = canvas.shape
        # Build sample grid: identity + displacement
        y = torch.linspace(-1, 1, H, device=self.device)
        x = torch.linspace(-1, 1, W, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Convert pixel displacement to normalized coords
        dx = flow_field[:, 0:1].permute(0, 2, 3, 1) / (W / 2)  # (B, H, W, 1)
        dy = flow_field[:, 1:2].permute(0, 2, 3, 1) / (H / 2)
        offset = torch.cat([dx, dy], dim=-1)  # (B, H, W, 2)

        grid = base_grid + offset
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    @torch.no_grad()
    def generate_sequence(self, batch_size, T=8,
                           use_physics=True, use_rotation=True,
                           use_zoom=True, use_fade=True,
                           use_viewport=True, use_fluid=False,
                           pan_strength=0.5, motion_strength=0.4,
                           viewport_pan=0.3, viewport_zoom=0.15,
                           viewport_rotation=0.2, fluid_strength=1.0):
        """Generate animated clips with physics, viewport effects, and fluid motion.

        Returns: (B, T, 3, H, W) tensor in [0, 1] on self.device.
        """
        if self.base_layers is None:
            self.build_banks()
        self._maybe_refresh_dynamic()

        B = batch_size
        H, W = self.H, self.W

        # --- Background ---
        bg = self._sample_colors(B).view(B, 3, 1, 1)

        # --- Layer setup ---
        n_layers = torch.randint(
            self.layers_per_image[0], self.layers_per_image[1] + 1,
            (B,), device=self.device)
        max_layers = n_layers.max().item()

        layer_idx = torch.randint(0, self.n_base_layers, (B, max_layers),
                                   device=self.device)
        pan_dx = (torch.rand(B, max_layers, device=self.device) - 0.5) * W * pan_strength
        pan_dy = (torch.rand(B, max_layers, device=self.device) - 0.5) * H * pan_strength
        pan_speed = torch.rand(B, max_layers, device=self.device) * 0.8 + 0.2

        opacity_start = torch.rand(B, max_layers, device=self.device) * 0.5 + 0.3
        opacity_end = opacity_start.clone() if not use_fade else \
            torch.rand(B, max_layers, device=self.device) * 0.5 + 0.3

        color_shift = (torch.rand(B, max_layers, 3, device=self.device) - 0.5) * 0.3

        # Layer zoom
        if use_zoom:
            zoom_start = torch.rand(B, max_layers, device=self.device) * 0.3 + 0.85
            zoom_end = torch.rand(B, max_layers, device=self.device) * 0.3 + 0.85
        else:
            zoom_start = torch.ones(B, max_layers, device=self.device)
            zoom_end = zoom_start

        # --- Stamp setup with physics ---
        n_stamps = torch.randint(
            self.stamps_per_image[0], self.stamps_per_image[1] + 1,
            (B,), device=self.device)
        max_stamps = n_stamps.max().item()

        stamp_idx = torch.randint(0, self.shape_bank.shape[0],
                                   (B, max_stamps), device=self.device)

        if use_physics:
            trajectories = self._simulate_physics(B, max_stamps, T)
        else:
            # Simple linear motion
            trajectories = torch.zeros(B, max_stamps, T, 4, device=self.device)
            sx = torch.rand(B, max_stamps, device=self.device) * W
            sy = torch.rand(B, max_stamps, device=self.device) * H
            dx = (torch.rand(B, max_stamps, device=self.device) - 0.5) * W * motion_strength
            dy = (torch.rand(B, max_stamps, device=self.device) - 0.5) * H * motion_strength
            sc = torch.rand(B, max_stamps, device=self.device) * 1.2 + 0.3
            for ti in range(T):
                t_frac = ti / max(T - 1, 1)
                trajectories[:, :, ti, 0] = sx + dx * t_frac
                trajectories[:, :, ti, 1] = sy + dy * t_frac
                trajectories[:, :, ti, 2] = sc
                trajectories[:, :, ti, 3] = 0

        # Pre-transform stamps (consistent across frames for temporal coherence)
        stamp_shapes = []
        for si in range(max_stamps):
            per_batch = []
            for bi in range(B):
                rgba = self._transform_bank_shape(
                    self.shape_bank[stamp_idx[bi, si]].clone())
                per_batch.append(rgba)
            stamp_shapes.append(torch.stack(per_batch))  # (B, 4, S, S)

        # Pre-generate fine-grain texture layer (consistent across frames)
        fine_layer = None
        fine_opacity = 0
        if torch.rand(1).item() < 0.5:
            fidx = torch.randint(0, self.n_base_layers, (B,), device=self.device)
            fine_src = self.base_layers[fidx].clone()
            tile_f = torch.randint(3, 7, (1,)).item()
            fh = (H + tile_f - 1) // tile_f
            fw = (W + tile_f - 1) // tile_f
            small = F.interpolate(fine_src, (fh, fw),
                                  mode="bilinear", align_corners=False)
            fine_layer = small.repeat(1, 1, tile_f, tile_f)[:, :, :H, :W]
            cs = (torch.rand(B, 3, 1, 1, device=self.device) - 0.5) * 0.4
            fine_layer = (fine_layer + cs).clamp(0, 1)
            fine_opacity = torch.rand(1, device=self.device).item() * 0.4 + 0.15

        # Pre-generate micro-stamp positions (consistent shapes, moving positions)
        n_micro = 0
        micro_shapes = []
        micro_start_x = None
        micro_start_y = None
        micro_dx = None
        micro_dy = None
        if torch.rand(1).item() < 0.5 and self.shape_bank is not None:
            n_micro = torch.randint(20, 60, (1,)).item()
            for _ in range(n_micro):
                sidx = torch.randint(0, self.shape_bank.shape[0], (B,),
                                      device=self.device)
                s = self.shape_bank[sidx].clone()
                for bi in range(B):
                    s[bi] = self._transform_bank_shape(s[bi])
                micro_shapes.append(s)
            micro_start_x = torch.rand(B, n_micro, device=self.device) * W
            micro_start_y = torch.rand(B, n_micro, device=self.device) * H
            micro_dx = (torch.rand(B, n_micro, device=self.device) - 0.5) * W * 0.15
            micro_dy = (torch.rand(B, n_micro, device=self.device) - 0.5) * H * 0.15

        # Consistent post-processing
        pp_gamma = torch.rand(B, 1, 1, 1, device=self.device) * 0.7 + 0.7

        # Viewport effects
        if use_viewport:
            vp_pan = (torch.rand(B, 2, device=self.device) - 0.5) * \
                torch.tensor([W, H], device=self.device).float() * viewport_pan
            vp_zoom = torch.ones(B, 2, device=self.device)
            vp_zoom[:, 0] = 1.0 - viewport_zoom / 2
            vp_zoom[:, 1] = 1.0 + viewport_zoom / 2
            vp_rot = (torch.rand(B, 2, device=self.device) - 0.5) * viewport_rotation
        else:
            vp_pan = torch.zeros(B, 2, device=self.device)
            vp_zoom = torch.ones(B, 2, device=self.device)
            vp_rot = torch.zeros(B, 2, device=self.device)

        # Fluid flow field
        flow_fields = None
        if use_fluid:
            flow_fields = self._generate_flow_field(B, T) * fluid_strength

        # Scene template (consistent across all frames in clip)
        template = self._pick_template()
        has_template = template != "random"

        # --- Render T frames ---
        frames = []
        for ti in range(T):
            t_frac = ti / max(T - 1, 1)
            canvas = bg.expand(B, 3, H, W).clone()

            # Apply scene template
            if has_template:
                canvas = self._apply_scene_template(canvas, template, B)

            # --- Layers with pan + zoom ---
            for li in range(max_layers):
                active = (li < n_layers).float().view(B, 1, 1, 1)
                layer = self.base_layers[layer_idx[:, li]].clone()
                cs = color_shift[:, li].view(B, 3, 1, 1)
                layer = (layer + cs).clamp(0, 1)

                dx = pan_dx[:, li] * pan_speed[:, li] * t_frac
                dy = pan_dy[:, li] * pan_speed[:, li] * t_frac
                zoom = zoom_start[:, li] * (1 - t_frac) + zoom_end[:, li] * t_frac

                theta = torch.zeros(B, 2, 3, device=self.device)
                theta[:, 0, 0] = 1.0 / zoom
                theta[:, 1, 1] = 1.0 / zoom
                theta[:, 0, 2] = -dx / (W / 2)
                theta[:, 1, 2] = -dy / (H / 2)
                grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
                layer = F.grid_sample(layer, grid, mode="bilinear",
                                       padding_mode="reflection", align_corners=False)

                opacity = (opacity_start[:, li] * (1 - t_frac) +
                           opacity_end[:, li] * t_frac).view(B, 1, 1, 1)
                alpha = opacity * active
                canvas = canvas * (1 - alpha) + layer * alpha

            # --- Stamps with physics trajectories ---
            for si in range(max_stamps):
                active_s = (si < n_stamps).float()
                shapes = stamp_shapes[si]  # (B, 4, S, S)

                for bi in range(B):
                    if active_s[bi] < 0.5:
                        continue

                    px_f = trajectories[bi, si, ti, 0].item()
                    py_f = trajectories[bi, si, ti, 1].item()
                    sc = trajectories[bi, si, ti, 2].item()
                    rot_angle = trajectories[bi, si, ti, 3].item()

                    rgba = shapes[bi]  # (4, S, S)

                    # Apply rotation if enabled
                    if use_rotation and abs(rot_angle) > 0.01:
                        cos_r = math.cos(rot_angle)
                        sin_r = math.sin(rot_angle)
                        rot_theta = torch.tensor(
                            [[cos_r, -sin_r, 0], [sin_r, cos_r, 0]],
                            device=self.device).unsqueeze(0)
                        rot_grid = F.affine_grid(
                            rot_theta, (1, 4, self.shape_res, self.shape_res),
                            align_corners=False)
                        rgba = F.grid_sample(
                            rgba.unsqueeze(0), rot_grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)[0]

                    rgb = rgba[:3]
                    alpha_s = rgba[3:4]

                    th = max(3, int(self.shape_res * sc * H / 360))
                    tw = max(3, int(self.shape_res * sc * W / 640))
                    th, tw = min(th, H // 2), min(tw, W // 2)

                    rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                          mode="bilinear", align_corners=False)[0]
                    a_r = F.interpolate(alpha_s.unsqueeze(0), (th, tw),
                                        mode="bilinear", align_corners=False)[0]

                    px = int(px_f) - tw // 2
                    py = int(py_f) - th // 2

                    sx = max(0, -px)
                    sy = max(0, -py)
                    ex = min(tw, W - px)
                    ey = min(th, H - py)
                    if ex <= sx or ey <= sy:
                        continue
                    cx, cy = max(0, px), max(0, py)
                    rh, rw = ey - sy, ex - sx

                    a = a_r[:, sy:ey, sx:ex]
                    canvas[bi, :, cy:cy+rh, cx:cx+rw] = \
                        canvas[bi, :, cy:cy+rh, cx:cx+rw] * (1 - a) + \
                        rgb_r[:, sy:ey, sx:ex] * a

            # Fine-grain texture overlay (moves with viewport)
            if fine_layer is not None:
                canvas = canvas * (1 - fine_opacity) + fine_layer * fine_opacity

            # Micro-stamps with linear motion
            if n_micro > 0:
                for mi in range(n_micro):
                    shapes_m = micro_shapes[mi]  # (B, 4, S, S)
                    for bi in range(B):
                        rgba = shapes_m[bi]
                        rgb = rgba[:3]
                        alpha_m = rgba[3:4]
                        sc = 0.1 + torch.rand(1).item() * 0.15
                        th = max(3, int(self.shape_res * sc * H / 360))
                        tw = max(3, int(self.shape_res * sc * W / 640))
                        th, tw = min(th, H // 3), min(tw, W // 3)
                        rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                              mode="bilinear", align_corners=False)[0]
                        a_r = F.interpolate(alpha_m.unsqueeze(0), (th, tw),
                                            mode="bilinear", align_corners=False)[0]
                        px = int(micro_start_x[bi, mi].item() +
                                 micro_dx[bi, mi].item() * t_frac) - tw // 2
                        py = int(micro_start_y[bi, mi].item() +
                                 micro_dy[bi, mi].item() * t_frac) - th // 2
                        if 0 <= px < W - tw and 0 <= py < H - th:
                            canvas[bi, :, py:py+th, px:px+tw] = \
                                canvas[bi, :, py:py+th, px:px+tw] * (1 - a_r) + \
                                rgb_r * a_r

            # Viewport transform (pan + zoom + rotation on entire canvas)
            if use_viewport:
                canvas = self._apply_viewport(canvas, ti, T, vp_pan, vp_zoom, vp_rot)

            # Fluid advection
            if flow_fields is not None:
                canvas = self._apply_fluid(canvas, flow_fields[:, ti])

            canvas = canvas.clamp(1e-6, 1).pow(pp_gamma)
            frames.append(canvas.clamp(0, 1))

        return torch.stack(frames, dim=1)  # (B, T, 3, H, W)

    # ------------------------------------------------------------------
    # Stage 3: 9ch temporal generation (RGB + depth + flow + semantic)
    # ------------------------------------------------------------------

    # Semantic role colors (approximate C-RADIO PCA mapping)
    _SEM_BG = torch.tensor([0.3, 0.2, 0.8])      # background: blue-pink
    _SEM_LAYER = torch.tensor([0.8, 0.5, 0.3])    # static layers: orange
    _SEM_STAMP = torch.tensor([0.2, 0.8, 0.3])    # moving stamps: green
    _SEM_MICRO = torch.tensor([0.3, 0.6, 0.6])    # micro detail: cyan

    @torch.no_grad()
    def generate_sequence_9ch(self, batch_size, T=8, **kwargs):
        """Generate animated clips with all 9 channels.

        Returns: (B, T, 9, H, W) tensor in [0, 1] on self.device.
        Channels: RGB(3) + depth(1) + flow(2) + semantic(3)
        """
        if self.base_layers is None:
            self.build_banks()
        self._maybe_refresh_dynamic()

        B = batch_size
        H, W = self.H, self.W
        dev = self.device

        # Semantic role colors on device
        sem_bg = self._SEM_BG.to(dev).view(3, 1, 1)
        sem_layer = self._SEM_LAYER.to(dev).view(3, 1, 1)
        sem_stamp = self._SEM_STAMP.to(dev).view(3, 1, 1)
        sem_micro = self._SEM_MICRO.to(dev).view(3, 1, 1)

        # --- Scene setup (same as generate_sequence) ---
        bg_rgb = self._sample_colors(B).view(B, 3, 1, 1)
        bg_depth = torch.ones(B, 1, 1, 1, device=dev)  # far
        bg_sem = sem_bg.unsqueeze(0).expand(B, 3, 1, 1)

        # Layer params
        n_layers = torch.randint(
            self.layers_per_image[0], self.layers_per_image[1] + 1,
            (B,), device=dev)
        max_layers = n_layers.max().item()
        layer_idx = torch.randint(0, self.n_base_layers, (B, max_layers), device=dev)
        pan_str = kwargs.get("pan_strength", 0.5)
        pan_dx = (torch.rand(B, max_layers, device=dev) - 0.5) * W * pan_str
        pan_dy = (torch.rand(B, max_layers, device=dev) - 0.5) * H * pan_str
        pan_speed = torch.rand(B, max_layers, device=dev) * 0.8 + 0.2
        opacity_start = torch.rand(B, max_layers, device=dev) * 0.5 + 0.3
        opacity_end = torch.rand(B, max_layers, device=dev) * 0.5 + 0.3
        color_shift = (torch.rand(B, max_layers, 3, device=dev) - 0.5) * 0.3
        zoom_start = torch.rand(B, max_layers, device=dev) * 0.3 + 0.85
        zoom_end = torch.rand(B, max_layers, device=dev) * 0.3 + 0.85

        # Stamp params with physics
        n_stamps = torch.randint(
            self.stamps_per_image[0], self.stamps_per_image[1] + 1,
            (B,), device=dev)
        max_stamps = n_stamps.max().item()
        stamp_idx = torch.randint(0, self.shape_bank.shape[0],
                                   (B, max_stamps), device=dev)
        trajectories = self._simulate_physics(B, max_stamps, T)

        # Pre-transform stamps
        stamp_shapes = []
        for si in range(max_stamps):
            per_batch = []
            for bi in range(B):
                rgba = self._transform_bank_shape(
                    self.shape_bank[stamp_idx[bi, si]].clone())
                per_batch.append(rgba)
            stamp_shapes.append(torch.stack(per_batch))

        # Viewport params
        use_vp = kwargs.get("use_viewport", True)
        if use_vp:
            vp_pan_s = kwargs.get("viewport_pan", 0.3)
            vp_pan = (torch.rand(B, 2, device=dev) - 0.5) * \
                torch.tensor([W, H], device=dev).float() * vp_pan_s
            vp_zoom_s = kwargs.get("viewport_zoom", 0.15)
            vp_zoom = torch.ones(B, 2, device=dev)
            vp_zoom[:, 0] = 1.0 - vp_zoom_s / 2
            vp_zoom[:, 1] = 1.0 + vp_zoom_s / 2
            vp_rot_s = kwargs.get("viewport_rotation", 0.2)
            vp_rot = (torch.rand(B, 2, device=dev) - 0.5) * vp_rot_s

        pp_gamma = torch.rand(B, 1, 1, 1, device=dev) * 0.7 + 0.7

        # Depth values per layer/stamp
        layer_depths = torch.linspace(0.9, 0.6, max_layers, device=dev)
        stamp_depths = torch.linspace(0.5, 0.2, max_stamps, device=dev)

        # Scene template (consistent across all frames)
        template = self._pick_template()
        has_template = template != "random"

        # --- Render T frames with all channels ---
        all_frames = []  # list of (B, 9, H, W)
        prev_rgb = None

        for ti in range(T):
            t_frac = ti / max(T - 1, 1)

            # Init canvases
            rgb = bg_rgb.expand(B, 3, H, W).clone()
            depth = bg_depth.expand(B, 1, H, W).clone()
            sem = bg_sem.expand(B, 3, H, W).clone()

            # Apply scene template
            if has_template:
                rgb = self._apply_scene_template(rgb, template, B)

            # --- Layer compositing ---
            for li in range(max_layers):
                active = (li < n_layers).float().view(B, 1, 1, 1)
                layer = self.base_layers[layer_idx[:, li]].clone()
                cs = color_shift[:, li].view(B, 3, 1, 1)
                layer = (layer + cs).clamp(0, 1)

                dx = pan_dx[:, li] * pan_speed[:, li] * t_frac
                dy = pan_dy[:, li] * pan_speed[:, li] * t_frac
                zoom = zoom_start[:, li] * (1 - t_frac) + zoom_end[:, li] * t_frac

                theta = torch.zeros(B, 2, 3, device=dev)
                theta[:, 0, 0] = 1.0 / zoom
                theta[:, 1, 1] = 1.0 / zoom
                theta[:, 0, 2] = -dx / (W / 2)
                theta[:, 1, 2] = -dy / (H / 2)
                grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
                layer = F.grid_sample(layer, grid, mode="bilinear",
                                       padding_mode="reflection", align_corners=False)

                opacity = (opacity_start[:, li] * (1 - t_frac) +
                           opacity_end[:, li] * t_frac).view(B, 1, 1, 1)
                alpha = opacity * active

                # Parallel compositing: RGB + depth + semantic
                rgb = rgb * (1 - alpha) + layer * alpha
                d_val = layer_depths[li].view(1, 1, 1, 1)
                depth = depth * (1 - alpha) + d_val * alpha
                sem_c = (sem_layer + torch.rand(3, 1, 1, device=dev) * 0.1
                         ).unsqueeze(0)
                sem = sem * (1 - alpha) + sem_c * alpha

            # --- Stamp compositing ---
            for si in range(max_stamps):
                active_s = (si < n_stamps).float()
                shapes = stamp_shapes[si]

                for bi in range(B):
                    if active_s[bi] < 0.5:
                        continue

                    px_f = trajectories[bi, si, ti, 0].item()
                    py_f = trajectories[bi, si, ti, 1].item()
                    sc = trajectories[bi, si, ti, 2].item()
                    rot_angle = trajectories[bi, si, ti, 3].item()

                    rgba = shapes[bi]
                    if kwargs.get("use_rotation", True) and abs(rot_angle) > 0.01:
                        cos_r = math.cos(rot_angle)
                        sin_r = math.sin(rot_angle)
                        rot_theta = torch.tensor(
                            [[cos_r, -sin_r, 0], [sin_r, cos_r, 0]],
                            device=dev).unsqueeze(0)
                        rot_grid = F.affine_grid(
                            rot_theta, (1, 4, self.shape_res, self.shape_res),
                            align_corners=False)
                        rgba = F.grid_sample(
                            rgba.unsqueeze(0), rot_grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)[0]

                    s_rgb = rgba[:3]
                    s_alpha = rgba[3:4]

                    th = max(3, int(self.shape_res * sc * H / 360))
                    tw = max(3, int(self.shape_res * sc * W / 640))
                    th, tw = min(th, H // 2), min(tw, W // 2)

                    rgb_r = F.interpolate(s_rgb.unsqueeze(0), (th, tw),
                                          mode="bilinear", align_corners=False)[0]
                    a_r = F.interpolate(s_alpha.unsqueeze(0), (th, tw),
                                        mode="bilinear", align_corners=False)[0]

                    px = int(px_f) - tw // 2
                    py = int(py_f) - th // 2
                    if 0 <= px < W - tw and 0 <= py < H - th:
                        a = a_r
                        rgb[bi, :, py:py+th, px:px+tw] = \
                            rgb[bi, :, py:py+th, px:px+tw] * (1 - a) + rgb_r * a
                        d_val = stamp_depths[min(si, len(stamp_depths) - 1)]
                        depth[bi, :, py:py+th, px:px+tw] = \
                            depth[bi, :, py:py+th, px:px+tw] * (1 - a) + d_val * a
                        sem_c = sem_stamp + torch.rand(3, 1, 1, device=dev) * 0.1
                        sem[bi, :, py:py+th, px:px+tw] = \
                            sem[bi, :, py:py+th, px:px+tw] * (1 - a) + sem_c * a

            # Viewport transform (applied to all channels)
            if use_vp:
                rgb = self._apply_viewport(rgb, ti, T, vp_pan, vp_zoom, vp_rot)
                depth = self._apply_viewport(depth, ti, T, vp_pan, vp_zoom, vp_rot)
                sem = self._apply_viewport(sem, ti, T, vp_pan, vp_zoom, vp_rot)

            # Gamma on RGB only
            rgb = rgb.clamp(1e-6, 1).pow(pp_gamma)

            # --- Flow computation ---
            if prev_rgb is not None:
                # Approximate flow from pixel difference
                # Real flow comes from motion params but this is a simpler start
                flow = torch.zeros(B, 2, H, W, device=dev)

                # Global flow from viewport
                if use_vp:
                    per_frame_dx = vp_pan[:, 0] / max(T - 1, 1)
                    per_frame_dy = vp_pan[:, 1] / max(T - 1, 1)
                    flow[:, 0] = (per_frame_dx / (W / 2)).view(B, 1, 1).expand(B, H, W)
                    flow[:, 1] = (per_frame_dy / (H / 2)).view(B, 1, 1).expand(B, H, W)

                # Layer pan flow (additive, weighted by layer alpha)
                for li in range(max_layers):
                    active_l = (li < n_layers).float().view(B, 1, 1)
                    per_frame_dx = pan_dx[:, li] * pan_speed[:, li] / max(T - 1, 1)
                    per_frame_dy = pan_dy[:, li] * pan_speed[:, li] / max(T - 1, 1)
                    w = opacity_start[:, li].view(B, 1, 1) * active_l * 0.3
                    flow[:, 0] += (per_frame_dx / (W / 2)).view(B, 1, 1) * w
                    flow[:, 1] += (per_frame_dy / (H / 2)).view(B, 1, 1) * w

                # Stamp flow (local, at stamp positions)
                for si in range(max_stamps):
                    for bi in range(B):
                        if (si < n_stamps[bi]).item():
                            dx_s = (trajectories[bi, si, ti, 0] -
                                    trajectories[bi, si, ti-1, 0]).item()
                            dy_s = (trajectories[bi, si, ti, 1] -
                                    trajectories[bi, si, ti-1, 1]).item()
                            sc_s = trajectories[bi, si, ti, 2].item()
                            sx = int(trajectories[bi, si, ti, 0].item())
                            sy = int(trajectories[bi, si, ti, 1].item())
                            r = max(3, int(self.shape_res * sc_s * H / 720))
                            x0 = max(0, sx - r)
                            y0 = max(0, sy - r)
                            x1 = min(W, sx + r)
                            y1 = min(H, sy + r)
                            if x1 > x0 and y1 > y0:
                                flow[bi, 0, y0:y1, x0:x1] += dx_s / (W / 2)
                                flow[bi, 1, y0:y1, x0:x1] += dy_s / (H / 2)

                # Normalize to [0, 1] from [-1, 1]
                flow = (flow.clamp(-1, 1) + 1) / 2
            else:
                flow = torch.full((B, 2, H, W), 0.5, device=dev)  # neutral

            prev_rgb = rgb.clone()

            # Combine all channels
            frame_9ch = torch.cat([
                rgb.clamp(0, 1),
                depth.clamp(0, 1),
                flow.clamp(0, 1),
                sem.clamp(0, 1),
            ], dim=1)  # (B, 9, H, W)
            all_frames.append(frame_9ch)

        return torch.stack(all_frames, dim=1)  # (B, T, 9, H, W)

    # ------------------------------------------------------------------
    # Motion clip pool (Stage 2 speed optimization)
    # ------------------------------------------------------------------

    def _generate_recipe(self, T=8, **seq_kwargs):
        """Generate a single motion recipe — lightweight parameter dict.

        A recipe stores which shapes/layers to use and how they move,
        NOT the rendered pixels. ~1KB per recipe vs ~9MB per rendered clip.
        """
        H, W = self.H, self.W
        n_layers = torch.randint(
            self.layers_per_image[0], self.layers_per_image[1] + 1, (1,)).item()
        n_stamps = torch.randint(
            self.stamps_per_image[0], self.stamps_per_image[1] + 1, (1,)).item()

        recipe = {
            "T": T,
            "bg_color": torch.rand(3).tolist(),
            # Layers
            "layer_idx": torch.randint(0, self.n_base_layers, (n_layers,)).tolist(),
            "pan_dx": ((torch.rand(n_layers) - 0.5) * W * seq_kwargs.get("pan_strength", 0.5)).tolist(),
            "pan_dy": ((torch.rand(n_layers) - 0.5) * H * seq_kwargs.get("pan_strength", 0.5)).tolist(),
            "pan_speed": (torch.rand(n_layers) * 0.8 + 0.2).tolist(),
            "opacity_start": (torch.rand(n_layers) * 0.5 + 0.3).tolist(),
            "opacity_end": (torch.rand(n_layers) * 0.5 + 0.3).tolist(),
            "color_shift": ((torch.rand(n_layers, 3) - 0.5) * 0.3).tolist(),
            "zoom_start": (torch.rand(n_layers) * 0.3 + 0.85).tolist(),
            "zoom_end": (torch.rand(n_layers) * 0.3 + 0.85).tolist(),
            # Stamps
            "stamp_idx": torch.randint(0, max(self.bank_size, 1), (n_stamps,)).tolist(),
            "stamp_x": (torch.rand(n_stamps) * W).tolist(),
            "stamp_y": (torch.rand(n_stamps) * H * 0.5).tolist(),
            "stamp_vx": ((torch.rand(n_stamps) - 0.5) * 15).tolist(),
            "stamp_vy": ((torch.rand(n_stamps) - 0.5) * 5).tolist(),
            "stamp_gravity": (torch.rand(n_stamps) * 3 + 1).tolist(),
            "stamp_scale": (torch.rand(n_stamps) * 1.0 + 0.3).tolist(),
            "stamp_rot": (torch.rand(n_stamps) * 6.28).tolist(),
            "stamp_rot_speed": ((torch.rand(n_stamps) - 0.5) * 0.3).tolist(),
            # Viewport
            "vp_pan": ((torch.rand(2) - 0.5) * torch.tensor([W, H]).float() *
                        seq_kwargs.get("viewport_pan", 0.3)).tolist(),
            "vp_zoom": [1.0 - seq_kwargs.get("viewport_zoom", 0.15) / 2,
                         1.0 + seq_kwargs.get("viewport_zoom", 0.15) / 2],
            "vp_rot": ((torch.rand(2) - 0.5) * seq_kwargs.get("viewport_rotation", 0.2)).tolist(),
            # Fine grain
            "fine_layer_idx": torch.randint(0, self.n_base_layers, (1,)).item(),
            "fine_tile": torch.randint(3, 7, (1,)).item(),
            "fine_opacity": torch.rand(1).item() * 0.4 + 0.15,
            "use_fine": torch.rand(1).item() < 0.5,
            # Micro stamps
            "n_micro": torch.randint(20, 60, (1,)).item() if torch.rand(1).item() < 0.5 else 0,
            "micro_idx": torch.randint(0, max(self.bank_size, 1), (60,)).tolist(),
            "micro_x": (torch.rand(60) * W).tolist(),
            "micro_y": (torch.rand(60) * H).tolist(),
            "micro_dx": ((torch.rand(60) - 0.5) * W * 0.15).tolist(),
            "micro_dy": ((torch.rand(60) - 0.5) * H * 0.15).tolist(),
            # Settings
            "gamma": torch.rand(1).item() * 0.7 + 0.7,
            "seq_kwargs": seq_kwargs,
        }
        return recipe

    def build_motion_pool(self, n_clips=200, T=8, **seq_kwargs):
        """Generate motion recipes (not pixels). Lightweight, ~1KB each.

        Args:
            n_clips: number of recipes
            T: frames per clip
            **seq_kwargs: motion settings
        """
        print(f"Building motion recipe pool ({n_clips} recipes, T={T})...",
              flush=True)
        recipes = []
        for i in range(n_clips):
            recipes.append(self._generate_recipe(T=T, **seq_kwargs))
            if (i + 1) % 100 == 0 or i == n_clips - 1:
                print(f"  [{i+1}/{n_clips}]", flush=True)
        self._recipe_pool = recipes
        self._motion_pool_T = T
        self._motion_pool_call_count = 0
        print(f"Recipe pool done: {n_clips} recipes", flush=True)

    def _refresh_motion_pool(self, swap_frac=0.2, **seq_kwargs):
        """Swap a fraction of recipes with fresh ones."""
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return
        N = len(self._recipe_pool)
        T = self._motion_pool_T
        n_swap = max(1, int(N * swap_frac))
        for _ in range(n_swap):
            idx = torch.randint(0, N, (1,)).item()
            self._recipe_pool[idx] = self._generate_recipe(
                T=T, **seq_kwargs)

    def _render_recipe(self, recipe):
        """Render a single recipe into a (T, 3, H, W) clip on GPU."""
        T = recipe["T"]
        # Reconstruct all the params and call generate_sequence with B=1
        # For simplicity, just call generate_sequence which already handles everything
        # But seed it with the recipe's parameters
        # For now, render fresh with the stored seq_kwargs
        clip = self.generate_sequence(1, T=T, **recipe.get("seq_kwargs", {}))
        return clip[0]  # (T, 3, H, W)

    def generate_from_pool(self, batch_size, refresh_interval=100):
        """Render B clips from recipe pool.

        Picks random recipes, renders them live from the shape bank.
        Applies anti-memorization transforms after rendering.

        Returns: (B, T, 3, H, W) tensor in [0, 1] on self.device.
        """
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            raise RuntimeError("Call build_motion_pool() first")

        B = batch_size
        N = len(self._recipe_pool)

        # Periodic refresh
        self._motion_pool_call_count += 1
        if self._motion_pool_call_count % refresh_interval == 0:
            self._refresh_motion_pool()

        # Render B recipes
        clips = []
        for bi in range(B):
            idx = torch.randint(0, N, (1,)).item()
            clip = self._render_recipe(self._recipe_pool[idx])
            clips.append(clip)
        clips = torch.stack(clips)  # (B, T, 3, H, W)

        # Anti-memorization transforms (fast, on GPU)
        for bi in range(B):
            if torch.rand(1).item() < 0.5:
                clips[bi] = clips[bi].flip(-1)
            if torch.rand(1).item() < 0.2:
                clips[bi] = clips[bi].flip(-2)
            if torch.rand(1).item() < 0.3:
                clips[bi] = clips[bi].flip(0)  # temporal reverse
            shift = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
            clips[bi] = (clips[bi] + shift).clamp(0, 1)
            if torch.rand(1).item() < 0.3:
                clips[bi] = clips[bi].roll(torch.randint(1, 3, (1,)).item(), dims=1)

        return clips.clamp(0, 1)

    def save_motion_pool(self, path):
        """Save recipe pool to disk with timestamp. Accumulates, doesn't overwrite."""
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return
        import json, time as _time
        # Timestamped filename in same directory
        d = os.path.dirname(path)
        ts = _time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(d, f"recipes_{ts}.json")
        os.makedirs(d, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(self._recipe_pool, f)
        kb = os.path.getsize(out) / 1e3
        print(f"Saved {len(self._recipe_pool)} recipes to {out} "
              f"({kb:.0f} KB)", flush=True)

    def load_motion_pool(self, path):
        """Load recipe pool from disk. If path is a directory, loads all recipe files."""
        import json
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path)
                           if f.startswith("recipes_") and f.endswith(".json"))
            if not files:
                print(f"No recipe files in {path}", flush=True)
                return
            for f in files:
                fp = os.path.join(path, f)
                with open(fp) as fh:
                    recipes = json.load(fh)
                self._recipe_pool.extend(recipes)
                print(f"  Loaded {len(recipes)} recipes from {f}", flush=True)
        else:
            with open(path) as f:
                recipes = json.load(f)
            self._recipe_pool.extend(recipes)
        self._motion_pool_T = self._recipe_pool[0]["T"] if self._recipe_pool else 8
        self._motion_pool_call_count = 0
        print(f"Total recipes: {len(self._recipe_pool)}", flush=True)

    def motion_pool_stats(self):
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return None
        return {
            "count": len(self._recipe_pool),
            "T": self._motion_pool_T,
            "kb": len(self._recipe_pool),  # ~1KB each
        }

    # ------------------------------------------------------------------
    # Shape transform on sample (anti-memorization)
    # ------------------------------------------------------------------

    def _transform_bank_shape(self, rgba):
        """Apply random transforms to a shape sampled from the bank.
        Input/output: (4, H, W) RGBA tensor.
        """
        # Random hue shift (rotate RGB channels)
        if torch.rand(1).item() < 0.5:
            shift = torch.randint(0, 3, (1,)).item()
            if shift > 0:
                rgba[:3] = rgba[:3].roll(shift, dims=0)

        # Random color remap: contrast + brightness jitter
        if torch.rand(1).item() < 0.6:
            contrast = torch.rand(1, device=self.device).item() * 0.6 + 0.7  # 0.7-1.3
            brightness = (torch.rand(1, device=self.device).item() - 0.5) * 0.3
            rgba[:3] = (rgba[:3] * contrast + brightness).clamp(0, 1)

        # Random invert (15% chance)
        if torch.rand(1).item() < 0.15:
            rgba[:3] = 1.0 - rgba[:3]

        # Spatial: random flip
        if torch.rand(1).item() < 0.5:
            rgba = rgba.flip(-1)  # horizontal
        if torch.rand(1).item() < 0.3:
            rgba = rgba.flip(-2)  # vertical

        # Random 90-degree rotation (25% chance)
        if torch.rand(1).item() < 0.25:
            k = torch.randint(1, 4, (1,)).item()
            rgba = torch.rot90(rgba, k, [-2, -1])

        return rgba

    # ------------------------------------------------------------------
    # Post-processing mutations (anti-memorization)
    # ------------------------------------------------------------------

    def _post_process(self, canvas):
        """Apply random mutations to final canvas. (B, 3, H, W) -> (B, 3, H, W)."""
        B = canvas.shape[0]

        # Per-image random gamma (0.7 - 1.4)
        gamma = torch.rand(B, 1, 1, 1, device=self.device) * 0.7 + 0.7
        canvas = canvas.clamp(1e-6, 1).pow(gamma)

        # Per-image HSV hue jitter
        if torch.rand(1).item() < 0.4:
            shift = torch.randint(0, 3, (1,)).item()
            if shift > 0:
                canvas = canvas.roll(shift, dims=1)  # roll RGB channels

        # Gaussian noise (30% chance, mild)
        if torch.rand(1).item() < 0.3:
            noise_std = torch.rand(1, device=self.device).item() * 0.05
            canvas = canvas + torch.randn_like(canvas) * noise_std

        # Random local contrast (per-image brightness wave)
        if torch.rand(1).item() < 0.25:
            freq = torch.rand(1, device=self.device).item() * 4 + 1
            phase = torch.rand(1, device=self.device).item() * 6.28
            H, W = canvas.shape[2], canvas.shape[3]
            x = torch.linspace(0, 1, W, device=self.device)
            wave = (torch.sin(x * freq * 6.28 + phase) * 0.1).view(1, 1, 1, W)
            canvas = canvas + wave

        # Random rectangular erasure (20% chance)
        if torch.rand(1).item() < 0.2:
            H, W = canvas.shape[2], canvas.shape[3]
            for bi in range(B):
                if torch.rand(1).item() < 0.5:
                    eh = torch.randint(10, H // 4, (1,)).item()
                    ew = torch.randint(10, W // 4, (1,)).item()
                    ey = torch.randint(0, H - eh, (1,)).item()
                    ex = torch.randint(0, W - ew, (1,)).item()
                    # Erase to random color (not just black)
                    c = torch.rand(3, 1, 1, device=self.device)
                    canvas[bi, :, ey:ey+eh, ex:ex+ew] = c

        # Random vignette (15% chance)
        if torch.rand(1).item() < 0.15:
            H, W = canvas.shape[2], canvas.shape[3]
            y = torch.linspace(-1, 1, H, device=self.device)
            x = torch.linspace(-1, 1, W, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            dist = torch.sqrt(xx ** 2 + yy ** 2)
            strength = torch.rand(1, device=self.device).item() * 0.5 + 0.2
            vignette = 1.0 - dist * strength
            canvas = canvas * vignette.view(1, 1, H, W).clamp(0.3, 1.0)

        # Barrel/pincushion distortion (10% chance)
        if torch.rand(1).item() < 0.10:
            H, W = canvas.shape[2], canvas.shape[3]
            y = torch.linspace(-1, 1, H, device=self.device)
            x = torch.linspace(-1, 1, W, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            r2 = xx ** 2 + yy ** 2
            k = (torch.rand(1, device=self.device).item() - 0.5) * 0.3
            xx_d = xx * (1 + k * r2)
            yy_d = yy * (1 + k * r2)
            grid = torch.stack([xx_d, yy_d], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            canvas = F.grid_sample(canvas, grid, mode="bilinear",
                                    padding_mode="reflection", align_corners=True)

        return canvas

    # ------------------------------------------------------------------
    # Bank management
    # ------------------------------------------------------------------

    def refresh_base_layers(self):
        """Regenerate base layers (keeps shape bank)."""
        self.build_base_layers()

    def refresh_all(self):
        """Regenerate everything."""
        self.build_banks()

    def bank_stats(self):
        """Return stats about current banks."""
        stats = {}
        if self.shape_bank is not None:
            stats["shape_bank"] = {
                "count": self.shape_bank.shape[0],
                "res": self.shape_res,
                "mb": self.shape_bank.element_size() * self.shape_bank.nelement() / 1e6,
            }
        if self.base_layers is not None:
            stats["base_layers"] = {
                "count": self.base_layers.shape[0],
                "res": f"{self.H}x{self.W}",
                "mb": self.base_layers.element_size() * self.base_layers.nelement() / 1e6,
            }
        return stats

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def save_shape_bank(self, path):
        """Save shape bank to disk."""
        if self.shape_bank is None:
            return
        torch.save(self.shape_bank.cpu().half(), path)
        n = self.shape_bank.shape[0]
        mb = os.path.getsize(path) / 1e6
        print(f"Saved {n} shapes to {path} ({mb:.1f} MB)", flush=True)

    def load_shape_bank(self, path):
        """Load shape bank from disk, appending to existing bank."""
        loaded = torch.load(path, map_location=self.device, weights_only=True).float()
        if self.shape_bank is None:
            self.shape_bank = loaded
        else:
            self.shape_bank = torch.cat([self.shape_bank, loaded], dim=0)
        self.bank_size = self.shape_bank.shape[0]
        print(f"Loaded {loaded.shape[0]} shapes from {path} "
              f"(total: {self.bank_size})", flush=True)

    def save_base_layers(self, path):
        """Save base layers to disk."""
        if self.base_layers is None:
            return
        torch.save(self.base_layers.cpu().half(), path)
        n = self.base_layers.shape[0]
        mb = os.path.getsize(path) / 1e6
        print(f"Saved {n} layers to {path} ({mb:.1f} MB)", flush=True)

    def load_base_layers(self, path):
        """Load base layers from disk, appending to existing."""
        loaded = torch.load(path, map_location=self.device, weights_only=True).float()
        if self.base_layers is None:
            self.base_layers = loaded
        else:
            self.base_layers = torch.cat([self.base_layers, loaded], dim=0)
        self.n_base_layers = self.base_layers.shape[0]
        print(f"Loaded {loaded.shape[0]} layers from {path} "
              f"(total: {self.n_base_layers})", flush=True)

    def load_bank_dir(self, bank_dir):
        """Load all .pt files from a directory, accumulating into banks."""
        if not os.path.isdir(bank_dir):
            print(f"Bank dir not found: {bank_dir}", flush=True)
            return
        shape_files = sorted(f for f in os.listdir(bank_dir)
                             if f.startswith("shapes_") and f.endswith(".pt"))
        layer_files = sorted(f for f in os.listdir(bank_dir)
                             if f.startswith("layers_") and f.endswith(".pt"))
        for f in shape_files:
            self.load_shape_bank(os.path.join(bank_dir, f))
        for f in layer_files:
            self.load_base_layers(os.path.join(bank_dir, f))
        if not shape_files and not layer_files:
            print(f"No bank files found in {bank_dir}", flush=True)

    # ------------------------------------------------------------------
    # Dynamic bank loading (stream from disk, keep working set in VRAM)
    # ------------------------------------------------------------------

    def setup_dynamic_bank(self, bank_dir, working_size=1000,
                            refresh_interval=50):
        """Configure dynamic bank loading from disk.

        Keeps `working_size` shapes in VRAM. Every `refresh_interval`
        generate() calls, swaps a random portion of the working set
        with shapes from disk.

        Args:
            bank_dir: directory containing shapes_*.pt files
            working_size: shapes to keep in VRAM at once
            refresh_interval: batches between partial swaps
        """
        self._dyn_bank_dir = bank_dir
        self._dyn_working_size = working_size
        self._dyn_refresh_interval = refresh_interval
        self._dyn_call_count = 0

        # Load all shapes to CPU (cheap — system RAM)
        shape_files = sorted(f for f in os.listdir(bank_dir)
                             if f.startswith("shapes_") and f.endswith(".pt"))
        if not shape_files:
            print(f"No shape files in {bank_dir}", flush=True)
            return

        all_shapes = []
        for f in shape_files:
            loaded = torch.load(os.path.join(bank_dir, f),
                                map_location="cpu", weights_only=True).float()
            all_shapes.append(loaded)
            print(f"  Loaded {loaded.shape[0]} shapes from {f} (CPU)",
                  flush=True)
        self._dyn_cpu_bank = torch.cat(all_shapes, dim=0)
        total = self._dyn_cpu_bank.shape[0]
        print(f"Dynamic bank: {total} shapes on CPU, "
              f"{working_size} working set on GPU", flush=True)

        # Initial working set: random sample
        self._dyn_refresh_working_set(full=True)

    def _dyn_refresh_working_set(self, full=False, swap_frac=0.25):
        """Swap part of the GPU working set with random CPU shapes."""
        total = self._dyn_cpu_bank.shape[0]
        ws = min(self._dyn_working_size, total)

        if full or self.shape_bank is None:
            # Full refresh
            idx = torch.randperm(total)[:ws]
            self.shape_bank = self._dyn_cpu_bank[idx].to(self.device)
            self.bank_size = ws
        else:
            # Partial swap
            n_swap = max(1, int(ws * swap_frac))
            # Pick random positions in working set to replace
            replace_idx = torch.randperm(ws)[:n_swap]
            # Pick random shapes from CPU bank
            source_idx = torch.randint(0, total, (n_swap,))
            self.shape_bank[replace_idx] = \
                self._dyn_cpu_bank[source_idx].to(self.device)

    def _maybe_refresh_dynamic(self):
        """Called each generate() — refreshes working set periodically."""
        if not hasattr(self, '_dyn_cpu_bank') or self._dyn_cpu_bank is None:
            return
        self._dyn_call_count += 1
        if self._dyn_call_count % self._dyn_refresh_interval == 0:
            self._dyn_refresh_working_set(swap_frac=0.25)
            print(f"  [bank refresh: swapped 25% of working set]", flush=True)

    def save_to_bank_dir(self, bank_dir):
        """Save current banks to directory with timestamped names."""
        import time
        os.makedirs(bank_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.shape_bank is not None:
            self.save_shape_bank(
                os.path.join(bank_dir, f"shapes_{ts}.pt"))
        if self.base_layers is not None:
            self.save_base_layers(
                os.path.join(bank_dir, f"layers_{ts}.pt"))


def main():
    """Quick visual test and benchmark."""
    import time
    from PIL import Image
    import numpy as np

    gen = SynthyperGenerator(360, 640, device="cuda",
                              bank_size=500, n_base_layers=128)
    gen.build_banks()

    # Warm up
    _ = gen.generate(1)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    N = 20
    for _ in range(N):
        batch = gen.generate(4)
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    ips = N * 4 / elapsed
    print(f"Generated {N * 4} images in {elapsed:.2f}s ({ips:.0f} img/s)")
    print(f"Shape: {batch.shape}, range: [{batch.min():.3f}, {batch.max():.3f}]")
    print(f"VRAM: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

    stats = gen.bank_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save samples
    for i in range(min(4, batch.shape[0])):
        img = (batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(f"synthyper_sample_{i}.png")
    print("Saved samples")


if __name__ == "__main__":
    main()

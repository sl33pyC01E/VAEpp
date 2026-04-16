#!/usr/bin/env python3
"""VAEpp0r procedural image generator.

Three-tier architecture for fast generation:
  Tier 1: Shape Bank — 1000+ pre-rendered RGBA shape textures
  Tier 2: Base Layers — composited scenes from shape bank
  Tier 3: Final Output — fast layer compositing with transforms

Stage 1: 2D static (single frames, RGB only).
"""

import math
import os
import time as _time
import torch
import torch.nn.functional as F

from core.patterns import PatternBank
from core import pattern_collage

from core.generator.shapes import ShapesMixin
from core.generator.templates import TemplateMixin
from core.generator.motion import MotionMixin
from core.generator.recipes import RecipesMixin
from core.generator.io import IOMixin
from core.generator.fluid import FluidMixin
from core.generator.effects import EffectsMixin
from core.generator.text import TextMixin
from core.generator.signage import SignageMixin
from core.generator.particles import ParticlesMixin
from core.generator.raymarch import RaymarchMixin
from core.generator.arcade import ArcadeMixin
from core.generator.extras import ExtrasMixin


class VAEpp0rGenerator(
    ShapesMixin,
    TemplateMixin,
    MotionMixin,
    RecipesMixin,
    IOMixin,
    FluidMixin,
    EffectsMixin,
    TextMixin,
    SignageMixin,
    ParticlesMixin,
    RaymarchMixin,
    ArcadeMixin,
    ExtrasMixin,
):
    """GPU-accelerated procedural image generator.

    Usage:
        gen = VAEpp0rGenerator(360, 640, device="cuda")
        gen.build_banks()  # one-time setup
        batch = gen.generate(4)  # (4, 3, 360, 640) on GPU, [0,1]
    """

    _SHAPE_NAMES = ["circle", "rect", "triangle", "ellipse", "blob",
                     "line", "stroke", "hatch", "stipple", "fractal"]
    _TEX_NAMES = ["flat", "perlin", "gradient", "voronoi"]
    _EDGE_NAMES = ["hard", "soft", "textured"]

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
        # Disco quadrant mode: 8 classes weighted for latency balance rather
        # than uniform distribution. All classes stay at >= 6% so every
        # motion family is seen during training, but cheap classes get
        # more budget so average-throughput stays high.
        #   Q0 pure pattern                    (cheap)   0.15
        #   Q1 pattern+collage+shapes          (cheap)   0.14
        #   Q2 dense random compositing        (medium)  0.10
        #   Q3 structured scenes               (medium)  0.13
        #   Q4 fluid (ripples + raindrops)     (medium)  0.13
        #   Q5 3D raymarch                     (expens)  0.06
        #   Q6 arcade scenes                   (cheap)   0.14
        #   Q7 signage + text                  (medium)  0.15
        self.disco_quadrant = False
        self._disco_weights = torch.tensor(
            [0.15, 0.14, 0.10, 0.13, 0.13, 0.06, 0.14, 0.15], device=device)

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
    # Tier 1: Shape Bank Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _render_one_shape(self):
        """Render a single RGBA shape texture at shape_res x shape_res."""
        S = self.shape_res

        # Sample shape type
        shape_type = torch.multinomial(self.shape_probs, 1).item()
        # Sample size (Pareto distribution via inverse CDF)
        u = torch.rand(1, device=self.device).item()
        r_frac = self.min_r_frac / max(u ** (1.0 / self.alpha), 1e-6)
        r_frac = min(r_frac, self.max_r_frac)
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

        # Micro-stamps (per-image decision)
        use_micro = torch.rand(B, device=self.device) < 0.5
        if use_micro.any():
            n_micro = torch.randint(n_micro_range[0], n_micro_range[1] + 1,
                                     (1,)).item()
            for _ in range(n_micro):
                for bi in range(B):
                    if not use_micro[bi]:
                        continue
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
            # Use local probs to avoid mutating shared state
            q3_probs = self.template_probs.clone()
            q3_probs[0] = 0
            q3_probs = q3_probs / (q3_probs.sum() + 1e-8)
            idx = torch.multinomial(q3_probs, 1).item()
            template = self.template_names[idx]
            bg = self._sample_colors(n3)
            c3 = bg.view(n3, 3, 1, 1).expand(n3, 3, H, W).clone()
            c3 = self._apply_scene_template(c3, template, n3)
            c3 = self._overlay_shapes_on_canvas(c3, n3,
                                                 n_stamps_range=(2, 6),
                                                 n_micro_range=(10, 30))
            c3 = self._post_process(c3)
            canvas[q3_mask] = c3.clamp(0, 1)

        # Q4-Q7: For each new class, build a MINIMAL base so the signature
        # effect dominates. Full-noise base drowns signage/raymarch/arcade
        # out and makes the class read as another "dense random". Instead,
        # each uses a simpler backdrop tuned for its hero effect.
        def _soft_scene(n, darken=0.4):
            """Single base layer + mild post. Keeps some texture but far
            less chaotic than _generate_disco Q2."""
            bg = self._sample_colors(n) * darken
            c = bg.view(n, 3, 1, 1).expand(n, 3, H, W).clone()
            layer_idx = torch.randint(0, self.base_layers.shape[0],
                                       (n,), device=self.device)
            layer = self.base_layers[layer_idx].clone()
            opacity = 0.35
            c = c * (1 - opacity) + layer * opacity
            return c.clamp(0, 1)

        # Q4: fluid ripple — needs texture to warp, keep soft base
        q4_mask = (quadrants == 4)
        n4 = q4_mask.sum().item()
        if n4 > 0:
            c4 = _soft_scene(n4, darken=0.7)
            # Crank up warp so ripples are visible in a single frame
            fp = self._sample_fluid_recipe(
                T=1, n_drops=4,
                warp_strength=16.0,                         # bigger displacement
                amp_range=(0.03, 0.08),                     # bigger impact amps
                gerstner_amp_range=(0.015, 0.04),           # bigger standing waves
            )
            c4 = self._apply_ripples(c4, 0, 1, fp)
            canvas[q4_mask] = c4.clamp(0, 1)

        # Q5: raymarched spheres — dark backdrop so spheres pop
        q5_mask = (quadrants == 5)
        n5 = q5_mask.sum().item()
        if n5 > 0:
            # Gradient sky backdrop: top dark, bottom a subtle color
            y = torch.linspace(0, 1, H, device=self.device).view(1, 1, H, 1)
            bg_top = torch.tensor([0.02, 0.03, 0.08], device=self.device).view(1, 3, 1, 1)
            bg_bot = torch.tensor([0.10, 0.05, 0.15], device=self.device).view(1, 3, 1, 1)
            c5 = (bg_top * (1 - y) + bg_bot * y).expand(n5, 3, H, W).clone()
            rm = self._sample_raymarch_recipe(T=1, n_spheres=3, march_steps=28)
            c5 = self._apply_raymarch(c5, 0, rm)
            canvas[q5_mask] = c5.clamp(0, 1)

        # Q6: arcade — arcade itself dims its own background, so a soft
        # underlying base + arcade overlay looks "game-over-scene" correctly
        q6_mask = (quadrants == 6)
        n6 = q6_mask.sum().item()
        if n6 > 0:
            c6 = _soft_scene(n6, darken=0.25)
            ap = self._sample_arcade_recipe(T=24, mode="auto")
            c6 = self._apply_arcade(c6, 12, ap)
            canvas[q6_mask] = c6.clamp(0, 1)

        # Q7: signage + text — dark base so the readable overlay stands out
        q7_mask = (quadrants == 7)
        n7 = q7_mask.sum().item()
        if n7 > 0:
            c7 = _soft_scene(n7, darken=0.2)
            if torch.rand(1).item() < 0.5:
                sp = self._sample_signage_recipe(T=1, mode="auto", font_size=36)
                c7 = self._apply_signage(c7, 0, sp)
            else:
                tp = self._sample_text_recipe(
                    T=1, mode="typing", language="mixed", font_size=32, cps=12.0)
                c7 = self._apply_text(c7, 0, tp)
            canvas[q7_mask] = c7.clamp(0, 1)

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

        # --- Optional fluid ripple (static: frozen snapshot of ti=0) ---
        if getattr(self, "static_ripple", False):
            fp = self._sample_fluid_recipe(
                T=1,
                n_drops=int(getattr(self, "static_ripple_n_drops", 3)),
                warp_strength=float(getattr(self, "static_ripple_warp_strength", 8.0)),
            )
            canvas = self._apply_ripples(canvas, 0, 1, fp)

        # --- Optional camera shake (frozen single frame) ---
        if getattr(self, "static_shake", False):
            sp = self._sample_shake_recipe(
                T=1,
                amp_xy=float(getattr(self, "static_shake_amp_xy", 0.02)),
                amp_rot=float(getattr(self, "static_shake_amp_rot", 0.02)),
                mode=str(getattr(self, "static_shake_mode", "vibrate")),
            )
            canvas = self._apply_camera_shake(canvas, 0, sp)

        # --- Optional kaleidoscope (frozen single frame) ---
        if getattr(self, "static_kaleido", False):
            kp = self._sample_kaleido_recipe(
                n_slices=int(getattr(self, "static_kaleido_slices", 6)),
                rot_per_frame=0.0,  # static: no rotation
            )
            canvas = self._apply_kaleidoscope(canvas, 0, kp)

        # --- Optional palette shift (frozen single hue rotation) ---
        if getattr(self, "static_palette", False):
            shift = float(getattr(self, "static_palette_shift", 0.25))
            # Build a trivial recipe with speed=0 and phase=shift so ti=0 applies the shift once
            pp = {"enable": True, "speed": 0.0, "phase0": shift, "sat_boost": 1.0}
            canvas = self._apply_palette_cycle(canvas, 0, pp)

        # --- Optional one-off flash (forces a flash at ti=0) ---
        if getattr(self, "static_flash", False):
            fp = self._sample_flash_recipe(T=1, n_flashes=1,
                                            strobe_rate=0.0, strobe_strength=0.0)
            # Force event to ti=0 so the single-frame path always triggers
            if fp.get("flashes"):
                fp["flashes"][0]["t"] = 0
            canvas = self._apply_flash(canvas, 0, fp)

        # --- Optional text overlay (single frozen frame) ---
        if getattr(self, "static_text", False):
            tp = self._sample_text_recipe(
                T=1,
                mode=str(getattr(self, "static_text_mode", "typing")),
                language=str(getattr(self, "static_text_lang", "mixed")),
                font_size=int(getattr(self, "static_text_size", 24)),
                cps=float(getattr(self, "static_text_cps", 12.0)),
            )
            # For static typing: show the fully typed string at ti=0
            canvas = self._apply_text(canvas, 0, tp)

        # --- Optional signage overlay (single frozen frame) ---
        if getattr(self, "static_signage", False):
            sp = self._sample_signage_recipe(
                T=1,
                mode=str(getattr(self, "static_signage_mode", "auto")),
                font_size=int(getattr(self, "static_signage_size", 32)),
            )
            canvas = self._apply_signage(canvas, 0, sp)

        # --- Optional particles (frozen single frame) ---
        if getattr(self, "static_particles", False):
            pp = self._sample_particles_recipe(
                T=16,
                preset=str(getattr(self, "static_particles_preset", "auto")),
                n_particles=int(getattr(self, "static_particles_n", 200)),
            )
            # Sample at mid-life so particles are distributed
            canvas = self._apply_particles(canvas, 8, pp)

        # --- Optional raymarched 3D primitives ---
        if getattr(self, "static_raymarch", False):
            rm = self._sample_raymarch_recipe(
                T=1,
                n_spheres=int(getattr(self, "static_raymarch_spheres", 2)),
                n_boxes=int(getattr(self, "static_raymarch_boxes", 0)),
                n_tori=int(getattr(self, "static_raymarch_tori", 0)),
                march_steps=int(getattr(self, "static_raymarch_steps", 24)),
            )
            canvas = self._apply_raymarch(canvas, 0, rm)

        # --- Optional arcade scene (single frozen frame at ti=0) ---
        if getattr(self, "static_arcade", False):
            ap = self._sample_arcade_recipe(
                T=24, mode=str(getattr(self, "static_arcade_mode", "auto")))
            canvas = self._apply_arcade(canvas, 12, ap)

        # --- Optional extras (fire / vortex / starfield / eq) ---
        if getattr(self, "static_fire", False):
            fp = self._sample_fire_recipe(T=1, intensity=0.8)
            canvas = self._apply_fire(canvas, 0, fp)
        if getattr(self, "static_vortex", False):
            vp = self._sample_vortex_recipe(T=1, strength=0.6)
            canvas = self._apply_vortex(canvas, 0, vp)
        if getattr(self, "static_starfield", False):
            sf = self._sample_starfield_recipe(T=24, n_stars=150)
            canvas = self._apply_starfield(canvas, 12, sf)
        if getattr(self, "static_eq", False):
            ep = self._sample_eq_recipe(T=1, n_bars=24)
            canvas = self._apply_eq_bars(canvas, 0, ep)

        return canvas.clamp(0, 1)

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
        for bi in range(B):
            if torch.rand(1).item() < 0.4:
                shift = torch.randint(1, 3, (1,)).item()
                canvas[bi] = canvas[bi].roll(shift, dims=0)

        # Random local contrast (per-image brightness wave)
        if torch.rand(1).item() < 0.25:
            freq = torch.rand(1, device=self.device).item() * 4 + 1
            phase = torch.rand(1, device=self.device).item() * 2 * math.pi
            H, W = canvas.shape[2], canvas.shape[3]
            x = torch.linspace(0, 1, W, device=self.device)
            wave = (torch.sin(x * freq * 2 * math.pi + phase) * 0.1).view(1, 1, 1, W)
            canvas = canvas + wave

        # Random rectangular erasure (20% chance)
        if torch.rand(1).item() < 0.2:
            H, W = canvas.shape[2], canvas.shape[3]
            for bi in range(B):
                if torch.rand(1).item() < 0.5:
                    eh = torch.randint(10, max(11, H // 4), (1,)).item()
                    ew = torch.randint(10, max(11, W // 4), (1,)).item()
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


def main():
    """Quick visual test and benchmark."""
    import time
    from PIL import Image
    import numpy as np

    gen = VAEpp0rGenerator(360, 640, device="cuda",
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
        Image.fromarray(img).save(f"vaepp_sample_{i}.png")
    print("Saved samples")


if __name__ == "__main__":
    main()

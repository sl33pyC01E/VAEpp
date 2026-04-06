#!/usr/bin/env python3
"""PatternBank — procedural full-canvas pattern generators for SynthyperVAE.

38+ mathematical/structural pattern generators that produce clean, coherent
spatial patterns (gradients, tilings, waves, surfaces, etc.) with NO shape
compositing.  Each method returns (B, 3, H, W) on device in [0, 1].

Used by SynthyperGenerator's disco mode for the "pure pattern" and
"pattern + light noise" quadrants.
"""

import math
import torch
import torch.nn.functional as F


class PatternBank:
    def __init__(self, H, W, device, color_fn):
        """
        Args:
            H, W: output resolution (e.g. 360, 640)
            device: torch device
            color_fn: callable(n) → (n, 3) RGB colors in [0,1]
        """
        self.H, self.W = H, W
        self.device = device
        self.color_fn = color_fn

        # Coordinate grids — cached on device
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        self.sy, self.sx = torch.meshgrid(y, x, indexing="ij")  # [-1, 1]

        uy = torch.linspace(0, 1, H, device=device)
        ux = torch.linspace(0, 1, W, device=device)
        self.uy, self.ux = torch.meshgrid(uy, ux, indexing="ij")  # [0, 1]

        self.r = torch.sqrt(self.sx ** 2 + self.sy ** 2)
        self.theta = torch.atan2(self.sy, self.sx)

        # Full-res frequency grid for Perlin
        fy = torch.fft.fftfreq(H, device=device)
        fx = torch.fft.rfftfreq(W, device=device)
        fy_g, fx_g = torch.meshgrid(fy, fx, indexing="ij")
        self.freq_r = torch.sqrt(fy_g ** 2 + fx_g ** 2)

        # Pattern registry
        self._patterns = [
            # Gradients (5)
            ("linear_gradient", self._pat_linear_gradient),
            ("radial_gradient", self._pat_radial_gradient),
            ("angular_gradient", self._pat_angular_gradient),
            ("diamond_gradient", self._pat_diamond_gradient),
            ("multi_gradient", self._pat_multi_gradient),
            # Tilings (9)
            ("checkerboard", self._pat_checkerboard),
            ("stripes", self._pat_stripes),
            ("hexagonal", self._pat_hexagonal),
            ("brick", self._pat_brick),
            ("herringbone", self._pat_herringbone),
            ("basketweave", self._pat_basketweave),
            ("fish_scale", self._pat_fish_scale),
            ("chevron", self._pat_chevron),
            ("argyle", self._pat_argyle),
            # Waves (5)
            ("sine_wave", self._pat_sine_wave),
            ("interference", self._pat_interference),
            ("concentric", self._pat_concentric),
            ("spiral", self._pat_spiral),
            ("ripple", self._pat_ripple),
            # Math surfaces (5)
            ("quadratic", self._pat_quadratic),
            ("lissajous", self._pat_lissajous),
            ("rose_curve", self._pat_rose_curve),
            ("spirograph", self._pat_spirograph),
            ("julia", self._pat_julia),
            # Symmetry / Op art (3)
            ("kaleidoscope", self._pat_kaleidoscope),
            ("op_art_grid", self._pat_op_art_grid),
            ("islamic_star", self._pat_islamic_star),
            # Procedural natural (5)
            ("reaction_diffusion", self._pat_reaction_diffusion),
            ("contour_map", self._pat_contour_map),
            ("wood_grain", self._pat_wood_grain),
            ("marble", self._pat_marble),
            ("cracked_earth", self._pat_cracked_earth),
            # Art exercises (4)
            ("zentangle", self._pat_zentangle),
            ("maze", self._pat_maze),
            ("contour_lines", self._pat_contour_lines),
            ("squiggle_fill", self._pat_squiggle_fill),
            # Fine-grain (3)
            ("halftone", self._pat_halftone),
            ("ordered_dither", self._pat_ordered_dither),
            ("stipple", self._pat_stipple),
        ]
        n = len(self._patterns)
        self.pattern_weights = torch.ones(n, device=device) / n
        self.pattern_names = [name for name, _ in self._patterns]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, B):
        """Generate B random patterns (one pattern type per image)."""
        indices = torch.multinomial(self.pattern_weights, B, replacement=True)
        canvas = torch.zeros(B, 3, self.H, self.W, device=self.device)
        for idx in indices.unique():
            mask = indices == idx
            n = mask.sum().item()
            _, fn = self._patterns[idx.item()]
            canvas[mask] = fn(n)
        return canvas.clamp(0, 1)

    def generate_specific(self, B, name):
        """Generate B images of a specific pattern by name."""
        for pname, fn in self._patterns:
            if pname == name:
                return fn(B).clamp(0, 1)
        raise ValueError(f"Unknown pattern: {name}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _colors(self, n):
        return self.color_fn(n)

    def _perlin(self, B, beta=1.5):
        """Generate full-res Perlin noise (B, H, W) in [0, 1]."""
        H, W = self.H, self.W
        phase = torch.rand(B, H, W // 2 + 1, device=self.device) * 2 * math.pi
        amp = 1.0 / (self.freq_r.unsqueeze(0) + 1e-6) ** beta
        amp[:, 0, 0] = 0
        spectrum = amp * torch.exp(1j * phase)
        noise = torch.fft.irfft2(spectrum, s=(H, W))
        # Normalize per-image
        n_flat = noise.reshape(B, -1)
        lo = n_flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
        hi = n_flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
        return ((noise - lo) / (hi - lo + 1e-8))

    def _lerp_colors(self, t, c1, c2):
        """t: (B, H, W), c1/c2: (B, 3) → (B, 3, H, W)."""
        t = t.unsqueeze(1)  # (B, 1, H, W)
        c1 = c1.view(-1, 3, 1, 1)
        c2 = c2.view(-1, 3, 1, 1)
        return c1 * (1 - t) + c2 * t

    def _rotated_coords(self, B, angle=None):
        """Rotate sx/sy by per-image angle. Returns (B, H, W) rx, ry."""
        if angle is None:
            angle = torch.rand(B, device=self.device) * 2 * math.pi
        cos_a = angle.cos().view(B, 1, 1)
        sin_a = angle.sin().view(B, 1, 1)
        rx = self.sx.unsqueeze(0) * cos_a - self.sy.unsqueeze(0) * sin_a
        ry = self.sx.unsqueeze(0) * sin_a + self.sy.unsqueeze(0) * cos_a
        return rx, ry

    def _line_mask(self, coords, thickness=0.02):
        """Render thin lines from coordinate field. coords: (B, H, W) → (B, H, W)."""
        return (-coords.abs() / thickness).exp()

    # ==================================================================
    # GRADIENTS (5)
    # ==================================================================

    def _pat_linear_gradient(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        angle = torch.rand(B, device=self.device) * 2 * math.pi
        cos_a = angle.cos().view(B, 1, 1)
        sin_a = angle.sin().view(B, 1, 1)
        t = self.sx.unsqueeze(0) * cos_a + self.sy.unsqueeze(0) * sin_a
        t = (t - t.reshape(B, -1).min(1).values.view(B, 1, 1)) / \
            (t.reshape(B, -1).max(1).values.view(B, 1, 1) -
             t.reshape(B, -1).min(1).values.view(B, 1, 1) + 1e-8)
        return self._lerp_colors(t, c1, c2)

    def _pat_radial_gradient(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        cx = (torch.rand(B, device=self.device) - 0.5).view(B, 1, 1)
        cy = (torch.rand(B, device=self.device) - 0.5).view(B, 1, 1)
        r = torch.sqrt((self.sx.unsqueeze(0) - cx) ** 2 +
                        (self.sy.unsqueeze(0) - cy) ** 2)
        t = (r / (r.reshape(B, -1).max(1).values.view(B, 1, 1) + 1e-8)).clamp(0, 1)
        return self._lerp_colors(t, c1, c2)

    def _pat_angular_gradient(self, B):
        c1, c2, c3 = self._colors(B), self._colors(B), self._colors(B)
        cx = (torch.rand(B, device=self.device) * 0.4 - 0.2).view(B, 1, 1)
        cy = (torch.rand(B, device=self.device) * 0.4 - 0.2).view(B, 1, 1)
        angle = torch.atan2(self.sy.unsqueeze(0) - cy, self.sx.unsqueeze(0) - cx)
        t = (angle / (2 * math.pi) + 0.5)  # [0, 1]
        # Three-way lerp: c1→c2→c3→c1
        t3 = t * 3
        seg = t3.floor().long() % 3
        f = t3 - t3.floor()
        colors = torch.stack([c1, c2, c3, c1], dim=1)  # (B, 4, 3)
        out = torch.zeros(B, 3, self.H, self.W, device=self.device)
        for s in range(3):
            mask = (seg == s).unsqueeze(1).float()
            ca = colors[:, s].view(B, 3, 1, 1)
            cb = colors[:, s + 1].view(B, 3, 1, 1)
            out += mask * (ca * (1 - f.unsqueeze(1)) + cb * f.unsqueeze(1))
        return out

    def _pat_diamond_gradient(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        t = (self.sx.unsqueeze(0).abs() + self.sy.unsqueeze(0).abs())
        t = (t / (t.reshape(B, -1).max(1).values.view(B, 1, 1) + 1e-8)).clamp(0, 1)
        return self._lerp_colors(t, c1, c2)

    def _pat_multi_gradient(self, B):
        n_stops = torch.randint(3, 6, (1,)).item()
        colors = [self._colors(B) for _ in range(n_stops)]
        angle = torch.rand(B, device=self.device) * 2 * math.pi
        cos_a = angle.cos().view(B, 1, 1)
        sin_a = angle.sin().view(B, 1, 1)
        t = self.sx.unsqueeze(0) * cos_a + self.sy.unsqueeze(0) * sin_a
        t = (t - t.reshape(B, -1).min(1).values.view(B, 1, 1)) / \
            (t.reshape(B, -1).max(1).values.view(B, 1, 1) -
             t.reshape(B, -1).min(1).values.view(B, 1, 1) + 1e-8)
        # Multi-stop lerp
        t_scaled = t * (n_stops - 1)
        seg = t_scaled.floor().long().clamp(0, n_stops - 2)
        f = (t_scaled - seg.float()).unsqueeze(1)
        out = torch.zeros(B, 3, self.H, self.W, device=self.device)
        for s in range(n_stops - 1):
            mask = (seg == s).unsqueeze(1).float()
            ca = colors[s].view(B, 3, 1, 1)
            cb = colors[s + 1].view(B, 3, 1, 1)
            out += mask * (ca * (1 - f) + cb * f)
        return out

    # ==================================================================
    # TILINGS (9)
    # ==================================================================

    def _pat_checkerboard(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        freq = (torch.randint(2, 16, (B,), device=self.device).float()
                .view(B, 1, 1))
        rx, ry = self._rotated_coords(B)
        check = ((rx * freq).floor() + (ry * freq).floor()) % 2  # (B, H, W)
        return self._lerp_colors(check, c1, c2)

    def _pat_stripes(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        freq = (torch.randint(2, 20, (B,), device=self.device).float()
                .view(B, 1, 1))
        rx, ry = self._rotated_coords(B)
        stripe = ((rx * freq).floor() % 2).float()
        return self._lerp_colors(stripe, c1, c2)

    def _pat_hexagonal(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        scale = (torch.randint(3, 12, (B,), device=self.device).float()
                 .view(B, 1, 1))
        sx = self.ux.unsqueeze(0) * scale
        sy = self.uy.unsqueeze(0) * scale * 1.1547  # 2/sqrt(3)
        # Offset every other row
        row = sy.floor().long()
        sx_off = sx + (row % 2).float() * 0.5
        # Distance to nearest hex center
        cx = (sx_off + 0.5).floor()
        cy = (sy + 0.5).floor()
        dx = sx_off - cx
        dy = sy - cy
        d = (dx.abs() + dy.abs() * 0.577).clamp(0, 0.5)  # hex distance approx
        t = (d < 0.35).float()
        return self._lerp_colors(t, c1, c2)

    def _pat_brick(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        rows = torch.randint(5, 15, (B,), device=self.device).float().view(B, 1, 1)
        cols = rows * 2
        sy = self.uy.unsqueeze(0) * rows
        row_idx = sy.floor().long()
        sx = self.ux.unsqueeze(0) * cols + (row_idx % 2).float() * 0.5
        # Mortar lines
        gy = (sy % 1.0 - 0.5).abs()
        gx = (sx % 1.0 - 0.5).abs()
        mortar = ((gy > 0.42) | (gx > 0.42)).float()
        return self._lerp_colors(mortar, c1, c2)

    def _pat_herringbone(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        scale = torch.randint(4, 12, (B,), device=self.device).float().view(B, 1, 1)
        rx, ry = self._rotated_coords(B)
        sx = rx * scale
        sy = ry * scale
        block = ((sx + sy).floor() + (sx - sy).floor()) % 2
        return self._lerp_colors(block.float(), c1, c2)

    def _pat_basketweave(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        scale = torch.randint(3, 10, (B,), device=self.device).float().view(B, 1, 1)
        sx = self.ux.unsqueeze(0) * scale * 2
        sy = self.uy.unsqueeze(0) * scale * 2
        bx = sx.floor().long() % 2
        by = sy.floor().long() % 2
        cell = (bx + by) % 2
        # Within cell: alternating H/V stripes
        inner_x = (sx * 3).floor().long() % 2
        inner_y = (sy * 3).floor().long() % 2
        pat = torch.where(cell == 0, inner_x, inner_y)
        return self._lerp_colors(pat.float(), c1, c2)

    def _pat_fish_scale(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        scale = torch.randint(4, 12, (B,), device=self.device).float().view(B, 1, 1)
        sx = self.ux.unsqueeze(0) * scale
        sy = self.uy.unsqueeze(0) * scale
        row = sy.floor().long()
        sx_off = sx + (row % 2).float() * 0.5
        # Distance from cell bottom-center
        cx = (sx_off + 0.5).floor()
        dy = sy % 1.0
        dx = sx_off - cx
        d = torch.sqrt(dx ** 2 + (dy - 1.0) ** 2)
        t = (d < 0.55).float()
        return self._lerp_colors(t, c1, c2)

    def _pat_chevron(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        freq = torch.randint(3, 12, (B,), device=self.device).float().view(B, 1, 1)
        sy = self.uy.unsqueeze(0) * freq
        sx = self.ux.unsqueeze(0) * freq
        v = (sy + sx.abs() * 2).floor() % 2
        return self._lerp_colors(v.float(), c1, c2)

    def _pat_argyle(self, B):
        c1, c2, c3 = self._colors(B), self._colors(B), self._colors(B)
        scale = torch.randint(3, 8, (B,), device=self.device).float().view(B, 1, 1)
        rx, ry = self._rotated_coords(B, angle=torch.full((B,), math.pi / 4,
                                                           device=self.device))
        sx = rx * scale
        sy = ry * scale
        diamond = ((sx.floor() + sy.floor()) % 2).float()
        base = self._lerp_colors(diamond, c1, c2)
        # Thin crossing lines
        line_x = ((sx % 1.0 - 0.5).abs() < 0.03).float()
        line_y = ((sy % 1.0 - 0.5).abs() < 0.03).float()
        line = (line_x + line_y).clamp(0, 1).unsqueeze(1)
        c3v = c3.view(B, 3, 1, 1)
        return base * (1 - line) + c3v * line

    # ==================================================================
    # WAVES & CIRCLES (5)
    # ==================================================================

    def _pat_sine_wave(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        freq = (torch.rand(B, device=self.device) * 8 + 2).view(B, 1, 1)
        phase = (torch.rand(B, device=self.device) * 2 * math.pi).view(B, 1, 1)
        rx, ry = self._rotated_coords(B)
        wave = (torch.sin(rx * freq * math.pi + phase) * 0.5 + 0.5)
        return self._lerp_colors(wave, c1, c2)

    def _pat_interference(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        f1 = (torch.rand(B, device=self.device) * 6 + 2).view(B, 1, 1)
        f2 = (torch.rand(B, device=self.device) * 6 + 2).view(B, 1, 1)
        a1 = (torch.rand(B, device=self.device) * 2 * math.pi).view(B, 1, 1)
        a2 = a1 + (torch.rand(B, device=self.device) * 0.5 + 0.3).view(B, 1, 1)
        sx = self.sx.unsqueeze(0)
        sy = self.sy.unsqueeze(0)
        w1 = torch.sin((sx * a1.cos() + sy * a1.sin()) * f1 * math.pi)
        w2 = torch.sin((sx * a2.cos() + sy * a2.sin()) * f2 * math.pi)
        t = ((w1 + w2) * 0.25 + 0.5).clamp(0, 1)
        return self._lerp_colors(t, c1, c2)

    def _pat_concentric(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        cx = (torch.rand(B, device=self.device) * 1.2 - 0.6).view(B, 1, 1)
        cy = (torch.rand(B, device=self.device) * 1.2 - 0.6).view(B, 1, 1)
        freq = (torch.rand(B, device=self.device) * 8 + 3).view(B, 1, 1)
        r = torch.sqrt((self.sx.unsqueeze(0) - cx) ** 2 +
                        (self.sy.unsqueeze(0) - cy) ** 2)
        ring = ((r * freq).floor() % 2).float()
        return self._lerp_colors(ring, c1, c2)

    def _pat_spiral(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        arms = torch.randint(1, 6, (B,), device=self.device).float().view(B, 1, 1)
        tightness = (torch.rand(B, device=self.device) * 3 + 1).view(B, 1, 1)
        r = self.r.unsqueeze(0)
        theta = self.theta.unsqueeze(0)
        spiral = ((theta * arms / (2 * math.pi) + r * tightness) % 1.0)
        t = (spiral > 0.5).float()
        return self._lerp_colors(t, c1, c2)

    def _pat_ripple(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        cx = (torch.rand(B, device=self.device) * 0.8 - 0.4).view(B, 1, 1)
        cy = (torch.rand(B, device=self.device) * 0.8 - 0.4).view(B, 1, 1)
        freq = (torch.rand(B, device=self.device) * 10 + 4).view(B, 1, 1)
        r = torch.sqrt((self.sx.unsqueeze(0) - cx) ** 2 +
                        (self.sy.unsqueeze(0) - cy) ** 2)
        decay = (-r * 2).exp()
        wave = (torch.sin(r * freq * math.pi) * decay * 0.5 + 0.5)
        return self._lerp_colors(wave, c1, c2)

    # ==================================================================
    # MATHEMATICAL SURFACES (5)
    # ==================================================================

    def _pat_quadratic(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        # Random surface type per image
        kind = torch.randint(0, 3, (B,), device=self.device)
        sx = self.sx.unsqueeze(0)
        sy = self.sy.unsqueeze(0)
        # Bowl, saddle, ridge
        bowl = sx ** 2 + sy ** 2
        saddle = sx ** 2 - sy ** 2
        ridge = sx ** 2
        surface = torch.where(kind.view(B, 1, 1) == 0, bowl,
                  torch.where(kind.view(B, 1, 1) == 1, saddle, ridge))
        # Contour lines
        freq = (torch.rand(B, device=self.device) * 6 + 3).view(B, 1, 1)
        contour = ((surface * freq).floor() % 2).float()
        return self._lerp_colors(contour, c1, c2)

    def _pat_lissajous(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        bg = c1.view(B, 3, 1, 1).expand(B, 3, self.H, self.W).clone()
        a = torch.randint(1, 6, (B,), device=self.device).float()
        b = torch.randint(1, 6, (B,), device=self.device).float()
        delta = torch.rand(B, device=self.device) * math.pi
        N = 2000
        t = torch.linspace(0, 2 * math.pi, N, device=self.device)
        for i in range(B):
            px = torch.sin(a[i] * t + delta[i])
            py = torch.sin(b[i] * t)
            # Map to pixel coords
            ix = ((px * 0.9 + 1) / 2 * (self.W - 1)).long().clamp(0, self.W - 1)
            iy = ((py * 0.9 + 1) / 2 * (self.H - 1)).long().clamp(0, self.H - 1)
            bg[i, :, iy, ix] = c2[i].view(3, 1)
        # Thicken with small blur
        bg = F.avg_pool2d(F.pad(bg, (1, 1, 1, 1), mode='reflect'), 3, 1)
        return bg

    def _pat_rose_curve(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        bg = c1.view(B, 3, 1, 1).expand(B, 3, self.H, self.W).clone()
        k = torch.randint(2, 8, (B,), device=self.device).float()
        N = 3000
        t = torch.linspace(0, 2 * math.pi, N, device=self.device)
        for i in range(B):
            r = (torch.cos(k[i] * t) * 0.85)
            px = r * torch.cos(t)
            py = r * torch.sin(t)
            ix = ((px + 1) / 2 * (self.W - 1)).long().clamp(0, self.W - 1)
            iy = ((py + 1) / 2 * (self.H - 1)).long().clamp(0, self.H - 1)
            bg[i, :, iy, ix] = c2[i].view(3, 1)
        bg = F.avg_pool2d(F.pad(bg, (1, 1, 1, 1), mode='reflect'), 3, 1)
        return bg

    def _pat_spirograph(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        bg = c1.view(B, 3, 1, 1).expand(B, 3, self.H, self.W).clone()
        R = 1.0
        r_inner = (torch.rand(B, device=self.device) * 0.6 + 0.2)
        d = (torch.rand(B, device=self.device) * 0.5 + 0.3)
        N = 5000
        t = torch.linspace(0, 20 * math.pi, N, device=self.device)
        for i in range(B):
            ri, di = r_inner[i], d[i]
            px = (R - ri) * torch.cos(t) + di * torch.cos((R - ri) / ri * t)
            py = (R - ri) * torch.sin(t) - di * torch.sin((R - ri) / ri * t)
            scale = max(px.abs().max(), py.abs().max()) + 1e-6
            px, py = px / scale * 0.85, py / scale * 0.85
            ix = ((px + 1) / 2 * (self.W - 1)).long().clamp(0, self.W - 1)
            iy = ((py + 1) / 2 * (self.H - 1)).long().clamp(0, self.H - 1)
            bg[i, :, iy, ix] = c2[i].view(3, 1)
        bg = F.avg_pool2d(F.pad(bg, (2, 2, 2, 2), mode='reflect'), 5, 1)
        return bg

    def _pat_julia(self, B):
        c1, c2, c3 = self._colors(B), self._colors(B), self._colors(B)
        # Render at 1/4 res then upsample
        h4, w4 = self.H // 4, self.W // 4
        y = torch.linspace(-1.5, 1.5, h4, device=self.device)
        x = torch.linspace(-2.0, 2.0, w4, device=self.device)
        yg, xg = torch.meshgrid(y, x, indexing="ij")
        out = torch.zeros(B, 3, h4, w4, device=self.device)
        for i in range(B):
            cr = torch.rand(1, device=self.device).item() * 1.5 - 0.75
            ci = torch.rand(1, device=self.device).item() * 1.5 - 0.75
            zr, zi = xg.clone(), yg.clone()
            mask = torch.ones(h4, w4, device=self.device, dtype=torch.bool)
            escape = torch.zeros(h4, w4, device=self.device)
            for it in range(64):
                zr2 = zr * zr - zi * zi + cr
                zi2 = 2 * zr * zi + ci
                zr, zi = zr2, zi2
                diverged = (zr * zr + zi * zi > 4) & mask
                escape[diverged] = it
                mask &= ~diverged
            t = (escape / 64).unsqueeze(0)  # (1, h4, w4)
            frame = (c1[i].view(3, 1, 1) * (1 - t) + c2[i].view(3, 1, 1) * t)
            frame[:, mask] = c3[i].view(3, 1)  # interior
            out[i] = frame
        return F.interpolate(out, size=(self.H, self.W), mode='bilinear',
                             align_corners=False)

    # ==================================================================
    # SYMMETRY & OP ART (3)
    # ==================================================================

    def _pat_kaleidoscope(self, B):
        # Generate noise, then fold into N-fold symmetry
        n_folds = torch.randint(3, 9, (B,), device=self.device)
        base_noise = self._perlin(B, beta=1.2)  # (B, H, W)
        c1, c2 = self._colors(B), self._colors(B)
        theta = self.theta.unsqueeze(0)  # (1, H, W)
        r = self.r.unsqueeze(0)
        out = torch.zeros(B, 3, self.H, self.W, device=self.device)
        for i in range(B):
            nf = n_folds[i].item()
            sector = 2 * math.pi / nf
            folded_theta = (theta[0] % sector)
            folded_theta = torch.where(folded_theta > sector / 2,
                                        sector - folded_theta, folded_theta)
            # Map folded polar back to cartesian for sampling
            fx = (r[0] * folded_theta.cos()).clamp(-1, 1)
            fy = (r[0] * folded_theta.sin()).clamp(-1, 1)
            # Sample from perlin using folded coords
            grid = torch.stack([fx, fy], dim=-1).unsqueeze(0)
            noise_2d = base_noise[i:i+1].unsqueeze(1)  # (1, 1, H, W)
            sampled = F.grid_sample(noise_2d, grid, align_corners=True,
                                     mode='bilinear', padding_mode='reflection')
            t = sampled.squeeze(1).squeeze(0)
            out[i] = c1[i].view(3, 1, 1) * (1 - t) + c2[i].view(3, 1, 1) * t
        return out

    def _pat_op_art_grid(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        freq = (torch.rand(B, device=self.device) * 8 + 4).view(B, 1, 1)
        # Warped grid — bulge in center
        sx = self.sx.unsqueeze(0)
        sy = self.sy.unsqueeze(0)
        r = torch.sqrt(sx ** 2 + sy ** 2).clamp(1e-6)
        warp = (torch.rand(B, device=self.device) * 0.4 + 0.1).view(B, 1, 1)
        factor = 1 + warp * (1 - r)
        wx = sx * factor
        wy = sy * factor
        grid = ((wx * freq).floor() + (wy * freq).floor()) % 2
        return self._lerp_colors(grid.float(), c1, c2)

    def _pat_islamic_star(self, B):
        c1, c2, c3 = self._colors(B), self._colors(B), self._colors(B)
        scale = torch.randint(2, 6, (B,), device=self.device).float().view(B, 1, 1)
        sx = self.ux.unsqueeze(0) * scale
        sy = self.uy.unsqueeze(0) * scale
        # Star pattern from overlaid rotated grids
        star = torch.zeros(B, self.H, self.W, device=self.device)
        for angle_deg in [0, 45, 90, 135]:
            a = math.radians(angle_deg)
            rx = sx * math.cos(a) - sy * math.sin(a)
            ry = sx * math.sin(a) + sy * math.cos(a)
            lines = ((rx % 1.0 - 0.5).abs() < 0.06).float()
            star = star + lines
        star = star.clamp(0, 1)
        bg = self._lerp_colors(
            ((sx.floor() + sy.floor()) % 2).float(), c1, c2)
        c3v = c3.view(B, 3, 1, 1)
        return bg * (1 - star.unsqueeze(1)) + c3v * star.unsqueeze(1)

    # ==================================================================
    # PROCEDURAL NATURAL (5)
    # ==================================================================

    def _pat_reaction_diffusion(self, B):
        """Gray-Scott reaction-diffusion at 1/4 res."""
        c1, c2 = self._colors(B), self._colors(B)
        h4, w4 = self.H // 4, self.W // 4
        u = torch.ones(B, h4, w4, device=self.device)
        v = torch.zeros(B, h4, w4, device=self.device)
        # Seed: random squares
        for i in range(B):
            n_seeds = torch.randint(3, 8, (1,)).item()
            for _ in range(n_seeds):
                sy = torch.randint(0, h4 - 5, (1,)).item()
                sx = torch.randint(0, w4 - 5, (1,)).item()
                v[i, sy:sy+5, sx:sx+5] = 1.0
                u[i, sy:sy+5, sx:sx+5] = 0.5
        # Laplacian kernel
        lap = torch.tensor([[0.05, 0.2, 0.05],
                            [0.2, -1.0, 0.2],
                            [0.05, 0.2, 0.05]], device=self.device)
        lap = lap.view(1, 1, 3, 3)
        f = torch.rand(B, 1, 1, device=self.device) * 0.02 + 0.03
        k = torch.rand(B, 1, 1, device=self.device) * 0.02 + 0.05
        Du, Dv = 0.21, 0.105
        for _ in range(300):
            u_pad = F.pad(u.unsqueeze(1), (1, 1, 1, 1), mode='circular')
            v_pad = F.pad(v.unsqueeze(1), (1, 1, 1, 1), mode='circular')
            lu = F.conv2d(u_pad, lap).squeeze(1)
            lv = F.conv2d(v_pad, lap).squeeze(1)
            uvv = u * v * v
            u = u + Du * lu - uvv + f * (1 - u)
            v = v + Dv * lv + uvv - (f + k) * v
            u = u.clamp(0, 1)
            v = v.clamp(0, 1)
        t = F.interpolate(v.unsqueeze(1), size=(self.H, self.W),
                          mode='bilinear', align_corners=False).squeeze(1)
        return self._lerp_colors(t, c1, c2)

    def _pat_contour_map(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        noise = self._perlin(B, beta=1.8)
        n_levels = torch.randint(6, 20, (B,), device=self.device).float().view(B, 1, 1)
        quantized = (noise * n_levels).floor() / n_levels
        # Edge detection for contour lines
        qp = F.pad(quantized.unsqueeze(1), (1, 1, 1, 1), mode='reflect')
        dx = (qp[:, :, :, 2:] - qp[:, :, :, :-2]).abs()
        dy = (qp[:, :, 2:, :] - qp[:, :, :-2, :]).abs()
        edge = (dx[:, 0, 1:-1, :] + dy[:, 0, :, 1:-1]).clamp(0, 1)
        edge = (edge > 0.001).float()
        fill = self._lerp_colors(quantized, c1, c2)
        # Darken contour lines
        return fill * (1 - edge.unsqueeze(1) * 0.8)

    def _pat_wood_grain(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        noise = self._perlin(B, beta=1.0)
        freq = (torch.rand(B, device=self.device) * 15 + 8).view(B, 1, 1)
        # Distorted concentric pattern
        r = self.r.unsqueeze(0) + noise * 0.3
        grain = (torch.sin(r * freq * math.pi) * 0.5 + 0.5)
        return self._lerp_colors(grain, c1, c2)

    def _pat_marble(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        noise = self._perlin(B, beta=1.5)
        freq = (torch.rand(B, device=self.device) * 6 + 3).view(B, 1, 1)
        rx, ry = self._rotated_coords(B)
        marble = (torch.sin((rx + noise * 2) * freq * math.pi) * 0.5 + 0.5)
        return self._lerp_colors(marble, c1, c2)

    def _pat_cracked_earth(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        n_cells = torch.randint(8, 25, (B,), device=self.device)
        out = torch.zeros(B, self.H, self.W, device=self.device)
        for i in range(B):
            nc = n_cells[i].item()
            centers_x = torch.rand(nc, device=self.device) * 2 - 1
            centers_y = torch.rand(nc, device=self.device) * 2 - 1
            # Distance to nearest center
            dx = self.sx.unsqueeze(0) - centers_x.view(nc, 1, 1)
            dy = self.sy.unsqueeze(0) - centers_y.view(nc, 1, 1)
            dists = torch.sqrt(dx ** 2 + dy ** 2)
            sorted_d, _ = dists.sort(dim=0)
            # Crack = where two nearest cells are close in distance
            crack = (sorted_d[1] - sorted_d[0]) < 0.04
            out[i] = crack.float()
        return self._lerp_colors(out, c1, c2)

    # ==================================================================
    # ART EXERCISES (4)
    # ==================================================================

    def _pat_zentangle(self, B):
        """Canvas divided into Voronoi regions, each filled with a different pattern."""
        n_regions = torch.randint(4, 9, (B,), device=self.device)
        out = torch.zeros(B, 3, self.H, self.W, device=self.device)
        # Simple patterns for fills
        fill_fns = [
            lambda b: self._pat_stripes(b),
            lambda b: self._pat_checkerboard(b),
            lambda b: self._pat_concentric(b),
            lambda b: self._pat_sine_wave(b),
            lambda b: self._pat_linear_gradient(b),
        ]
        for i in range(B):
            nr = n_regions[i].item()
            cx = torch.rand(nr, device=self.device) * 2 - 1
            cy = torch.rand(nr, device=self.device) * 2 - 1
            dx = self.sx.unsqueeze(0) - cx.view(nr, 1, 1)
            dy = self.sy.unsqueeze(0) - cy.view(nr, 1, 1)
            dists = torch.sqrt(dx ** 2 + dy ** 2)
            region_map = dists.argmin(dim=0)  # (H, W)
            for r in range(nr):
                mask = (region_map == r).unsqueeze(0).unsqueeze(0).float()
                fn = fill_fns[r % len(fill_fns)]
                fill = fn(1)  # (1, 3, H, W)
                out[i:i+1] += fill * mask
        return out

    def _pat_maze(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        cell_size = torch.randint(8, 24, (B,), device=self.device)
        out = torch.zeros(B, self.H, self.W, device=self.device)
        for i in range(B):
            cs = cell_size[i].item()
            rows = self.H // cs
            cols = self.W // cs
            # Simple binary maze via random DFS
            maze = torch.ones(rows * 2 + 1, cols * 2 + 1, device=self.device)
            visited = torch.zeros(rows, cols, dtype=torch.bool, device=self.device)
            stack = [(0, 0)]
            visited[0, 0] = True
            maze[1, 1] = 0
            while stack:
                cr, cc = stack[-1]
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                        neighbors.append((nr, nc, dr, dc))
                if not neighbors:
                    stack.pop()
                    continue
                idx = torch.randint(0, len(neighbors), (1,)).item()
                nr, nc, dr, dc = neighbors[idx]
                maze[1 + cr * 2 + dr, 1 + cc * 2 + dc] = 0
                maze[1 + nr * 2, 1 + nc * 2] = 0
                visited[nr, nc] = True
                stack.append((nr, nc))
            # Upsample maze to full res
            maze_img = F.interpolate(maze.unsqueeze(0).unsqueeze(0),
                                     size=(self.H, self.W),
                                     mode='nearest').squeeze()
            out[i] = maze_img
        return self._lerp_colors(out, c1, c2)

    def _pat_contour_lines(self, B):
        """Parallel offset curves from a seed shape (Perlin blob)."""
        c1, c2 = self._colors(B), self._colors(B)
        noise = self._perlin(B, beta=2.0)
        n_lines = (torch.rand(B, device=self.device) * 15 + 5).view(B, 1, 1)
        # Create contour levels
        levels = (noise * n_lines).floor() / n_lines
        # Detect edges between levels
        lp = F.pad(levels.unsqueeze(1), (1, 1, 1, 1), mode='reflect')
        dx = (lp[:, :, :, 2:] - lp[:, :, :, :-2]).abs()
        dy = (lp[:, :, 2:, :] - lp[:, :, :-2, :]).abs()
        edge = (dx[:, 0, 1:-1, :] + dy[:, 0, :, 1:-1])
        lines = (edge > 0.001).float()
        bg = c1.view(B, 3, 1, 1).expand(B, 3, self.H, self.W)
        fg = c2.view(B, 3, 1, 1).expand(B, 3, self.H, self.W)
        return bg * (1 - lines.unsqueeze(1)) + fg * lines.unsqueeze(1)

    def _pat_squiggle_fill(self, B):
        """Random walk curves that expand outward."""
        c1, c2 = self._colors(B), self._colors(B)
        canvas = c1.view(B, 3, 1, 1).expand(B, 3, self.H, self.W).clone()
        n_walks = torch.randint(5, 15, (B,), device=self.device)
        for i in range(B):
            for _ in range(n_walks[i].item()):
                # Random walk
                steps = torch.randint(100, 400, (1,)).item()
                x = torch.zeros(steps, device=self.device)
                y = torch.zeros(steps, device=self.device)
                x[0] = torch.rand(1, device=self.device).item() * 2 - 1
                y[0] = torch.rand(1, device=self.device).item() * 2 - 1
                angle = torch.rand(1, device=self.device).item() * 2 * math.pi
                for s in range(1, steps):
                    angle += (torch.rand(1, device=self.device).item() - 0.5) * 0.8
                    x[s] = (x[s-1] + math.cos(angle) * 0.015).clamp(-1, 1)
                    y[s] = (y[s-1] + math.sin(angle) * 0.015).clamp(-1, 1)
                # Render path
                ix = ((x + 1) / 2 * (self.W - 1)).long().clamp(0, self.W - 1)
                iy = ((y + 1) / 2 * (self.H - 1)).long().clamp(0, self.H - 1)
                canvas[i, :, iy, ix] = c2[i].view(3, 1)
        # Thicken
        canvas = F.avg_pool2d(F.pad(canvas, (1, 1, 1, 1), mode='reflect'), 3, 1)
        return canvas

    # ==================================================================
    # FINE-GRAIN / NOISE (3)
    # ==================================================================

    def _pat_halftone(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        # Dot grid at angle
        freq = (torch.rand(B, device=self.device) * 15 + 5).view(B, 1, 1)
        rx, ry = self._rotated_coords(B)
        # Brightness field from perlin
        brightness = self._perlin(B, beta=1.5)
        # Dot centers
        cx = (rx * freq + 0.5).floor() / freq
        cy = (ry * freq + 0.5).floor() / freq
        dx = rx - cx
        dy = ry - cy
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        # Dot radius proportional to brightness
        max_r = 0.4 / freq
        dot_r = brightness * max_r
        dots = (dist < dot_r).float()
        return self._lerp_colors(dots, c1, c2)

    def _pat_ordered_dither(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        # Bayer 8×8 matrix
        bayer2 = torch.tensor([[0, 2], [3, 1]], device=self.device, dtype=torch.float32)
        bayer4 = torch.zeros(4, 4, device=self.device)
        for r in range(2):
            for c in range(2):
                bayer4[r*2:(r+1)*2, c*2:(c+1)*2] = bayer2 * 4 + \
                    torch.tensor([[0, 2], [3, 1]], device=self.device).float() + \
                    bayer2[r, c] * 4
        bayer4 = bayer4 / 16.0
        # Tile to full res
        scale = torch.randint(1, 6, (B,), device=self.device)
        brightness = self._perlin(B, beta=1.5)
        out = torch.zeros(B, self.H, self.W, device=self.device)
        for i in range(B):
            s = scale[i].item()
            bayer_scaled = bayer4.repeat(1, 1)  # base
            # Tile
            reps_h = self.H // (4 * s) + 1
            reps_w = self.W // (4 * s) + 1
            tiled = bayer4.repeat(reps_h, reps_w)
            tiled = F.interpolate(tiled.unsqueeze(0).unsqueeze(0),
                                  size=(self.H, self.W), mode='nearest').squeeze()
            out[i] = (brightness[i] > tiled).float()
        return self._lerp_colors(out, c1, c2)

    def _pat_stipple(self, B):
        c1, c2 = self._colors(B), self._colors(B)
        bg = c1.view(B, 3, 1, 1).expand(B, 3, self.H, self.W).clone()
        brightness = self._perlin(B, beta=1.8)
        n_dots = torch.randint(2000, 8000, (B,), device=self.device)
        for i in range(B):
            nd = n_dots[i].item()
            # Sample dot positions weighted by brightness
            flat_bright = brightness[i].reshape(-1)
            probs = flat_bright / (flat_bright.sum() + 1e-8)
            indices = torch.multinomial(probs, nd, replacement=True)
            iy = indices // self.W
            ix = indices % self.W
            bg[i, :, iy, ix] = c2[i].view(3, 1)
        return bg

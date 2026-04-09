#!/usr/bin/env python3
"""Shape SDF definitions for VAEpp0r generator.

Signed distance functions for 10 shape types + fractal layout renderer.
"""

import math
import torch
import torch.nn.functional as F


class ShapesMixin:
    """Mixin providing shape SDF methods for VAEpp0rGenerator."""

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
        else:
            px.fill_(S // 2)
        if py.max() > py.min():
            py = (py - py.min()) / (py.max() - py.min()) * (S - 1)
        else:
            py.fill_(S // 2)
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
            else:
                pts[:, 0] = W * 0.5
            if pts[:, 1].max() > pts[:, 1].min():
                pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / \
                    (pts[:, 1].max() - pts[:, 1].min()) * H * 0.8 + H * 0.1
            else:
                pts[:, 1] = H * 0.5

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

#!/usr/bin/env python3
"""Scene template methods for VAEpp generator.

19 scene templates for structured image composition.
"""

import math
import torch
import torch.nn.functional as F


class TemplateMixin:
    """Mixin providing scene template methods for VAEppGenerator."""

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
        vp_y = torch.randint(int(H * 0.1), max(int(H * 0.1) + 1, int(H * 0.35)), (1,)).item()
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
                rw = W - W // 2  # right half width (handles odd W)
                canvas[bi, :, :, :W//2] = half[bi, :, :, :W//2]
                canvas[bi, :, :, W//2:] = half[bi, :, :, :rw].flip(-1)
            else:
                # Horizontal mirror
                bh = H - H // 2  # bottom half height (handles odd H)
                canvas[bi, :, :H//2] = half[bi, :, :H//2]
                canvas[bi, :, H//2:] = half[bi, :, :bh].flip(-2)
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
                peak_y = torch.randint(int(H * 0.15), max(int(H * 0.15) + 1, horizon), (1,)).item()
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
                canvas[bi, :, max(0, ty-trunk_h):ty, max(0,tx-tw):tx+tw] = \
                    torch.tensor([0.4, 0.25, 0.1], device=self.device).view(3, 1, 1)
                # Canopy (approximate circle via square)
                cy = ty - trunk_h - canopy_r
                cx = tx
                y0, y1 = max(0, cy - canopy_r), min(H, max(0, cy + canopy_r))
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
            wall_h = floor_y - ceil_h
            if torch.rand(1).item() < 0.7 and wall_h > H // 4:
                ww = torch.randint(W // 8, W // 3, (1,)).item()
                wh_max = max(H // 8 + 1, wall_h * 2 // 3)
                wh = torch.randint(H // 8, wh_max, (1,)).item()
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
                trunk_top = torch.randint(int(H * 0.1), max(int(H * 0.1) + 1, int(H * 0.4)), (1,)).item()
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

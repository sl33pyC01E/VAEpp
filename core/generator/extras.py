#!/usr/bin/env python3
"""Additional motion priors — catalog fillers.

Four cheap per-frame effects that round out the 8x temporal training
distribution with visually-distinct classes:

  - fire flames    : vertical advection of noise with cool->hot palette
  - vortex swirl   : polar UV rotation that warps by (r, θ) of distance
  - starfield zoom : radial streaming points with trails
  - eq bars        : animated bouncing frequency bars

All follow the recipe-sample/apply pattern.
"""

import math
import random
import torch
import torch.nn.functional as F


class ExtrasMixin:
    """Mixin providing additional motion-prior effects."""

    # ------------------------------------------------------------------
    # Fire flames
    # ------------------------------------------------------------------
    def _sample_fire_recipe(self, T, intensity=0.8):
        return {
            "enable": True,
            "intensity": float(intensity),
            "seed": int(torch.randint(0, 2**31 - 1, (1,)).item()),
        }

    def _apply_fire(self, canvas, ti, fp):
        if fp is None or not fp.get("enable", False):
            return canvas
        B, C, H, W = canvas.shape
        dev = canvas.device
        # Animated Perlin-like noise advected upward
        seed = int(fp["seed"])
        g = torch.Generator(device=dev).manual_seed(seed ^ ti)
        low_h, low_w = H // 8, W // 8
        noise = torch.rand(1, 1, low_h, low_w, device=dev, generator=g)
        noise = F.interpolate(noise, (H, W), mode="bilinear", align_corners=False)
        # Vertical gradient: hot at bottom, cool at top
        ys = torch.linspace(1.0, 0.0, H, device=dev).view(1, 1, H, 1)
        gradient = ys * ys * ys  # bias energy toward bottom
        # Flame = noise * gradient
        flame = (noise * gradient).clamp(0, 1) * float(fp["intensity"])
        # Palette: black -> red -> orange -> yellow -> white
        r = flame.clamp(0, 1)
        g_c = (flame - 0.3).clamp(0, 1) * 1.3
        b_c = (flame - 0.7).clamp(0, 1) * 3.0
        fire = torch.cat([r, g_c, b_c], dim=1).clamp(0, 1)
        # Additive composite using flame as alpha
        alpha = flame
        return canvas * (1 - alpha) + fire * alpha

    # ------------------------------------------------------------------
    # Vortex swirl
    # ------------------------------------------------------------------
    def _sample_vortex_recipe(self, T, strength=0.6):
        return {
            "enable": True,
            "strength": float(strength),
            "speed": float(torch.empty(1).uniform_(0.02, 0.1).item()),
            "cx": float(torch.empty(1).uniform_(0.3, 0.7).item()),
            "cy": float(torch.empty(1).uniform_(0.3, 0.7).item()),
        }

    def _apply_vortex(self, canvas, ti, vp):
        if vp is None or not vp.get("enable", False):
            return canvas
        self._ensure_effects_grids()
        B, C, H, W = canvas.shape
        cx = (float(vp["cx"]) - 0.5) * 2
        cy = (float(vp["cy"]) - 0.5) * 2
        strength = float(vp["strength"])
        speed = float(vp["speed"])
        nx = self._eff_nx - cx
        ny = self._eff_ny - cy
        r = torch.sqrt(nx * nx + ny * ny + 1e-8)
        theta = torch.atan2(ny, nx) + strength * (1 - r).clamp(0, 1) + speed * ti
        src_x = cx + r * torch.cos(theta)
        src_y = cy + r * torch.sin(theta)
        grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0)
        if B > 1:
            grid = grid.expand(B, -1, -1, -1)
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    # ------------------------------------------------------------------
    # Starfield zoom
    # ------------------------------------------------------------------
    def _sample_starfield_recipe(self, T, n_stars=150):
        rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        stars = []
        for _ in range(n_stars):
            stars.append({
                "ang": rng.uniform(0, 2 * math.pi),
                "r0": rng.uniform(0.05, 0.9),
                "speed": rng.uniform(0.01, 0.08),
                "phase": rng.uniform(0, T),
            })
        return {"enable": True, "stars": stars}

    def _apply_starfield(self, canvas, ti, sf):
        if sf is None or not sf.get("enable", False):
            return canvas
        H, W = self.H, self.W
        out = canvas
        cx, cy = W * 0.5, H * 0.5
        for s in sf["stars"]:
            # Star streams outward from center with wrap
            t_eff = (ti + float(s["phase"])) % 40
            r = (float(s["r0"]) + float(s["speed"]) * t_eff) % 1.0
            rr = r * max(H, W) * 0.5
            ang = float(s["ang"])
            x = cx + rr * math.cos(ang)
            y = cy + rr * math.sin(ang)
            brightness = min(1.0, r * 1.5)  # dimmer near center, brighter at edge
            # Trail: previous frame position
            tr = max(0, t_eff - 1)
            r_t = (float(s["r0"]) + float(s["speed"]) * tr) % 1.0
            rr_t = r_t * max(H, W) * 0.5
            xt = cx + rr_t * math.cos(ang)
            yt = cy + rr_t * math.sin(ang)
            # Draw streak
            n_samples = 4
            for k in range(n_samples):
                alpha = k / max(n_samples - 1, 1)
                px = x * (1 - alpha) + xt * alpha
                py = y * (1 - alpha) + yt * alpha
                b = brightness * (1 - alpha * 0.7)
                if 0 <= int(px) < W and 0 <= int(py) < H:
                    out[:, :, int(py), int(px)] = b
        return out

    # ------------------------------------------------------------------
    # EQ bars
    # ------------------------------------------------------------------
    def _sample_eq_recipe(self, T, n_bars=24):
        rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        phases = [rng.uniform(0, 2 * math.pi) for _ in range(n_bars)]
        freqs = [rng.uniform(0.1, 0.4) for _ in range(n_bars)]
        return {
            "enable": True,
            "n_bars": int(n_bars),
            "phases": phases,
            "freqs": freqs,
            "color": [float(torch.empty(1).uniform_(0.4, 1.0).item()) for _ in range(3)],
        }

    def _apply_eq_bars(self, canvas, ti, ep):
        if ep is None or not ep.get("enable", False):
            return canvas
        H, W = self.H, self.W
        n = int(ep["n_bars"])
        bar_w = W // n
        gap = 2
        origin_y = H - 20
        out = canvas
        color = ep["color"]
        for i in range(n):
            # Height from sinusoid + random modulation
            phase = ep["phases"][i]
            freq = ep["freqs"][i]
            amp = 0.5 + 0.5 * math.sin(freq * ti + phase)
            # Plus higher-freq shiver
            shiver = 0.15 * math.sin(freq * 3.1 * ti + phase * 1.7)
            height = int(max(0.05, min(1.0, amp + shiver)) * (H * 0.5))
            x = i * bar_w + gap
            y = origin_y - height
            self._draw_rect(out, x, y, bar_w - gap * 2, height, color)
        return out

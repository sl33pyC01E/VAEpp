#!/usr/bin/env python3
"""Particle system for VAEpp0r generator.

A single state tensor `(N, 8)` = [x, y, vx, vy, life, size, color_idx, kind].
Closed-form trajectory: particle position at frame ti is computed directly
from initial state + simple kinematics (gravity / drag / wind), no step-
by-step integration. This makes per-frame rendering stateless.

Each particle stamps a small gaussian RGB kernel onto the canvas; stamp
kernels are pre-built once per-canvas-size/device and reused.

6 presets, each a different sampler that populates the (N, 8) state:
  - confetti      : upward burst then gravity, vibrant palette
  - fireworks     : point explosion + trailing sparks, warm palette
  - sparks        : short-life upward streaks
  - snow          : slow downward drift with horizontal jitter
  - rain          : fast downward angled streaks
  - embers        : slow upward drift with brightness flicker
"""

import math
import random
import torch


_PRESETS = ["confetti", "fireworks", "sparks", "snow", "rain", "embers"]


class ParticleSystem:
    """Serializable particle state + closed-form step.
    Kept a plain class (not a mixin) so it can be instance-cached on the
    generator. Methods take a device for GPU placement."""

    def __init__(self, state_np, colors_np, kind, params):
        # state_np: (N, 4) -> [x, y, vx, vy] initial
        # life_np:  (N,) in [0, T]
        # size_np:  (N,)
        # colors_np: (N, 3) in [0, 1]
        self.state = state_np
        self.colors = colors_np
        self.kind = kind
        self.params = params  # extra scalars (gravity, drag, wind, palette)

    def to_device(self, device):
        self.state = self.state.to(device)
        self.colors = self.colors.to(device)
        return self

    def positions_at(self, ti):
        """Closed-form (x, y, alpha) at frame ti. alpha drops when life < ti."""
        p = self.params
        gravity = float(p.get("gravity", 0.0))
        drag = float(p.get("drag", 0.0))
        wind = float(p.get("wind", 0.0))
        s = self.state  # (N, 5): x, y, vx, vy, life
        t = float(ti)
        x0 = s[:, 0]
        y0 = s[:, 1]
        vx0 = s[:, 2]
        vy0 = s[:, 3]
        life = s[:, 4]
        # Simple kinematics with linear drag: v(t) = v0 * exp(-drag*t)
        # + gravity integral
        if drag > 1e-6:
            damp = math.exp(-drag * t)
            xv = x0 + (vx0 / drag) * (1 - damp) + wind * t
            yv = y0 + (vy0 / drag) * (1 - damp) + 0.5 * gravity * t * t
        else:
            xv = x0 + vx0 * t + wind * t
            yv = y0 + vy0 * t + 0.5 * gravity * t * t
        alive = (life > t).float()
        # Fade in / out at ends of life
        fade = ((life - t).clamp(0, 3) / 3.0) * (t / 2.0).__class__.__mul__(1, 1)  # placeholder
        # Simple: alpha = alive * (1 - t / max_life)
        return xv, yv, alive, self.state[:, 5] if s.shape[1] > 5 else None


class ParticlesMixin:
    """Mixin providing particle system presets + renderer."""

    # ------------------------------------------------------------------
    # Stamp kernel cache
    # ------------------------------------------------------------------
    def _get_particle_kernel(self, size_px):
        """Cached small gaussian kernel for particle stamping."""
        key = int(max(3, size_px))
        if not hasattr(self, "_part_kernel_cache"):
            self._part_kernel_cache = {}
        if key in self._part_kernel_cache:
            return self._part_kernel_cache[key]
        k = key
        xs = torch.arange(-(k // 2), k // 2 + 1, device=self.device).float()
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        sigma = max(k / 4.0, 0.8)
        g = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        g = g / g.max()
        self._part_kernel_cache[key] = g  # (k, k)
        return g

    # ------------------------------------------------------------------
    # Recipe sampling
    # ------------------------------------------------------------------
    def _sample_particles_recipe(self, T, preset="auto", n_particles=200):
        """Build a serializable particle recipe.

        Deterministic: stores initial state lists; render is closed-form.
        """
        if preset == "auto":
            preset = _PRESETS[int(torch.randint(0, len(_PRESETS), (1,)).item())]
        H, W = self.H, self.W
        rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        n = int(n_particles)
        xs, ys, vxs, vys, lives, sizes, colors = [], [], [], [], [], [], []

        if preset == "confetti":
            # Burst from a point, upward velocity, bright palette, gravity on
            cx, cy = rng.uniform(W * 0.3, W * 0.7), rng.uniform(H * 0.6, H * 0.9)
            palette = [[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.4, 1],
                       [1, 1, 0.2], [1, 0.4, 1], [0.3, 1, 1]]
            for _ in range(n):
                ang = rng.uniform(-math.pi * 0.8, -math.pi * 0.2)  # up-ish
                spd = rng.uniform(3, 10)
                xs.append(cx + rng.uniform(-5, 5))
                ys.append(cy)
                vxs.append(math.cos(ang) * spd)
                vys.append(math.sin(ang) * spd)
                lives.append(rng.uniform(T * 0.6, T * 1.1))
                sizes.append(rng.randint(3, 6))
                colors.append(palette[rng.randint(0, len(palette) - 1)])
            params = {"gravity": 0.5, "drag": 0.02, "wind": 0.0}

        elif preset == "fireworks":
            n_bursts = rng.randint(2, 4)
            per_burst = n // n_bursts
            palette = [[1, 0.5, 0.2], [1, 0.8, 0.3], [1, 0.3, 0.3],
                       [1, 0.7, 0.7], [0.9, 0.9, 0.3]]
            for _ in range(n_bursts):
                cx = rng.uniform(W * 0.2, W * 0.8)
                cy = rng.uniform(H * 0.15, H * 0.5)
                for _ in range(per_burst):
                    ang = rng.uniform(0, 2 * math.pi)
                    spd = rng.uniform(4, 12)
                    xs.append(cx)
                    ys.append(cy)
                    vxs.append(math.cos(ang) * spd)
                    vys.append(math.sin(ang) * spd)
                    lives.append(rng.uniform(T * 0.3, T * 0.7))
                    sizes.append(rng.randint(2, 4))
                    colors.append(palette[rng.randint(0, len(palette) - 1)])
            params = {"gravity": 0.2, "drag": 0.08, "wind": 0.0}

        elif preset == "sparks":
            cx, cy = rng.uniform(W * 0.2, W * 0.8), rng.uniform(H * 0.7, H * 0.95)
            palette = [[1, 0.85, 0.3], [1, 0.6, 0.2], [1, 1, 0.6]]
            for _ in range(n):
                ang = rng.uniform(-math.pi * 0.85, -math.pi * 0.15)
                spd = rng.uniform(6, 14)
                xs.append(cx + rng.uniform(-3, 3))
                ys.append(cy)
                vxs.append(math.cos(ang) * spd)
                vys.append(math.sin(ang) * spd)
                lives.append(rng.uniform(T * 0.2, T * 0.5))
                sizes.append(rng.randint(2, 3))
                colors.append(palette[rng.randint(0, len(palette) - 1)])
            params = {"gravity": 0.3, "drag": 0.15, "wind": 0.0}

        elif preset == "snow":
            palette = [[0.95, 0.95, 1.0], [0.85, 0.9, 1.0], [1, 1, 1]]
            for _ in range(n):
                xs.append(rng.uniform(0, W))
                ys.append(rng.uniform(-H, H))
                vxs.append(rng.uniform(-0.5, 0.5))
                vys.append(rng.uniform(0.5, 2.0))
                lives.append(T * 2)  # lives through whole clip
                sizes.append(rng.randint(3, 6))
                colors.append(palette[rng.randint(0, len(palette) - 1)])
            params = {"gravity": 0.02, "drag": 0.01, "wind": rng.uniform(-0.3, 0.3)}

        elif preset == "rain":
            palette = [[0.6, 0.75, 0.95], [0.5, 0.65, 0.85]]
            for _ in range(n):
                xs.append(rng.uniform(0, W * 1.3) - W * 0.15)
                ys.append(rng.uniform(-H, H))
                vxs.append(rng.uniform(-2, -0.5))  # angled left
                vys.append(rng.uniform(10, 20))
                lives.append(T * 2)
                sizes.append(rng.randint(2, 3))
                colors.append(palette[rng.randint(0, len(palette) - 1)])
            params = {"gravity": 0.1, "drag": 0.0, "wind": 0.0}

        elif preset == "embers":
            cx, cy = rng.uniform(W * 0.2, W * 0.8), rng.uniform(H * 0.8, H * 0.95)
            palette = [[1, 0.4, 0.1], [1, 0.6, 0.2], [1, 0.3, 0.05],
                       [0.9, 0.7, 0.3]]
            for _ in range(n):
                ang = rng.uniform(-math.pi * 0.7, -math.pi * 0.3)
                spd = rng.uniform(0.5, 2)
                xs.append(cx + rng.uniform(-W * 0.1, W * 0.1))
                ys.append(cy)
                vxs.append(math.cos(ang) * spd)
                vys.append(math.sin(ang) * spd)
                lives.append(rng.uniform(T * 0.8, T * 1.3))
                sizes.append(rng.randint(3, 5))
                colors.append(palette[rng.randint(0, len(palette) - 1)])
            params = {"gravity": -0.03, "drag": 0.03, "wind": rng.uniform(-0.2, 0.2)}

        else:
            return {"enable": False}

        # Randomize scene orientation by flipping the whole trajectory set
        # (initial positions, velocities, and gravity). Gives 4 equally-
        # likely variants per sample: normal / mirror-x / flip-y / both.
        H_, W_ = self.H, self.W
        flip_x = rng.random() < 0.5
        flip_y = rng.random() < 0.5
        if flip_x:
            xs = [W_ - x for x in xs]
            vxs = [-vx for vx in vxs]
            params["wind"] = -params.get("wind", 0.0)
        if flip_y:
            ys = [H_ - y for y in ys]
            vys = [-vy for vy in vys]
            params["gravity"] = -params.get("gravity", 0.0)

        return {
            "enable": True,
            "preset": preset,
            "n": n,
            "xs": xs, "ys": ys, "vxs": vxs, "vys": vys,
            "lives": lives, "sizes": sizes, "colors": colors,
            "params": params,
            "flip_x": flip_x, "flip_y": flip_y,
        }

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def _apply_particles(self, canvas, ti, pp):
        if pp is None or not pp.get("enable", False):
            return canvas
        B, C, H, W = canvas.shape
        n = int(pp["n"])
        if n == 0:
            return canvas
        dev = self.device
        # Load state tensors
        x0 = torch.tensor(pp["xs"], device=dev, dtype=torch.float32)
        y0 = torch.tensor(pp["ys"], device=dev, dtype=torch.float32)
        vx0 = torch.tensor(pp["vxs"], device=dev, dtype=torch.float32)
        vy0 = torch.tensor(pp["vys"], device=dev, dtype=torch.float32)
        life = torch.tensor(pp["lives"], device=dev, dtype=torch.float32)
        sizes = torch.tensor(pp["sizes"], device=dev, dtype=torch.int64)
        colors = torch.tensor(pp["colors"], device=dev, dtype=torch.float32)  # (N, 3)

        params = pp["params"]
        g = float(params.get("gravity", 0.0))
        drag = float(params.get("drag", 0.0))
        wind = float(params.get("wind", 0.0))
        t = float(ti)

        if drag > 1e-6:
            damp = math.exp(-drag * t)
            xv = x0 + (vx0 / drag) * (1 - damp) + wind * t
            yv = y0 + (vy0 / drag) * (1 - damp) + 0.5 * g * t * t
        else:
            xv = x0 + vx0 * t + wind * t
            yv = y0 + vy0 * t + 0.5 * g * t * t

        alive = (life > t).float()
        # Simple fade: full brightness until last 20% of life, then linear
        fade_start = life * 0.8
        fade = ((life - t) / (life - fade_start + 1e-6)).clamp(0, 1)
        alpha = (alive * fade).unsqueeze(1)  # (N, 1)

        # Now splat each particle. For speed, process unique sizes in batches.
        out = canvas.clone()
        xi = xv.round().long()
        yi = yv.round().long()
        for size_val in sizes.unique().tolist():
            size_val = int(size_val)
            mask = (sizes == size_val) & alive.bool()
            if not mask.any():
                continue
            idxs = torch.where(mask)[0]
            kern = self._get_particle_kernel(size_val)
            kh, kw = kern.shape
            for idx in idxs.tolist():
                x = int(xi[idx].item())
                y = int(yi[idx].item())
                a = float(alpha[idx].item())
                if a <= 0:
                    continue
                y1 = max(0, y - kh // 2); y2 = min(H, y - kh // 2 + kh)
                x1 = max(0, x - kw // 2); x2 = min(W, x - kw // 2 + kw)
                if y2 <= y1 or x2 <= x1:
                    continue
                ky1 = y1 - (y - kh // 2); ky2 = ky1 + (y2 - y1)
                kx1 = x1 - (x - kw // 2); kx2 = kx1 + (x2 - x1)
                k = kern[ky1:ky2, kx1:kx2] * a
                col = colors[idx].view(3, 1, 1)
                out[:, :, y1:y2, x1:x2] = out[:, :, y1:y2, x1:x2] * (1 - k) + col * k
        return out.clamp(0, 1)

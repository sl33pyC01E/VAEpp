#!/usr/bin/env python3
"""Arcade-style scene renderers for VAEpp0r generator.

Six deterministic mini-games rendered on a plain canvas:
  - pong           : two paddles + bouncing ball
  - breakout       : ball + brick wall that shrinks over time
  - invaders       : rows of marching enemies + descending bullets
  - snake          : growing chain following a path
  - tetris         : falling tetrominoes stacking at the bottom
  - asteroids      : rotating polygonal rocks drifting with wrap

All games simulate from a seed and recipe params at each frame using
closed-form or simple iterative logic. Rendered with a shared rectangle-
fill helper, no fonts needed.
"""

import math
import random
import torch


_ARCADE_MODES = ["pong", "breakout", "invaders", "snake", "tetris", "asteroids"]


class ArcadeMixin:
    """Mixin providing arcade-style scene renderers."""

    def _sample_arcade_recipe(self, T, mode="auto"):
        """Pick a mode and seed it; rendering reads just this dict."""
        if mode == "auto":
            mode = _ARCADE_MODES[int(torch.randint(0, len(_ARCADE_MODES), (1,)).item())]
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        return {
            "enable": True,
            "mode": mode,
            "seed": seed,
            "bg_color": [float(torch.empty(1).uniform_(0.0, 0.08).item()) for _ in range(3)],
            "fg_color": [float(torch.empty(1).uniform_(0.7, 1.0).item()) for _ in range(3)],
            "accent": [float(torch.empty(1).uniform_(0.5, 1.0).item()) for _ in range(3)],
        }

    # ------------------------------------------------------------------
    # Apply dispatch
    # ------------------------------------------------------------------
    def _apply_arcade(self, canvas, ti, ap):
        if ap is None or not ap.get("enable", False):
            return canvas
        mode = ap.get("mode", "pong")
        # Paint a dark game background then overlay the scene
        bg = torch.tensor(ap["bg_color"], device=self.device).view(1, 3, 1, 1)
        canvas = canvas * 0.15 + bg.expand_as(canvas) * 0.85
        dispatch = {
            "pong": self._arcade_pong,
            "breakout": self._arcade_breakout,
            "invaders": self._arcade_invaders,
            "snake": self._arcade_snake,
            "tetris": self._arcade_tetris,
            "asteroids": self._arcade_asteroids,
        }
        fn = dispatch.get(mode, self._arcade_pong)
        return fn(canvas, ti, ap)

    # ------------------------------------------------------------------
    # Rect fill helper (in-place on an alpha layer) + composite utility
    # ------------------------------------------------------------------
    def _draw_rect(self, canvas, x, y, w, h, color):
        """color: list/tuple of 3 floats."""
        H, W = canvas.shape[-2], canvas.shape[-1]
        x1 = max(0, int(x)); y1 = max(0, int(y))
        x2 = min(W, int(x + w)); y2 = min(H, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return
        c = torch.tensor(color, device=self.device).view(1, 3, 1, 1)
        canvas[:, :, y1:y2, x1:x2] = c.expand(canvas.shape[0], 3, y2 - y1, x2 - x1)

    def _draw_circle(self, canvas, cx, cy, r, color):
        H, W = canvas.shape[-2], canvas.shape[-1]
        x1 = max(0, int(cx - r - 1)); y1 = max(0, int(cy - r - 1))
        x2 = min(W, int(cx + r + 2)); y2 = min(H, int(cy + r + 2))
        if x2 <= x1 or y2 <= y1:
            return
        xs = torch.arange(x1, x2, device=self.device).float()
        ys = torch.arange(y1, y2, device=self.device).float()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        c = torch.tensor(color, device=self.device).view(3, 1, 1)
        region = canvas[:, :, y1:y2, x1:x2]
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        canvas[:, :, y1:y2, x1:x2] = region * (1 - mask) + c.unsqueeze(0) * mask

    # ------------------------------------------------------------------
    # Pong
    # ------------------------------------------------------------------
    def _arcade_pong(self, canvas, ti, ap):
        H, W = self.H, self.W
        rng = random.Random(ap["seed"])
        fg = ap["fg_color"]
        # Two paddles oscillating up/down, ball bouncing between them
        pad_w, pad_h = 6, int(H * 0.15)
        ball_r = 6
        speed_y = rng.uniform(1.5, 3.5)
        bvx = rng.uniform(3.5, 6.0)
        bvy = rng.uniform(-2.5, 2.5)
        # Closed-form: ball pos with bouncing off top/bottom
        bx = (W * 0.5 + bvx * ti) % (W - 2 * pad_w - ball_r * 2 - 8) + pad_w + ball_r + 4
        by = H * 0.5 + math.sin(ti * 0.2) * (H * 0.35)
        # Paddles follow ball's y
        pady_l = (by - pad_h * 0.5) + math.sin(ti * 0.12) * 10
        pady_r = (by - pad_h * 0.5) + math.cos(ti * 0.12) * 10
        self._draw_rect(canvas, 10, pady_l, pad_w, pad_h, fg)
        self._draw_rect(canvas, W - 10 - pad_w, pady_r, pad_w, pad_h, fg)
        # Center line
        for dy in range(0, H, 16):
            self._draw_rect(canvas, W // 2 - 1, dy, 2, 8, [0.5, 0.5, 0.5])
        self._draw_circle(canvas, bx, by, ball_r, fg)
        return canvas

    # ------------------------------------------------------------------
    # Breakout
    # ------------------------------------------------------------------
    def _arcade_breakout(self, canvas, ti, ap):
        H, W = self.H, self.W
        rng = random.Random(ap["seed"])
        fg = ap["fg_color"]
        rows, cols = 5, 10
        brick_w = (W - 40) // cols
        brick_h = 14
        # Bricks disappear over time (one per frame, in a random order)
        order = list(range(rows * cols))
        rng.shuffle(order)
        remaining = max(0, rows * cols - ti)
        alive = set(order[:remaining])
        palette = [[1, 0.3, 0.3], [1, 0.8, 0.3], [0.4, 1, 0.5],
                   [0.3, 0.7, 1], [0.8, 0.4, 1]]
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx not in alive:
                    continue
                x = 20 + c * brick_w
                y = 40 + r * (brick_h + 3)
                self._draw_rect(canvas, x + 2, y, brick_w - 4, brick_h,
                                palette[r % len(palette)])
        # Paddle
        pad_w, pad_h = 64, 8
        pad_x = (W - pad_w) // 2 + math.sin(ti * 0.15) * (W * 0.3)
        self._draw_rect(canvas, pad_x, H - 40, pad_w, pad_h, fg)
        # Ball
        bx = W * 0.5 + math.sin(ti * 0.3) * (W * 0.4)
        by = H * 0.5 + math.cos(ti * 0.2) * (H * 0.2)
        self._draw_circle(canvas, bx, by, 5, fg)
        return canvas

    # ------------------------------------------------------------------
    # Space Invaders
    # ------------------------------------------------------------------
    def _arcade_invaders(self, canvas, ti, ap):
        H, W = self.H, self.W
        rng = random.Random(ap["seed"])
        fg = ap["fg_color"]
        acc = ap["accent"]
        rows, cols = 4, 8
        inv_w, inv_h = 24, 16
        step = 20
        # Marching motion: left/right sweep + drop
        march_cycle = 40
        sweep = math.sin(ti * 2 * math.pi / march_cycle)
        offset_x = sweep * 30
        drop_y = (ti // march_cycle) * 6
        origin_x = (W - cols * (inv_w + 6)) // 2
        origin_y = 30
        for r in range(rows):
            for c in range(cols):
                x = origin_x + c * (inv_w + 6) + offset_x
                y = origin_y + r * (inv_h + 8) + drop_y
                if y > H - 80:
                    continue
                color = fg if r % 2 == 0 else acc
                self._draw_rect(canvas, x, y, inv_w, inv_h, color)
        # Player ship
        ship_w, ship_h = 40, 14
        ship_x = (W - ship_w) // 2 + math.sin(ti * 0.2) * (W * 0.2)
        self._draw_rect(canvas, ship_x, H - 30, ship_w, ship_h, fg)
        # Bullets (one every few frames)
        n_bullets = min(5, ti // 3)
        for bi in range(n_bullets):
            bx = (bi * 113 + ti * 17) % (W - 10) + 5
            by = ((ti - bi * 3) * 8) % H
            self._draw_rect(canvas, bx, by, 3, 10, acc)
        return canvas

    # ------------------------------------------------------------------
    # Snake
    # ------------------------------------------------------------------
    def _arcade_snake(self, canvas, ti, ap):
        H, W = self.H, self.W
        rng = random.Random(ap["seed"])
        fg = ap["fg_color"]
        acc = ap["accent"]
        cell = 16
        cols = W // cell; rows = H // cell
        # Snake follows a Lissajous-ish path, growing over time
        length = min(3 + ti, cols + rows)
        for i in range(length):
            t = ti - i
            cx = cols * 0.5 + math.sin(t * 0.15) * cols * 0.35
            cy = rows * 0.5 + math.cos(t * 0.12) * rows * 0.4
            cx = int(cx) % cols
            cy = int(cy) % rows
            col = fg if i == 0 else acc
            self._draw_rect(canvas, cx * cell + 2, cy * cell + 2,
                           cell - 4, cell - 4, col)
        # "Food" dot
        fx = (ap["seed"] % cols) * cell + cell // 2
        fy = ((ap["seed"] // cols) % rows) * cell + cell // 2
        self._draw_circle(canvas, fx, fy, 4, [1, 0.3, 0.3])
        return canvas

    # ------------------------------------------------------------------
    # Tetris
    # ------------------------------------------------------------------
    def _arcade_tetris(self, canvas, ti, ap):
        H, W = self.H, self.W
        rng = random.Random(ap["seed"])
        fg = ap["fg_color"]
        cell = 18
        cols = (W - 80) // cell
        rows = (H - 40) // cell
        origin_x = 40
        origin_y = 20
        palette = [[0.2, 1, 1], [1, 1, 0.2], [1, 0.4, 1],
                   [0.2, 1, 0.2], [1, 0.2, 0.2], [0.2, 0.4, 1],
                   [1, 0.6, 0.2]]
        # Stacked pieces at bottom (one more per frame until board full)
        stack_height = min(rows - 1, ti // 3)
        for r in range(rows - stack_height, rows):
            # Random row pattern from deterministic seed + row idx
            row_seed = ap["seed"] ^ (r * 0x9E3779B9)
            row_rng = random.Random(row_seed)
            for c in range(cols):
                if row_rng.random() < 0.75:
                    col = palette[(c + r) % len(palette)]
                    self._draw_rect(canvas, origin_x + c * cell, origin_y + r * cell,
                                   cell - 2, cell - 2, col)
        # Falling piece (uses a simple L-tetromino)
        if ti < rows * 3:
            piece_r = (ti // 3) % (rows - stack_height - 2)
            piece_c = (ti * 7 + ap["seed"]) % (cols - 2)
            col = palette[ap["seed"] % len(palette)]
            for dr, dc in [(0, 0), (1, 0), (2, 0), (2, 1)]:
                self._draw_rect(canvas, origin_x + (piece_c + dc) * cell,
                               origin_y + (piece_r + dr) * cell,
                               cell - 2, cell - 2, col)
        # Frame
        for r in range(rows):
            self._draw_rect(canvas, origin_x - 3, origin_y + r * cell, 3, cell, fg)
            self._draw_rect(canvas, origin_x + cols * cell, origin_y + r * cell, 3, cell, fg)
        return canvas

    # ------------------------------------------------------------------
    # Asteroids
    # ------------------------------------------------------------------
    def _arcade_asteroids(self, canvas, ti, ap):
        H, W = self.H, self.W
        rng = random.Random(ap["seed"])
        fg = ap["fg_color"]
        n_rocks = rng.randint(4, 7)
        for i in range(n_rocks):
            # Deterministic per-rock params
            rseed = ap["seed"] ^ (i * 0x85EBCA6B)
            r_rng = random.Random(rseed)
            r = r_rng.randint(12, 30)
            vx = r_rng.uniform(-3, 3)
            vy = r_rng.uniform(-2, 2)
            rot_speed = r_rng.uniform(-0.1, 0.1)
            x0 = r_rng.uniform(0, W)
            y0 = r_rng.uniform(0, H)
            x = (x0 + vx * ti) % W
            y = (y0 + vy * ti) % H
            rot = rot_speed * ti
            # Octagonal asteroid via rect splats
            n_sides = 8
            for k in range(n_sides):
                ang = rot + k * 2 * math.pi / n_sides
                px = x + r * math.cos(ang)
                py = y + r * math.sin(ang)
                self._draw_rect(canvas, px - 3, py - 3, 6, 6, fg)
            # Center dot
            self._draw_circle(canvas, x, y, 3, ap["accent"])
        # Player ship
        ship_x = W * 0.5
        ship_y = H * 0.5
        ang = ti * 0.08
        tip_x = ship_x + 12 * math.cos(ang)
        tip_y = ship_y + 12 * math.sin(ang)
        self._draw_rect(canvas, tip_x - 2, tip_y - 2, 4, 4, fg)
        self._draw_circle(canvas, ship_x, ship_y, 4, fg)
        return canvas

#!/usr/bin/env python3
"""Signage effects for VAEpp0r generator.

A catalog of simulated displays and signs. All of them share the same
recipe/apply pattern: _sample_signage_recipe picks a mode and per-mode
params, _apply_signage dispatches to the right renderer at each frame.

Modes:
  - "led_matrix"  : text rendered then masked through a dot-grid
  - "seven_seg"   : digit counter or clock, drawn as 7-segment bars
  - "marquee"     : horizontally scrolling text inside a bordered panel
  - "neon"        : text with color pulse and soft glow
  - "ticker"      : stock-ticker-style scrolling text with +/- arrows
  - "warning"     : full-field color alternation (red/yellow/hazard)
  - "test_card"   : static SMPTE-style color bars
  - "loading"     : caption + animated progress bar

Uses DejaVu fonts from Phase 4 for text modes.
"""

import math
import os
import random
import torch
import torch.nn.functional as F


class SignageMixin:
    """Mixin providing display/sign simulations."""

    _SIGNAGE_MODES = ["led_matrix", "seven_seg", "marquee", "neon",
                     "ticker", "warning", "test_card", "loading"]

    # ------------------------------------------------------------------
    # Recipe sampler
    # ------------------------------------------------------------------
    def _sample_signage_recipe(self, T, mode="auto",
                               font_size=32, color=None):
        """Build a serializable signage recipe. mode='auto' picks random."""
        if mode == "auto":
            mode = self._SIGNAGE_MODES[
                int(torch.randint(0, len(self._SIGNAGE_MODES), (1,)).item())]
        if color is None:
            color = [float(torch.empty(1).uniform_(0.6, 1.0).item()) for _ in range(3)]
        p = {
            "enable": True,
            "mode": mode,
            "font_size": int(font_size),
            "color": color,
            "origin_xy": [
                int(torch.randint(20, max(self.W - 200, 40), (1,)).item()),
                int(torch.randint(20, max(self.H - 80, 40), (1,)).item()),
            ],
            "seed": int(torch.randint(0, 2**31 - 1, (1,)).item()),
        }
        # Mode-specific params
        if mode == "led_matrix":
            p["dot_size"] = int(torch.randint(2, 5, (1,)).item())
            p["dot_gap"] = int(torch.randint(1, 3, (1,)).item())
            p["text"] = _random_text(seed=p["seed"], length=12)
            p["scroll_pxpf"] = float(torch.empty(1).uniform_(0, 6).item())
        elif mode == "seven_seg":
            p["mode_7seg"] = ["counter_up", "counter_down", "clock",
                             "random"][int(torch.randint(0, 4, (1,)).item())]
            p["digits"] = int(torch.randint(3, 7, (1,)).item())
            p["seg_w"] = int(font_size * 0.6)
            p["seg_h"] = int(font_size)
            p["start"] = int(torch.randint(0, 10000, (1,)).item())
            p["step_per_frame"] = int(torch.randint(1, 20, (1,)).item())
        elif mode == "marquee":
            p["text"] = _random_text(seed=p["seed"], length=40)
            p["scroll_pxpf"] = float(torch.empty(1).uniform_(4, 14).item())
            p["panel_pad"] = 8
        elif mode == "neon":
            p["text"] = _random_text(seed=p["seed"], length=10)
            p["pulse_hz"] = float(torch.empty(1).uniform_(0.05, 0.3).item())
            p["glow_sigma"] = float(torch.empty(1).uniform_(2, 6).item())
        elif mode == "ticker":
            p["n_items"] = int(torch.randint(4, 9, (1,)).item())
            p["scroll_pxpf"] = float(torch.empty(1).uniform_(4, 10).item())
            rng = random.Random(p["seed"])
            items = []
            syms = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "META",
                    "NVDA", "ORCL", "IBM", "ACME", "CORP", "XYZ"]
            for _ in range(p["n_items"]):
                items.append({
                    "sym": rng.choice(syms),
                    "price": round(rng.uniform(10, 500), 2),
                    "delta": round(rng.uniform(-5, 5), 2),
                })
            p["items"] = items
        elif mode == "warning":
            p["colors"] = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
            p["hz"] = float(torch.empty(1).uniform_(1.5, 4.0).item())
            p["pattern"] = ["stripes_diag", "solid", "chevron"][
                int(torch.randint(0, 3, (1,)).item())]
        elif mode == "test_card":
            # SMPTE-ish bar colors
            p["bars"] = [[0.75, 0.75, 0.75], [0.75, 0.75, 0],
                        [0, 0.75, 0.75], [0, 0.75, 0],
                        [0.75, 0, 0.75], [0.75, 0, 0], [0, 0, 0.75]]
        elif mode == "loading":
            p["caption"] = _random_text(seed=p["seed"], length=20,
                                        language="latin")
            p["bar_width"] = int(self.W * 0.6)
            p["bar_height"] = 14
        return p

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _apply_signage(self, canvas, ti, sp):
        if sp is None or not sp.get("enable", False):
            return canvas
        mode = sp.get("mode", "marquee")
        if mode == "led_matrix":
            return self._render_led_matrix(canvas, ti, sp)
        if mode == "seven_seg":
            return self._render_seven_seg(canvas, ti, sp)
        if mode == "marquee":
            return self._render_marquee(canvas, ti, sp)
        if mode == "neon":
            return self._render_neon(canvas, ti, sp)
        if mode == "ticker":
            return self._render_ticker(canvas, ti, sp)
        if mode == "warning":
            return self._render_warning(canvas, ti, sp)
        if mode == "test_card":
            return self._render_test_card(canvas, ti, sp)
        if mode == "loading":
            return self._render_loading(canvas, ti, sp)
        return canvas

    # ------------------------------------------------------------------
    # Individual renderers
    # ------------------------------------------------------------------
    def _render_led_matrix(self, canvas, ti, sp):
        # Render text as alpha mask, then multiply by a dot grid
        font = self._get_font(sp["font_size"], "mono")
        origin = list(sp["origin_xy"])
        dx = int(sp.get("scroll_pxpf", 0) * ti)
        origin[0] = (origin[0] - dx) % max(self.W, 1)
        alpha, rgb = self._rasterize_text(sp["text"], font, origin,
                                          sp["color"], with_cursor=False)
        # Dot grid mask
        d = int(sp.get("dot_size", 3))
        g = int(sp.get("dot_gap", 1))
        step = max(d + g, 2)
        H, W = self.H, self.W
        yy = torch.arange(H, device=self.device) % step
        xx = torch.arange(W, device=self.device) % step
        dot_mask_y = (yy < d).float().view(H, 1)
        dot_mask_x = (xx < d).float().view(1, W)
        dot_mask = dot_mask_y * dot_mask_x  # (H, W)
        alpha = alpha * dot_mask.unsqueeze(0)
        rgb = rgb * dot_mask.unsqueeze(0)
        alpha_b = alpha.unsqueeze(0)
        rgb_b = rgb.unsqueeze(0)
        return canvas * (1 - alpha_b) + rgb_b * alpha_b

    def _render_seven_seg(self, canvas, ti, sp):
        # Compute the displayed number
        mode = sp.get("mode_7seg", "counter_up")
        digits = int(sp.get("digits", 4))
        start = int(sp.get("start", 0))
        step = int(sp.get("step_per_frame", 1))
        if mode == "counter_up":
            value = start + step * ti
        elif mode == "counter_down":
            value = start - step * ti
        elif mode == "clock":
            # Encode as HHMM or MMSS
            total_s = start + ti
            m, s = (total_s // 60) % 60, total_s % 60
            value = int(f"{m:02d}{s:02d}")
        else:
            rng = random.Random(sp["seed"] ^ ti)
            value = rng.randint(0, 10 ** digits - 1)
        text = str(abs(value) % (10 ** digits)).zfill(digits)
        return self._composite_7seg(canvas, text, sp)

    def _composite_7seg(self, canvas, text, sp):
        """Draw the digit string as stylized 7-segment bars directly into
        an alpha mask, then composite. Each digit is `seg_w` wide."""
        seg_w = int(sp.get("seg_w", 20))
        seg_h = int(sp.get("seg_h", 32))
        bar = max(3, int(seg_w * 0.15))
        ox, oy = int(sp["origin_xy"][0]), int(sp["origin_xy"][1])
        color = torch.tensor(sp["color"], device=self.device).view(3, 1, 1)
        # Segment lookup: for each digit 0-9, which of a-g are lit
        seg_map = {
            "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
            "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
            "8": "abcdefg", "9": "abcdfg"
        }
        H, W = self.H, self.W
        alpha = torch.zeros(1, H, W, device=self.device)
        for i, ch in enumerate(text):
            if ch not in seg_map:
                continue
            d_ox = ox + i * (seg_w + int(seg_w * 0.2))
            lit = seg_map[ch]
            # a: top horizontal
            if 'a' in lit: _fill_rect(alpha, d_ox, oy, seg_w, bar)
            # b: top-right vertical
            if 'b' in lit: _fill_rect(alpha, d_ox + seg_w - bar, oy, bar, seg_h // 2)
            # c: bottom-right vertical
            if 'c' in lit: _fill_rect(alpha, d_ox + seg_w - bar, oy + seg_h // 2, bar, seg_h // 2)
            # d: bottom horizontal
            if 'd' in lit: _fill_rect(alpha, d_ox, oy + seg_h - bar, seg_w, bar)
            # e: bottom-left vertical
            if 'e' in lit: _fill_rect(alpha, d_ox, oy + seg_h // 2, bar, seg_h // 2)
            # f: top-left vertical
            if 'f' in lit: _fill_rect(alpha, d_ox, oy, bar, seg_h // 2)
            # g: middle horizontal
            if 'g' in lit: _fill_rect(alpha, d_ox, oy + seg_h // 2 - bar // 2, seg_w, bar)
        rgb = color.expand(3, H, W) * alpha
        alpha_b = alpha.unsqueeze(0)
        rgb_b = rgb.unsqueeze(0)
        return canvas * (1 - alpha_b) + rgb_b * alpha_b

    def _render_marquee(self, canvas, ti, sp):
        # Scrolling text inside a dark panel with border
        font = self._get_font(sp["font_size"], "mono")
        pad = int(sp.get("panel_pad", 8))
        ox, oy = int(sp["origin_xy"][0]), int(sp["origin_xy"][1])
        dx = -int(sp["scroll_pxpf"] * ti)
        text_origin = [ox + dx, oy]
        alpha, rgb = self._rasterize_text(sp["text"], font, text_origin,
                                          sp["color"], with_cursor=False)
        # Build panel rectangle at fixed position (not scrolling)
        panel_h = int(sp["font_size"] + pad * 2)
        panel_w = int(sp.get("panel_w", self.W - ox - pad))
        H, W = self.H, self.W
        panel_alpha = torch.zeros(1, H, W, device=self.device)
        y1 = max(0, oy - pad)
        y2 = min(H, oy + panel_h - pad)
        x1 = max(0, ox - pad)
        x2 = min(W, ox + panel_w)
        panel_alpha[0, y1:y2, x1:x2] = 1.0
        # Fill panel first (dark bg)
        panel_color = torch.tensor([0.05, 0.05, 0.08], device=self.device).view(3, 1, 1)
        panel_rgb = panel_color.expand(3, H, W) * panel_alpha
        canvas = canvas * (1 - panel_alpha.unsqueeze(0) * 0.85) + panel_rgb.unsqueeze(0) * 0.85
        # Clip text to panel
        alpha = alpha * panel_alpha
        rgb = rgb * panel_alpha
        return canvas * (1 - alpha.unsqueeze(0)) + rgb.unsqueeze(0) * alpha.unsqueeze(0)

    def _render_neon(self, canvas, ti, sp):
        # Render text, build a gaussian glow, pulse intensity over time
        font = self._get_font(sp["font_size"], "sans")
        origin = list(sp["origin_xy"])
        alpha, rgb = self._rasterize_text(sp["text"], font, origin,
                                          sp["color"], with_cursor=False)
        # Glow: blur alpha with a gaussian
        sigma = float(sp.get("glow_sigma", 4.0))
        k = max(int(sigma * 3) | 1, 3)
        # Separable blur using conv2d
        xs = torch.arange(-(k // 2), k // 2 + 1, device=self.device).float()
        g = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
        g = g / g.sum()
        g_row = g.view(1, 1, 1, k)
        g_col = g.view(1, 1, k, 1)
        a_4d = alpha.unsqueeze(0)
        glow = F.conv2d(a_4d, g_row, padding=(0, k // 2))
        glow = F.conv2d(glow, g_col, padding=(k // 2, 0))
        # Pulse amplitude
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * sp["pulse_hz"] * ti)
        glow_strength = 0.6 + 0.4 * pulse
        glow = glow.clamp(0, 1) * glow_strength
        color_t = torch.tensor(sp["color"], device=self.device).view(1, 3, 1, 1)
        glow_rgb = color_t * glow  # (1, 3, H, W)
        # Composite: additive glow, then solid text on top
        out = canvas + glow_rgb * 0.8
        a_b = alpha.unsqueeze(0)
        r_b = rgb.unsqueeze(0)
        out = out * (1 - a_b) + r_b * a_b
        return out.clamp(0, 1)

    def _render_ticker(self, canvas, ti, sp):
        # Horizontal stock ticker scrolling left
        font = self._get_font(sp["font_size"], "mono")
        # Build one flat string with colored +/- markers embedded (render twice: base + colored deltas)
        items = sp["items"]
        texts = []
        for it in items:
            arrow = "+" if it["delta"] >= 0 else ""
            texts.append(f"{it['sym']} {it['price']:.2f}  {arrow}{it['delta']:.2f}")
        full_text = "   ".join(texts) + "   "
        ox, oy = int(sp["origin_xy"][0]), int(sp["origin_xy"][1])
        dx = -int(sp["scroll_pxpf"] * ti)
        origin = [ox + dx, oy]
        alpha, rgb = self._rasterize_text(full_text, font, origin,
                                          [0.9, 0.9, 0.9], with_cursor=False)
        a_b = alpha.unsqueeze(0)
        r_b = rgb.unsqueeze(0)
        # Dark background strip behind the ticker
        band_h = int(sp["font_size"] + 12)
        band = torch.zeros(1, 1, self.H, self.W, device=self.device)
        band[:, :, max(0, oy - 6):min(self.H, oy + band_h)] = 1.0
        band_rgb = torch.zeros(1, 3, self.H, self.W, device=self.device)
        canvas = canvas * (1 - band * 0.7) + band_rgb * (band * 0.7)
        return canvas * (1 - a_b) + r_b * a_b

    def _render_warning(self, canvas, ti, sp):
        # Full-field alternation at hz between two colors + pattern overlay
        hz = float(sp.get("hz", 2.0))
        toggle = int(math.floor(hz * ti)) % 2
        col = sp["colors"][toggle]
        c = torch.tensor(col, device=self.device).view(1, 3, 1, 1)
        full = c.expand(canvas.shape[0], 3, self.H, self.W)
        pattern = sp.get("pattern", "solid")
        if pattern == "solid":
            mask = torch.ones(1, 1, self.H, self.W, device=self.device)
        elif pattern == "stripes_diag":
            yy = torch.arange(self.H, device=self.device).view(self.H, 1)
            xx = torch.arange(self.W, device=self.device).view(1, self.W)
            mask = (((yy + xx) // 20) % 2).float().unsqueeze(0).unsqueeze(0)
        elif pattern == "chevron":
            yy = torch.arange(self.H, device=self.device).view(self.H, 1)
            xx = torch.arange(self.W, device=self.device).view(1, self.W)
            v = ((yy - xx).abs() // 24) % 2
            mask = v.float().unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.ones(1, 1, self.H, self.W, device=self.device)
        blend = 0.85 * mask
        return canvas * (1 - blend) + full * blend

    def _render_test_card(self, canvas, ti, sp):
        # SMPTE-style bars filling the full canvas
        bars = sp["bars"]
        n = len(bars)
        band_w = self.W // n
        out = canvas.clone()
        for i, col in enumerate(bars):
            x0 = i * band_w
            x1 = (i + 1) * band_w if i < n - 1 else self.W
            c = torch.tensor(col, device=self.device).view(3, 1, 1)
            out[:, :, :, x0:x1] = c.expand(3, self.H, x1 - x0)
        return out

    def _render_loading(self, canvas, ti, sp):
        # Caption + horizontal progress bar that fills linearly over the clip
        font = self._get_font(sp["font_size"], "sans")
        origin = list(sp["origin_xy"])
        alpha, rgb = self._rasterize_text(sp["caption"], font, origin,
                                          sp["color"], with_cursor=False)
        a_b = alpha.unsqueeze(0)
        r_b = rgb.unsqueeze(0)
        out = canvas * (1 - a_b) + r_b * a_b
        # Progress bar just below caption
        bar_w = int(sp["bar_width"])
        bar_h = int(sp["bar_height"])
        bx = int(origin[0])
        by = int(origin[1] + sp["font_size"] + 8)
        # Estimate total T from the scheduled recipe (fallback to plausible constant)
        T_est = 24
        progress = min(1.0, (ti + 1) / max(T_est, 1))
        # Frame: outline
        frame = torch.zeros(1, 1, self.H, self.W, device=self.device)
        y1 = max(0, by)
        y2 = min(self.H, by + bar_h)
        x1 = max(0, bx)
        x2 = min(self.W, bx + bar_w)
        frame[:, :, y1:y2, x1:x2] = 0.2
        # Fill portion
        fill_x2 = min(x2, x1 + int(bar_w * progress))
        frame[:, :, y1:y2, x1:fill_x2] = 1.0
        col = torch.tensor(sp["color"], device=self.device).view(1, 3, 1, 1)
        fill_rgb = col * frame
        return out * (1 - frame) + fill_rgb * frame


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------
def _fill_rect(alpha, x, y, w, h):
    """In-place fill of alpha rectangle, clipping to bounds."""
    H, W = alpha.shape[-2], alpha.shape[-1]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 > x1 and y2 > y1:
        alpha[..., y1:y2, x1:x2] = 1.0


def _random_text(seed, length=20, language="latin"):
    """Plain-ASCII random text for signage labels."""
    rng = random.Random(seed)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
    if language == "mixed":
        chars = chars + "абвгде αβγδε"
    return "".join(rng.choice(chars) for _ in range(length))

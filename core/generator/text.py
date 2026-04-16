#!/usr/bin/env python3
"""Typing / scrolling text overlays for VAEpp0r generator.

Two modes:
  - "typing": character-by-character reveal, random backspaces, cursor blink.
              Rendered as a HUD that stays with the camera (applied post-viewport).
  - "scroll_left" / "scroll_right": long string sliding horizontally, looped.

Text rendering uses PIL + bundled DejaVu fonts (Latin/Cyrillic/Greek at minimum;
DejaVuSans covers more scripts). The full schedule of what substring is visible
at each frame is baked into the recipe at sample time so rendering is a pure
lookup.
"""

import math
import os
import random
import torch


# ---------------------------------------------------------------------------
# Bundled font paths
# ---------------------------------------------------------------------------
_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_MONO = os.path.join(_FONT_DIR, "DejaVuSansMono.ttf")
FONT_SANS = os.path.join(_FONT_DIR, "DejaVuSans.ttf")


# Unicode sample ranges for different languages. (start, end_exclusive) pairs.
# DejaVu covers these; uncovered glyphs just render as tofu (fine for training).
_LANG_RANGES = {
    "latin":    [(0x0041, 0x005B), (0x0061, 0x007B), (0x0030, 0x003A)],
    "cyrillic": [(0x0410, 0x0450)],
    "greek":    [(0x0391, 0x03CA)],
    "hebrew":   [(0x05D0, 0x05EB)],
    "arabic":   [(0x0621, 0x0649)],
    "ascii_sym":[(0x0021, 0x002F), (0x003A, 0x0040), (0x005B, 0x0061)],
    "digits":   [(0x0030, 0x003A)],
}
_LANG_NAMES = list(_LANG_RANGES.keys())


def _sample_random_string(length, language="mixed", seed=None):
    """Random unicode string of length `length` from the given language."""
    rng = random.Random(seed)
    if language == "mixed":
        langs = ["latin", "cyrillic", "greek", "ascii_sym", "digits"]
    else:
        langs = [language] if language in _LANG_RANGES else ["latin"]
    all_chars = []
    for lg in langs:
        for lo, hi in _LANG_RANGES[lg]:
            all_chars.extend(chr(c) for c in range(lo, hi))
    # Insert occasional spaces
    all_chars.extend([" "] * max(1, len(all_chars) // 20))
    out = "".join(rng.choice(all_chars) for _ in range(length))
    return out


class TextMixin:
    """Mixin adding typing / scrolling text overlays."""

    # ------------------------------------------------------------------
    # Font cache (ImageFont instances by size; PIL fonts are cheap to cache)
    # ------------------------------------------------------------------
    def _get_font(self, size, family="mono"):
        key = (family, int(size))
        if not hasattr(self, "_text_font_cache"):
            self._text_font_cache = {}
        if key in self._text_font_cache:
            return self._text_font_cache[key]
        from PIL import ImageFont
        path = FONT_MONO if family == "mono" else FONT_SANS
        try:
            font = ImageFont.truetype(path, int(size))
        except Exception:
            font = ImageFont.load_default()
        self._text_font_cache[key] = font
        return font

    # ------------------------------------------------------------------
    # Recipe sampler
    # ------------------------------------------------------------------
    def _sample_text_recipe(self, T,
                            mode="typing",
                            language="mixed",
                            font_size=24,
                            family="mono",
                            cps=12.0,
                            backspace_rate=0.05,
                            cursor_blink_hz=2.0,
                            scroll_pxpf=8.0,
                            string_length=None,
                            origin_xy=None,
                            color=None):
        """Build a serializable text-overlay recipe.

        Args:
            T: total clip frames (for scheduling typed/scrolled length)
            mode: "typing", "scroll_left", "scroll_right"
            language: one of _LANG_NAMES plus "mixed"
            font_size: pixel size
            family: "mono" or "sans"
            cps: typing chars-per-frame-second (used in typing mode)
            backspace_rate: poisson rate of backspace events per frame
            cursor_blink_hz: cursor blink frequency (cycles/frame)
            scroll_pxpf: pixels-per-frame for scrolling modes
            string_length: override final string length
            origin_xy: (x, y) in pixels; random if None
            color: (r, g, b) in [0,1]; random if None
        """
        if origin_xy is None:
            origin_xy = [
                int(torch.randint(10, max(self.W - 50, 20), (1,)).item()),
                int(torch.randint(10, max(self.H - int(font_size) - 10, 20), (1,)).item()),
            ]
        if color is None:
            color = [float(torch.empty(1).uniform_(0.6, 1.0).item()) for _ in range(3)]
        if string_length is None:
            if mode == "typing":
                # Enough chars to cover the whole clip at cps
                string_length = max(8, int(cps * T) + 4)
            else:
                # Scroll mode: long enough to fill the width at the scroll speed
                char_w = max(int(font_size * 0.6), 1)
                n_chars = int((self.W + T * scroll_pxpf) / char_w) + 4
                string_length = max(8, n_chars)

        # Deterministic RNG for this recipe
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        string = _sample_random_string(string_length, language=language, seed=seed)

        # Backspace schedule (typing mode only): each event is (t, n_chars_deleted)
        backspace_events = []
        if mode == "typing" and backspace_rate > 0:
            rng = random.Random(seed ^ 0xBACCE)
            for ti in range(T):
                if rng.random() < backspace_rate:
                    n = rng.randint(1, 4)
                    backspace_events.append([int(ti), int(n)])

        return {
            "enable": True,
            "mode": mode,
            "language": language,
            "string": string,
            "font_size": int(font_size),
            "family": family,
            "cps": float(cps),
            "backspace_events": backspace_events,
            "cursor_blink_hz": float(cursor_blink_hz),
            "scroll_pxpf": float(scroll_pxpf),
            "origin_xy": [int(origin_xy[0]), int(origin_xy[1])],
            "color": [float(c) for c in color],
        }

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _typing_substring_at(text_params, ti):
        """Compute the currently-typed substring at frame ti in typing mode."""
        full = text_params["string"]
        cps = float(text_params["cps"])
        target = min(int(cps * ti + 0.5), len(full))
        # Apply backspaces: each event deletes n chars from current target (not full)
        cur = target
        for t_evt, n in text_params.get("backspace_events", []):
            if t_evt <= ti:
                cur = max(0, cur - int(n))
        return full[:cur]

    # ------------------------------------------------------------------
    # Rasterize text to an alpha mask + tint canvas
    # ------------------------------------------------------------------
    def _rasterize_text(self, text, font, origin_xy, color, with_cursor=False):
        """Render `text` to a (H, W) alpha map and (3, H, W) color. CPU via PIL,
        then moved to device. Returns (alpha (1,H,W), rgb (3,H,W)) both [0,1]."""
        from PIL import Image, ImageDraw
        import numpy as np
        H, W = self.H, self.W
        img = Image.new("L", (W, H), 0)  # alpha-only mask
        draw = ImageDraw.Draw(img)
        x, y = int(origin_xy[0]), int(origin_xy[1])
        draw.text((x, y), text, fill=255, font=font)
        if with_cursor:
            # Measure current text width to append cursor
            try:
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = int(len(text) * font.size * 0.6)
            cx0 = x + tw
            # Cursor: a thin vertical bar at current typed position
            cursor_w = max(2, font.size // 10)
            draw.rectangle([cx0, y, cx0 + cursor_w, y + font.size], fill=255)
        arr = torch.from_numpy(np.array(img, dtype=np.uint8)).to(
            self.device).float() / 255.0  # (H, W)
        alpha = arr.unsqueeze(0)  # (1, H, W)
        c = torch.tensor(color, device=self.device, dtype=torch.float32).view(3, 1, 1)
        rgb = c.expand(3, H, W) * alpha  # pre-multiplied
        return alpha, rgb

    def _apply_text(self, canvas, ti, text_params):
        """Composite a text overlay onto canvas for frame ti.
        canvas: (B, 3, H, W) in [0, 1]. Returns same shape."""
        if text_params is None or not text_params.get("enable", False):
            return canvas
        B, C, H, W = canvas.shape
        mode = text_params.get("mode", "typing")
        font = self._get_font(text_params["font_size"], text_params.get("family", "mono"))
        color = text_params["color"]
        origin = list(text_params["origin_xy"])

        if mode == "typing":
            text = self._typing_substring_at(text_params, ti)
            # Cursor blinks at cursor_blink_hz cycles per frame
            blink = float(text_params.get("cursor_blink_hz", 2.0))
            # Period = 1/blink frames; visible on even halves
            period = max(1.0 / max(blink, 1e-4), 2.0)
            cursor_on = (int(ti % period) < int(period / 2))
            alpha, rgb = self._rasterize_text(
                text, font, origin, color, with_cursor=cursor_on)
        elif mode in ("scroll_left", "scroll_right"):
            direction = -1 if mode == "scroll_left" else 1
            scroll_pxpf = float(text_params.get("scroll_pxpf", 8.0))
            # Offset text horizontally based on ti
            dx = direction * int(scroll_pxpf * ti)
            origin[0] = int(origin[0] + dx)
            alpha, rgb = self._rasterize_text(
                text_params["string"], font, origin, color, with_cursor=False)
        else:
            return canvas

        # Alpha-composite onto each batch item
        alpha = alpha.unsqueeze(0)  # (1, 1, H, W)
        rgb = rgb.unsqueeze(0)      # (1, 3, H, W)
        return canvas * (1 - alpha) + rgb * alpha

#!/usr/bin/env python3
"""Isolate each generator effect on a minimal background so we can see
exactly what each one does. Outputs PNG per effect into examples/isolation/

Background: a soft linear gradient (gray to near-white). No shape bank,
no disco — just the raw effect call on a clean canvas.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generator import VAEpp0rGenerator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "isolation")
os.makedirs(OUT, exist_ok=True)

H, W = 360, 640


def save(canvas, name):
    img = (canvas[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    path = os.path.join(OUT, f"{name}.png")
    Image.fromarray(img).save(path)
    return path


def soft_bg(gen, textured=False):
    """Minimal backdrop. If textured, overlay a checkerboard so warps show."""
    device = gen.device
    y = torch.linspace(0.25, 0.7, H, device=device)  # (H,)
    # Build a (3, H, W) gradient then broadcast to batch
    r = (y * 0.7 + 0.25).view(H, 1).expand(H, W)
    g = (y * 0.5 + 0.3).view(H, 1).expand(H, W)
    b = (y * 0.9 + 0.15).view(H, 1).expand(H, W)
    c = torch.stack([r, g, b], dim=0).unsqueeze(0).clone()  # (1, 3, H, W)
    if textured:
        yy = torch.arange(H, device=device).view(H, 1)
        xx = torch.arange(W, device=device).view(1, W)
        cb = (((yy // 20) + (xx // 20)) % 2).float().unsqueeze(0).unsqueeze(0)
        c = c * 0.6 + cb * 0.4
    return c


def main():
    gen = VAEpp0rGenerator(H, W, device="cuda", bank_size=1, n_base_layers=1)
    # skip build_banks — we don't need shape bank for isolation

    print(f"Output: {OUT}\n")

    # === Fluid ripples (needs texture) ===
    bg_tex = soft_bg(gen, textured=True)
    save(bg_tex, "00_baseline_textured")

    for ws in [4, 8, 16, 32]:
        fp = gen._sample_fluid_recipe(
            T=1, n_drops=6, warp_strength=ws,
            amp_range=(0.05, 0.10),
            gerstner_amp_range=(0.03, 0.06),
        )
        out = gen._apply_ripples(bg_tex, 0, 1, fp)
        p = save(out, f"fluid__ws{ws:02d}")
        print(f"  {p}")

    # === Camera shake ===
    for amp in [0.02, 0.05, 0.1]:
        sp = gen._sample_shake_recipe(T=8, amp_xy=amp, amp_rot=amp)
        out = gen._apply_camera_shake(bg_tex, 4, sp)
        save(out, f"shake__amp{amp}")

    # === Kaleidoscope ===
    for n in [4, 6, 12]:
        kp = gen._sample_kaleido_recipe(n_slices=n, rot_per_frame=0.0)
        out = gen._apply_kaleidoscope(bg_tex, 0, kp)
        save(out, f"kaleido__n{n}")

    # === Flash / strobe ===
    fp = gen._sample_flash_recipe(T=8, n_flashes=1, strobe_rate=0.0)
    fp["flashes"][0]["t"] = 0
    for mode in ["white", "black", "invert", "color"]:
        fp["flashes"][0]["mode"] = mode
        fp["flashes"][0]["strength"] = 0.8
        out = gen._apply_flash(bg_tex, 0, fp)
        save(out, f"flash__{mode}")

    # === Palette cycle ===
    for shift in [0.15, 0.33, 0.5, 0.75]:
        pp = {"enable": True, "speed": 0.0, "phase0": shift, "sat_boost": 1.0}
        out = gen._apply_palette_cycle(bg_tex, 0, pp)
        save(out, f"palette__shift{shift}")

    # === Text overlay ===
    bg = soft_bg(gen, textured=False)
    for lang in ["latin", "cyrillic", "greek", "mixed"]:
        for size in [18, 32]:
            tp = gen._sample_text_recipe(
                T=1, mode="typing", language=lang,
                font_size=size, cps=100.0)  # huge cps so ti=0 shows full string
            out = gen._apply_text(bg, 0, tp)
            save(out, f"text__{lang}_size{size}")

    # === Signage (all 8 modes) ===
    for mode in ["led_matrix", "seven_seg", "marquee", "neon",
                 "ticker", "warning", "test_card", "loading"]:
        sp = gen._sample_signage_recipe(T=1, mode=mode, font_size=40)
        out = gen._apply_signage(bg.clone(), 0, sp)
        save(out, f"signage__{mode}")

    # === Particles (mid-life) ===
    for preset in ["confetti", "fireworks", "sparks", "snow", "rain", "embers"]:
        pp = gen._sample_particles_recipe(T=16, preset=preset, n_particles=300)
        out = gen._apply_particles(bg.clone(), 8, pp)
        save(out, f"particles__{preset}")

    # === Raymarch (on dark bg) ===
    dark = torch.zeros(1, 3, H, W, device=gen.device) + 0.05
    for n_sph in [1, 2, 3, 4]:
        rm = gen._sample_raymarch_recipe(T=1, n_spheres=n_sph, march_steps=32)
        out = gen._apply_raymarch(dark.clone(), 0, rm)
        save(out, f"raymarch__{n_sph}spheres")

    # === Arcade ===
    for mode in ["pong", "breakout", "invaders", "snake", "tetris", "asteroids"]:
        ap = gen._sample_arcade_recipe(T=24, mode=mode)
        out = gen._apply_arcade(bg.clone(), 12, ap)
        save(out, f"arcade__{mode}")

    # === Glitch / chromatic / scanlines ===
    gp = gen._sample_glitch_recipe(T=8, n_bursts=1)
    out = gen._apply_glitch(bg_tex.clone(), 0, gp)
    save(out, "glitch")

    cp = gen._sample_chromatic_recipe(T=1, strength=0.02)
    out = gen._apply_chromatic(bg_tex.clone(), 0, cp)
    save(out, "chromatic")

    sl = gen._sample_scanline_recipe(T=1, intensity=0.3, grain_strength=0.08)
    out = gen._apply_scanlines(bg_tex.clone(), 0, sl)
    save(out, "scanlines_grain")

    # === Extras ===
    fr = gen._sample_fire_recipe(T=1, intensity=0.9)
    out = gen._apply_fire(bg.clone(), 0, fr)
    save(out, "fire")

    vx = gen._sample_vortex_recipe(T=1, strength=0.8)
    out = gen._apply_vortex(bg_tex.clone(), 5, vx)
    save(out, "vortex")

    sf = gen._sample_starfield_recipe(T=24, n_stars=200)
    out = gen._apply_starfield(dark.clone(), 12, sf)
    save(out, "starfield")

    eq = gen._sample_eq_recipe(T=1, n_bars=24)
    out = gen._apply_eq_bars(bg.clone(), 0, eq)
    save(out, "eq_bars")

    print(f"\nDone. Files in {OUT}")


if __name__ == "__main__":
    main()

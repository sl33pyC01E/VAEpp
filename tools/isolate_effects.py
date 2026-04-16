#!/usr/bin/env python3
"""Isolate each generator effect on a minimal background so we can see
exactly what each one does. Outputs PNG (single frame) + MP4 (animated
T-frame loop) per effect into examples/isolation/

Background: a soft linear gradient (or textured checkerboard for warp-type
effects). No shape bank, no disco — just the raw effect call on a clean
canvas, once per frame.
"""

import os
import subprocess
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
T = 24  # frames per MP4


def save(canvas, name):
    img = (canvas[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    path = os.path.join(OUT, f"{name}.png")
    Image.fromarray(img).save(path)
    return path


def save_video(frames_bchw, name, fps=12):
    """frames_bchw: (T, 3, H, W) float [0,1]. Writes MP4 via ffmpeg pipe."""
    path = os.path.join(OUT, f"{name}.mp4")
    T_ = frames_bchw.shape[0]
    h_, w_ = frames_bchw.shape[-2], frames_bchw.shape[-1]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w_}x{h_}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-v", "error", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for ti in range(T_):
        frame = (frames_bchw[ti].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                 * 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return path


def run_temporal(apply_fn, backdrop, params, frames=T):
    """Call apply_fn(backdrop, ti, params) for each ti. Returns (T, 3, H, W)."""
    out = []
    for ti in range(frames):
        out.append(apply_fn(backdrop.clone(), ti, params)[0])
    return torch.stack(out, dim=0)


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


def _emit(name, png_canvas, video_frames=None):
    """Write PNG (single-frame) and optional MP4."""
    save(png_canvas, name)
    if video_frames is not None:
        save_video(video_frames, name)
    print(f"  {name}")


def main():
    gen = VAEpp0rGenerator(H, W, device="cuda", bank_size=1, n_base_layers=1)
    # skip build_banks — we don't need shape bank for isolation

    print(f"Output: {OUT}\n")

    bg_tex = soft_bg(gen, textured=True)
    bg = soft_bg(gen, textured=False)
    dark = torch.zeros(1, 3, H, W, device=gen.device) + 0.05
    save(bg_tex, "00_baseline_textured")
    save(bg, "00_baseline_plain")

    # === Fluid ripples ===
    for ws in [4, 8, 16, 32]:
        fp = gen._sample_fluid_recipe(
            T=T, n_drops=6, warp_strength=ws,
            amp_range=(0.05, 0.10),
            gerstner_amp_range=(0.03, 0.06),
        )
        png = gen._apply_ripples(bg_tex, 0, T, fp)
        vid = run_temporal(lambda c, ti, p: gen._apply_ripples(c, ti, T, p), bg_tex, fp)
        _emit(f"fluid__ws{ws:02d}", png, vid)

    # === Camera shake ===
    for mode in ["vibrate", "earthquake", "handheld"]:
        sp = gen._sample_shake_recipe(T=T, amp_xy=0.04, amp_rot=0.04, mode=mode)
        png = gen._apply_camera_shake(bg_tex, 0, sp)
        vid = run_temporal(gen._apply_camera_shake, bg_tex, sp)
        _emit(f"shake__{mode}", png, vid)

    # === Kaleidoscope ===
    for n in [4, 6, 12]:
        kp = gen._sample_kaleido_recipe(n_slices=n, rot_per_frame=0.06)
        png = gen._apply_kaleidoscope(bg_tex, 0, kp)
        vid = run_temporal(gen._apply_kaleidoscope, bg_tex, kp)
        _emit(f"kaleido__n{n}", png, vid)

    # === Flash / strobe ===
    # Mode-specific demo with a single flash at ti=4
    for mode in ["white", "black", "invert", "color"]:
        fp = gen._sample_flash_recipe(T=T, n_flashes=1, strobe_rate=0.0)
        fp["flashes"][0]["t"] = 4
        fp["flashes"][0]["mode"] = mode
        fp["flashes"][0]["strength"] = 0.9
        png = gen._apply_flash(bg_tex, 4, fp)
        vid = run_temporal(gen._apply_flash, bg_tex, fp)
        _emit(f"flash__{mode}", png, vid)
    # Strobe: periodic
    fp_strobe = gen._sample_flash_recipe(T=T, n_flashes=0, strobe_rate=3.0,
                                          strobe_strength=0.5)
    png = gen._apply_flash(bg_tex, 0, fp_strobe)
    vid = run_temporal(gen._apply_flash, bg_tex, fp_strobe)
    _emit("flash__strobe", png, vid)

    # === Palette cycle (animated hue rotation) ===
    pp = {"enable": True, "speed": 0.05, "phase0": 0.0, "sat_boost": 1.2}
    png = gen._apply_palette_cycle(bg_tex, 0, pp)
    vid = run_temporal(gen._apply_palette_cycle, bg_tex, pp)
    _emit("palette__cycle", png, vid)

    # === Text overlay (typing + scrolling) ===
    for lang in ["latin", "cyrillic", "greek", "mixed"]:
        tp = gen._sample_text_recipe(
            T=T, mode="typing", language=lang, font_size=32, cps=2.5)
        png = gen._apply_text(bg, T - 1, tp)  # end-of-clip = full string
        vid = run_temporal(gen._apply_text, bg, tp)
        _emit(f"text__typing_{lang}", png, vid)
    for direction in ["scroll_left", "scroll_right"]:
        tp = gen._sample_text_recipe(
            T=T, mode=direction, language="mixed",
            font_size=40, scroll_pxpf=20.0)
        png = gen._apply_text(bg, T // 2, tp)
        vid = run_temporal(gen._apply_text, bg, tp)
        _emit(f"text__{direction}", png, vid)

    # === Signage (all 8 modes animated) ===
    for mode in ["led_matrix", "seven_seg", "marquee", "neon",
                 "ticker", "warning", "test_card", "loading"]:
        sp = gen._sample_signage_recipe(T=T, mode=mode, font_size=40)
        png = gen._apply_signage(bg.clone(), T // 2, sp)
        vid = run_temporal(gen._apply_signage, bg, sp)
        _emit(f"signage__{mode}", png, vid)

    # === Particles ===
    for preset in ["confetti", "fireworks", "sparks", "snow", "rain", "embers"]:
        pp = gen._sample_particles_recipe(T=T, preset=preset, n_particles=300)
        png = gen._apply_particles(bg.clone(), T // 2, pp)
        vid = run_temporal(gen._apply_particles, bg, pp)
        _emit(f"particles__{preset}", png, vid)

    # === Raymarch (dark bg for contrast) ===
    for n_sph in [1, 2, 3, 4]:
        rm = gen._sample_raymarch_recipe(T=T, n_spheres=n_sph, march_steps=32)
        png = gen._apply_raymarch(dark.clone(), 0, rm)
        vid = run_temporal(gen._apply_raymarch, dark, rm)
        _emit(f"raymarch__{n_sph}spheres", png, vid)

    # === Arcade ===
    for mode in ["pong", "breakout", "invaders", "snake", "tetris", "asteroids"]:
        ap = gen._sample_arcade_recipe(T=T, mode=mode)
        png = gen._apply_arcade(bg.clone(), T // 2, ap)
        vid = run_temporal(gen._apply_arcade, bg, ap)
        _emit(f"arcade__{mode}", png, vid)

    # === Glitch / chromatic / scanlines ===
    gp = gen._sample_glitch_recipe(T=T, n_bursts=3)
    png = gen._apply_glitch(bg_tex.clone(), 0, gp)
    vid = run_temporal(gen._apply_glitch, bg_tex, gp)
    _emit("glitch", png, vid)

    cp = gen._sample_chromatic_recipe(T=T, strength=0.02)
    # Chromatic pulses; set a modest hz
    cp["pulse_hz"] = 0.1
    png = gen._apply_chromatic(bg_tex.clone(), 0, cp)
    vid = run_temporal(gen._apply_chromatic, bg_tex, cp)
    _emit("chromatic", png, vid)

    sl = gen._sample_scanline_recipe(T=T, intensity=0.3, grain_strength=0.08)
    png = gen._apply_scanlines(bg_tex.clone(), 0, sl)
    vid = run_temporal(gen._apply_scanlines, bg_tex, sl)
    _emit("scanlines_grain", png, vid)

    # === Extras ===
    fr = gen._sample_fire_recipe(T=T, intensity=0.9)
    png = gen._apply_fire(bg.clone(), 0, fr)
    vid = run_temporal(gen._apply_fire, bg, fr)
    _emit("fire", png, vid)

    vx = gen._sample_vortex_recipe(T=T, strength=0.8)
    png = gen._apply_vortex(bg_tex.clone(), 0, vx)
    vid = run_temporal(gen._apply_vortex, bg_tex, vx)
    _emit("vortex", png, vid)

    sf = gen._sample_starfield_recipe(T=T, n_stars=200)
    png = gen._apply_starfield(dark.clone(), T // 2, sf)
    vid = run_temporal(gen._apply_starfield, dark, sf)
    _emit("starfield", png, vid)

    eq = gen._sample_eq_recipe(T=T, n_bars=24)
    png = gen._apply_eq_bars(bg.clone(), 0, eq)
    vid = run_temporal(gen._apply_eq_bars, bg, eq)
    _emit("eq_bars", png, vid)

    print(f"\nDone. Files in {OUT}")


if __name__ == "__main__":
    main()

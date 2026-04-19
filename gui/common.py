#!/usr/bin/env python3
"""Shared GUI infrastructure — theme, helpers, process runner, chunked inference."""

import os
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk

# Pillow 10+ removed Image.BILINEAR; use Image.Resampling.BILINEAR instead
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAEPP_ROOT = PROJECT_ROOT  # bank/log paths should be project-relative, not gui-relative
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

# -- Theme ---------------------------------------------------------------------
BG       = "#0e0e16"
BG_PANEL = "#161622"
BG_INPUT = "#1c1c2e"
BG_LOG   = "#0a0a12"
FG       = "#c8c8d0"
FG_DIM   = "#55556a"
RED      = "#e05555"
GREEN    = "#44cc77"
BLUE     = "#5588dd"
ACCENT   = "#44aacc"

FONT       = ("Consolas", 10)
FONT_BOLD  = ("Consolas", 10, "bold")
FONT_TITLE = ("Consolas", 14, "bold")
FONT_SMALL = ("Consolas", 9)


# -- Process runner ------------------------------------------------------------
class ProcRunner:
    def __init__(self, log_widget):
        self.log = log_widget
        self.proc = None
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def run(self, cmd, cwd=None):
        # If a previous thread is still alive, give it up to 2 s to
        # finish — this covers the common "just killed, subprocess is
        # dying, reader thread is draining stdout" window. Earlier
        # check used `self.proc is None` which only becomes true
        # AFTER the finally block runs, so it never matched during
        # the window it was meant to catch.
        if self.running:
            self._thread.join(timeout=2.0)
        if self.running:
            self._append("[Already running]\n")
            return
        self.log.delete("1.0", tk.END)
        self._append(f"$ {' '.join(cmd)}\n\n")
        self._thread = threading.Thread(target=self._run, args=(cmd, cwd), daemon=True)
        self._thread.start()

    def _run(self, cmd, cwd):
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=cwd or PROJECT_ROOT, text=True, bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32" else 0)
            for line in self.proc.stdout:
                self._append(line)
            self.proc.wait()
            self._append(f"\n[exit {self.proc.returncode}]\n")
        except Exception as e:
            self._append(f"\nERROR: {e}\n")
        finally:
            if self.proc and self.proc.stdout:
                self.proc.stdout.close()
            self.proc = None

    def stop(self):
        proc = self.proc  # snapshot to avoid race with finally block
        if proc:
            try:
                if sys.platform == "win32":
                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                self._append("\n[Stopping...]\n")
            except (ProcessLookupError, OSError):
                pass

    def kill(self):
        proc = self.proc  # snapshot to avoid race with finally block
        if proc:
            if sys.platform == "win32":
                # Tree-kill: taskkill /F /T hits the entire process
                # tree so DataLoader workers, ffmpeg subprocesses,
                # etc. die atomically with the parent. Without this,
                # children hold stdout open and our reader thread
                # hangs, making the next Run spuriously fire
                # "[Already running]".
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True, timeout=5)
                except Exception:
                    pass
            # Belt + suspenders: proc.kill() in case taskkill missed
            # (wrong PATH, access denied, etc.) or the platform isn't
            # Windows. Idempotent if already dead.
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
            self._append("\n[Killed]\n")

    def _append(self, text):
        self.log.after(0, self._do_append, text)

    def _do_append(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)


# -- Helpers -------------------------------------------------------------------
def parse_arch_config(config):
    """Extract architecture params from checkpoint config dict.

    Returns (encoder_channels, decoder_channels) with proper types,
    falling back to defaults for old checkpoints.
    """
    enc_ch = config.get("encoder_channels", 64)
    if isinstance(enc_ch, str):
        if "," in enc_ch:
            enc_ch = tuple(int(x.strip()) for x in enc_ch.split(","))
        else:
            enc_ch = int(enc_ch)
    dec_ch = config.get("decoder_channels", (256, 128, 64))
    if isinstance(dec_ch, str):
        dec_ch = tuple(int(x.strip()) for x in dec_ch.split(","))
    return enc_ch, dec_ch


def run_with_log(tab, fn, on_done=None):
    """Run fn() on a background thread with stdout tee'd to tab.log.
    Polling starts on the main thread before the worker launches.
    on_done() is called on the main thread when fn() completes."""
    import queue as _q
    q = _q.Queue()
    original_stdout = sys.stdout
    done = [False]

    class _Tee:
        def write(self, s):
            if original_stdout:
                original_stdout.write(s)
            if s.strip():
                q.put(s.rstrip())
        def flush(self):
            if original_stdout:
                original_stdout.flush()

    def _poll():
        batch = []
        try:
            while True:
                batch.append(q.get_nowait())
        except _q.Empty:
            pass
        if batch:
            tab.log.insert(tk.END, "\n".join(batch) + "\n")
            tab.log.see(tk.END)
        if not done[0]:
            tab.after(50, _poll)
        else:
            # Final flush
            rest = []
            try:
                while True:
                    rest.append(q.get_nowait())
            except _q.Empty:
                pass
            if rest:
                tab.log.insert(tk.END, "\n".join(rest) + "\n")
                tab.log.see(tk.END)

    def _worker():
        sys.stdout = _Tee()
        try:
            fn()
        finally:
            sys.stdout = original_stdout
            done[0] = True
        if on_done:
            tab.after(0, on_done)

    # Start polling on main thread FIRST, then launch worker
    tab.after(50, _poll)
    threading.Thread(target=_worker, daemon=True).start()


def make_log(parent):
    return tk.Text(parent, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                   insertbackground=FG, height=10, wrap=tk.WORD,
                   borderwidth=0, highlightthickness=0)

def make_btn(parent, text, command, color=ACCENT, width=14):
    return tk.Button(parent, text=text, command=command,
                     bg=color, fg="#ffffff", font=FONT_BOLD,
                     activebackground=color, activeforeground="#ffffff",
                     borderwidth=0, padx=8, pady=3, width=width)

def make_spin(parent, label, from_=0, to=99999, default=0, width=8):
    frame = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL, anchor="w").pack(anchor="w")
    var = tk.IntVar(value=default)
    tk.Spinbox(frame, textvariable=var, from_=from_, to=to,
               bg=BG_INPUT, fg=FG, font=FONT, width=width,
               buttonbackground=BG_PANEL, borderwidth=0).pack(fill="x")
    return frame, var

def make_float(parent, label, default=0.0, width=8):
    frame = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL, anchor="w").pack(anchor="w")
    var = tk.StringVar(value=str(default))
    tk.Entry(frame, textvariable=var, bg=BG_INPUT, fg=FG, font=FONT,
             width=width, borderwidth=0, insertbackground=FG).pack(fill="x")
    return frame, var

def make_slider(parent, label, from_=0.0, to=1.0, default=0.5, resolution=0.01):
    frame = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL, anchor="w").pack(anchor="w")
    var = tk.DoubleVar(value=default)
    s = tk.Scale(frame, variable=var, from_=from_, to=to, resolution=resolution,
                 orient="horizontal", bg=BG_PANEL, fg=FG, troughcolor=BG_INPUT,
                 highlightthickness=0, font=FONT_SMALL, length=120,
                 activebackground=ACCENT, sliderrelief="flat")
    s.pack(fill="x")
    return frame, var


# -- Overfit-mode row ----------------------------------------------------------
# One-clip overfit toggle for training tabs. The training script is expected
# to support `--overfit-video PATH --overfit-frame-skip N` args which bypass
# the generator and train on a single cached clip (or single frame for
# static tabs). Video tabs use T frames starting at frame_skip; static tabs
# use just frame N.
def make_overfit_row(parent):
    """Return (frame, toggle_var, path_var, skip_var).

    Caller packs `frame` wherever it wants. In subprocess launcher,
    do: `if toggle_var.get() and path_var.get().strip(): cmd +=
    ["--overfit-video", path_var.get().strip(), "--overfit-frame-skip",
    str(skip_var.get())]`.
    """
    from tkinter import filedialog
    frame = tk.Frame(parent, bg=BG_PANEL)
    toggle_var = tk.BooleanVar(value=False)
    path_var = tk.StringVar(value="")
    skip_var = tk.IntVar(value=0)

    tk.Checkbutton(
        frame, text="Overfit on video:",
        variable=toggle_var, bg=BG_PANEL, fg=FG,
        selectcolor=BG_INPUT, font=FONT_SMALL,
        activebackground=BG_PANEL, activeforeground=FG,
    ).pack(side="left", padx=(0, 4))

    tk.Entry(
        frame, textvariable=path_var,
        bg=BG_INPUT, fg=FG, insertbackground=FG,
        font=FONT_SMALL,
    ).pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _pick():
        p = filedialog.askopenfilename(
            title="Select overfit video",
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm *.avi"),
                       ("All", "*.*")])
        if p:
            path_var.set(p)

    make_btn(frame, "Browse", _pick, BLUE, 8).pack(side="left", padx=(0, 4))
    make_btn(frame, "Clear", lambda: path_var.set(""), RED, 6).pack(
        side="left", padx=(0, 8))
    tk.Label(frame, text="Skip frames", bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL).pack(side="left", padx=(0, 4))
    tk.Spinbox(frame, from_=0, to=10**7, textvariable=skip_var,
               width=6, bg=BG_INPUT, fg=FG, insertbackground=FG,
               font=FONT_SMALL).pack(side="left")
    return frame, toggle_var, path_var, skip_var


# -- Inference saving helper ---------------------------------------------------
def save_inference_output(images, logdir, prefix="inference", label="output"):
    """Save inference images to logdir/inference/ with timestamp."""
    inf_dir = os.path.join(logdir, "inference")
    os.makedirs(inf_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, img_np in enumerate(images):
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = img_np.transpose(1, 2, 0)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
        pil = Image.fromarray(img_np)
        path = os.path.join(inf_dir, f"{prefix}_{label}_{ts}_{i:02d}.png")
        pil.save(path)
    return inf_dir


# -- Haar wavelet helpers -------------------------------------------------------

def haar_down(x):
    """2x spatial downscale: (B, C, H, W) -> (B, 4C, H/2, W/2). Lossless."""
    x = x.contiguous()
    a = x[:, :, 0::2, 0::2].contiguous()
    b = x[:, :, 0::2, 1::2].contiguous()
    c = x[:, :, 1::2, 0::2].contiguous()
    d = x[:, :, 1::2, 1::2].contiguous()
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)

def haar_down_video(x, n):
    """Apply haar_down_n to a (B, T, C, H, W) tensor without looping over T."""
    if n == 0:
        return x
    B, T, C, H, W = x.shape
    flat = x.reshape(B * T, C, H, W)
    for _ in range(n):
        flat = haar_down(flat)
    _, C2, H2, W2 = flat.shape
    return flat.reshape(B, T, C2, H2, W2)

def haar_up(x):
    """2x spatial upscale: (B, 4C, H, W) -> (B, C, H*2, W*2). Lossless inverse."""
    x = x.contiguous()
    C = x.shape[1] // 4
    ll = x[:, 0*C:1*C].contiguous()
    lh = x[:, 1*C:2*C].contiguous()
    hl = x[:, 2*C:3*C].contiguous()
    hh = x[:, 3*C:4*C].contiguous()
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    B, C2, H, W = ll.shape
    out = torch.zeros(B, C2, H * 2, W * 2, device=x.device, dtype=x.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out

def haar_up_video(x, n):
    """Apply haar_up_n to a (B, T, C, H, W) tensor without looping over T."""
    if n == 0:
        return x
    B, T, C, H, W = x.shape
    flat = x.reshape(B * T, C, H, W)
    for _ in range(n):
        flat = haar_up(flat)
    _, C2, H2, W2 = flat.shape
    return flat.reshape(B, T, C2, H2, W2)

def haar_down_n(x, n):
    for _ in range(n): x = haar_down(x)
    return x

def haar_up_n(x, n):
    for _ in range(n): x = haar_up(x)
    return x


# -- Chunked video inference helper --------------------------------------------
CHUNK_SIZE = 24

@torch.no_grad()
def chunked_vae_inference(model, x, chunk_size=None, amp_dtype=torch.bfloat16):
    """Run VAE encode+decode in overlapping chunks with cross-fade blending.

    chunk_size:
        - None (default): use `model.train_T` if it was set at load time
          (the T the checkpoint was trained at), else fall back to
          `max(2 * model.t_downscale, 8)`, aligned to a multiple of
          `t_downscale`.
        - int: explicit override, aligned to t_downscale.

    Uses large overlap (half the output length) and linearly blends in the
    overlap region to eliminate flicker at chunk boundaries.

    Returns:
        recon: aligned to input[trim:], so T_out = T_in - trim
        latent: concatenated latents from all chunks
    """
    T = x.shape[1]
    trim = getattr(model, 'frames_to_trim', 0)
    t_ds = getattr(model, 't_downscale', 1) or 1

    # Resolve chunk_size from the model's training T when not given
    if chunk_size is None:
        chunk_size = getattr(model, 'train_T', 0) or max(8, 2 * t_ds)
    # Align to a multiple of t_downscale so chunks cover full latent slots
    chunk_size = max(t_ds, (int(chunk_size) // t_ds) * t_ds)

    if T <= chunk_size:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon, latent = model(x)
        return recon, latent

    output_per_chunk = chunk_size - trim
    # Overlap by half the output length for smooth blending
    overlap = max(output_per_chunk // 2, trim)
    stride = output_per_chunk - overlap
    target_len = T - trim

    # Collect all chunk outputs with their positions
    chunks_out = []  # list of (start_frame, recon_tensor)
    all_latent = []

    chunk_start = 0
    while chunk_start < T:
        chunk_end = min(chunk_start + chunk_size, T)
        chunk = x[:, chunk_start:chunk_end]

        if chunk.shape[1] < model.t_downscale:
            break

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            rc, lat = model(chunk)

        # This chunk's output covers input frames [chunk_start+trim .. chunk_end)
        # which maps to output frames [chunk_start .. chunk_start + rc.shape[1])
        # (relative to the trim-offset output timeline)
        out_start = chunk_start  # in output frame space (0-indexed from trim)
        chunks_out.append((out_start, rc.float().cpu()))
        all_latent.append(lat.float().cpu())
        del rc, lat
        torch.cuda.empty_cache()

        if chunk_end >= T:
            break

        chunk_start += stride

    # Blend overlapping chunks
    if not chunks_out:
        return torch.zeros(x.shape[0], 0, *x.shape[2:]), torch.cat(all_latent, dim=1)

    # Allocate output buffer
    B = x.shape[0]
    sample_rc = chunks_out[0][1]
    C_out = sample_rc.shape[2]
    H_out = sample_rc.shape[3]
    W_out = sample_rc.shape[4]

    recon = torch.zeros(B, target_len, C_out, H_out, W_out)
    weight = torch.zeros(B, target_len, 1, 1, 1)

    for out_start, rc in chunks_out:
        n_frames = rc.shape[1]
        # Ramp weights: fade in at start, fade out at end
        w = torch.ones(n_frames)
        fade_len = min(overlap, n_frames // 2)
        if fade_len > 0 and out_start > 0:
            # Fade in (not for first chunk)
            w[:fade_len] = torch.linspace(0, 1, fade_len)
        # Note: we don't fade out — the next chunk's fade-in handles blending

        for t in range(n_frames):
            out_t = out_start + t
            if 0 <= out_t < target_len:
                wt = w[t]
                recon[:, out_t] += rc[:, t] * wt
                weight[:, out_t] += wt

    # Normalize by total weight
    weight = weight.clamp(min=1e-6)
    recon = recon / weight

    return recon, torch.cat(all_latent, dim=1)


@torch.no_grad()
def chunked_flatten_inference(vae, bottleneck, x, chunk_size=None,
                               amp_dtype=torch.bfloat16,
                               encode_fn=None, decode_fn=None):
    """Run VAE encode -> flatten/deflatten -> VAE decode in chunks with cross-fade.
    encode_fn/decode_fn override vae paths when provided (for FSQ).

    chunk_size defaults to the VAE's `train_T` (its training-time T), else
    `max(8, 2 * vae.t_downscale)`, aligned to a multiple of t_downscale.
    """
    _encode = encode_fn or (lambda c: vae.encode_video(c))
    _decode = decode_fn or (lambda z: vae.decode_video(z))
    T = x.shape[1]
    t_ds = getattr(vae, 't_downscale', 1) or 1
    if chunk_size is None:
        chunk_size = getattr(vae, 'train_T', 0) or max(8, 2 * t_ds)
    chunk_size = max(t_ds, (int(chunk_size) // t_ds) * t_ds)
    if T <= chunk_size:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = _encode(x)
            recon_vae = _decode(lat)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            lat_recon, _ = bottleneck(lat_flat)
            lat_recon = lat_recon.reshape(B, Tp, C, Hl, Wl)
            recon_flat = _decode(lat_recon)
        return recon_vae, recon_flat, lat

    trim = getattr(vae, 'frames_to_trim', 0)
    output_per_chunk = chunk_size - trim
    overlap = max(output_per_chunk // 2, trim)
    stride = output_per_chunk - overlap
    target_len = T - trim

    chunks_vae = []
    chunks_flat = []
    all_lat = []

    chunk_start = 0
    while chunk_start < T:
        chunk_end = min(chunk_start + chunk_size, T)
        chunk = x[:, chunk_start:chunk_end]

        if chunk.shape[1] < getattr(vae, 't_downscale', 1):
            break

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = _encode(chunk)
            rc_vae = _decode(lat)
            B, Tp, C, Hl, Wl = lat.shape
            lat_f = lat.reshape(B * Tp, C, Hl, Wl)
            lat_r, _ = bottleneck(lat_f)
            lat_r = lat_r.reshape(B, Tp, C, Hl, Wl)
            rc_flat = _decode(lat_r)

        chunks_vae.append((chunk_start, rc_vae.float().cpu()))
        chunks_flat.append((chunk_start, rc_flat.float().cpu()))
        all_lat.append(lat.float().cpu())
        del rc_vae, rc_flat, lat, lat_r, lat_f
        torch.cuda.empty_cache()

        if chunk_end >= T:
            break
        chunk_start += stride

    def _blend(chunks_list, target_len, B):
        if not chunks_list:
            return torch.zeros(B, 0)
        sample = chunks_list[0][1]
        C_o, H_o, W_o = sample.shape[2], sample.shape[3], sample.shape[4]
        out = torch.zeros(B, target_len, C_o, H_o, W_o)
        wgt = torch.zeros(B, target_len, 1, 1, 1)
        for out_start, rc in chunks_list:
            n = rc.shape[1]
            w = torch.ones(n)
            fade = min(overlap, n // 2)
            if fade > 0 and out_start > 0:
                w[:fade] = torch.linspace(0, 1, fade)
            for t in range(n):
                ot = out_start + t
                if 0 <= ot < target_len:
                    out[:, ot] += rc[:, t] * w[t]
                    wgt[:, ot] += w[t]
        return out / wgt.clamp(min=1e-6)

    B = x.shape[0]
    recon_vae = _blend(chunks_vae, target_len, B)
    recon_flat = _blend(chunks_flat, target_len, B)

    return recon_vae, recon_flat, torch.cat(all_lat, dim=1)


@torch.no_grad()
def chunked_fsq_inference(vae, fsq_layer, pre_quant, post_quant, x,
                           chunk_size=None, amp_dtype=torch.bfloat16):
    """Run VAE encode -> FSQ -> VAE decode in chunks with cross-fade.

    chunk_size defaults to the VAE's `train_T`, else `max(8, 2*t_downscale)`,
    aligned to a multiple of t_downscale. Half-chunk overlap preserved
    by the blend logic below.

    Returns:
        recon_fsq: FSQ reconstruction (aligned to input[trim:])
    """
    T = x.shape[1]
    t_ds = getattr(vae, 't_downscale', 1) or 1
    if chunk_size is None:
        chunk_size = getattr(vae, 'train_T', 0) or max(8, 2 * t_ds)
    chunk_size = max(t_ds, (int(chunk_size) // t_ds) * t_ds)
    if T <= chunk_size:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = vae.encode_video(x)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            z_proj = pre_quant(lat_flat)
            z_q, _ = fsq_layer(z_proj)
            lat_q = post_quant(z_q)
            lat_q = lat_q.reshape(B, Tp, C, Hl, Wl)
            recon_fsq = vae.decode_video(lat_q)
        return recon_fsq

    trim = getattr(vae, 'frames_to_trim', 0)
    output_per_chunk = chunk_size - trim
    overlap = max(output_per_chunk // 2, trim)
    stride = output_per_chunk - overlap
    target_len = T - trim

    chunks_out = []

    chunk_start = 0
    while chunk_start < T:
        chunk_end = min(chunk_start + chunk_size, T)
        chunk = x[:, chunk_start:chunk_end]

        if chunk.shape[1] < getattr(vae, 't_downscale', 1):
            break

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = vae.encode_video(chunk)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            z_proj = pre_quant(lat_flat)
            z_q, _ = fsq_layer(z_proj)
            lat_q = post_quant(z_q)
            lat_q = lat_q.reshape(B, Tp, C, Hl, Wl)
            rc_fsq = vae.decode_video(lat_q)

        chunks_out.append((chunk_start, rc_fsq.float().cpu()))
        del rc_fsq, lat, lat_flat, z_proj, z_q, lat_q
        torch.cuda.empty_cache()

        if chunk_end >= T:
            break
        chunk_start += stride

    if not chunks_out:
        return torch.zeros(x.shape[0], 0, *x.shape[2:])

    sample = chunks_out[0][1]
    B = x.shape[0]
    C_o, H_o, W_o = sample.shape[2], sample.shape[3], sample.shape[4]
    out = torch.zeros(B, target_len, C_o, H_o, W_o)
    wgt = torch.zeros(B, target_len, 1, 1, 1)

    for out_start, rc in chunks_out:
        n = rc.shape[1]
        w = torch.ones(n)
        fade = min(overlap, n // 2)
        if fade > 0 and out_start > 0:
            w[:fade] = torch.linspace(0, 1, fade)
        for t in range(n):
            ot = out_start + t
            if 0 <= ot < target_len:
                out[:, ot] += rc[:, t] * w[t]
                wgt[:, ot] += w[t]

    return out / wgt.clamp(min=1e-6)


# -- MiniVAE3D presets ---------------------------------------------------------
# Each preset configures a MiniVAE3D. Keys match the argparse flags of
# train_video3d.py. "params" is an approximate total parameter count for
# display-only purposes; the actual value is recomputed after instantiation.
MINIVAE3D_PRESETS = {
    "Custom": None,  # sentinel: no auto-populate

    "Cosmos-Large (~97M, 16s 8t, Haar+FSQ)": {
        "base_ch": 128, "ch_mult": "1,2,4,4", "num_res_blocks": 2,
        "temporal_down": "true,false,false,false",
        "spatial_down":  "true,true,false,false",
        "haar_levels": 2, "fsq": True, "latent_ch": 16,
        "params_hint": 97_280_118,
    },
    "Cosmos-Base (~97M, 16s 8t, Haar)": {
        "base_ch": 128, "ch_mult": "1,2,4,4", "num_res_blocks": 2,
        "temporal_down": "true,false,false,false",
        "spatial_down":  "true,true,false,false",
        "haar_levels": 2, "fsq": False, "latent_ch": 16,
        "params_hint": 97_279_904,
    },
    "Cosmos-Small (~25M, 16s 8t, Haar+FSQ)": {
        "base_ch": 64, "ch_mult": "1,2,4,4", "num_res_blocks": 2,
        "temporal_down": "true,false,false,false",
        "spatial_down":  "true,true,false,false",
        "haar_levels": 2, "fsq": True, "latent_ch": 16,
        "params_hint": 24_570_614,
    },
    "Medium-Haar (~26M, 16s 8t, Haar)": {
        "base_ch": 64, "ch_mult": "1,2,4,4", "num_res_blocks": 2,
        "temporal_down": "true,true,false,false",
        "spatial_down":  "true,true,true,false",
        "haar_levels": 1, "fsq": False, "latent_ch": 16,
        "params_hint": 25_693_968,
    },
    "Medium (~27M, 16s 8t, no Haar)": {
        "base_ch": 64, "ch_mult": "1,2,4,4", "num_res_blocks": 2,
        "temporal_down": "true,true,true,false",
        "spatial_down":  "true,true,true,true",
        "haar_levels": 0, "fsq": False, "latent_ch": 16,
        "params_hint": 27_241_921,
    },
    "Small-FSQ (~10M, 8s 4t, Haar+FSQ)": {
        "base_ch": 48, "ch_mult": "1,2,4", "num_res_blocks": 2,
        "temporal_down": "true,false,false",
        "spatial_down":  "true,true,false",
        "haar_levels": 1, "fsq": True, "latent_ch": 16,
        "params_hint": 10_058_886,
    },
    "Small-Haar (~10M, 8s 4t, Haar)": {
        "base_ch": 48, "ch_mult": "1,2,4", "num_res_blocks": 2,
        "temporal_down": "true,false,false",
        "spatial_down":  "true,true,false",
        "haar_levels": 1, "fsq": False, "latent_ch": 16,
        "params_hint": 10_058_672,
    },
    "Small (~11M, 8s 4t, no Haar)": {
        "base_ch": 48, "ch_mult": "1,2,4", "num_res_blocks": 2,
        "temporal_down": "true,true,false",
        "spatial_down":  "true,true,true",
        "haar_levels": 0, "fsq": False, "latent_ch": 16,
        "params_hint": 10_841_249,
    },
    "Tiny-Haar (~3.5M, 8s 4t, Haar)": {
        "base_ch": 32, "ch_mult": "1,2,4", "num_res_blocks": 1,
        "temporal_down": "true,false,false",
        "spatial_down":  "true,true,false",
        "haar_levels": 1, "fsq": False, "latent_ch": 16,
        "params_hint": 3_458_000,
    },
    "Tiny (~3.8M, 8s 4t, no Haar)": {
        "base_ch": 32, "ch_mult": "1,2,4", "num_res_blocks": 1,
        "temporal_down": "true,true,false",
        "spatial_down":  "true,true,true",
        "haar_levels": 0, "fsq": False, "latent_ch": 16,
        "params_hint": 3_800_961,
    },
    "Pico (~2.1M, 8s 2t, no Haar)": {
        "base_ch": 24, "ch_mult": "1,2,4", "num_res_blocks": 1,
        "temporal_down": "true,false,false",
        "spatial_down":  "true,true,true",
        "haar_levels": 0, "fsq": False, "latent_ch": 16,
        "params_hint": 2_113_121,
    },
}

MINIVAE3D_PRESET_NAMES = list(MINIVAE3D_PRESETS.keys())
MINIVAE3D_DEFAULT_PRESET = "Small-Haar (~10M, 8s 4t, Haar)"


# -- ElasticVideoTokenizer presets ---------------------------------------------
# Mirrors MINIVAE3D_PRESETS. Keys match the argparse flags of
# train_tokenizer.py. params_hint is a rough display-only estimate; actual
# count is computed by instantiating on meta device.
TOKENIZER_PRESETS = {
    "Custom": None,

    "Mini (~5M, N_q=64, d8)": {
        "n_queries": 64, "min_keep": 16,
        "dim": 192, "depth": 4, "heads": 4, "d_bottleneck": 8,
        "params_hint": 5_000_000,
    },
    "Small (~12M, N_q=128, d8)": {
        "n_queries": 128, "min_keep": 32,
        "dim": 256, "depth": 5, "heads": 4, "d_bottleneck": 8,
        "params_hint": 12_000_000,
    },
    "Medium (~25M, N_q=128, d8)": {
        "n_queries": 128, "min_keep": 32,
        "dim": 384, "depth": 6, "heads": 6, "d_bottleneck": 8,
        "params_hint": 25_000_000,
    },
    "Large (~60M, N_q=256, d12)": {
        "n_queries": 256, "min_keep": 64,
        "dim": 512, "depth": 8, "heads": 8, "d_bottleneck": 12,
        "params_hint": 60_000_000,
    },
}

TOKENIZER_PRESET_NAMES = list(TOKENIZER_PRESETS.keys())
TOKENIZER_DEFAULT_PRESET = "Medium (~25M, N_q=128, d8)"


def estimate_tokenizer_dims(n_queries: int, T: int, t_downscale: int,
                            d_bottleneck: int, H: int = 360, W: int = 640,
                            s_downscale: int = 16):
    """Compute tokens-per-clip for the tokenizer at the given VAE t_downscale.

    The tokenizer emits N_q query tokens per latent-temporal slot. VAE slot
    count for T frames is ceil(T / t_downscale) (Cosmos convention). Each
    token is a d_bottleneck-dim float vector in v1 (continuous bottleneck);
    when FSQ replaces the bottleneck later, bits/token becomes log2(codebook).
    """
    t_lat = max(1, -(-int(T) // max(int(t_downscale), 1)))
    total_tokens = int(n_queries) * t_lat
    # v1 bottleneck is continuous bf16 — d_bottleneck floats per token
    vals_per_token = int(d_bottleneck)
    total_vals = total_tokens * vals_per_token
    bits = total_vals * 16
    raw_bits = int(T) * H * W * 3 * 8
    ratio = raw_bits / bits if bits else 0.0
    label = (f"Tokenizer(T={int(T)}->T'={t_lat}): "
             f"{int(n_queries)}x{t_lat} = {total_tokens:,} tokens "
             f"({total_vals:,} floats, {bits/8/1024:.1f} KB)  |  "
             f"Raw {int(T)}f: {raw_bits/8/1024/1024:.1f} MB  |  "
             f"Compression: {ratio:.0f}x")
    return {
        "t_lat": t_lat,
        "total_tokens": total_tokens,
        "total_vals": total_vals,
        "bits": bits,
        "ratio": ratio,
        "label": label,
    }


_PARAM_CACHE = {}


def estimate_param_count(latent_ch, base_ch=None, ch_mult=None,
                         num_res_blocks=2,
                         temporal_down=None, spatial_down=None,
                         haar_levels=0, fsq=False,
                         fsq_levels=(8, 8, 8, 5, 5, 5), fsq_stages=4,
                         enc_channels=None, dec_channels=None,
                         use_attention=True, use_groupnorm=True,
                         residual_shortcut=False,
                         attn_heads=8, gn_groups=1):
    """Instantiate a MiniVAE3D on meta device and return exact param count.
    Results cached by config tuple.

    Two ways to specify the channel schedule:
      - `enc_channels` (shallow->deep) + `dec_channels` (deep->shallow), matching
        VideoTrainTab's Enc ch / Dec ch convention. Preferred.
      - Legacy `base_ch` + `ch_mult` (symmetric). `enc = base*mult`,
        `dec = reversed(enc)`.
    """
    import torch
    # Resolve explicit enc/dec
    if enc_channels is None or dec_channels is None:
        if base_ch is None or ch_mult is None:
            raise ValueError(
                "Provide either (enc_channels, dec_channels) or "
                "(base_ch, ch_mult) to estimate_param_count.")
        enc_channels = tuple(base_ch * int(m) for m in ch_mult)
        dec_channels = tuple(reversed(enc_channels))
    else:
        enc_channels = tuple(int(x) for x in enc_channels)
        dec_channels = tuple(int(x) for x in dec_channels)

    key = (latent_ch, enc_channels, dec_channels, num_res_blocks,
           tuple(temporal_down or ()), tuple(spatial_down or ()),
           haar_levels, bool(fsq), tuple(fsq_levels), fsq_stages,
           bool(use_attention), bool(use_groupnorm),
           bool(residual_shortcut), int(attn_heads), int(gn_groups))
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    try:
        from core.model import MiniVAE3D
        with torch.device("meta"):
            m = MiniVAE3D(
                latent_channels=latent_ch, image_channels=3, output_channels=3,
                enc_channels=enc_channels, dec_channels=dec_channels,
                num_res_blocks=num_res_blocks,
                temporal_downsample=tuple(temporal_down),
                spatial_downsample=tuple(spatial_down),
                haar_levels=haar_levels, fsq=fsq,
                fsq_levels=tuple(fsq_levels), fsq_stages=fsq_stages,
                use_attention=bool(use_attention),
                use_groupnorm=bool(use_groupnorm),
                residual_shortcut=bool(residual_shortcut),
                attn_heads=int(attn_heads), gn_groups=int(gn_groups),
            )
        p = sum(x.numel() for x in m.parameters())
    except Exception:
        p = 0
    _PARAM_CACHE[key] = p
    return p


def estimate_latent_dims(
    latent_ch: int, s_downscale: int, t_downscale: int,
    fsq: bool = False, fsq_levels=(8, 8, 8, 5, 5, 5), fsq_stages: int = 4,
    H: int = 360, W: int = 640,
    T: int | None = None,
):
    """Compute latent shape + per-slot AND per-clip dim count + compression.

    Per-slot math:
        h_lat, w_lat  - ceil(H/s_downscale), ceil(W/s_downscale)
        n_channels    - fsq_stages if fsq else latent_ch
        per_slot_vals = n_channels * h_lat * w_lat
        per_slot_bits = per_slot_vals * (log2 codebook if fsq else 16)

    Per-clip math (when T is provided):
        t_lat         - how many latent temporal slots for T input frames.
                        MiniVAE3D pads T up to a multiple of t_downscale
                        (see encode_video()'s `(-T) % t_downscale`), so
                        t_lat = ceil(T / t_downscale).
        clip_vals     = per_slot_vals * t_lat
        clip_bits     = per_slot_bits * t_lat
        raw_clip_bits = T * H * W * 3 * 8  (the real input, not padded)

    Raw bit baseline for compression:
        with T: use the actual T input frames (what the user sees)
        without T: use t_downscale frames (one-slot equivalent, legacy)

    Returns dict with all of the above plus a compact `label` string.
    """
    import math
    # Pad to nearest multiple of s_downscale (ceil)
    h_lat = -(-H // s_downscale)
    w_lat = -(-W // s_downscale)

    if fsq:
        n_channels = fsq_stages
        codebook = 1
        for lv in fsq_levels:
            codebook *= int(lv)
        bits_per_val = math.log2(codebook)
        kind = "tokens"
    else:
        n_channels = latent_ch
        bits_per_val = 16  # bf16/fp16
        kind = "values"

    per_slot_vals = n_channels * h_lat * w_lat
    per_slot_bits = per_slot_vals * bits_per_val

    if T is not None and T > 0:
        # encode_video pads: t_lat = ceil(T / t_downscale)
        t_lat = -(-int(T) // max(t_downscale, 1))
        clip_vals = per_slot_vals * t_lat
        clip_bits = per_slot_bits * t_lat
        raw_bits = int(T) * H * W * 3 * 8
        ratio = raw_bits / clip_bits if clip_bits > 0 else 0.0

        shape_str = f"{n_channels}x{t_lat}x{h_lat}x{w_lat}"
        kb = clip_bits / 8 / 1024
        raw_kb = raw_bits / 8 / 1024
        if raw_kb >= 1024:
            raw_str = f"{raw_kb/1024:.1f} MB"
        else:
            raw_str = f"{raw_kb:.0f} KB"
        label = (f"Latent(T={int(T)}->T'={t_lat}): "
                 f"{shape_str} = {clip_vals:,} {kind} ({kb:.1f} KB)  |  "
                 f"Raw {int(T)}f: {raw_str}  |  "
                 f"Compression: {ratio:.0f}x")
    else:
        # Legacy per-slot label (one latent temporal slot = t_downscale input frames)
        t_lat = None
        clip_vals = None
        clip_bits = None
        raw_t = max(t_downscale, 1)
        raw_bits = raw_t * H * W * 3 * 8
        ratio = raw_bits / per_slot_bits if per_slot_bits > 0 else 0.0
        shape_str = f"{n_channels}x{h_lat}x{w_lat}"
        label = (f"Latent/slot: {shape_str} = {per_slot_vals:,} {kind}  "
                 f"({per_slot_bits/8/1024:.1f} KB)  |  "
                 f"Raw {raw_t}f: {raw_bits/8/1024:.1f} KB  |  "
                 f"Compression: {ratio:.0f}x")

    return {
        "h_lat": h_lat, "w_lat": w_lat,
        "t_lat": t_lat,
        "per_slot_vals": per_slot_vals,
        "per_slot_bits": per_slot_bits,
        "clip_vals": clip_vals,
        "clip_bits": clip_bits,
        "raw_frame_bits": H * W * 3 * 8,
        "ratio": ratio,
        "label": label,
    }

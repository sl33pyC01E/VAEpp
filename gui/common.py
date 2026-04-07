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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAEPP_ROOT = os.path.dirname(os.path.abspath(__file__))
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
            try:
                proc.kill()
                self._append("\n[Killed]\n")
            except (ProcessLookupError, OSError):
                pass

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
        enc_ch = int(enc_ch)
    dec_ch = config.get("decoder_channels", (256, 128, 64))
    if isinstance(dec_ch, str):
        dec_ch = tuple(int(x.strip()) for x in dec_ch.split(","))
    return enc_ch, dec_ch


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


# -- Chunked video inference helper --------------------------------------------
CHUNK_SIZE = 24

@torch.no_grad()
def chunked_vae_inference(model, x, chunk_size=CHUNK_SIZE, amp_dtype=torch.bfloat16):
    """Run VAE encode+decode in overlapping chunks with 1:1 frame alignment.

    Each chunk of `chunk_size` input frames produces `chunk_size - trim`
    output frames (trim = frames_to_trim). The `trim` leading input frames
    are consumed as temporal context.

    To maintain alignment, each chunk's input overlaps the previous by `trim`
    frames so those context frames come from real data. Output stride equals
    `chunk_size - trim` frames per chunk.

    Returns:
        recon: aligned to input[trim:], so T_out = T_in - trim
        latent: concatenated latents from all chunks
    """
    T = x.shape[1]
    trim = getattr(model, 'frames_to_trim', 0)

    if T <= chunk_size:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon, latent = model(x)
        return recon, latent

    all_recon = []
    all_latent = []
    output_per_chunk = chunk_size - trim
    target_len = T - trim  # total recon frames we want

    chunk_start = 0
    collected = 0
    while chunk_start < T and collected < target_len:
        chunk_end = min(chunk_start + chunk_size, T)
        chunk = x[:, chunk_start:chunk_end]

        if chunk.shape[1] < model.t_downscale:
            break

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            rc, lat = model(chunk)

        # How many frames we still need
        need = target_len - collected
        keep = min(rc.shape[1], need)
        all_recon.append(rc[:, :keep].float().cpu())
        all_latent.append(lat.float().cpu())
        collected += keep
        del rc, lat
        torch.cuda.empty_cache()

        if chunk_end >= T or collected >= target_len:
            break

        chunk_start += output_per_chunk

    return torch.cat(all_recon, dim=1), torch.cat(all_latent, dim=1)


@torch.no_grad()
def chunked_flatten_inference(vae, bottleneck, x, chunk_size=CHUNK_SIZE,
                               amp_dtype=torch.bfloat16):
    """Run VAE encode → flatten/deflatten → VAE decode in chunks."""
    T = x.shape[1]
    if T <= chunk_size:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon_vae, _ = vae(x)
            lat = vae.encode_video(x)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            lat_recon, _ = bottleneck(lat_flat)
            lat_recon = lat_recon.reshape(B, Tp, C, Hl, Wl)
            recon_flat = vae.decode_video(lat_recon)
        return recon_vae, recon_flat, lat

    all_vae = []
    all_flat = []
    all_lat = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = x[:, start:end]
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            rc_vae, _ = vae(chunk)
            lat = vae.encode_video(chunk)
            B, Tp, C, Hl, Wl = lat.shape
            lat_f = lat.reshape(B * Tp, C, Hl, Wl)
            lat_r, _ = bottleneck(lat_f)
            lat_r = lat_r.reshape(B, Tp, C, Hl, Wl)
            rc_flat = vae.decode_video(lat_r)

        all_vae.append(rc_vae.float().cpu())
        all_flat.append(rc_flat.float().cpu())
        all_lat.append(lat.float().cpu())
        del rc_vae, rc_flat, lat, lat_r, lat_f
        torch.cuda.empty_cache()

    return torch.cat(all_vae, dim=1), torch.cat(all_flat, dim=1), \
           torch.cat(all_lat, dim=1)

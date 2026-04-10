#!/usr/bin/env python3
"""Standalone GUI for CPU VAE experiment.  4 tabs.

Tabs:
  - S1 Train:   train UnrolledPatchVAE (fresh / resume / extend)
  - Refiner:    latent smoothing refiner training
  - S2 Train:   flatten bottleneck training
  - Inference:  load any pipeline checkpoint, GT | Recon

Usage:
    python -m experiments.cpu_vae_gui
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gui.common import (
    BG, BG_PANEL, BG_INPUT, BG_LOG, FG, FG_DIM, RED, GREEN, BLUE, ACCENT,
    FONT, FONT_BOLD, FONT_TITLE, FONT_SMALL, BILINEAR,
    ProcRunner, make_log, make_btn, make_spin, make_float,
    VENV_PYTHON,
)

from PIL import Image, ImageTk


# =============================================================================
# Preview watcher mixin
# =============================================================================

class PreviewWatcher:
    """Mixin for tabs that auto-refresh a preview image."""

    def init_preview(self, preview_dir, label_widget):
        self._preview_dir = preview_dir
        self._preview_label = label_widget
        self._preview_photo = None
        self._preview_mtime = 0
        self._check_preview()

    def _check_preview(self):
        try:
            # Check both synthetic and real preview, show whichever is newer
            candidates = [
                os.path.join(self._preview_dir, "preview_latest.png"),
                os.path.join(self._preview_dir, "real_preview_latest.png"),
            ]
            best_path, best_mt = None, 0
            for p in candidates:
                if os.path.exists(p):
                    mt = os.path.getmtime(p)
                    if mt > best_mt:
                        best_path, best_mt = p, mt

            if best_path and best_mt > self._preview_mtime:
                self._preview_mtime = best_mt
                pil = Image.open(best_path)
                # Scale to fit available space
                w, h = pil.size
                try:
                    avail_w = self._preview_label.winfo_width()
                    avail_h = self._preview_label.winfo_height()
                    if avail_w < 100:
                        avail_w = self.winfo_width() - 20
                    if avail_h < 100:
                        avail_h = self.winfo_height() - 300
                except Exception:
                    avail_w, avail_h = 1000, 600
                scale = min(avail_w / w, avail_h / h, 1.0)
                if scale < 1.0:
                    pil = pil.resize((int(w * scale), int(h * scale)),
                                     BILINEAR)
                self._preview_photo = ImageTk.PhotoImage(pil)
                self._preview_label.config(image=self._preview_photo)
        except Exception:
            pass
        self.after(2000, self._check_preview)


# =============================================================================
# Helper: file browse entry row
# =============================================================================

def _make_file_row(parent, label_text, default="", width=50):
    """Create a labelled entry + Browse button for file paths.
    Returns (frame, StringVar)."""
    row = tk.Frame(parent, bg=BG_PANEL)
    f = tk.Frame(row, bg=BG_PANEL)
    tk.Label(f, text=label_text, bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL).pack(anchor="w")
    var = tk.StringVar(value=default)
    ef = tk.Frame(f, bg=BG_PANEL)
    tk.Entry(ef, textvariable=var, bg=BG_INPUT, fg=FG, font=FONT,
             width=width, borderwidth=0, insertbackground=FG
             ).pack(side="left", fill="x", expand=True)

    def _browse():
        path = filedialog.askopenfilename(
            title=f"Select {label_text}",
            filetypes=[("Checkpoint / Image",
                        "*.pt *.pth *.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            var.set(path)

    make_btn(ef, "Browse", _browse, ACCENT, width=7
             ).pack(side="left", padx=(5, 0))
    ef.pack(fill="x")
    f.pack(side="left", fill="x", expand=True)
    return row, var


def _make_preview_row(parent):
    """Preview image entry + Browse.  Returns (frame, StringVar)."""
    row = tk.Frame(parent, bg=BG_PANEL)
    var = tk.StringVar(value="")
    f = tk.Frame(row, bg=BG_PANEL)
    tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL).pack(anchor="w")
    ef = tk.Frame(f, bg=BG_PANEL)
    tk.Entry(ef, textvariable=var, bg=BG_INPUT, fg=FG, font=FONT,
             width=45, borderwidth=0, insertbackground=FG
             ).pack(side="left", fill="x", expand=True)

    def _browse():
        path = filedialog.askopenfilename(
            title="Select preview image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            var.set(path)

    make_btn(ef, "Browse", _browse, ACCENT, width=7
             ).pack(side="left", padx=(5, 0))
    ef.pack(fill="x")
    f.pack(side="left", fill="x", expand=True)
    return row, var


def _make_walk_dropdown(parent, default="hilbert"):
    """Walk order dropdown.  Returns (frame, StringVar)."""
    wf = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(wf, text="Walk order", bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL).pack(anchor="w")
    var = tk.StringVar(value=default)
    menu = tk.OptionMenu(wf, var, "raster", "hilbert", "morton")
    menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                activebackground=BG_PANEL, activeforeground=FG,
                highlightthickness=0, borderwidth=0)
    menu.pack(anchor="w")
    return wf, var


def _make_mode_dropdown(parent, default="fresh"):
    """Mode dropdown for S1 (fresh / resume / extend).  Returns (frame, StringVar)."""
    mf = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(mf, text="Mode", bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL).pack(anchor="w")
    var = tk.StringVar(value=default)
    menu = tk.OptionMenu(mf, var, "fresh", "resume", "extend")
    menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                activebackground=BG_PANEL, activeforeground=FG,
                highlightthickness=0, borderwidth=0)
    menu.pack(anchor="w")
    return mf, var


# =============================================================================
# Tab 1: S1TrainTab
# =============================================================================

class S1TrainTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="S1: Train UnrolledPatchVAE", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Train encoder/decoder. Fresh start, resume "
                 "training, or extend with new stage.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Mode + input checkpoint
        row0 = tk.Frame(top, bg=BG_PANEL)
        row0.pack(fill="x", pady=(5, 0))
        f, self.mode_var = _make_mode_dropdown(row0)
        f.pack(side="left", padx=(0, 10))
        fr, self.input_ckpt_var = _make_file_row(row0, "Input checkpoint")
        fr.pack(side="left", fill="x", expand=True)

        # Architecture row
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.patch_size = make_spin(row1, "Patch size", default=3)
        f.pack(side="left", padx=(0, 10))
        f, self.latent_ch = make_spin(row1, "Latent ch", default=3)
        f.pack(side="left", padx=(0, 10))
        f, self.inner_dim = make_spin(row1, "Inner dim", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.hidden_dim_var = make_spin(row1, "Hidden dim", default=32)
        f.pack(side="left", padx=(0, 10))
        f, self.overlap_var = make_spin(row1, "Overlap", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.post_kernel = make_spin(row1, "Post kernel", default=5)
        f.pack(side="left")

        # Training row
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row2, "LR", "2e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=30000)
        f.pack(side="left", padx=(0, 10))
        f, self.w_l1 = make_float(row2, "w_l1", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_mse = make_float(row2, "w_mse", "0.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_lpips = make_float(row2, "w_lpips", "0.5")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Save row
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row3, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row3, "Grad accum", default=1)
        f.pack(side="left")

        # Options row
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(0, 10))
        self.loose_load_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Loose load", variable=self.loose_load_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left")

        # Preview image
        row5, self.preview_img_var = _make_preview_row(top)
        row5.pack(fill="x", pady=(5, 0))

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        # Log at bottom
        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        # Preview fills remaining
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_logs"),
                          self.preview_label)

    def start(self):
        mode = self.mode_var.get()
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "s1",
               "--mode", mode,
               "--patch-size", str(self.patch_size.get()),
               "--latent-ch", str(self.latent_ch.get()),
               "--inner-dim", str(self.inner_dim.get()),
               "--hidden-dim", str(self.hidden_dim_var.get()),
               "--overlap", str(self.overlap_var.get()),
               "--post-kernel", str(self.post_kernel.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-l1", self.w_l1.get(),
               "--w-mse", self.w_mse.get(),
               "--w-lpips", self.w_lpips.get(),
               "--precision", self.prec_var.get(),
               "--save-every", str(self.save_every.get()),
               "--preview-every", str(self.preview_every.get()),
               "--grad-accum", str(self.grad_accum.get())]
        input_ckpt = self.input_ckpt_var.get().strip()
        if input_ckpt:
            cmd.extend(["--input-ckpt", input_ckpt])
        preview_img = self.preview_img_var.get().strip()
        if preview_img:
            cmd.extend(["--preview-image", preview_img])
        if self.fresh_opt_var.get():
            cmd.append("--fresh-opt")
        if self.loose_load_var.get():
            cmd.append("--loose-load")
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        stop_file = os.path.join(PROJECT_ROOT, "cpu_vae_logs", ".stop")
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch()

    def kill(self):
        self.runner.kill()


# =============================================================================
# Tab 2: RefinerTrainTab
# =============================================================================

class RefinerTrainTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Refiner: Latent Smoothing", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Residual Conv1d blocks on latent grid. "
                 "Smooths patch boundary artifacts.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Input checkpoint
        row0 = tk.Frame(top, bg=BG_PANEL)
        row0.pack(fill="x", pady=(5, 0))
        fr, self.input_ckpt_var = _make_file_row(
            row0, "Input checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_logs", "latest.pt"))
        fr.pack(side="left", fill="x", expand=True)

        # Config row
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))

        # Type dropdown
        tf = tk.Frame(row1, bg=BG_PANEL)
        tk.Label(tf, text="Type", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        self.refiner_type_var = tk.StringVar(value="attention")
        type_menu = tk.OptionMenu(tf, self.refiner_type_var,
                                  "attention", "conv1d")
        type_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                         activebackground=BG_PANEL, activeforeground=FG,
                         highlightthickness=0, borderwidth=0)
        type_menu.pack(anchor="w")
        tf.pack(side="left", padx=(0, 10))

        f, self.n_blocks = make_spin(row1, "Blocks", default=2)
        f.pack(side="left", padx=(0, 10))
        # Attention params
        f, self.n_heads = make_spin(row1, "Heads", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.embed_dim = make_spin(row1, "Embed dim", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.attn_patch = make_spin(row1, "Attn patch", default=3)
        f.pack(side="left", padx=(0, 10))
        f, self.attn_overlap = make_spin(row1, "Attn overlap", default=1)
        f.pack(side="left", padx=(0, 10))
        # Conv1d params
        f, self.hidden_ch = make_spin(row1, "Hidden ch", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.kernel_var = make_spin(row1, "Kernel", default=5)
        f.pack(side="left", padx=(0, 10))
        f, self.dropout_var = make_float(row1, "Dropout", "0.0")
        f.pack(side="left", padx=(0, 10))
        wf, self.walk_var = _make_walk_dropdown(row1)
        wf.pack(side="left")

        # Training row
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row2, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=10000)
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row2, "w_pixel", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_reg = make_float(row2, "w_reg", "0.01")
        f.pack(side="left", padx=(0, 10))
        f, self.blur_sigma = make_float(row2, "Blur sigma", "0.0")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Save row
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row3, "Save every", default=2000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row3, "Grad accum", default=1)
        f.pack(side="left")

        # Options row
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row4, "Resume checkpoint", "", width=40)
        f.pack(side="left", padx=(0, 10))
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(0, 10))
        self.finetune_dec_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Finetune decoder",
                       variable=self.finetune_dec_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left")

        # Preview image
        row5, self.preview_img_var = _make_preview_row(top)
        row5.pack(fill="x", pady=(5, 0))

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        # Log at bottom
        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        # Preview fills remaining
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_refiner_logs"),
                          self.preview_label)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "refiner",
               "--input-ckpt", self.input_ckpt_var.get(),
               "--refiner-type", self.refiner_type_var.get(),
               "--n-blocks", str(self.n_blocks.get()),
               "--n-heads", str(self.n_heads.get()),
               "--embed-dim", str(self.embed_dim.get()),
               "--attn-patch-size", str(self.attn_patch.get()),
               "--attn-patch-overlap", str(self.attn_overlap.get()),
               "--hidden-channels", str(self.hidden_ch.get()),
               "--kernel-size", str(self.kernel_var.get()),
               "--walk-order", self.walk_var.get(),
               "--dropout", self.dropout_var.get(),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-pixel", self.w_pix.get(),
               "--w-reg", self.w_reg.get(),
               "--blur-sigma", self.blur_sigma.get(),
               "--precision", self.prec_var.get(),
               "--save-every", str(self.save_every.get()),
               "--preview-every", str(self.preview_every.get()),
               "--grad-accum", str(self.grad_accum.get())]
        preview_img = self.preview_img_var.get().strip()
        if preview_img:
            cmd.extend(["--preview-image", preview_img])
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        if self.fresh_opt_var.get():
            cmd.append("--fresh-opt")
        if self.finetune_dec_var.get():
            cmd.append("--finetune-decoder")
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        stop_file = os.path.join(PROJECT_ROOT, "cpu_vae_refiner_logs", ".stop")
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch()

    def kill(self):
        self.runner.kill()


# =============================================================================
# Tab 3: S2TrainTab
# =============================================================================

class S2TrainTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="S2: Flatten Bottleneck", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Conv1d flatten/deflatten for 1D serialization.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Input checkpoint
        row0 = tk.Frame(top, bg=BG_PANEL)
        row0.pack(fill="x", pady=(5, 0))
        fr, self.input_ckpt_var = _make_file_row(
            row0, "Input checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_logs", "latest.pt"))
        fr.pack(side="left", fill="x", expand=True)

        # Config row
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.bottleneck_ch = make_spin(row1, "Bottleneck ch", default=1)
        f.pack(side="left", padx=(0, 10))
        wf, self.walk_var = _make_walk_dropdown(row1)
        wf.pack(side="left", padx=(0, 10))
        f, self.kernel_var = make_spin(row1, "Kernel", default=10)
        f.pack(side="left", padx=(0, 10))
        f, self.deflatten_hidden = make_spin(row1, "Defl hidden", default=0)
        f.pack(side="left")

        # Training row
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row2, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=10000)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lat = make_float(row2, "w_latent", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row2, "w_pixel", "0.5")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Resolution row
        row2b = tk.Frame(top, bg=BG_PANEL)
        row2b.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row2b, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row2b, "W", default=640)
        f.pack(side="left")

        # Save row
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row3, "Save every", default=2000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row3, "Grad accum", default=1)
        f.pack(side="left")

        # Options row
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row4, "Resume checkpoint", "", width=40)
        f.pack(side="left", padx=(0, 10))
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left")

        # Preview image
        row5, self.preview_img_var = _make_preview_row(top)
        row5.pack(fill="x", pady=(5, 0))

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        # Log at bottom
        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        # Preview fills remaining
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_flatten_logs"),
                          self.preview_label)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "s2",
               "--input-ckpt", self.input_ckpt_var.get(),
               "--bottleneck-ch", str(self.bottleneck_ch.get()),
               "--walk-order", self.walk_var.get(),
               "--kernel-size", str(self.kernel_var.get()),
               "--deflatten-hidden", str(self.deflatten_hidden.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--w-latent", self.w_lat.get(),
               "--w-pixel", self.w_pix.get(),
               "--precision", self.prec_var.get(),
               "--save-every", str(self.save_every.get()),
               "--preview-every", str(self.preview_every.get()),
               "--grad-accum", str(self.grad_accum.get())]
        preview_img = self.preview_img_var.get().strip()
        if preview_img:
            cmd.extend(["--preview-image", preview_img])
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        if self.fresh_opt_var.get():
            cmd.append("--fresh-opt")
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        stop_file = os.path.join(PROJECT_ROOT, "cpu_vae_flatten_logs", ".stop")
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch()

    def kill(self):
        self.runner.kill()


# =============================================================================
# Tab 4: InferenceTab
# =============================================================================

class InferenceTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._image_paths = []
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Inference", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Load any pipeline checkpoint. Shows GT | "
                 "Reconstruction with per-stage latency.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Checkpoint
        row0 = tk.Frame(top, bg=BG_PANEL)
        row0.pack(fill="x", pady=(5, 0))
        fr, self.ckpt_var = _make_file_row(
            row0, "Checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_logs", "latest.pt"))
        fr.pack(side="left", fill="x", expand=True)

        # Resolution
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row1, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row1, "W", default=640)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row1, "Precision", "bf16")
        f.pack(side="left")

        # Image browse label
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        self.files_label = tk.Label(
            row2, text="Images: (none — will use synthetic)",
            bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL)
        self.files_label.pack(side="left", fill="x", expand=True)

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Run", self.run_infer, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Browse", self.browse_images, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Clear", self.clear_images, BLUE).pack(side="left")

        # Log at bottom
        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        # Preview fills remaining
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_logs"),
                          self.preview_label)

    def browse_images(self):
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if paths:
            self._image_paths = list(paths)
            names = [os.path.basename(p) for p in self._image_paths]
            display = ", ".join(names[:4])
            if len(names) > 4:
                display += f" (+{len(names)-4} more)"
            self.files_label.config(text=f"Images: {display}")

    def clear_images(self):
        self._image_paths = []
        self.files_label.config(text="Images: (none — will use synthetic)")

    def run_infer(self):
        if self._image_paths:
            self._run_real_infer()
        else:
            # Synthetic: run via subprocess
            cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "infer",
                   "--ckpt", self.ckpt_var.get(),
                   "--H", str(self.H_var.get()),
                   "--W", str(self.W_var.get()),
                   "--precision", self.prec_var.get()]
            self.runner.run(cmd, cwd=PROJECT_ROOT)

    def _run_real_infer(self):
        """Run inference on browsed real images in-process."""
        import torch
        import numpy as np
        from experiments.cpu_vae import _load_pipeline

        ckpt_path = self.ckpt_var.get()
        H, W = self.H_var.get(), self.W_var.get()
        prec = self.prec_var.get()
        paths = list(self._image_paths)

        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, f"Loading {ckpt_path}...\n")

        def _work():
            amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                         "fp32": torch.float32}[prec]
            device = torch.device("cuda:0" if torch.cuda.is_available()
                                  else "cpu")
            pipeline = _load_pipeline(ckpt_path, device)
            print(f"Running on {len(paths)} real image(s)...", flush=True)

            from torchvision import transforms as T
            to_tensor = T.Compose([
                T.Resize((H, W)),
                T.ToTensor(),
            ])

            logdir = os.path.dirname(ckpt_path) or "cpu_vae_logs"
            os.makedirs(logdir, exist_ok=True)

            for img_path in paths:
                pil = Image.open(img_path).convert("RGB")
                x = to_tensor(pil).unsqueeze(0).to(device)  # [1,3,H,W]

                import time as _time
                t0 = _time.perf_counter()
                with torch.no_grad(), torch.amp.autocast(
                        device.type, dtype=amp_dtype):
                    z = pipeline.encode(x)
                    recon = pipeline.decode(z)
                dt = _time.perf_counter() - t0
                print(f"  {os.path.basename(img_path)}: "
                      f"{dt*1000:.1f}ms  latent {list(z.shape)}", flush=True)

                # Save GT | Recon side by side
                gt_np = x[0].cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
                rc_np = recon[0].cpu().float().clamp(0, 1).permute(
                    1, 2, 0).numpy()
                combo = np.concatenate([gt_np, rc_np], axis=1)
                combo = (combo * 255).astype(np.uint8)
                out_pil = Image.fromarray(combo)
                out_path = os.path.join(logdir, "real_preview_latest.png")
                out_pil.save(out_path)

            self._preview_mtime = 0  # force refresh
            self._preview_dir = logdir

        from gui.common import run_with_log
        run_with_log(self, _work)


# =============================================================================
# Main
# =============================================================================

def main():
    root = tk.Tk()
    root.title("CPU VAE Experiment")
    root.geometry("1100x800")
    root.configure(bg=BG)

    # Style for dark notebook tabs
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Dark.TNotebook", background=BG, borderwidth=0)
    style.configure("Dark.TNotebook.Tab",
                    background=BG_PANEL, foreground=FG,
                    padding=[12, 4], font=FONT_BOLD)
    style.map("Dark.TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])

    nb = ttk.Notebook(root, style="Dark.TNotebook")
    nb.pack(fill="both", expand=True, padx=5, pady=5)

    nb.add(S1TrainTab(nb), text="S1 Train")
    nb.add(RefinerTrainTab(nb), text="Refiner")
    nb.add(S2TrainTab(nb), text="S2 Train")
    nb.add(InferenceTab(nb), text="Inference")

    root.mainloop()


if __name__ == "__main__":
    main()

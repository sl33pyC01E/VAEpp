#!/usr/bin/env python3
"""Standalone GUI for CPU VAE experiment.

4 tabs:
  - Stage 1 Train: train PatchVAE end-to-end
  - Stage 1 Infer: load PatchVAE checkpoint, show GT | Recon
  - Stage 2 Train: train FlattenDeflatten on frozen PatchVAE
  - Stage 2 Infer: show GT | PatchVAE | Flatten

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
# Stage 1 Train Tab
# =============================================================================

class Stage1TrainTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Stage 1: Train PatchVAE", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Unfold+Linear encoder, Linear+Fold decoder. "
                 "No Conv2d. End-to-end training on synthetic data.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Row 1: architecture
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.patch_size = make_spin(row1, "Patch size", default=8)
        f.pack(side="left", padx=(0, 10))
        f, self.latent_ch = make_spin(row1, "Latent ch", default=32)
        f.pack(side="left", padx=(0, 10))
        f, self.hidden_dim = make_spin(row1, "Hidden dim", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.overlap_var = make_spin(row1, "Overlap", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.H_var = make_spin(row1, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row1, "W", default=640)
        f.pack(side="left")

        # Row 2: training params
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row2, "LR", "2e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=30000)
        f.pack(side="left", padx=(0, 10))
        f, self.w_mse = make_float(row2, "w_mse", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_lpips = make_float(row2, "w_lpips", "0.5")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Row 3: save/log
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row3, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row3, "Grad accum", default=1)
        f.pack(side="left")

        # Row 4: resume
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row4, "Resume checkpoint", "", width=50)
        f.pack(side="left", fill="x", expand=True)
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(10, 0))

        # Row 5: preview target image
        row5 = tk.Frame(top, bg=BG_PANEL)
        row5.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(row5, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        make_btn(ef, "Browse", self._browse_preview, ACCENT, width=7
                 ).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_logs"),
                          self.preview_label)

    def _browse_preview(self):
        path = filedialog.askopenfilename(
            title="Select preview image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self.preview_img_var.set(path)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "stage1",
               "--patch-size", str(self.patch_size.get()),
               "--latent-ch", str(self.latent_ch.get()),
               "--hidden-dim", str(self.hidden_dim.get()),
               "--overlap", str(self.overlap_var.get()),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-mse", self.w_mse.get(),
               "--w-lpips", self.w_lpips.get(),
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
        stop_file = os.path.join(PROJECT_ROOT, "cpu_vae_logs", ".stop")
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch()

    def kill(self):
        self.runner.kill()


# =============================================================================
# Stage 1 Inference Tab
# =============================================================================

class Stage1InferTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._image_paths = []
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Stage 1: PatchVAE Inference", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Load trained PatchVAE. Run on synthetic data or "
                 "browse real images for GT | Reconstruction.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.ckpt_var = make_float(row1, "PatchVAE checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row2, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row2, "W", default=640)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Image browse
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        self.files_label = tk.Label(row3, text="Images: (none — will use synthetic)",
                                    bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL)
        self.files_label.pack(side="left", fill="x", expand=True)

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Browse", self.browse_images, ACCENT).pack(side="left", padx=(0, 5))
        make_btn(btn, "Clear", self.clear_images, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Run", self.run_infer, GREEN).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

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
            cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "infer1",
                   "--patch-ckpt", self.ckpt_var.get(),
                   "--H", str(self.H_var.get()),
                   "--W", str(self.W_var.get()),
                   "--precision", self.prec_var.get()]
            self.runner.run(cmd, cwd=PROJECT_ROOT)

    def _run_real_infer(self):
        """Run inference on browsed real images in a background thread."""
        import torch
        from experiments.cpu_vae import _load_model, save_real_preview_stage1

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
            model, _, cfg = _load_model(ckpt_path, device)
            model.eval()
            logdir = os.path.dirname(ckpt_path) or "cpu_vae_logs"
            os.makedirs(logdir, exist_ok=True)
            print(f"Running on {len(paths)} real image(s)...", flush=True)
            out = save_real_preview_stage1(model, paths, H, W, logdir,
                                           device, amp_dtype)
            if out:
                self._preview_mtime = 0  # force refresh
                self._preview_dir = logdir

        from gui.common import run_with_log
        run_with_log(self, _work)


# =============================================================================
# Stage 2 Train Tab
# =============================================================================

class Stage2TrainTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Stage 2: Train Flatten Bottleneck", bg=BG_PANEL,
                 fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Freeze PatchVAE. Train Conv1d flatten/deflatten "
                 "bottleneck in latent space.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Row 1: checkpoints
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.patch_ckpt = make_float(row1, "Encoder checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row1b = tk.Frame(top, bg=BG_PANEL)
        row1b.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row1b, "Resume flatten checkpoint",
                                        "", width=50)
        f.pack(side="left", fill="x", expand=True)

        # Row 2: bottleneck config
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.bottleneck_ch = make_spin(row2, "Bottleneck ch", default=1)
        f.pack(side="left", padx=(0, 10))

        wf = tk.Frame(row2, bg=BG_PANEL)
        tk.Label(wf, text="Walk order", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        self.walk_var = tk.StringVar(value="hilbert")
        walk_menu = tk.OptionMenu(wf, self.walk_var, "raster", "hilbert",
                                  "morton")
        walk_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                         activebackground=BG_PANEL, activeforeground=FG,
                         highlightthickness=0, borderwidth=0)
        walk_menu.pack(anchor="w")
        wf.pack(side="left", padx=(0, 10))

        f, self.kernel_var = make_spin(row2, "Kernel", default=10)
        f.pack(side="left", padx=(0, 10))
        f, self.deflatten_hidden = make_spin(row2, "Defl hidden", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.lr_var = make_float(row2, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=10000)
        f.pack(side="left")

        # Row 3: loss weights, resolution, precision
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row3, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row3, "W", default=640)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lat = make_float(row3, "w_latent", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row3, "w_pixel", "0.5")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row3, "Precision", "bf16")
        f.pack(side="left")

        # Row 4: save/log
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row4, "Save every", default=2000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row4, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row4, "Grad accum", default=1)
        f.pack(side="left")
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(10, 0))

        # Row 5: preview target image
        row5 = tk.Frame(top, bg=BG_PANEL)
        row5.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(row5, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        make_btn(ef, "Browse", self._browse_preview, ACCENT, width=7
                 ).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_flatten_logs"),
                          self.preview_label)

    def _browse_preview(self):
        path = filedialog.askopenfilename(
            title="Select preview image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self.preview_img_var.set(path)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "stage2",
               "--patch-ckpt", self.patch_ckpt.get(),
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
# Stage 2 Inference Tab
# =============================================================================

class Stage2InferTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._image_paths = []
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Stage 2: Flatten Inference", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Encoder + FlattenDeflatten. Run on synthetic "
                 "or browse real images for GT | Encoder | Flatten.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.patch_ckpt = make_float(row1, "Encoder checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.flatten_ckpt = make_float(row2, "Flatten checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_flatten_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row3, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row3, "W", default=640)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row3, "Precision", "bf16")
        f.pack(side="left")

        # Image browse
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        self.files_label = tk.Label(row4, text="Images: (none — will use synthetic)",
                                    bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL)
        self.files_label.pack(side="left", fill="x", expand=True)

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Browse", self.browse_images, ACCENT).pack(side="left", padx=(0, 5))
        make_btn(btn, "Clear", self.clear_images, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Run", self.run_infer, GREEN).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_flatten_logs"),
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
            cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "infer2",
                   "--patch-ckpt", self.patch_ckpt.get(),
                   "--flatten-ckpt", self.flatten_ckpt.get(),
                   "--H", str(self.H_var.get()),
                   "--W", str(self.W_var.get()),
                   "--precision", self.prec_var.get()]
            self.runner.run(cmd, cwd=PROJECT_ROOT)

    def _run_real_infer(self):
        import torch
        from experiments.cpu_vae import _load_model, save_real_preview_stage2
        from experiments.flatten import FlattenDeflatten

        patch_ckpt = self.patch_ckpt.get()
        flatten_ckpt = self.flatten_ckpt.get()
        H, W = self.H_var.get(), self.W_var.get()
        prec = self.prec_var.get()
        paths = list(self._image_paths)

        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, f"Loading encoder + flatten...\n")

        def _work():
            amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                         "fp32": torch.float32}[prec]
            device = torch.device("cuda:0" if torch.cuda.is_available()
                                  else "cpu")
            patch_vae, _, cfg = _load_model(patch_ckpt, device)
            patch_vae.eval()

            fk = torch.load(flatten_ckpt, map_location="cpu",
                            weights_only=False)
            fcfg = fk.get("config", {})
            bottleneck = FlattenDeflatten(
                latent_channels=fcfg.get("latent_channels", 32),
                bottleneck_channels=fcfg.get("bottleneck_channels", 6),
                spatial_h=fcfg.get("spatial_h", H // 8),
                spatial_w=fcfg.get("spatial_w", W // 8),
                walk_order=fcfg.get("walk_order", "raster"),
                kernel_size=fcfg.get("kernel_size", 1),
                deflatten_hidden=fcfg.get("deflatten_hidden", 0),
            ).to(device)
            bottleneck.load_state_dict(fk["bottleneck"])
            bottleneck.eval()

            logdir = os.path.dirname(flatten_ckpt) or "cpu_vae_flatten_logs"
            os.makedirs(logdir, exist_ok=True)
            print(f"Running on {len(paths)} real image(s)...", flush=True)
            out = save_real_preview_stage2(patch_vae, bottleneck, paths,
                                            H, W, logdir, device, amp_dtype)
            if out:
                self._preview_mtime = 0
                self._preview_dir = logdir

        from gui.common import run_with_log
        run_with_log(self, _work)


# =============================================================================
# Unrolled Stage 1 Train Tab
# =============================================================================

class UnrolledTrainTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Stage 1: Train UnrolledPatchVAE", bg=BG_PANEL,
                 fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Unroll patches to pixel lines with positional "
                 "embeddings + channel IDs. Per-patch Conv1d compression.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Row 1: architecture
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.patch_size = make_spin(row1, "Patch size", default=8)
        f.pack(side="left", padx=(0, 10))
        f, self.latent_ch = make_spin(row1, "Latent ch", default=32)
        f.pack(side="left", padx=(0, 10))
        f, self.inner_dim = make_spin(row1, "Inner dim", default=8)
        f.pack(side="left", padx=(0, 10))
        f, self.hidden_dim_var = make_spin(row1, "Hidden dim", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.overlap_var = make_spin(row1, "Overlap", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.post_kernel = make_spin(row1, "Post kernel", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.H_var = make_spin(row1, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row1, "W", default=640)
        f.pack(side="left")

        # Row 2: training params
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row2, "LR", "2e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=30000)
        f.pack(side="left", padx=(0, 10))
        f, self.w_mse = make_float(row2, "w_mse", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_lpips = make_float(row2, "w_lpips", "0.5")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Row 3: save/log
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row3, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row3, "Grad accum", default=1)
        f.pack(side="left")

        # Row 4: resume
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row4, "Resume checkpoint", "", width=50)
        f.pack(side="left", fill="x", expand=True)
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(10, 0))
        self.loose_load_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Loose load", variable=self.loose_load_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(10, 0))

        # Row 5: preview target image
        row5 = tk.Frame(top, bg=BG_PANEL)
        row5.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(row5, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        make_btn(ef, "Browse", self._browse_preview, ACCENT, width=7
                 ).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs"),
                          self.preview_label)

    def _browse_preview(self):
        path = filedialog.askopenfilename(
            title="Select preview image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self.preview_img_var.set(path)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "stage1",
               "--model-type", "unrolled",
               "--patch-size", str(self.patch_size.get()),
               "--latent-ch", str(self.latent_ch.get()),
               "--inner-dim", str(self.inner_dim.get()),
               "--hidden-dim", str(self.hidden_dim_var.get()),
               "--overlap", str(self.overlap_var.get()),
               "--post-kernel", str(self.post_kernel.get()),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-mse", self.w_mse.get(),
               "--w-lpips", self.w_lpips.get(),
               "--precision", self.prec_var.get(),
               "--save-every", str(self.save_every.get()),
               "--preview-every", str(self.preview_every.get()),
               "--grad-accum", str(self.grad_accum.get()),
               "--logdir", "cpu_vae_unrolled_logs"]
        preview_img = self.preview_img_var.get().strip()
        if preview_img:
            cmd.extend(["--preview-image", preview_img])
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        if self.fresh_opt_var.get():
            cmd.append("--fresh-opt")
        if self.loose_load_var.get():
            cmd.append("--loose-load")
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        stop_file = os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs", ".stop")
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch()

    def kill(self):
        self.runner.kill()


# =============================================================================
# Unrolled Stage 1 Inference Tab
# =============================================================================

class UnrolledInferTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._image_paths = []
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Unrolled: Inference", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Load trained UnrolledPatchVAE. Run on synthetic "
                 "data or browse real images.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.ckpt_var = make_float(row1, "Checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row2, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row2, "W", default=640)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Image browse
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        self.files_label = tk.Label(row3, text="Images: (none — will use synthetic)",
                                    bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL)
        self.files_label.pack(side="left", fill="x", expand=True)

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Browse", self.browse_images, ACCENT).pack(side="left", padx=(0, 5))
        make_btn(btn, "Clear", self.clear_images, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Run", self.run_infer, GREEN).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs"),
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
            cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "infer1",
                   "--patch-ckpt", self.ckpt_var.get(),
                   "--H", str(self.H_var.get()),
                   "--W", str(self.W_var.get()),
                   "--precision", self.prec_var.get(),
                   "--logdir", "cpu_vae_unrolled_logs"]
            self.runner.run(cmd, cwd=PROJECT_ROOT)

    def _run_real_infer(self):
        import torch
        from experiments.cpu_vae import _load_model, save_real_preview_stage1

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
            model, _, cfg = _load_model(ckpt_path, device)
            model.eval()
            logdir = os.path.dirname(ckpt_path) or "cpu_vae_unrolled_logs"
            os.makedirs(logdir, exist_ok=True)
            print(f"Running on {len(paths)} real image(s)...", flush=True)
            out = save_real_preview_stage1(model, paths, H, W, logdir,
                                           device, amp_dtype)
            if out:
                self._preview_mtime = 0
                self._preview_dir = logdir

        from gui.common import run_with_log
        run_with_log(self, _work)


# =============================================================================
# Stage 1.5 Train Tab
# =============================================================================

class Stage15TrainTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Stage 1.5: Cascaded Spatial Compression",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Freeze S1. Train second UnrolledPatchVAE on "
                 "S1 latent grid for spatial compression.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Row 1: S1 checkpoint
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.s1_ckpt = make_float(row1, "S1 checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        # Row 2: S1.5 architecture
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.patch_size = make_spin(row2, "Patch size", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.latent_ch = make_spin(row2, "Latent ch", default=3)
        f.pack(side="left", padx=(0, 10))
        f, self.inner_dim = make_spin(row2, "Inner dim", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.hidden_dim_var = make_spin(row2, "Hidden dim", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.overlap_var = make_spin(row2, "Overlap", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.post_kernel = make_spin(row2, "Post kernel", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.H_var = make_spin(row2, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row2, "W", default=640)
        f.pack(side="left")

        # Row 3: training params
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row3, "LR", "2e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row3, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row3, "Steps", default=20000)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lat = make_float(row3, "w_latent", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row3, "w_pixel", "0.5")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row3, "Precision", "bf16")
        f.pack(side="left")

        # Row 4: save/log
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row4, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row4, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row4, "Grad accum", default=1)
        f.pack(side="left")

        # Row 5: resume
        row5 = tk.Frame(top, bg=BG_PANEL)
        row5.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row5, "Resume checkpoint", "", width=50)
        f.pack(side="left", fill="x", expand=True)
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row5, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(10, 0))
        self.loose_load_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row5, text="Loose load", variable=self.loose_load_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(10, 0))

        # Row 6: preview image
        row6 = tk.Frame(top, bg=BG_PANEL)
        row6.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(row6, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        make_btn(ef, "Browse", self._browse_preview, ACCENT, width=7
                 ).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_s1_5_logs"),
                          self.preview_label)

    def _browse_preview(self):
        path = filedialog.askopenfilename(
            title="Select preview image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self.preview_img_var.set(path)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "stage1_5",
               "--s1-ckpt", self.s1_ckpt.get(),
               "--patch-size", str(self.patch_size.get()),
               "--latent-ch", str(self.latent_ch.get()),
               "--inner-dim", str(self.inner_dim.get()),
               "--hidden-dim", str(self.hidden_dim_var.get()),
               "--overlap", str(self.overlap_var.get()),
               "--post-kernel", str(self.post_kernel.get()),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
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
        if self.loose_load_var.get():
            cmd.append("--loose-load")
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        stop_file = os.path.join(PROJECT_ROOT, "cpu_vae_s1_5_logs", ".stop")
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch()

    def kill(self):
        self.runner.kill()


# =============================================================================
# Stage 1.5 Inference Tab
# =============================================================================

class Stage15InferTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._image_paths = []
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Stage 1.5: Inference", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="S1 + S1.5 cascaded. "
                 "Shows GT | S1 Recon | S1.5 Recon.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.s1_ckpt = make_float(row1, "S1 checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.s1_5_ckpt = make_float(row2, "S1.5 checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_s1_5_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row3, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row3, "W", default=640)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row3, "Precision", "bf16")
        f.pack(side="left")

        # Image browse
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        self.files_label = tk.Label(row4,
            text="Images: (none — will use synthetic)",
            bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL)
        self.files_label.pack(side="left", fill="x", expand=True)

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Browse", self.browse_images, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Clear", self.clear_images, BLUE).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Run", self.run_infer, GREEN).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_s1_5_logs"),
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
            cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "infer1_5",
                   "--s1-ckpt", self.s1_ckpt.get(),
                   "--s1-5-ckpt", self.s1_5_ckpt.get(),
                   "--H", str(self.H_var.get()),
                   "--W", str(self.W_var.get()),
                   "--precision", self.prec_var.get()]
            self.runner.run(cmd, cwd=PROJECT_ROOT)

    def _run_real_infer(self):
        import torch
        from experiments.cpu_vae import (_load_model, load_real_images,
                                         save_preview_stage1_5)

        s1_path = self.s1_ckpt.get()
        s1_5_path = self.s1_5_ckpt.get()
        H, W = self.H_var.get(), self.W_var.get()
        prec = self.prec_var.get()
        paths = list(self._image_paths)

        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, f"Loading S1 + S1.5...\n")

        def _work():
            amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                         "fp32": torch.float32}[prec]
            device = torch.device("cuda:0" if torch.cuda.is_available()
                                  else "cpu")
            s1, _, _ = _load_model(s1_path, device)
            s1.eval()
            s1_5, _, _ = _load_model(s1_5_path, device)
            s1_5.eval()

            # Build a minimal gen-like object for preview
            class _FakeGen:
                pass
            fg = _FakeGen()
            fg.H, fg.W = H, W
            fg.generate = lambda n: load_real_images(paths[:n], H, W, device)

            logdir = os.path.dirname(s1_5_path) or "cpu_vae_s1_5_logs"
            os.makedirs(logdir, exist_ok=True)
            print(f"Running on {len(paths)} real image(s)...", flush=True)
            save_preview_stage1_5(s1, s1_5, fg, logdir, 0,
                                   device, amp_dtype)
            self._preview_mtime = 0
            self._preview_dir = logdir

        from gui.common import run_with_log
        run_with_log(self, _work)


# =============================================================================
# Main window
# =============================================================================

# =============================================================================
# Refiner Train Tab
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
                 "Smooths patch boundary artifacts without changing dims.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        # Row 1: upstream checkpoints
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.s1_ckpt = make_float(row1, "S1 checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row1b = tk.Frame(top, bg=BG_PANEL)
        row1b.pack(fill="x", pady=(5, 0))
        f, self.s1_5_ckpt = make_float(row1b, "S1.5 checkpoint (optional)",
            os.path.join(PROJECT_ROOT, "cpu_vae_s1_5_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        # Row 2: refiner config
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.n_blocks = make_spin(row2, "Blocks", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.kernel_var = make_spin(row2, "Kernel", default=5)
        f.pack(side="left", padx=(0, 10))

        wf = tk.Frame(row2, bg=BG_PANEL)
        tk.Label(wf, text="Walk order", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        self.walk_var = tk.StringVar(value="hilbert")
        walk_menu = tk.OptionMenu(wf, self.walk_var, "raster", "hilbert",
                                  "morton")
        walk_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                         activebackground=BG_PANEL, activeforeground=FG,
                         highlightthickness=0, borderwidth=0)
        walk_menu.pack(anchor="w")
        wf.pack(side="left", padx=(0, 10))

        f, self.H_var = make_spin(row2, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row2, "W", default=640)
        f.pack(side="left")

        # Row 3: training params
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row3, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row3, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row3, "Steps", default=10000)
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row3, "w_pixel", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_reg = make_float(row3, "w_reg", "0.1")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row3, "Precision", "bf16")
        f.pack(side="left")

        # Row 4: save/log
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row4, "Save every", default=2000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row4, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.grad_accum = make_spin(row4, "Grad accum", default=1)
        f.pack(side="left")

        # Row 5: resume
        row5 = tk.Frame(top, bg=BG_PANEL)
        row5.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row5, "Resume checkpoint", "", width=50)
        f.pack(side="left", fill="x", expand=True)
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row5, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(10, 0))

        # Row 6: preview image
        row6 = tk.Frame(top, bg=BG_PANEL)
        row6.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(row6, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        make_btn(ef, "Browse", self._browse_preview, ACCENT, width=7
                 ).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

        self.runner = ProcRunner(self.log)
        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_refiner_logs"),
                          self.preview_label)

    def _browse_preview(self):
        path = filedialog.askopenfilename(
            title="Select preview image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self.preview_img_var.set(path)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "refiner",
               "--s1-ckpt", self.s1_ckpt.get(),
               "--n-blocks", str(self.n_blocks.get()),
               "--kernel-size", str(self.kernel_var.get()),
               "--walk-order", self.walk_var.get(),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-pixel", self.w_pix.get(),
               "--w-reg", self.w_reg.get(),
               "--precision", self.prec_var.get(),
               "--save-every", str(self.save_every.get()),
               "--preview-every", str(self.preview_every.get()),
               "--grad-accum", str(self.grad_accum.get())]
        s1_5 = self.s1_5_ckpt.get().strip()
        if s1_5 and os.path.exists(s1_5):
            cmd.extend(["--s1-5-ckpt", s1_5])
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
        stop_file = os.path.join(PROJECT_ROOT, "cpu_vae_refiner_logs", ".stop")
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch()

    def kill(self):
        self.runner.kill()


# =============================================================================
# Refiner Inference Tab
# =============================================================================

class RefinerInferTab(tk.Frame, PreviewWatcher):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._image_paths = []
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Refiner: Inference", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Shows GT | Raw decode | Refined decode.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.s1_ckpt = make_float(row1, "S1 checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_unrolled_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row1b = tk.Frame(top, bg=BG_PANEL)
        row1b.pack(fill="x", pady=(5, 0))
        f, self.s1_5_ckpt = make_float(row1b, "S1.5 checkpoint (optional)",
            os.path.join(PROJECT_ROOT, "cpu_vae_s1_5_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.refiner_ckpt = make_float(row2, "Refiner checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_refiner_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(row3, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(row3, "W", default=640)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row3, "Precision", "bf16")
        f.pack(side="left")

        # Image browse
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        self.files_label = tk.Label(row4,
            text="Images: (none — will use synthetic)",
            bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL)
        self.files_label.pack(side="left", fill="x", expand=True)

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Browse", self.browse_images, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Clear", self.clear_images, BLUE).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Run", self.run_infer, GREEN).pack(side="left")

        self.log = tk.Text(self, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                           insertbackground=FG, height=6, wrap=tk.WORD,
                           borderwidth=0, highlightthickness=0)
        self.log.pack(fill="x", side="bottom", padx=5, pady=5)

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

        self.runner = ProcRunner(self.log)
        self.init_preview(os.path.join(PROJECT_ROOT, "cpu_vae_refiner_logs"),
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
        cmd = [VENV_PYTHON, "-m", "experiments.cpu_vae", "infer_refiner",
               "--s1-ckpt", self.s1_ckpt.get(),
               "--refiner-ckpt", self.refiner_ckpt.get(),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--precision", self.prec_var.get()]
        s1_5 = self.s1_5_ckpt.get().strip()
        if s1_5 and os.path.exists(s1_5):
            cmd.extend(["--s1-5-ckpt", s1_5])
        self.runner.run(cmd, cwd=PROJECT_ROOT)


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

    nb.add(Stage1TrainTab(nb), text="S1 Train")
    nb.add(Stage1InferTab(nb), text="S1 Infer")
    nb.add(UnrolledTrainTab(nb), text="Unrolled Train")
    nb.add(UnrolledInferTab(nb), text="Unrolled Infer")
    nb.add(Stage15TrainTab(nb), text="S1.5 Train")
    nb.add(Stage15InferTab(nb), text="S1.5 Infer")
    nb.add(RefinerTrainTab(nb), text="Refiner Train")
    nb.add(RefinerInferTab(nb), text="Refiner Infer")
    nb.add(Stage2TrainTab(nb), text="S2 Train")
    nb.add(Stage2InferTab(nb), text="S2 Infer")

    root.mainloop()


if __name__ == "__main__":
    main()

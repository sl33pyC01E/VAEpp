#!/usr/bin/env python3
"""Compression tabs -- Flatten, Flatten Inference, FSQ."""

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageTk

from gui.common import *

class FlattenTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._preview_photo = None
        self._last_mtime = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Flatten/Deflatten Experiment", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Freeze VAE encoder+decoder. Train 1D kernel-1 conv "
                 "bottleneck in latent space.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL,
                 justify="left").pack(anchor="w", pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.vae_ckpt = make_float(row1, "VAE checkpoint",
            os.path.join(PROJECT_ROOT, "synthyper_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row1b = tk.Frame(top, bg=BG_PANEL)
        row1b.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row1b, "Resume", "", width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.bottleneck_ch = make_spin(row2, "Bottleneck ch", default=6)
        f.pack(side="left", padx=(0, 10))
        f, self.walk_var = make_float(row2, "Walk order", "raster")
        f.pack(side="left", padx=(0, 10))
        f, self.lr_var = make_float(row2, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=10000)
        f.pack(side="left")

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.w_lat = make_float(row3, "w_latent", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row3, "w_pixel", 0.5)
        f.pack(side="left", padx=(0, 10))
        f, self.log_every = make_spin(row3, "Log", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview", default=200)
        f.pack(side="left")

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        # Preview: GT | VAE | Flatten
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)

        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)

        self._check_preview()

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.flatten",
               "--vae-ckpt", self.vae_ckpt.get(),
               "--bottleneck-ch", str(self.bottleneck_ch.get()),
               "--walk-order", self.walk_var.get(),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-latent", self.w_lat.get(),
               "--w-pixel", self.w_pix.get(),
               "--log-every", str(self.log_every.get()),
               "--preview-every", str(self.preview_every.get())]
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        logdir = os.path.join(PROJECT_ROOT, "flatten_logs")
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()

    def _check_preview(self):
        preview = os.path.join(PROJECT_ROOT, "flatten_logs", "preview_latest.png")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    img = Image.open(preview)
                    scale = min(900 / img.width, 400 / img.height, 1.0)
                    if scale < 1:
                        img = img.resize((int(img.width * scale),
                                          int(img.height * scale)),
                                         Image.BILINEAR)
                    self._preview_photo = ImageTk.PhotoImage(img)
                    self.preview_label.config(image=self._preview_photo)
            except Exception:
                pass
        self.after(5000, self._check_preview)


# -- Flatten Inference Tab -----------------------------------------------------

class FlattenInferenceTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.vae = None
        self.bottleneck = None
        self._preview_photo = None
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Flatten Inference (GT | VAE | Flatten)",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(10, 0))
        f, self.vae_ckpt = make_float(row1, "VAE checkpoint",
            os.path.join(PROJECT_ROOT, "synthyper_logs", "latest.pt"), width=40)
        f.pack(side="left", fill="x", expand=True, padx=(0, 5))
        f, self.bn_ckpt = make_float(row1, "Bottleneck ckpt",
            os.path.join(PROJECT_ROOT, "flatten_logs", "latest.pt"), width=40)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.input_path = make_float(row2, "Image/folder (drop or browse)",
                                        "", width=40)
        f.pack(side="left", fill="x", expand=True, padx=(0, 5))
        make_btn(row2, "Browse", self._browse_image, FG_DIM, width=7).pack(
            side="left", pady=(15, 0))

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        make_btn(row3, "Load", self.load_model, GREEN, width=8).pack(
            side="left", padx=(0, 10), pady=(0, 0))
        make_btn(row3, "Test Synthetic", self.test_synthetic, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(row3, "Test Image", self.test_image, BLUE).pack(
            side="left")

        self.status = tk.Label(top, text="No model loaded", bg=BG_PANEL,
                                fg=FG_DIM, font=FONT_SMALL)
        self.status.pack(fill="x", pady=(5, 0))

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

        # Enable drag-and-drop (files dropped onto the input field)
        self._setup_drop()

    def _setup_drop(self):
        """Try to enable tkdnd drag-and-drop. Falls back silently."""
        try:
            # tkinterdnd2 must be installed: pip install tkinterdnd2
            from tkinterdnd2 import DND_FILES
            # Need TkinterDnD root — check if available
            root = self.winfo_toplevel()
            if hasattr(root, 'drop_target_register'):
                self.drop_target_register(DND_FILES)
                self.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass  # No drag-and-drop support — that's fine

    def _on_drop(self, event):
        path = event.data.strip().strip('{}')
        self.input_path.set(path)

    def _browse_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All", "*.*")])
        if path:
            self.input_path.set(path)

    def load_model(self):
        sys.path.insert(0, PROJECT_ROOT)
        from core.model import MiniVAE
        from experiments.flatten import FlattenDeflatten

        try:
            # Load VAE
            vae_path = self.vae_ckpt.get().strip()
            if not os.path.isabs(vae_path):
                vae_path = os.path.join(PROJECT_ROOT, vae_path)
            ckpt = torch.load(vae_path, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            ch = config.get("image_channels", 3)
            lat = config.get("latent_channels", 32)

            self.vae = MiniVAE(
                latent_channels=lat, image_channels=ch, output_channels=ch,
                encoder_channels=64, decoder_channels=(256, 128, 64),
                encoder_time_downscale=(False, False, False),
                decoder_time_upscale=(False, False, False),
            ).cuda()
            src_sd = ckpt["model"] if "model" in ckpt else ckpt
            target_sd = self.vae.state_dict()
            for k, v in src_sd.items():
                if k in target_sd and v.shape == target_sd[k].shape:
                    target_sd[k] = v
            self.vae.load_state_dict(target_sd)
            self.vae.eval()
            self.vae.requires_grad_(False)

            # Probe latent dims
            with torch.no_grad():
                dummy = torch.randn(1, 1, ch, 360, 640, device="cuda")
                lat_d = self.vae.encode_video(dummy)
                _, _, lat_C, lat_H, lat_W = lat_d.shape
                del dummy, lat_d

            # Load bottleneck
            bn_path = self.bn_ckpt.get().strip()
            if not os.path.isabs(bn_path):
                bn_path = os.path.join(PROJECT_ROOT, bn_path)
            bn_ckpt = torch.load(bn_path, map_location="cpu", weights_only=False)
            bn_cfg = bn_ckpt.get("config", {})
            bn_ch = bn_cfg.get("bottleneck_channels", 6)
            walk = bn_cfg.get("walk_order", "raster")

            self.bottleneck = FlattenDeflatten(
                latent_channels=lat_C, bottleneck_channels=bn_ch,
                spatial_h=lat_H, spatial_w=lat_W, walk_order=walk,
            ).cuda()
            self.bottleneck.load_state_dict(bn_ckpt["bottleneck"])
            self.bottleneck.eval()

            self._image_channels = ch
            step = bn_ckpt.get("step", "?")
            self.status.config(
                text=f"VAE: {ch}ch lat={lat} | Bottleneck: {bn_ch}ch "
                     f"({lat_C}→{bn_ch}→{lat_C}), step {step}, "
                     f"walk={walk}")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    @torch.no_grad()
    def test_synthetic(self):
        if self.vae is None or self.bottleneck is None:
            self.status.config(text="Load models first")
            return

        sys.path.insert(0, PROJECT_ROOT)
        from core.generator import SynthyperGenerator

        self.status.config(text="Generating...")
        self.update()

        gen = SynthyperGenerator(360, 640, device="cuda", bank_size=200,
                                  n_base_layers=64)
        gen.build_banks()
        images = gen.generate(4)  # (4, 3, H, W)
        x = images.unsqueeze(1).cuda()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            recon_vae, _ = self.vae(x)
            lat = self.vae.encode_video(x).squeeze(1)
            lat_recon, _ = self.bottleneck(lat)
            recon_flat = self.vae.decode_video(lat_recon.unsqueeze(1))

        gt = images.cpu().numpy()
        rc_vae = recon_vae[:, -1, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[:, -1, :3].clamp(0, 1).float().cpu().numpy()
        self._display_grid(gt, rc_vae, rc_flat)
        self.status.config(text="GT | VAE | Flatten (synthetic)")

    @torch.no_grad()
    def test_image(self):
        """Test on a user-provided image file."""
        if self.vae is None or self.bottleneck is None:
            self.status.config(text="Load models first")
            return

        path = self.input_path.get().strip()
        if not path:
            # Fall back to file dialog
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                           ("All", "*.*")])
            if not path:
                return
            self.input_path.set(path)

        self.status.config(text=f"Loading {os.path.basename(path)}...")
        self.update()

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((640, 360), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

            ch = getattr(self, '_image_channels', 3)
            if ch > 3:
                t = torch.cat([t, torch.zeros(ch - 3, 360, 640)], dim=0)

            x = t.unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, C, H, W)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                recon_vae, _ = self.vae(x)
                lat = self.vae.encode_video(x).squeeze(1)
                lat_recon, _ = self.bottleneck(lat)
                recon_flat = self.vae.decode_video(lat_recon.unsqueeze(1))

            gt = t[:3].unsqueeze(0).numpy()  # (1, 3, H, W)
            rc_vae = recon_vae[:, -1, :3].clamp(0, 1).float().cpu().numpy()
            rc_flat = recon_flat[:, -1, :3].clamp(0, 1).float().cpu().numpy()
            self._display_grid(gt, rc_vae, rc_flat)
            self.status.config(text=f"GT | VAE | Flatten — {os.path.basename(path)}")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    def _display_grid(self, gt, rc_vae, rc_flat):
        """Show GT | VAE | Flatten grid for 1-4 images."""
        H, W = 360, 640
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        gap = np.full((4, W * 3 + 8, 3), 14, dtype=np.uint8)
        rows = []
        for i in range(min(4, len(gt))):
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            v = (rc_vae[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            f = (rc_flat[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            row = np.concatenate([g, sep, v, sep, f], axis=1)
            rows.append(row)

        grid = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], axis=0)
        pil_full = Image.fromarray(grid)
        # Save to inference dir
        import time as _time
        inf_dir = os.path.join(PROJECT_ROOT, "flatten_logs", "inference")
        os.makedirs(inf_dir, exist_ok=True)
        pil_full.save(os.path.join(inf_dir,
            f"flatten_inf_{int(_time.time())}.png"))

        scale = min(900 / pil_full.width, 500 / pil_full.height, 1.0)
        if scale < 1:
            pil_full = pil_full.resize(
                (int(pil_full.width * scale), int(pil_full.height * scale)),
                Image.BILINEAR)
        self._preview_photo = ImageTk.PhotoImage(pil_full)
        self.preview_label.config(image=self._preview_photo)


# -- Flatten Video Tab ---------------------------------------------------------

class FlattenVideoTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._video_frames = []
        self._play_gen = 0
        self._video_idx = 0
        self._last_mtime = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Flatten Video Experiment", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Freeze temporal VAE. Train per-frame 1D conv "
                 "bottleneck in latent space.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL,
                 justify="left").pack(anchor="w", pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.vae_ckpt = make_float(row1, "Temporal VAE ckpt",
            os.path.join(PROJECT_ROOT, "synthyper_video_logs", "latest.pt"),
            width=50)
        f.pack(side="left", fill="x", expand=True)

        row1b = tk.Frame(top, bg=BG_PANEL)
        row1b.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row1b, "Resume", "", width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.bottleneck_ch = make_spin(row2, "Bottleneck ch", default=6)
        f.pack(side="left", padx=(0, 10))
        f, self.walk_var = make_float(row2, "Walk order", "raster")
        f.pack(side="left", padx=(0, 10))
        f, self.lr_var = make_float(row2, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=10000)
        f.pack(side="left")

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.T_var = make_spin(row3, "T", default=24)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lat = make_float(row3, "w_latent", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row3, "w_pixel", 0.5)
        f.pack(side="left", padx=(0, 10))
        f, self.w_temp = make_float(row3, "w_temporal", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.log_every = make_spin(row3, "Log", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview", default=100)
        f.pack(side="left")

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        # Preview: video playback
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)

        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)

        self._check_preview()

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.flatten_video",
               "--vae-ckpt", self.vae_ckpt.get(),
               "--bottleneck-ch", str(self.bottleneck_ch.get()),
               "--walk-order", self.walk_var.get(),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--T", str(self.T_var.get()),
               "--w-latent", self.w_lat.get(),
               "--w-pixel", self.w_pix.get(),
               "--w-temporal", self.w_temp.get(),
               "--log-every", str(self.log_every.get()),
               "--preview-every", str(self.preview_every.get())]
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        logdir = os.path.join(PROJECT_ROOT, "flatten_video_logs")
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()

    def _check_preview(self):
        preview = os.path.join(PROJECT_ROOT, "flatten_video_logs",
                               "preview_latest.mp4")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    # Stop old playback
                    self._play_gen += 1
                    # Decode ALL frames for inline playback
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", preview],
                        capture_output=True, text=True)
                    parts = probe.stdout.strip().split(",")
                    if len(parts) >= 2:
                        w, h = int(parts[0]), int(parts[1])
                        cmd = ["ffmpeg", "-v", "quiet", "-i", preview,
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
                        raw = subprocess.run(cmd, capture_output=True).stdout
                        fs = w * h * 3
                        n = len(raw) // fs
                        if n > 0:
                            scale = min(900 / w, 350 / h, 1.0)
                            dw = int(w * scale) if scale < 1 else w
                            dh = int(h * scale) if scale < 1 else h
                            self._video_frames = []
                            for fi in range(n):
                                arr = np.frombuffer(
                                    raw[fi*fs:(fi+1)*fs],
                                    dtype=np.uint8).reshape(h, w, 3)
                                pil = Image.fromarray(arr)
                                if scale < 1:
                                    pil = pil.resize((dw, dh), Image.BILINEAR)
                                self._video_frames.append(
                                    ImageTk.PhotoImage(pil))
                            self._video_idx = 0
                            self._play_gen += 1
                            self._play_video_loop(self._play_gen)
            except Exception:
                pass
        self.after(5000, self._check_preview)

    def _play_video_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        idx = self._video_idx % len(self._video_frames)
        self.preview_label.config(image=self._video_frames[idx])
        self._video_idx = idx + 1
        self.after(33, self._play_video_loop, self._play_gen)


# -- Flatten Video Inference Tab -----------------------------------------------

class FlattenVideoInferenceTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.vae = None
        self.bottleneck = None
        self._video_frames = []
        self._play_gen = 0
        self._video_idx = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Flatten Video Inference (GT | VAE | Flatten)",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(10, 0))
        f, self.vae_ckpt = make_float(row1, "Temporal VAE ckpt",
            os.path.join(PROJECT_ROOT, "synthyper_video_logs", "latest.pt"),
            width=40)
        f.pack(side="left", fill="x", expand=True, padx=(0, 5))
        f, self.bn_ckpt = make_float(row1, "Bottleneck ckpt",
            os.path.join(PROJECT_ROOT, "flatten_video_logs", "latest.pt"),
            width=40)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.input_path = make_float(row2, "Video path (drop or browse)",
                                        "", width=40)
        f.pack(side="left", fill="x", expand=True, padx=(0, 5))
        make_btn(row2, "Browse", self._browse_video, FG_DIM, width=7).pack(
            side="left", pady=(15, 0))

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.T_var = make_spin(row3, "T (frames)", default=24, width=6)
        f.pack(side="left", padx=(0, 10))
        make_btn(row3, "Load", self.load_model, GREEN, width=8).pack(
            side="left", padx=(0, 10))
        make_btn(row3, "Test Synthetic", self.test_synthetic, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(row3, "Test Video", self.test_video, BLUE).pack(
            side="left")

        self.status = tk.Label(top, text="No model loaded", bg=BG_PANEL,
                                fg=FG_DIM, font=FONT_SMALL)
        self.status.pack(fill="x", pady=(5, 0))

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

        # Enable drag-and-drop
        self._setup_drop()

    def _setup_drop(self):
        try:
            from tkinterdnd2 import DND_FILES
            root = self.winfo_toplevel()
            if hasattr(root, 'drop_target_register'):
                self.drop_target_register(DND_FILES)
                self.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        path = event.data.strip().strip('{}')
        self.input_path.set(path)

    def _browse_video(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov *.webm"),
                       ("All", "*.*")])
        if path:
            self.input_path.set(path)

    def load_model(self):
        sys.path.insert(0, PROJECT_ROOT)
        from core.model import MiniVAE
        from experiments.flatten import FlattenDeflatten

        try:
            # Load temporal VAE
            vae_path = self.vae_ckpt.get().strip()
            if not os.path.isabs(vae_path):
                vae_path = os.path.join(PROJECT_ROOT, vae_path)
            ckpt = torch.load(vae_path, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            ch = config.get("image_channels", 3)
            lat = config.get("latent_channels", 32)

            self.vae = MiniVAE(
                latent_channels=lat, image_channels=ch, output_channels=ch,
                encoder_channels=64, decoder_channels=(256, 128, 64),
                encoder_time_downscale=(True, True, False),
                decoder_time_upscale=(False, True, True),
            ).cuda()
            src_sd = ckpt["model"] if "model" in ckpt else ckpt
            target_sd = self.vae.state_dict()
            loaded = 0
            for k, v in src_sd.items():
                if k in target_sd and v.shape == target_sd[k].shape:
                    target_sd[k] = v
                    loaded += 1
            self.vae.load_state_dict(target_sd)
            self.vae.eval()
            self.vae.requires_grad_(False)

            # Probe latent dims
            with torch.no_grad():
                dummy = torch.randn(1, 8, ch, 360, 640, device="cuda")
                lat_d = self.vae.encode_video(dummy)
                _, _, lat_C, lat_H, lat_W = lat_d.shape
                del dummy, lat_d

            # Load bottleneck
            bn_path = self.bn_ckpt.get().strip()
            if not os.path.isabs(bn_path):
                bn_path = os.path.join(PROJECT_ROOT, bn_path)
            bn_ckpt = torch.load(bn_path, map_location="cpu", weights_only=False)
            bn_cfg = bn_ckpt.get("config", {})
            bn_ch = bn_cfg.get("bottleneck_channels", 6)
            walk = bn_cfg.get("walk_order", "raster")

            self.bottleneck = FlattenDeflatten(
                latent_channels=lat_C, bottleneck_channels=bn_ch,
                spatial_h=lat_H, spatial_w=lat_W, walk_order=walk,
            ).cuda()
            self.bottleneck.load_state_dict(bn_ckpt["bottleneck"])
            self.bottleneck.eval()

            step = bn_ckpt.get("step", "?")
            self.status.config(
                text=f"VAE: {ch}ch lat={lat} temporal | Bottleneck: "
                     f"{bn_ch}ch ({lat_C}→{bn_ch}→{lat_C}), step {step}, "
                     f"walk={walk}, {loaded} VAE weights")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    @torch.no_grad()
    def test_synthetic(self):
        if self.vae is None or self.bottleneck is None:
            self.status.config(text="Load models first")
            return

        sys.path.insert(0, PROJECT_ROOT)
        from core.generator import SynthyperGenerator

        T = self.T_var.get()
        self.status.config(text=f"Generating T={T} clip...")
        self.update()

        gen = SynthyperGenerator(360, 640, device="cuda", bank_size=200,
                                  n_base_layers=64)
        gen.build_banks()
        clip = gen.generate_sequence(1, T=T)  # (1, T, 3, H, W)
        x = clip.cuda()

        recon_vae, recon_flat, lat = chunked_flatten_inference(
            self.vae, self.bottleneck, x)

        T_vae = recon_vae.shape[1]
        T_flat = recon_flat.shape[1]
        T_in = x.shape[1]
        T_show = min(T_vae, T_flat, T_in)

        gt = x[0, T_in - T_show:, :3].float().cpu().numpy()
        rc_vae = recon_vae[0, T_vae - T_show:, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[0, T_flat - T_show:, :3].clamp(0, 1).float().cpu().numpy()

        self.status.config(text=f"Input T={T_in}, Latent T'={lat.shape[1]}, "
                           f"Output T''={T_vae}, showing {T_show} frames")
        self._show_video(gt, rc_vae, rc_flat, T_show)

    @torch.no_grad()
    def test_video(self):
        """Test on user-provided video file (from path field or browse)."""
        if self.vae is None or self.bottleneck is None:
            self.status.config(text="Load models first")
            return

        path = self.input_path.get().strip()
        if not path:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov *.webm"),
                           ("All", "*.*")])
            if not path:
                return
            self.input_path.set(path)

        T = self.T_var.get()
        self.status.config(text=f"Loading {os.path.basename(path)}...")
        self.update()

        cmd = ["ffmpeg", "-v", "quiet", "-i", path,
               "-frames:v", str(T), "-vf", "scale=640:360",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        fs = 360 * 640 * 3
        n = min(len(raw) // fs, T)
        if n < 2:
            self.status.config(text="Not enough frames")
            return

        frames = np.frombuffer(raw[:n*fs], dtype=np.uint8).reshape(n, 360, 640, 3)
        clip = torch.from_numpy(frames.astype(np.float32) / 255.0
                                ).permute(0, 3, 1, 2).unsqueeze(0).cuda()

        ch = self.vae.image_channels
        if ch > 3:
            clip = torch.cat([clip, torch.zeros(1, n, ch - 3, 360, 640,
                                                 device="cuda")], dim=2)

        recon_vae, recon_flat, lat = chunked_flatten_inference(
            self.vae, self.bottleneck, clip)

        T_vae = recon_vae.shape[1]
        T_flat = recon_flat.shape[1]
        T_in = clip.shape[1]
        T_show = min(T_vae, T_flat, T_in)

        gt = clip[0, T_in - T_show:, :3].float().cpu().numpy()
        rc_vae = recon_vae[0, T_vae - T_show:, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[0, T_flat - T_show:, :3].clamp(0, 1).float().cpu().numpy()

        self.status.config(text=f"{os.path.basename(path)} — T={T_in}, "
                           f"Latent T'={lat.shape[1]}, showing {T_show} frames")
        self._show_video(gt, rc_vae, rc_flat, T_show)

    def _show_video(self, gt, rc_vae, rc_flat, T_show):
        """Play GT | VAE | Flatten side by side as looping video."""
        H, W = 360, 640
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        frame_w = W * 3 + 8  # 3 columns + 2 seps
        scale = min(900 / frame_w, 400 / H, 1.0)
        dw = int(frame_w * scale) if scale < 1 else frame_w
        dh = int(H * scale) if scale < 1 else H

        # Save inference video
        import time as _time
        inf_dir = os.path.join(PROJECT_ROOT, "flatten_video_logs", "inference")
        os.makedirs(inf_dir, exist_ok=True)
        ts = int(_time.time())
        vid_path = os.path.join(inf_dir, f"flatten_vid_inf_{ts}.mp4")
        cmd = ["ffmpeg", "-y", "-v", "quiet",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{frame_w}x{H}", "-r", "30",
               "-i", "pipe:0",
               "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", vid_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        self._video_frames = []
        for t in range(T_show):
            g = (gt[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            v = (rc_vae[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            f = (rc_flat[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            frame = np.concatenate([g, sep, v, sep, f], axis=1)
            proc.stdin.write(frame.tobytes())
            pil = Image.fromarray(frame)
            if scale < 1:
                pil = pil.resize((dw, dh), Image.BILINEAR)
            self._video_frames.append(ImageTk.PhotoImage(pil))

        proc.stdin.close()
        proc.wait()

        self._video_idx = 0
        self._play_gen += 1
        self._play_video_loop(self._play_gen)

    def _play_video_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        idx = self._video_idx % len(self._video_frames)
        self.preview_label.config(image=self._video_frames[idx])
        self._video_idx = idx + 1
        self.after(33, self._play_video_loop, self._play_gen)


# -- FSQ Convert Tab -----------------------------------------------------------
class FSQConvertTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="FSQ Convert (Continuous -> Quantized)",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Insert FSQ quantization layer into trained VAE. "
                 "Fine-tune with straight-through estimator.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w", pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.src_var = make_float(row1, "Source VAE checkpoint",
            os.path.join(PROJECT_ROOT, "synthyper_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.levels_var = make_spin(row2, "Levels", default=8)
        f.pack(side="left", padx=(0, 10))
        f, self.groups_var = make_spin(row2, "Groups", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.ch_per_group_var = make_spin(row2, "Ch/group", default=0)
        f.pack(side="left", padx=(0, 10))
        tk.Label(row2, text="(0 = auto from latent_ch / groups)",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(
                     side="left", pady=(15, 0))

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row3, "LR", "5e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row3, "Steps", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row3, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.logdir_var = make_float(row3, "Log dir", "fsq_logs")
        f.pack(side="left")

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Convert + Fine-tune", self.start, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.status = tk.Label(top, text="", bg=BG_PANEL, fg=FG_DIM,
                                font=FONT_SMALL)
        self.status.pack(fill="x", pady=(5, 0))

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)
        self._preview_photo = None

        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)

        self._check_preview()

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.fsq",
               "--vae-ckpt", self.src_var.get(),
               "--levels", str(self.levels_var.get()),
               "--n-groups", str(self.groups_var.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--logdir", self.logdir_var.get()]
        cpg = self.ch_per_group_var.get()
        if cpg > 0:
            cmd.extend(["--ch-per-group", str(cpg)])
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        logdir = os.path.join(PROJECT_ROOT, self.logdir_var.get())
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()

    def _check_preview(self):
        logdir = os.path.join(PROJECT_ROOT, self.logdir_var.get())
        preview = os.path.join(logdir, "preview_latest.png")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if not hasattr(self, '_last_mtime') or mtime != self._last_mtime:
                    self._last_mtime = mtime
                    img = Image.open(preview)
                    scale = min(900 / img.width, 400 / img.height, 1.0)
                    if scale < 1:
                        img = img.resize((int(img.width * scale),
                                          int(img.height * scale)),
                                         Image.BILINEAR)
                    self._preview_photo = ImageTk.PhotoImage(img)
                    self.preview_label.config(image=self._preview_photo)
            except Exception:
                pass
        self.after(5000, self._check_preview)


# -- FSQ Inference Tab ---------------------------------------------------------
class FSQInferenceTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.vae = None
        self.fsq = None
        self._preview_photo = None
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="FSQ Inference (GT | Continuous | Quantized)",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(10, 0))
        f, self.ckpt_var = make_float(row1, "FSQ checkpoint",
            os.path.join(PROJECT_ROOT, "fsq_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.input_path = make_float(row2, "Image (drop or browse)", "", width=40)
        f.pack(side="left", fill="x", expand=True, padx=(0, 5))
        make_btn(row2, "Browse", self._browse, FG_DIM, width=7).pack(
            side="left", pady=(15, 0))

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        make_btn(row3, "Load", self.load_model, GREEN, width=8).pack(
            side="left", padx=(0, 10))
        make_btn(row3, "Test Synthetic", self.test_synthetic, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(row3, "Test Image", self.test_image, BLUE).pack(
            side="left")

        self.status = tk.Label(top, text="No model loaded", bg=BG_PANEL,
                                fg=FG_DIM, font=FONT_SMALL)
        self.status.pack(fill="x", pady=(5, 0))

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

    def _browse(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if path:
            self.input_path.set(path)

    def load_model(self):
        sys.path.insert(0, PROJECT_ROOT)
        from core.model import MiniVAE
        from core.fsq import FSQ

        try:
            ckpt_path = self.ckpt_var.get().strip()
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            ch = config.get("image_channels", 3)
            lat = config.get("latent_channels", 32)

            self.vae = MiniVAE(
                latent_channels=lat, image_channels=ch, output_channels=ch,
                encoder_channels=config.get("encoder_channels", 64),
                decoder_channels=tuple(int(x) for x in
                    config.get("decoder_channels", "256,128,64").split(",")),
                encoder_time_downscale=(False, False, False),
                decoder_time_upscale=(False, False, False),
            ).cuda()

            src_sd = ckpt.get("model", ckpt)
            target_sd = self.vae.state_dict()
            loaded = 0
            for k, v in src_sd.items():
                if k in target_sd and v.shape == target_sd[k].shape:
                    target_sd[k] = v
                    loaded += 1
            self.vae.load_state_dict(target_sd)
            self.vae.eval()
            self.vae.requires_grad_(False)

            # Load FSQ
            fsq_cfg = config.get("fsq", {})
            levels = fsq_cfg.get("levels", 8)
            n_groups = fsq_cfg.get("n_groups", 1)
            cpg = fsq_cfg.get("channels_per_group", lat // n_groups)
            self.fsq = FSQ(levels=levels, n_groups=n_groups,
                           channels_per_group=cpg).cuda()

            step = ckpt.get("global_step", ckpt.get("step", "?"))
            codes = self.fsq.num_codes
            self.status.config(
                text=f"VAE: {ch}ch lat={lat} | FSQ: {levels} levels, "
                     f"{n_groups} groups, {cpg}ch/group, {codes:,} codes, "
                     f"step {step}, {loaded} weights")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    @torch.no_grad()
    def test_synthetic(self):
        if self.vae is None:
            self.status.config(text="Load a model first")
            return
        sys.path.insert(0, PROJECT_ROOT)
        from core.generator import SynthyperGenerator
        self.status.config(text="Generating...")
        self.update()
        gen = SynthyperGenerator(360, 640, device="cuda", bank_size=200,
                                  n_base_layers=64)
        gen.build_banks()
        images = gen.generate(4)
        self._run_inference(images)

    @torch.no_grad()
    def test_image(self):
        if self.vae is None:
            self.status.config(text="Load a model first")
            return
        path = self.input_path.get().strip()
        if not path:
            self._browse()
            path = self.input_path.get().strip()
            if not path:
                return
        try:
            img = Image.open(path).convert("RGB").resize((640, 360), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            self._run_inference(t)
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    def _run_inference(self, images):
        x = images.unsqueeze(1).cuda() if images.dim() == 4 else images.unsqueeze(1).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            recon_cont, _ = self.vae(x)
            lat = self.vae.encode_video(x).squeeze(1)
            lat_q, indices = self.fsq(lat)
            recon_fsq = self.vae.decode_video(lat_q.unsqueeze(1))

        gt = images[:, :3].cpu().numpy() if images.dim() == 4 else images[:, :, :3].cpu().numpy()
        if gt.ndim == 3:
            gt = gt[np.newaxis]
        rc_cont = recon_cont[:, -1, :3].clamp(0, 1).float().cpu().numpy()
        rc_fsq = recon_fsq[:, -1, :3].clamp(0, 1).float().cpu().numpy()

        H, W = 360, 640
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        gap = np.full((4, W * 3 + 8, 3), 14, dtype=np.uint8)
        rows = []
        for i in range(min(4, len(gt))):
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8) \
                if gt[i].shape[0] == 3 else (gt[i] * 255).clip(0, 255).astype(np.uint8)
            c = (rc_cont[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            q = (rc_fsq[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            row = np.concatenate([g, sep, c, sep, q], axis=1)
            rows.append(row)
        grid = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], axis=0)
        pil = Image.fromarray(grid)
        scale = min(900 / pil.width, 500 / pil.height, 1.0)
        if scale < 1:
            pil = pil.resize((int(pil.width * scale), int(pil.height * scale)),
                             Image.BILINEAR)
        self._preview_photo = ImageTk.PhotoImage(pil)
        self.preview_label.config(image=self._preview_photo)

        # Save inference output
        inf_dir = os.path.join(PROJECT_ROOT, "fsq_logs", "inference")
        os.makedirs(inf_dir, exist_ok=True)
        import time as _time
        pil.save(os.path.join(inf_dir, f"fsq_inf_{int(_time.time())}.png"))
        self.status.config(text="GT | Continuous | FSQ")


# -- Flatten FSQ Tab -----------------------------------------------------------
class FlattenFSQTab(tk.Frame):
    """Train flatten bottleneck on FSQ-quantized latents (static)."""
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._preview_photo = None
        self._last_mtime = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Flatten FSQ (freeze VAE+FSQ, train bottleneck)",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.fsq_ckpt = make_float(row1, "FSQ checkpoint",
            os.path.join(PROJECT_ROOT, "fsq_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row1b = tk.Frame(top, bg=BG_PANEL)
        row1b.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row1b, "Resume", "", width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.bottleneck_ch = make_spin(row2, "Bottleneck ch", default=6)
        f.pack(side="left", padx=(0, 10))
        f, self.walk_var = make_float(row2, "Walk order", "raster")
        f.pack(side="left", padx=(0, 10))
        f, self.lr_var = make_float(row2, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=10000)
        f.pack(side="left")

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.w_lat = make_float(row3, "w_latent", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row3, "w_pixel", 0.5)
        f.pack(side="left", padx=(0, 10))
        f, self.log_every = make_spin(row3, "Log", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview", default=200)
        f.pack(side="left")

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)

        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)
        self._check_preview()

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.flatten",
               "--vae-ckpt", self.fsq_ckpt.get(),
               "--bottleneck-ch", str(self.bottleneck_ch.get()),
               "--walk-order", self.walk_var.get(),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-latent", self.w_lat.get(),
               "--w-pixel", self.w_pix.get(),
               "--log-every", str(self.log_every.get()),
               "--preview-every", str(self.preview_every.get()),
               "--logdir", "flatten_fsq_logs"]
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        logdir = os.path.join(PROJECT_ROOT, "flatten_fsq_logs")
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()

    def _check_preview(self):
        preview = os.path.join(PROJECT_ROOT, "flatten_fsq_logs", "preview_latest.png")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    img = Image.open(preview)
                    scale = min(900 / img.width, 400 / img.height, 1.0)
                    if scale < 1:
                        img = img.resize((int(img.width * scale),
                                          int(img.height * scale)), Image.BILINEAR)
                    self._preview_photo = ImageTk.PhotoImage(img)
                    self.preview_label.config(image=self._preview_photo)
            except Exception:
                pass
        self.after(5000, self._check_preview)


# -- Flatten Video FSQ Tab -----------------------------------------------------
class FlattenVideoFSQTab(tk.Frame):
    """Train flatten bottleneck on FSQ-quantized temporal latents."""
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._video_frames = []
        self._play_gen = 0
        self._video_idx = 0
        self._last_mtime = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Flatten Video FSQ", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Freeze temporal VAE+FSQ. Train per-frame flatten "
                 "bottleneck on quantized latents.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w", pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.fsq_ckpt = make_float(row1, "Temporal FSQ checkpoint",
            os.path.join(PROJECT_ROOT, "fsq_video_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        row1b = tk.Frame(top, bg=BG_PANEL)
        row1b.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(row1b, "Resume", "", width=50)
        f.pack(side="left", fill="x", expand=True)

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.bottleneck_ch = make_spin(row2, "Bottleneck ch", default=6)
        f.pack(side="left", padx=(0, 10))
        f, self.walk_var = make_float(row2, "Walk order", "raster")
        f.pack(side="left", padx=(0, 10))
        f, self.lr_var = make_float(row2, "LR", "1e-3")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row2, "Batch", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row2, "Steps", default=10000)
        f.pack(side="left")

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.T_var = make_spin(row3, "T", default=24)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lat = make_float(row3, "w_latent", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_pix = make_float(row3, "w_pixel", 0.5)
        f.pack(side="left", padx=(0, 10))
        f, self.w_temp = make_float(row3, "w_temporal", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.log_every = make_spin(row3, "Log", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview", default=100)
        f.pack(side="left")

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Stop", self.stop, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)

        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)
        self._check_preview()

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.flatten_video",
               "--vae-ckpt", self.fsq_ckpt.get(),
               "--bottleneck-ch", str(self.bottleneck_ch.get()),
               "--walk-order", self.walk_var.get(),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--T", str(self.T_var.get()),
               "--w-latent", self.w_lat.get(),
               "--w-pixel", self.w_pix.get(),
               "--w-temporal", self.w_temp.get(),
               "--log-every", str(self.log_every.get()),
               "--preview-every", str(self.preview_every.get()),
               "--logdir", "flatten_video_fsq_logs"]
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        logdir = os.path.join(PROJECT_ROOT, "flatten_video_fsq_logs")
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()

    def _check_preview(self):
        preview = os.path.join(PROJECT_ROOT, "flatten_video_fsq_logs",
                               "preview_latest.mp4")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    self._play_gen += 1
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", preview],
                        capture_output=True, text=True)
                    parts = probe.stdout.strip().split(",")
                    if len(parts) >= 2:
                        w, h = int(parts[0]), int(parts[1])
                        cmd = ["ffmpeg", "-v", "quiet", "-i", preview,
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
                        raw = subprocess.run(cmd, capture_output=True).stdout
                        fs = w * h * 3
                        n = len(raw) // fs
                        if n > 0:
                            scale = min(900 / w, 350 / h, 1.0)
                            dw = int(w * scale) if scale < 1 else w
                            dh = int(h * scale) if scale < 1 else h
                            self._video_frames = []
                            for fi in range(n):
                                arr = np.frombuffer(
                                    raw[fi*fs:(fi+1)*fs],
                                    dtype=np.uint8).reshape(h, w, 3)
                                pil = Image.fromarray(arr)
                                if scale < 1:
                                    pil = pil.resize((dw, dh), Image.BILINEAR)
                                self._video_frames.append(ImageTk.PhotoImage(pil))
                            self._video_idx = 0
                            self._play_gen += 1
                            self._play_video_loop(self._play_gen)
            except Exception:
                pass
        self.after(5000, self._check_preview)

    def _play_video_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        idx = self._video_idx % len(self._video_frames)
        self.preview_label.config(image=self._video_frames[idx])
        self._video_idx = idx + 1
        self.after(33, self._play_video_loop, self._play_gen)

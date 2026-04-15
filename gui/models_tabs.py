#!/usr/bin/env python3
"""Model training + inference tabs."""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageTk

from gui.common import *

class TrainingTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="VAEpp0r Training", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")

        # Model architecture presets
        arch_row = tk.Frame(top, bg=BG_PANEL)
        arch_row.pack(fill="x", pady=(10, 0))
        self._presets = {
            "Pico (3ch, 1M, 4MB)":     {"image_ch": 3, "latent_ch": 4,  "enc_ch": "32",  "dec_ch": "64,32,16",         "haar": "none", "shortcut": False},
            "Nano (3ch, 1.2M, 5MB)":   {"image_ch": 3, "latent_ch": 8,  "enc_ch": "32",  "dec_ch": "64,48,32",         "haar": "none", "shortcut": False},
            "Tiny (3ch, 3.3M, 13MB)":  {"image_ch": 3, "latent_ch": 16, "enc_ch": "48",  "dec_ch": "128,64,32",        "haar": "none", "shortcut": False},
            "Small (3ch, 4M, 16MB)":   {"image_ch": 3, "latent_ch": 16, "enc_ch": "64",  "dec_ch": "128,64,48",        "haar": "none", "shortcut": False},
            "Medium (3ch, 11M, 43MB)": {"image_ch": 3, "latent_ch": 32, "enc_ch": "64",  "dec_ch": "256,128,64",       "haar": "none", "shortcut": False},
            "16x DC (4ch, 79M)":       {"image_ch": 3, "latent_ch": 4,  "enc_ch": "64,128,256,512", "dec_ch": "512,256,128,64", "haar": "none", "shortcut": True},
            "32x DC (16ch, 95M)":      {"image_ch": 3, "latent_ch": 16, "enc_ch": "64,128,256,256,512", "dec_ch": "512,256,256,128,64", "haar": "none", "shortcut": True},
            "32x H2 DC (16ch, 79M)":   {"image_ch": 3, "latent_ch": 16, "enc_ch": "64,128,256,512", "dec_ch": "512,256,128,64", "haar": "2x", "shortcut": True},
            "32x H4 DC (16ch, 20M)":   {"image_ch": 3, "latent_ch": 16, "enc_ch": "64,128,256", "dec_ch": "256,128,64", "haar": "4x", "shortcut": True},
            "64x DC (64ch, 157M)":     {"image_ch": 3, "latent_ch": 64, "enc_ch": "64,128,256,256,512,512", "dec_ch": "512,512,256,256,128,64", "haar": "none", "shortcut": True},
            "64x H2 DC (64ch, 95M)":   {"image_ch": 3, "latent_ch": 64, "enc_ch": "64,128,256,256,512", "dec_ch": "512,256,256,128,64", "haar": "2x", "shortcut": True},
            "64x H4 DC (64ch, 80M)":   {"image_ch": 3, "latent_ch": 64, "enc_ch": "64,128,256,512", "dec_ch": "512,256,128,64", "haar": "4x", "shortcut": True},
            "128x DC (256ch, 220M)":   {"image_ch": 3, "latent_ch": 256,"enc_ch": "64,128,256,256,512,512,512", "dec_ch": "512,512,512,256,256,128,64", "haar": "none", "shortcut": True},
            "128x H2 DC (256ch, 159M)":{"image_ch": 3, "latent_ch": 256,"enc_ch": "64,128,256,256,512,512", "dec_ch": "512,512,256,256,128,64", "haar": "2x", "shortcut": True},
            "128x H4 DC (256ch, 97M)": {"image_ch": 3, "latent_ch": 256,"enc_ch": "64,128,256,256,512", "dec_ch": "512,256,256,128,64", "haar": "4x", "shortcut": True},
            "Custom":                   None,
        }
        tk.Label(arch_row, text="Preset", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 5))
        self.preset_var = tk.StringVar(value="Medium (3ch, 11M, 43MB)")
        preset_menu = tk.OptionMenu(arch_row, self.preset_var,
                                     *self._presets.keys(),
                                     command=self._apply_preset)
        preset_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                           activebackground=BG_PANEL, activeforeground=FG,
                           highlightthickness=0, borderwidth=0)
        preset_menu.pack(side="left", padx=(0, 10))

        f, self.image_ch_var = make_spin(arch_row, "Image ch", default=3)
        f.pack(side="left", padx=(0, 10))
        f, self.latent_var = make_spin(arch_row, "Latent ch", default=32)
        f.pack(side="left", padx=(0, 10))
        f, self.enc_ch_var = make_float(arch_row, "Enc ch", "64", width=12)
        f.pack(side="left", padx=(0, 10))
        f, self.dec_ch_var = make_float(arch_row, "Dec ch", "256,128,64", width=12)
        f.pack(side="left")

        # Resolution
        res_row = tk.Frame(top, bg=BG_PANEL)
        res_row.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(res_row, "H", default=360)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(res_row, "W", default=640)
        f.pack(side="left")

        # Params
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row1, "LR", "2e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row1, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row1, "Total steps", default=30000)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row1, "Precision", "bf16")
        f.pack(side="left")

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.w_l1_var = make_float(row2, "w_l1", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_mse_var = make_float(row2, "w_mse", 0.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lpips_var = make_float(row2, "w_lpips", 0.5)
        f.pack(side="left", padx=(0, 10))
        f, self.w_gan_var = make_float(row2, "w_gan", 0.1)
        f.pack(side="left", padx=(0, 10))
        f, self.gan_start_var = make_spin(row2, "GAN start", default=1000)
        f.pack(side="left", padx=(0, 10))
        f, self.gan_warmup_var = make_spin(row2, "GAN warmup", default=2000)
        f.pack(side="left", padx=(0, 10))
        f, self.disc_lr_var = make_float(row2, "disc_lr", "none")
        f.pack(side="left", padx=(0, 10))
        f, self.min_shapes_var = make_spin(row2, "Bank size", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.max_shapes_var = make_spin(row2, "Layers", default=128)
        f.pack(side="left", padx=(0, 10))
        f, self.alpha_var = make_float(row2, "Alpha", 3.0)
        f.pack(side="left")

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.log_every_var = make_spin(row3, "Log every", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.save_every_var = make_spin(row3, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every_var = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.logdir_var = make_float(row3, "Log dir", "synthyper_logs")
        f.pack(side="left", padx=(0, 10))
        f, self.resume_var = make_float(row3, "Resume",
            os.path.join(PROJECT_ROOT, "synthyper_logs", "latest.pt"))
        f.pack(side="left")

        # Buttons
        # Preview image
        prev_row = tk.Frame(top, bg=BG_PANEL)
        prev_row.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(prev_row, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        from tkinter import filedialog
        make_btn(ef, "Browse",
                 lambda: self.preview_img_var.set(
                     filedialog.askopenfilename(
                         filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                                    ("All", "*.*")]) or self.preview_img_var.get()),
                 ACCENT, width=7).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

        btn_row = tk.Frame(top, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(10, 0))
        make_btn(btn_row, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn_row, "Stop (save)", self.stop_save, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn_row, "Kill", self.kill, RED).pack(side="left", padx=(0, 5))
        self.disco_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="Disco Quadrant", variable=self.disco_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left", padx=(0, 10))
        tk.Label(btn_row, text="Haar:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left")
        self.haar_var = tk.StringVar(value="none")
        haar_menu = tk.OptionMenu(btn_row, self.haar_var, "none", "2x", "4x")
        haar_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                         activebackground=BG_PANEL, activeforeground=FG,
                         highlightthickness=0, borderwidth=0, width=4)
        haar_menu.pack(side="left", padx=(0, 10))
        self.residual_shortcut_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="DC-AE Shortcut", variable=self.residual_shortcut_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left", padx=(0, 10))
        self.use_attention_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="Attention", variable=self.use_attention_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left", padx=(0, 10))
        self.use_groupnorm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="GroupNorm", variable=self.use_groupnorm_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left")

        # Preview
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)
        self._preview_photo = None

        # Log
        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)

        # Auto-refresh preview
        self._check_preview()

    def _apply_preset(self, name):
        """Apply architecture preset to the controls."""
        cfg = self._presets.get(name)
        if cfg is None:
            return  # Custom — don't change anything
        self.image_ch_var.set(cfg["image_ch"])
        self.latent_var.set(cfg["latent_ch"])
        self.enc_ch_var.set(str(cfg["enc_ch"]))
        self.dec_ch_var.set(cfg["dec_ch"])
        self.haar_var.set(cfg.get("haar", "none"))
        self.residual_shortcut_var.set(cfg.get("shortcut", False))
        self.use_attention_var.set(cfg.get("attention", False))
        self.use_groupnorm_var.set(cfg.get("groupnorm", False))

    def start(self):
        cmd = [VENV_PYTHON, "-m", "training.train_static",
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--image-ch", str(self.image_ch_var.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--precision", self.prec_var.get(),
               "--latent-ch", str(self.latent_var.get()),
               "--enc-ch", self.enc_ch_var.get(),
               "--dec-ch", self.dec_ch_var.get(),
               "--w-mse", self.w_mse_var.get(),
               "--w-l1", self.w_l1_var.get(),
               "--w-lpips", self.w_lpips_var.get(),
               "--w-gan", self.w_gan_var.get(),
               "--gan-start", str(self.gan_start_var.get()),
               "--gan-warmup", str(self.gan_warmup_var.get()),
               "--disc-lr", self.disc_lr_var.get(),
               "--bank-size", str(self.min_shapes_var.get()),
               "--n-layers", str(self.max_shapes_var.get()),
               "--alpha", self.alpha_var.get(),
               "--log-every", str(self.log_every_var.get()),
               "--save-every", str(self.save_every_var.get()),
               "--preview-every", str(self.preview_every_var.get()),
               "--logdir", self.logdir_var.get()]
        resume = self.resume_var.get().strip()
        if resume:
            cmd += ["--resume", resume]
        if self.disco_var.get():
            cmd.append("--disco")
        haar = self.haar_var.get()
        if haar != "none":
            cmd += ["--haar", haar]
        if self.residual_shortcut_var.get():
            cmd.append("--residual-shortcut")
        if self.use_attention_var.get():
            cmd.append("--use-attention")
        if self.use_groupnorm_var.get():
            cmd.append("--use-groupnorm")
        prev_img = self.preview_img_var.get().strip()
        if prev_img:
            cmd += ["--preview-image", prev_img]
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop_save(self):
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
                    # Scale
                    max_w = 900
                    scale = min(max_w / img.width, 1.0)
                    if scale < 1.0:
                        img = img.resize((int(img.width * scale),
                                          int(img.height * scale)), BILINEAR)
                    self._preview_photo = ImageTk.PhotoImage(img)
                    self.preview_label.config(image=self._preview_photo)
            except Exception:
                pass
        self.after(5000, self._check_preview)


# -- Inference Tab -------------------------------------------------------------

class InferenceTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._preview_photo = None
        self.model = None
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Inference (GT | Recon)", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(10, 0))
        f, self.ckpt_var = make_float(row1, "Checkpoint", "synthyper_logs/latest.pt", width=40)
        f.pack(side="left", fill="x", expand=True, padx=(0, 10))
        make_btn(row1, "Load", self.load_model, GREEN, width=8).pack(side="left", pady=(15, 0))

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.input_var = make_float(row2, "Image or directory", "", width=50)
        f.pack(side="left", fill="x", expand=True, padx=(0, 10))
        make_btn(row2, "Browse", self.browse, BLUE, width=8).pack(side="left", pady=(15, 0), padx=(0, 5))
        make_btn(row2, "Run", self.run_inference, ACCENT, width=8).pack(side="left", pady=(15, 0))

        self.status_label = tk.Label(top, text="No model loaded", bg=BG_PANEL,
                                      fg=FG_DIM, font=FONT_SMALL)
        self.status_label.pack(fill="x", pady=(5, 0))

        # Preview
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

    def browse(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if not path:
            path = filedialog.askdirectory()
        if path:
            self.input_var.set(path)

    def load_model(self):
        import torch
        sys.path.insert(0, PROJECT_ROOT)
        from core.model import MiniVAE

        ckpt_path = self.ckpt_var.get().strip()
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            ch = config.get("image_channels", 3)
            lat = config.get("latent_channels", 32)
            haar_mode = config.get("haar", "none")
            # Backward compat: old checkpoints stored haar as bool
            if haar_mode is True:
                haar_mode = "2x"
            elif haar_mode is False or haar_mode is None:
                haar_mode = "none"
            haar_rounds = {"none": 0, "2x": 1, "4x": 2}[haar_mode]
            haar_ch_mult = 4 ** haar_rounds
            vae_ch = ch * haar_ch_mult
            self._haar_rounds = haar_rounds

            enc_ch, dec_ch = parse_arch_config(config)
            n_stages = len(dec_ch) if isinstance(dec_ch, tuple) else 3
            sd = ckpt["model"] if "model" in ckpt else ckpt

            # Try all flag combos until strict load succeeds
            flags_to_try = [
                (config.get("residual_shortcut", False),
                 config.get("use_attention", False),
                 config.get("use_groupnorm", False)),
            ]
            # If config doesn't have these keys, also try with shortcut=True
            if "residual_shortcut" not in config:
                flags_to_try.append((True, False, False))

            best_result, best_model = None, None
            for shortcut, attention, groupnorm in flags_to_try:
                model = MiniVAE(
                    latent_channels=lat,
                    image_channels=vae_ch,
                    output_channels=vae_ch,
                    encoder_channels=enc_ch,
                    decoder_channels=dec_ch,
                    encoder_time_downscale=tuple([False] * n_stages),
                    decoder_time_upscale=tuple([False] * n_stages),
                    residual_shortcut=shortcut,
                    use_attention=attention,
                    use_groupnorm=groupnorm,
                ).cuda()
                result = model.load_state_dict(sd, strict=False)
                n_bad = len(result.missing_keys) + len(result.unexpected_keys)
                if n_bad == 0:
                    best_result, best_model = result, model
                    break
                if best_result is None or n_bad < len(best_result.missing_keys) + len(best_result.unexpected_keys):
                    best_result, best_model = result, model

            self.model = best_model
            self.model.eval()
            result = best_result
            step = ckpt.get("global_step", "?")
            pc = sum(p.numel() for p in self.model.parameters())
            info = f"Loaded: {ch}ch, {lat} latent, step {step}, {pc:,} params"
            if haar_rounds > 0:
                info += f", haar={haar_mode}"
            if result.missing_keys:
                info += f", missing={len(result.missing_keys)}"
            if result.unexpected_keys:
                info += f", unexpected={len(result.unexpected_keys)}"
            self.status_label.config(text=info)
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    def run_inference(self):
        if self.model is None:
            self.status_label.config(text="Load a model first")
            return

        input_path = self.input_var.get().strip()
        if not input_path:
            self.status_label.config(text="Set an image or directory path")
            return

        # Collect images
        img_paths = []
        if os.path.isfile(input_path):
            img_paths = [input_path]
        elif os.path.isdir(input_path):
            exts = {".png", ".jpg", ".jpeg", ".bmp"}
            img_paths = sorted(
                os.path.join(input_path, f) for f in os.listdir(input_path)
                if os.path.splitext(f)[1].lower() in exts)[:16]

        if not img_paths:
            self.status_label.config(text="No images found")
            return

        self.status_label.config(text="Running inference...")

        def _bg():
            try:
                import torch
                model = self.model
                haar_rounds = getattr(self, '_haar_rounds', 0)

                # Haar helpers (inline to avoid import issues)
                def _haar_down(x):
                    a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
                    c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
                    return torch.cat([(a+b+c+d)*0.5, (a-b+c-d)*0.5,
                                      (a+b-c-d)*0.5, (a-b-c+d)*0.5], dim=1)

                def _haar_up(x):
                    C = x.shape[1] // 4
                    ll, lh, hl, hh = x[:,0*C:1*C], x[:,1*C:2*C], x[:,2*C:3*C], x[:,3*C:4*C]
                    a, b = (ll+lh+hl+hh)*0.5, (ll-lh+hl-hh)*0.5
                    c, d = (ll+lh-hl-hh)*0.5, (ll-lh-hl+hh)*0.5
                    B, Ch, H, W = a.shape
                    out = torch.zeros(B, Ch, H*2, W*2, device=x.device, dtype=x.dtype)
                    out[:,:,0::2,0::2], out[:,:,0::2,1::2] = a, b
                    out[:,:,1::2,0::2], out[:,:,1::2,1::2] = c, d
                    return out

                # Load and process
                pairs = []
                with torch.no_grad():
                    for p in img_paths:
                        img = Image.open(p).convert("RGB")
                        img = img.resize((640, 360), BILINEAR)
                        arr = np.array(img, dtype=np.float32) / 255.0
                        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
                        if haar_rounds > 0:
                            for _ in range(haar_rounds):
                                t = _haar_down(t)
                        inp = t.unsqueeze(1)
                        recon, _ = model(inp)
                        rc = recon[0, -1]
                        if haar_rounds > 0:
                            hH, hW = 360 // (2 ** haar_rounds), 640 // (2 ** haar_rounds)
                            rc = rc[:, :hH, :hW].unsqueeze(0)
                            for _ in range(haar_rounds):
                                rc = _haar_up(rc)
                            rc = rc[0, :3, :360, :640]
                        else:
                            rc = rc[:3, :360, :640]
                        rc = rc.clamp(0, 1).cpu().numpy()
                        gt = arr.transpose(2, 0, 1)

                        gt_img = (gt.transpose(1, 2, 0) * 255).astype(np.uint8)
                        rc_img = (rc.transpose(1, 2, 0) * 255).astype(np.uint8)
                        pair = np.concatenate([gt_img, np.full((360, 4, 3), 14, dtype=np.uint8),
                                               rc_img], axis=1)
                        pairs.append(pair)

                # Stack vertically (up to 16)
                if len(pairs) == 1:
                    grid = pairs[0]
                else:
                    gap = np.full((4, pairs[0].shape[1], 3), 14, dtype=np.uint8)
                    rows = []
                    for i in range(0, len(pairs), 4):
                        chunk = pairs[i:i+4]
                        row = np.concatenate(
                            [np.concatenate([c, gap], axis=0) for c in chunk][:-1] + [chunk[-1]],
                            axis=0)
                        rows.append(row)
                    if len(rows) > 1:
                        max_w = max(r.shape[1] for r in rows)
                        padded = []
                        for r in rows:
                            if r.shape[1] < max_w:
                                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 14, dtype=np.uint8)
                                r = np.concatenate([r, pad], axis=1)
                            padded.append(r)
                        vgap = np.full((4, max_w, 3), 14, dtype=np.uint8)
                        grid = np.concatenate(
                            sum([[r, vgap] for r in padded], [])[:-1], axis=0)
                    else:
                        grid = rows[0]

                pil_full = Image.fromarray(grid)
                import time as _time
                inf_dir = os.path.join(PROJECT_ROOT, "synthyper_logs", "inference")
                os.makedirs(inf_dir, exist_ok=True)
                pil_full.save(os.path.join(inf_dir,
                    f"static_inf_{int(_time.time())}.png"))

                n_pairs = len(pairs)
                self.after(0, self._show_inference_result, pil_full, n_pairs)
            except Exception as e:
                _e = str(e)
                self.after(0, lambda: self.status_label.config(text=f"Error: {_e}"))

        threading.Thread(target=_bg, daemon=True).start()

    def _show_inference_result(self, pil_full, n_pairs):
        max_w = 1000
        max_h = 600
        scale = min(max_w / pil_full.width, max_h / pil_full.height, 1.0)
        if scale < 1.0:
            pil_full = pil_full.resize(
                (int(pil_full.width * scale), int(pil_full.height * scale)),
                BILINEAR)

        self._preview_photo = ImageTk.PhotoImage(pil_full)
        self.preview_label.config(image=self._preview_photo)
        self.status_label.config(text=f"Inference: {n_pairs} images (saved)")


# -- Convert Tab ---------------------------------------------------------------

class ConvertTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Checkpoint Converter", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")

        tk.Label(top, text="Inflate a Stage 1 (static) checkpoint into a "
                 "Stage 2 (temporal) checkpoint.\nSpatial weights transfer, "
                 "temporal weights initialize fresh.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL,
                 justify="left").pack(anchor="w", pady=(5, 10))

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.src_var = make_float(row1, "Source (Stage 1 checkpoint)",
            os.path.join(PROJECT_ROOT, "synthyper_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True, padx=(0, 10))
        make_btn(row1, "Browse", self._browse_src, BLUE, width=8).pack(
            side="left", pady=(15, 0))

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.dst_var = make_float(row2, "Output (Stage 2 checkpoint)",
            os.path.join(PROJECT_ROOT, "synthyper_video_logs", "converted.pt"), width=50)
        f.pack(side="left", fill="x", expand=True, padx=(0, 10))
        make_btn(row2, "Browse", self._browse_dst, BLUE, width=8).pack(
            side="left", pady=(15, 0))

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.latent_var = make_spin(row3, "Latent ch", default=32, width=6)
        f.pack(side="left", padx=(0, 10))

        for label, attr, default in [
            ("Enc time downscale", "enc_t_var", "1,1,0"),
            ("Dec time upscale",   "dec_t_var", "0,1,1"),
        ]:
            grp = tk.Frame(row3, bg=BG_PANEL)
            grp.pack(side="left", padx=(0, 10))
            tk.Label(grp, text=label, bg=BG_PANEL, fg=FG_DIM,
                     font=FONT_SMALL).pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            tk.Entry(grp, textvariable=var, width=12,
                     bg=BG, fg=FG, insertbackground=FG,
                     relief="flat").pack()

        btn_row = tk.Frame(top, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(10, 0))
        make_btn(btn_row, "Convert", self._convert, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(btn_row, "Verify", self._verify, ACCENT).pack(side="left")

        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)

    def _browse_src(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Checkpoints", "*.pt"), ("All", "*.*")])
        if path:
            self.src_var.set(path)

    def _browse_dst(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("Checkpoints", "*.pt"), ("All", "*.*")])
        if path:
            self.dst_var.set(path)

    def _log(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def _convert(self):
        self.log.delete("1.0", tk.END)
        src = self.src_var.get().strip()
        dst = self.dst_var.get().strip()
        lat = self.latent_var.get()

        if not src or not os.path.exists(src):
            self._log(f"Source not found: {src}")
            return

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        self._log(f"Loading source: {src}")
        try:
            ckpt = torch.load(src, map_location="cpu", weights_only=False)
            src_sd = ckpt["model"] if "model" in ckpt else ckpt
            src_config = ckpt.get("config", {})
            src_step = ckpt.get("global_step", 0)
            self._log(f"  Step: {src_step}")
            self._log(f"  Config: {src_config}")
        except Exception as e:
            self._log(f"  Error: {e}")
            return

        # Build temporal model
        self._log(f"\nBuilding temporal model (latent_ch={lat})...")
        sys.path.insert(0, PROJECT_ROOT)
        from core.model import MiniVAE
        enc_ch, dec_ch = parse_arch_config(src_config)
        try:
            enc_t = tuple(bool(int(x)) for x in self.enc_t_var.get().split(","))
            dec_t = tuple(bool(int(x)) for x in self.dec_t_var.get().split(","))
        except Exception:
            self._log("Bad enc/dec time pattern — use comma-separated 0/1 (e.g. 1,1,0)")
            return
        if len(enc_t) != len(dec_ch) or len(dec_t) != len(dec_ch):
            self._log(f"enc/dec time patterns must have {len(dec_ch)} values "
                      f"to match {len(dec_ch)}-stage model (got {len(enc_t)}, {len(dec_t)})")
            return
        haar_mode = src_config.get("haar", "none")
        if haar_mode is True: haar_mode = "2x"
        elif not haar_mode or haar_mode is False: haar_mode = "none"
        haar_rounds = {"none": 0, "2x": 1, "4x": 2}.get(haar_mode, 0)
        base_ch = src_config.get("image_channels", 3)
        # image_channels in static checkpoints is always the RGB count (3), not haar-expanded
        if base_ch == 3 and haar_rounds > 0:
            vae_in_ch = 3 * (4 ** haar_rounds)
        else:
            vae_in_ch = base_ch
        temporal_model = MiniVAE(
            latent_channels=lat,
            image_channels=vae_in_ch,
            output_channels=vae_in_ch,
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
            encoder_time_downscale=enc_t,
            decoder_time_upscale=dec_t,
            residual_shortcut=src_config.get("residual_shortcut", False),
            use_attention=src_config.get("use_attention", False),
            use_groupnorm=src_config.get("use_groupnorm", False),
        )
        pc = temporal_model.param_count()
        self._log(f"  Params: {pc['total']:,}")
        self._log(f"  t_downscale={temporal_model.t_downscale}, "
                  f"t_upscale={temporal_model.t_upscale}")

        # Load spatial weights — smart conversion for TPool/TGrow shape changes
        target_sd = temporal_model.state_dict()
        loaded, converted, skipped_size, skipped_missing = 0, 0, 0, 0
        for k, v in src_sd.items():
            if k not in target_sd:
                skipped_missing += 1
                continue
            t = target_sd[k]
            if v.shape == t.shape:
                target_sd[k] = v
                loaded += 1
            elif v.ndim == 4 and t.ndim == 4:
                # TPool: src (n_f, n_f, 1, 1) -> dst (n_f, 2*n_f, 1, 1)
                # Copy src into first half of input channels, zero the rest
                if t.shape[0] == v.shape[0] and t.shape[1] == v.shape[1] * 2:
                    t2 = torch.zeros_like(t)
                    t2[:, :v.shape[1]] = v
                    target_sd[k] = t2
                    converted += 1
                # TGrow: src (n_f, n_f, 1, 1) -> dst (2*n_f, n_f, 1, 1)
                # Copy src into both output halves so both frames start as copies
                elif t.shape[1] == v.shape[1] and t.shape[0] == v.shape[0] * 2:
                    t2 = torch.zeros_like(t)
                    t2[:v.shape[0]] = v
                    t2[v.shape[0]:] = v
                    target_sd[k] = t2
                    converted += 1
                else:
                    skipped_size += 1
                    self._log(f"  Size mismatch: {k} "
                              f"{list(v.shape)} -> {list(t.shape)}")
            else:
                skipped_size += 1
                self._log(f"  Size mismatch: {k} "
                          f"{list(v.shape)} -> {list(t.shape)}")
        temporal_model.load_state_dict(target_sd)
        new_keys = len(target_sd) - loaded - converted - skipped_size
        self._log(f"\nWeight transfer:")
        self._log(f"  Loaded: {loaded}")
        self._log(f"  Converted (TPool/TGrow): {converted}")
        self._log(f"  Size mismatch (reinit): {skipped_size}")
        self._log(f"  New temporal keys: {new_keys}")
        if skipped_missing:
            self._log(f"  Unexpected in source: {skipped_missing}")

        # Save
        out_ckpt = {
            "model": temporal_model.state_dict(),
            "optimizer": None,
            "global_step": 0,
            "config": {
                "latent_channels": lat,
                "image_channels": vae_in_ch,
                "output_channels": vae_in_ch,
                "encoder_channels": enc_ch,
                "decoder_channels": ",".join(str(c) for c in dec_ch),
                "encoder_time_downscale": ",".join(str(int(x)) for x in enc_t),
                "decoder_time_upscale": ",".join(str(int(x)) for x in dec_t),
                "residual_shortcut": src_config.get("residual_shortcut", False),
                "use_attention": src_config.get("use_attention", False),
                "use_groupnorm": src_config.get("use_groupnorm", False),
                "haar": haar_mode,
                "temporal": True,
                "synthyper_stage": 2,
                "converted_from": src,
                "converted_from_step": src_step,
            },
        }
        torch.save(out_ckpt, dst)
        size_mb = os.path.getsize(dst) / 1e6
        self._log(f"\nSaved: {dst} ({size_mb:.1f} MB)")
        self._log("Done. Use this as --resume for Video Training.")

    def _verify(self):
        """Verify a converted checkpoint loads correctly."""
        self.log.delete("1.0", tk.END)
        dst = self.dst_var.get().strip()
        if not dst or not os.path.exists(dst):
            self._log(f"File not found: {dst}")
            return

        self._log(f"Loading: {dst}")
        try:
            ckpt = torch.load(dst, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            step = ckpt.get("global_step", 0)
            self._log(f"  Step: {step}")
            self._log(f"  Config: {config}")

            sd = ckpt["model"]
            # Check key shapes
            checks = [
                ("encoder.0.weight", None),
                ("encoder.17.weight", None),
                ("decoder.1.weight", None),
                ("decoder.22.weight", None),
            ]
            self._log("\nKey shapes:")
            for k, _ in checks:
                if k in sd:
                    self._log(f"  {k}: {list(sd[k].shape)}")
                else:
                    self._log(f"  {k}: MISSING")

            # Test forward pass
            self._log("\nTest forward pass...")
            sys.path.insert(0, PROJECT_ROOT)
            from core.model import MiniVAE
            lat = config.get("latent_channels", 32)
            enc_ch, dec_ch = parse_arch_config(config)
            n_stages = len(dec_ch)
            enc_t_cfg = config.get("encoder_time_downscale", None)
            dec_t_cfg = config.get("decoder_time_upscale", None)
            try:
                enc_t_str = str(enc_t_cfg) if enc_t_cfg is not None else self.enc_t_var.get()
                dec_t_str = str(dec_t_cfg) if dec_t_cfg is not None else self.dec_t_var.get()
                enc_t = tuple(bool(int(x)) for x in enc_t_str.split(","))
                dec_t = tuple(bool(int(x)) for x in dec_t_str.split(","))
            except Exception:
                self._log("Bad enc/dec time pattern in config or UI fields")
                return
            haar_mode = config.get("haar", "none")
            if haar_mode is True: haar_mode = "2x"
            elif not haar_mode or haar_mode is False: haar_mode = "none"
            haar_rounds = {"none": 0, "2x": 1, "4x": 2}.get(haar_mode, 0)
            base_ch = config.get("image_channels", 3)
            if base_ch == 3 and haar_rounds > 0:
                vae_ch = 3 * (4 ** haar_rounds)
            else:
                vae_ch = base_ch
            model = MiniVAE(
                latent_channels=lat, image_channels=vae_ch, output_channels=vae_ch,
                encoder_channels=enc_ch, decoder_channels=dec_ch,
                encoder_time_downscale=enc_t,
                decoder_time_upscale=dec_t,
                residual_shortcut=config.get("residual_shortcut", False),
                use_attention=config.get("use_attention", False),
                use_groupnorm=config.get("use_groupnorm", False),
            )
            model.load_state_dict(sd, strict=True)
            model.eval()

            x = torch.randn(1, 8, vae_ch, 64, 64)
            with torch.no_grad():
                recon, latent = model(x)
            self._log(f"  Input:  {list(x.shape)}")
            self._log(f"  Latent: {list(latent.shape)}")
            self._log(f"  Recon:  {list(recon.shape)}")
            self._log("\nVerification PASSED")

        except Exception as e:
            import traceback
            self._log(f"  Error: {e}")
            traceback.print_exc()


# -- Video Generator Tab -------------------------------------------------------

class VideoTrainTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._video_frames = []
        self._video_playing = False
        self._video_idx = 0
        self._preview_photo = None
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Video Training (Stage 2)", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")

        # Architecture row
        arch_row = tk.Frame(top, bg=BG_PANEL)
        arch_row.pack(fill="x", pady=(10, 0))
        f, self.latent_var = make_spin(arch_row, "Latent ch", default=32)
        f.pack(side="left", padx=(0, 10))
        f, self.enc_ch_var = make_float(arch_row, "Enc ch", "64", width=12)
        f.pack(side="left", padx=(0, 10))
        f, self.dec_ch_var = make_float(arch_row, "Dec ch", "256,128,64", width=12)
        f.pack(side="left")

        # Temporal config row
        time_row = tk.Frame(top, bg=BG_PANEL)
        time_row.pack(fill="x", pady=(5, 0))
        f, self.enc_time_var = make_float(time_row, "Enc time", "true,true,false", width=16)
        f.pack(side="left", padx=(0, 10))
        f, self.dec_time_var = make_float(time_row, "Dec time", "false,true,true", width=16)
        f.pack(side="left", padx=(0, 10))
        f, self.T_var = make_spin(time_row, "T (frames)", default=24)
        f.pack(side="left")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row1, "LR", "2e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row1, "Batch", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row1, "Total steps", default=30000)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row1, "Precision", "bf16")
        f.pack(side="left")

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.w_l1_var = make_float(row2, "w_l1", 1.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_mse_var = make_float(row2, "w_mse", 0.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lpips_var = make_float(row2, "w_lpips", 0.5)
        f.pack(side="left", padx=(0, 10))
        f, self.w_temp_var = make_float(row2, "w_temporal", 2.0)
        f.pack(side="left", padx=(0, 10))
        f, self.w_gan_var = make_float(row2, "w_gan", 0.0)
        f.pack(side="left", padx=(0, 10))
        f, self.gan_start_var = make_spin(row2, "GAN start", default=1000)
        f.pack(side="left", padx=(0, 10))
        f, self.bank_var = make_spin(row2, "Bank size", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.layers_var = make_spin(row2, "Layers", default=128)
        f.pack(side="left")

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.log_every_var = make_spin(row3, "Log every", default=1)
        f.pack(side="left", padx=(0, 10))
        f, self.save_every_var = make_spin(row3, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every_var = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.logdir_var = make_float(row3, "Log dir", "synthyper_video_logs")
        f.pack(side="left", padx=(0, 10))
        f, self.resume_var = make_float(row3, "Resume",
            os.path.join(PROJECT_ROOT, "synthyper_video_logs", "converted.pt"))
        f.pack(side="left")

        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        self.use_latest_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Resume from latest.pt",
                       variable=self.use_latest_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, activebackground=BG_PANEL,
                       font=FONT, command=self._toggle_latest).pack(side="left")
        self.fresh_opt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row4, text="Fresh optimizer",
                       variable=self.fresh_opt_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, activebackground=BG_PANEL,
                       font=FONT).pack(side="left", padx=(0, 10))
        f_wu, self.warmup_steps_var = make_spin(row4, "Warmup steps", default=500)
        f_wu.pack(side="left")

        # Preview video
        prev_row = tk.Frame(top, bg=BG_PANEL)
        prev_row.pack(fill="x", pady=(5, 0))
        self.preview_vid_var = tk.StringVar(value="")
        f = tk.Frame(prev_row, bg=BG_PANEL)
        tk.Label(f, text="Preview video", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_vid_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=40, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        from tkinter import filedialog
        make_btn(ef, "Browse",
                 lambda: self.preview_vid_var.set(
                     filedialog.askopenfilename(
                         filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"),
                                    ("All", "*.*")]) or self.preview_vid_var.get()),
                 ACCENT, width=7).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True, padx=(0, 10))
        f2, self.frame_skip_var = make_spin(prev_row, "Frame skip", default=0)
        f2.pack(side="left", padx=(0, 10))
        f3, self.preview_T_var = make_spin(prev_row, "Preview T", default=0)
        f3.pack(side="left")

        btn_row = tk.Frame(top, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(10, 0))
        make_btn(btn_row, "Train", self.start, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(btn_row, "Stop (save)", self.stop_save, BLUE).pack(
            side="left", padx=(0, 5))
        make_btn(btn_row, "Kill", self.kill, RED).pack(side="left", padx=(0, 5))
        self.disco_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="Disco Quadrant", variable=self.disco_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left", padx=(0, 10))
        self.residual_shortcut_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="DC-AE Shortcut", variable=self.residual_shortcut_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left", padx=(0, 10))
        self.use_attention_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="Attention", variable=self.use_attention_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left", padx=(0, 10))
        self.use_groupnorm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btn_row, text="GroupNorm", variable=self.use_groupnorm_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, activeforeground=FG,
                       font=FONT_SMALL).pack(side="left", padx=(0, 10))
        tk.Label(btn_row, text="Haar", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left")
        self.haar_var = tk.StringVar(value="none")
        haar_menu = tk.OptionMenu(btn_row, self.haar_var, "none", "2x", "4x")
        haar_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                         activebackground=BG_INPUT, highlightthickness=0, borderwidth=0)
        haar_menu.pack(side="left", padx=(2, 0))

        # Preview
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)
        self._preview_photo = None

        # Log
        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)

        self._check_preview()

    def _toggle_latest(self):
        if self.use_latest_var.get():
            logdir = self.logdir_var.get()
            self.resume_var.set(os.path.join(PROJECT_ROOT, logdir, "latest.pt"))
            self.fresh_opt_var.set(False)

    def start(self):
        cmd = [VENV_PYTHON, "-m", "training.train_video",
               "--latent-ch", str(self.latent_var.get()),
               "--enc-ch", self.enc_ch_var.get(),
               "--dec-ch", self.dec_ch_var.get(),
               "--enc-time", self.enc_time_var.get(),
               "--dec-time", self.dec_time_var.get(),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--T", str(self.T_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--precision", self.prec_var.get(),
               "--w-l1", self.w_l1_var.get(),
               "--w-mse", self.w_mse_var.get(),
               "--w-lpips", self.w_lpips_var.get(),
               "--w-temporal", self.w_temp_var.get(),
               "--w-gan", self.w_gan_var.get(),
               "--gan-start", str(self.gan_start_var.get()),
               "--bank-size", str(self.bank_var.get()),
               "--n-layers", str(self.layers_var.get()),
               "--log-every", str(self.log_every_var.get()),
               "--save-every", str(self.save_every_var.get()),
               "--preview-every", str(self.preview_every_var.get()),
               "--logdir", self.logdir_var.get()]
        resume = self.resume_var.get().strip()
        if resume:
            cmd += ["--resume", resume]
        if self.fresh_opt_var.get():
            cmd += ["--fresh-opt"]
        cmd += ["--warmup-steps", str(self.warmup_steps_var.get())]
        if self.disco_var.get():
            cmd.append("--disco")
        if self.residual_shortcut_var.get():
            cmd.append("--residual-shortcut")
        if self.use_attention_var.get():
            cmd.append("--use-attention")
        if self.use_groupnorm_var.get():
            cmd.append("--use-groupnorm")
        haar = self.haar_var.get()
        if haar != "none":
            cmd += ["--haar", haar]
        prev_vid = self.preview_vid_var.get().strip()
        if prev_vid:
            cmd += ["--preview-image", prev_vid,
                    "--preview-frame-skip", str(self.frame_skip_var.get())]
        prev_T = self.preview_T_var.get()
        if prev_T > 0:
            cmd += ["--preview-T", str(prev_T)]
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop_save(self):
        logdir = os.path.join(PROJECT_ROOT, self.logdir_var.get())
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()

    def _check_preview(self):
        logdir = os.path.join(PROJECT_ROOT, self.logdir_var.get())
        preview = os.path.join(logdir, "preview_latest.mp4")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if not hasattr(self, '_last_mtime') or mtime != self._last_mtime:
                    self._last_mtime = mtime
                    self._video_playing = False
                    # Decode ALL frames for inline playback
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", preview],
                        capture_output=True, text=True)
                    parts = probe.stdout.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            w, h = int(parts[0]), int(parts[1])
                        except (ValueError, TypeError):
                            w, h = 0, 0
                        if w > 0 and h > 0:
                            cmd = ["ffmpeg", "-v", "quiet", "-i", preview,
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
                            raw = subprocess.run(cmd, capture_output=True).stdout
                            fs = w * h * 3
                            n = len(raw) // fs
                            if n > 0:
                                scale = min(700 / w, 300 / h, 1.0)
                                dw = int(w * scale) if scale < 1 else w
                                dh = int(h * scale) if scale < 1 else h
                                self._video_frames = []
                                for fi in range(n):
                                    arr = np.frombuffer(
                                        raw[fi*fs:(fi+1)*fs],
                                        dtype=np.uint8).reshape(h, w, 3)
                                    pil = Image.fromarray(arr)
                                    if scale < 1:
                                        pil = pil.resize((dw, dh), BILINEAR)
                                    self._video_frames.append(
                                        ImageTk.PhotoImage(pil))
                                self._video_idx = 0
                                self._play_gen = getattr(self, '_play_gen', 0) + 1
                                self._play_preview_loop(self._play_gen)
            except Exception:
                pass
        self.after(5000, self._check_preview)

    def _play_preview_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        self._video_idx = self._video_idx % len(self._video_frames)
        self.preview_label.config(image=self._video_frames[self._video_idx])
        self._video_idx += 1
        self.after(33, self._play_preview_loop, self._play_gen)



class VideoInferenceTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.model = None
        self.haar_rounds = 0
        self._video_frames = []
        self._play_gen = 0
        self._video_idx = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Video Inference (GT | Recon)", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(10, 0))
        f, self.ckpt_var = make_float(row1, "Checkpoint",
            os.path.join(PROJECT_ROOT, "synthyper_video_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True, padx=(0, 10))
        make_btn(row1, "Load", self.load_model, GREEN, width=8).pack(
            side="left", pady=(15, 0))

        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.T_var = make_spin(row2, "T (frames)", default=24, width=6)
        f.pack(side="left", padx=(0, 10))

        tk.Label(top, text="Feed synthetic video or mp4 file through the temporal VAE",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w", pady=(5, 0))

        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        make_btn(row3, "Test Synthetic", self.test_synthetic, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(row3, "Test MP4 File", self.test_mp4, BLUE).pack(
            side="left")

        self.status = tk.Label(top, text="No model loaded", bg=BG_PANEL,
                                fg=FG_DIM, font=FONT_SMALL)
        self.status.pack(fill="x", pady=(5, 0))

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

    def load_model(self):
        sys.path.insert(0, PROJECT_ROOT)
        from core.model import MiniVAE

        ckpt_path = self.ckpt_var.get().strip()
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            lat = config.get("latent_channels", 32)
            temporal = config.get("temporal", True)
            haar_mode = config.get("haar", "none")
            if haar_mode is True: haar_mode = "2x"
            elif not haar_mode or haar_mode is False: haar_mode = "none"
            haar_rounds = {"none": 0, "2x": 1, "4x": 2}.get(haar_mode, 0)
            vae_in_ch = 3 * (4 ** haar_rounds)
            self.haar_rounds = haar_rounds

            enc_ch, dec_ch = parse_arch_config(config)
            n_stages = len(dec_ch) if isinstance(dec_ch, tuple) else len(enc_ch) if isinstance(enc_ch, tuple) else 3
            if temporal:
                etd_str = config.get("encoder_time_downscale", "true,true,false")
                dtu_str = config.get("decoder_time_upscale", "false,true,true")
                if isinstance(etd_str, str):
                    etd = tuple(x.strip().lower() == "true" for x in etd_str.split(","))
                else:
                    etd = tuple(etd_str)
                if isinstance(dtu_str, str):
                    dtu = tuple(x.strip().lower() == "true" for x in dtu_str.split(","))
                else:
                    dtu = tuple(dtu_str)
            else:
                etd = tuple([False] * n_stages)
                dtu = tuple([False] * n_stages)

            self.model = MiniVAE(
                latent_channels=lat, image_channels=vae_in_ch, output_channels=vae_in_ch,
                encoder_channels=enc_ch, decoder_channels=dec_ch,
                encoder_time_downscale=etd, decoder_time_upscale=dtu,
                residual_shortcut=config.get("residual_shortcut", False),
                use_attention=config.get("use_attention", False),
                use_groupnorm=config.get("use_groupnorm", False),
            ).cuda()

            src_sd = ckpt["model"] if "model" in ckpt else ckpt
            target_sd = self.model.state_dict()
            loaded = 0
            for k, v in src_sd.items():
                if k in target_sd and v.shape == target_sd[k].shape:
                    target_sd[k] = v
                    loaded += 1
            self.model.load_state_dict(target_sd)
            self.model.eval()

            step = ckpt.get("global_step", "?")
            pc = sum(p.numel() for p in self.model.parameters())
            self.status.config(
                text=f"Loaded: haar={haar_mode}, in_ch={vae_in_ch}, lat={lat}, "
                     f"temporal={temporal}, step {step}, {pc:,} params, {loaded} weights")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    def test_synthetic(self):
        if self.model is None:
            self.status.config(text="Load a model first")
            return

        T = self.T_var.get()
        self.status.config(text=f"Generating T={T} synthetic clip...")

        def _bg():
            try:
                import torch
                sys.path.insert(0, PROJECT_ROOT)
                from core.generator import VAEpp0rGenerator

                gen = VAEpp0rGenerator(360, 640, device="cuda", bank_size=200,
                                          n_base_layers=64)
                gen.build_banks()

                hr = self.haar_rounds
                with torch.no_grad():
                    clip = gen.generate_sequence(1, T=T)
                    clip_rgb = clip.cuda()  # (1, T, 3, H, W)
                    x = haar_down_video(clip_rgb, hr) if hr > 0 else clip_rgb
                    recon, latent = chunked_vae_inference(self.model, x)

                trim = getattr(self.model, 'frames_to_trim', 0)
                T_out = recon.shape[1]
                T_in = x.shape[1]
                gt = clip_rgb[0, trim:trim + T_out, :3].float().cpu().numpy()
                if hr > 0:
                    recon_up = haar_up_video(recon.cuda() if not recon.is_cuda else recon, hr)
                    rc = recon_up[0, :, :3].clamp(0, 1).float().cpu().numpy()
                else:
                    rc = recon[0, :, :3].clamp(0, 1).float().cpu().numpy()

                self.after(0, lambda: self.status.config(
                    text=f"Input T={T_in}, Recon T={T_out}, trim={trim}"))
                self.after(0, self._show_video, gt, rc, T_out)
            except Exception as e:
                import traceback; traceback.print_exc()
                _e = str(e)
                self.after(0, lambda: self.status.config(text=f"Error: {_e}"))

        threading.Thread(target=_bg, daemon=True).start()

    def test_mp4(self):
        if self.model is None:
            self.status.config(text="Load a model first")
            return

        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.mkv *.avi"), ("All", "*.*")])
        if not path:
            return

        T = self.T_var.get()
        self.status.config(text=f"Loading {os.path.basename(path)}...")

        def _bg():
            try:
                import torch
                # Decode T frames
                cmd = ["ffmpeg", "-v", "quiet", "-i", path,
                       "-frames:v", str(T),
                       "-vf", "scale=640:360",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
                raw = subprocess.run(cmd, capture_output=True).stdout
                fs = 360 * 640 * 3
                n = min(len(raw) // fs, T)
                if n < 2:
                    self.after(0, lambda: self.status.config(text="Not enough frames"))
                    return

                frames = np.frombuffer(raw[:n*fs], dtype=np.uint8).reshape(n, 360, 640, 3)
                clip_rgb = torch.from_numpy(frames.astype(np.float32) / 255.0
                                            ).permute(0, 3, 1, 2).unsqueeze(0).cuda()

                hr = self.haar_rounds
                with torch.no_grad():
                    clip = haar_down_video(clip_rgb, hr) if hr > 0 else clip_rgb
                    recon, latent = chunked_vae_inference(self.model, clip)

                trim = getattr(self.model, 'frames_to_trim', 0)
                T_out = recon.shape[1]
                T_in = clip.shape[1]
                gt = clip_rgb[0, trim:trim + T_out, :3].float().cpu().numpy()
                if hr > 0:
                    recon_up = haar_up_video(recon.cuda() if not recon.is_cuda else recon, hr)
                    rc = recon_up[0, :, :3].clamp(0, 1).float().cpu().numpy()
                else:
                    rc = recon[0, :, :3].clamp(0, 1).float().cpu().numpy()

                self.after(0, lambda: self.status.config(
                    text=f"Input T={T_in}, Recon T={T_out}, trim={trim}"))
                self.after(0, self._show_video, gt, rc, T_out)
            except Exception as e:
                import traceback; traceback.print_exc()
                _e = str(e)
                self.after(0, lambda: self.status.config(text=f"Error: {_e}"))

        threading.Thread(target=_bg, daemon=True).start()

    def _show_video(self, gt, rc, T_show):
        """Play GT|Recon side by side as inline video loop."""
        H, W = 360, 640
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        frame_w = W * 2 + 4
        scale = min(700 / frame_w, 400 / H, 1.0)
        dw = int(frame_w * scale) if scale < 1 else frame_w
        dh = int(H * scale) if scale < 1 else H

        # Save inference video
        import time as _time
        inf_dir = os.path.join(PROJECT_ROOT, "synthyper_video_logs", "inference")
        os.makedirs(inf_dir, exist_ok=True)
        ts = int(_time.time())
        vid_path = os.path.join(inf_dir, f"video_inf_{ts}.mp4")
        cmd = ["ffmpeg", "-y", "-v", "quiet",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{frame_w}x{H}", "-r", "30",
               "-i", "pipe:0",
               "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", vid_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        self._video_frames = []
        try:
            for t in range(T_show):
                g = (gt[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                r = (rc[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                r = r[:g.shape[0], :g.shape[1]]
                frame = np.concatenate([g, sep, r], axis=1)
                proc.stdin.write(frame.tobytes())
                pil = Image.fromarray(frame)
                if scale < 1:
                    pil = pil.resize((dw, dh), BILINEAR)
                self._video_frames.append(ImageTk.PhotoImage(pil))
        except BrokenPipeError:
            pass

        proc.stdin.close()
        proc.wait()

        self._play_gen += 1
        self._play_video_loop(self._play_gen)

    def _play_video_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        idx = getattr(self, '_video_idx', 0) % len(self._video_frames)
        self.preview_label.config(image=self._video_frames[idx])
        self._video_idx = idx + 1
        self.after(33, self._play_video_loop, self._play_gen)


class FusionTrainTab(tk.Frame):
    """CPU VAE + MiniVAE fusion training tab."""
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._preview_photo = None
        self._last_mtime = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Fusion Training (CPU VAE + MiniVAE)", bg=BG_PANEL,
                 fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Frozen CPU VAE preprocessor → MiniVAE compression",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(2, 10))

        # CPU VAE checkpoint
        cpu_row = tk.Frame(top, bg=BG_PANEL)
        cpu_row.pack(fill="x", pady=(5, 0))
        f, self.cpu_ckpt_var = make_float(cpu_row, "CPU VAE checkpoint",
            os.path.join(PROJECT_ROOT, "cpu_vae_logs", "latest.pt"), width=50)
        f.pack(side="left", fill="x", expand=True)

        # MiniVAE architecture
        arch_row = tk.Frame(top, bg=BG_PANEL)
        arch_row.pack(fill="x", pady=(5, 0))
        f, self.latent_var = make_spin(arch_row, "Latent ch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.enc_ch_var = make_float(arch_row, "Enc ch", "64", width=12)
        f.pack(side="left", padx=(0, 10))
        f, self.dec_ch_var = make_float(arch_row, "Dec ch", "256,128,64", width=14)
        f.pack(side="left")

        # Training params
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(row1, "LR", "2e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(row1, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(row1, "Total steps", default=30000)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row1, "Precision", "bf16")
        f.pack(side="left")

        # Loss weights
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.w_l1_var = make_float(row2, "w_l1", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_mse_var = make_float(row2, "w_mse", "0.0")
        f.pack(side="left", padx=(0, 10))
        f, self.w_pixel_var = make_float(row2, "w_pixel", "0.5")
        f.pack(side="left", padx=(0, 10))
        f, self.w_lpips_var = make_float(row2, "w_lpips", "0.0")
        f.pack(side="left")

        # Save/log
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.save_every_var = make_spin(row3, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every_var = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.logdir_var = make_float(row3, "Log dir", "fusion_logs")
        f.pack(side="left", padx=(0, 10))
        f, self.resume_var = make_float(row3, "Resume", "")
        f.pack(side="left")

        # Options
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        self.disco_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row4, text="Disco Quadrant", variable=self.disco_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(0, 10))
        self.fresh_opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Fresh opt", variable=self.fresh_opt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left", padx=(0, 10))
        self.grad_ckpt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Grad checkpoint", variable=self.grad_ckpt_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       activebackground=BG_PANEL, font=FONT_SMALL
                       ).pack(side="left")

        # Preview image
        prev_row = tk.Frame(top, bg=BG_PANEL)
        prev_row.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(prev_row, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        from tkinter import filedialog
        make_btn(ef, "Browse",
                 lambda: self.preview_img_var.set(
                     filedialog.askopenfilename(
                         filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                                    ("All", "*.*")]) or self.preview_img_var.get()),
                 ACCENT, width=7).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

        # Buttons
        btn_row = tk.Frame(top, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(10, 0))
        make_btn(btn_row, "Train", self.start, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn_row, "Stop (save)", self.stop_save, BLUE).pack(side="left", padx=(0, 5))
        make_btn(btn_row, "Kill", self.kill, RED).pack(side="left")

        # Preview
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)

        # Log
        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)

        self._check_preview()

    def start(self):
        cmd = [VENV_PYTHON, "-m", "training.train_fusion",
               "--cpu-vae-ckpt", self.cpu_ckpt_var.get(),
               "--latent-ch", str(self.latent_var.get()),
               "--enc-ch", self.enc_ch_var.get(),
               "--dec-ch", self.dec_ch_var.get(),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--precision", self.prec_var.get(),
               "--w-l1", self.w_l1_var.get(),
               "--w-mse", self.w_mse_var.get(),
               "--w-pixel", self.w_pixel_var.get(),
               "--w-lpips", self.w_lpips_var.get(),
               "--save-every", str(self.save_every_var.get()),
               "--preview-every", str(self.preview_every_var.get()),
               "--logdir", self.logdir_var.get()]
        resume = self.resume_var.get().strip()
        if resume:
            cmd += ["--resume", resume]
        if self.disco_var.get():
            cmd.append("--disco")
        if self.fresh_opt_var.get():
            cmd.append("--fresh-opt")
        if self.grad_ckpt_var.get():
            cmd.append("--grad-checkpoint")
        prev_img = self.preview_img_var.get().strip()
        if prev_img:
            cmd += ["--preview-image", prev_img]
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop_save(self):
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
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    from PIL import Image as PILImg, ImageTk as PILTk
                    pil = PILImg.open(preview)
                    w, h = pil.size
                    if w > 900:
                        scale = 900 / w
                        pil = pil.resize((int(w * scale), int(h * scale)),
                                         BILINEAR)
                    self._preview_photo = PILTk.PhotoImage(pil)
                    self.preview_label.config(image=self._preview_photo)
            except Exception:
                pass
        self.after(2000, self._check_preview)


# -- Flatten Experiment Tab -----------------------------------------------------

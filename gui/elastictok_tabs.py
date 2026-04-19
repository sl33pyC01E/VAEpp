#!/usr/bin/env python3
"""ElasticTok training tab for the main VAEpp0r GUI.

Single `ElasticTokTrainTab` that does both generator-driven training and
single-video overfit testing, with an Overfit toggle at the top of the
config. Launches `training.train_elastictok` as a subprocess (via venv's
python from gui.common.VENV_PYTHON) and polls `preview_latest.mp4`.
"""

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog

import numpy as np
from PIL import Image, ImageTk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gui.common import (
    BG, BG_PANEL, BG_INPUT, FG, FG_DIM, ACCENT, GREEN, BLUE, RED,
    FONT, FONT_TITLE, FONT_BOLD, FONT_SMALL,
    VENV_PYTHON, BILINEAR,
    ProcRunner, make_btn, make_spin, make_float, make_log,
)
from core.elastictok.model import CONFIGS as _ELASTIC_CONFIGS


def _estimate_param_count(cfg_name, patch_size, in_channels,
                          bottleneck_type, vae_bn_dim, fsq_levels):
    """Analytic param count for ElasticTok. Matches the real model to
    within a few % — close enough for a live readout. No GPU alloc."""
    d = _ELASTIC_CONFIGS.get(cfg_name, {})
    H = int(d.get("hidden_size", 256))
    FFN = int(d.get("intermediate_size", 256))
    L_enc = int(d.get("num_encoder_layers", 2))
    L_dec = int(d.get("num_decoder_layers", 2))
    Tp, Hp, Wp = patch_size
    patch_dim = int(Tp * Hp * Wp * in_channels)
    # per-layer: attn (4*H*H) + SwiGLU MLP (3*H*FFN) + 2 RMSNorm scales (2*H)
    per_layer = 4 * H * H + 3 * H * FFN + 2 * H
    # bottleneck projection dims
    if bottleneck_type == "vae":
        proj_in = 2 * vae_bn_dim
        proj_out = vae_bn_dim
    else:  # fsq
        n_lvl = len(fsq_levels)
        proj_in = n_lvl
        proj_out = n_lvl
    # non-layer params: in_proj + out_proj + pre_quant + post_quant +
    # encoder ln_f + decoder ln_f + is_kept_embed + is_masked_embed
    other = (patch_dim * H            # in_proj
             + H * patch_dim          # out_proj
             + H * proj_in            # pre_quant
             + proj_out * H           # post_quant
             + 4 * H)                 # 2 ln_f + 2 emb tables (size H each)
    return (L_enc + L_dec) * per_layer + other


# =============================================================================
# MP4 preview watcher
# =============================================================================

class _Mp4PreviewMixin:
    """Verbatim copy of VideoTrainTab's mp4 preview pattern.

    Uses a generation counter (`_play_gen`) so that when a new mp4
    arrives, any in-flight playback loop on the previous frame list
    self-invalidates on its next tick. All work (ffprobe/ffmpeg/resize)
    runs on the Tk main thread — mirrors what the other tabs do and
    what the user has confirmed works."""
    def init_mp4_preview(self, label_widget):
        self._mp4_label = label_widget
        self._video_frames = []
        self._video_idx = 0
        self._play_gen = 0
        self._last_mtime = 0
        self._check_preview()

    def _mp4_logdir(self):
        if hasattr(self, "logdir_var"):
            v = self.logdir_var.get()
            if not os.path.isabs(v):
                v = os.path.join(PROJECT_ROOT, v)
            return v
        return "."

    def _check_preview(self):
        preview = os.path.join(self._mp4_logdir(), "preview_latest.mp4")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries",
                         "stream=width,height",
                         "-of", "csv=p=0", preview],
                        capture_output=True, text=True)
                    parts = probe.stdout.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            w, h = int(parts[0]), int(parts[1])
                        except (ValueError, TypeError):
                            w, h = 0, 0
                        if w > 0 and h > 0:
                            cmd = ["ffmpeg", "-v", "quiet", "-i", preview,
                                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "pipe:1"]
                            raw = subprocess.run(
                                cmd, capture_output=True).stdout
                            fs = w * h * 3
                            n = len(raw) // fs
                            if n > 0:
                                # ElasticTok previews are wider than
                                # FlattenVideoTab's (4-cell ref row +
                                # synth row), so the 900×350 cap left
                                # huge dead space once the log got
                                # compact. Bump to 1800×900 so the
                                # preview actually uses the freed
                                # vertical/horizontal room.
                                scale = min(1800 / w, 900 / h, 1.0)
                                dw = int(w * scale) if scale < 1 else w
                                dh = int(h * scale) if scale < 1 else h
                                self._video_frames = []
                                for fi in range(n):
                                    arr = np.frombuffer(
                                        raw[fi*fs:(fi+1)*fs],
                                        dtype=np.uint8).reshape(h, w, 3)
                                    pil = Image.fromarray(arr)
                                    if scale < 1:
                                        pil = pil.resize(
                                            (dw, dh), BILINEAR)
                                    self._video_frames.append(
                                        ImageTk.PhotoImage(pil))
                                self._video_idx = 0
                                self._play_gen += 1
                                self._play_preview_loop(self._play_gen)
            except Exception:
                pass
        self.after(5000, self._check_preview)

    def _play_preview_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        self._video_idx = self._video_idx % len(self._video_frames)
        self._mp4_label.config(image=self._video_frames[self._video_idx])
        self._video_idx += 1
        self.after(33, self._play_preview_loop, self._play_gen)


# =============================================================================
# Train tab
# =============================================================================

class ElasticTokTrainTab(tk.Frame, _Mp4PreviewMixin):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="ElasticTok — PyTorch port (LargeWorldModel)",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        # Info label — full width at top of the config block.
        self.info_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.info_var, bg=BG_PANEL,
                 fg=ACCENT, font=FONT_SMALL, anchor="w").pack(
                     fill="x", pady=(6, 0))

        # Two-column config: left and right both get rows. Uses the
        # horizontal space that was previously dead on the right side.
        cfg_split = tk.Frame(top, bg=BG_PANEL)
        cfg_split.pack(fill="x", pady=(8, 0))
        cfg_left = tk.Frame(cfg_split, bg=BG_PANEL)
        cfg_left.pack(side="left", fill="both", expand=True,
                       padx=(0, 10))
        cfg_right = tk.Frame(cfg_split, bg=BG_PANEL)
        cfg_right.pack(side="left", fill="both", expand=True)

        # --- Left column ---
        # Clip geometry
        clip_row = tk.Frame(cfg_left, bg=BG_PANEL)
        clip_row.pack(fill="x", pady=(0, 0))
        f, self.H_var = make_spin(clip_row, "H", default=368, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(clip_row, "W", default=640, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.T_var = make_spin(clip_row, "T (frames)", default=4, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.patch_var = make_float(
            clip_row, "patch_size Tp,Hp,Wp", "1,16,16", width=10)
        f.pack(side="left", padx=(0, 10))
        f, self.haar_var = make_spin(
            clip_row, "haar lvls (0=off)", default=0, width=4)
        f.pack(side="left")

        # Overfit-mode toggle row (left column).
        ov_row = tk.Frame(cfg_left, bg=BG_PANEL)
        ov_row.pack(fill="x", pady=(3, 0))
        self.overfit_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ov_row, text="Overfit on single video (instead of generator):",
            variable=self.overfit_var, bg=BG_PANEL, fg=FG,
            selectcolor=BG_INPUT, font=FONT_SMALL).pack(
                side="left", padx=(0, 8))
        self.overfit_vid_var = tk.StringVar(value="")
        tk.Entry(ov_row, textvariable=self.overfit_vid_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(ov_row, "Browse", self._pick_overfit_vid, BLUE, 8).pack(
            side="left", padx=(0, 4))
        make_btn(ov_row, "Clear",
                 lambda: self.overfit_vid_var.set(""), RED, 6).pack(
                     side="left", padx=(0, 8))
        f, self.overfit_skip_var = make_spin(
            ov_row, "Skip frames", default=0, width=5)
        f.pack(side="left")

        # Aux real-video dataset row (left column). Walks a directory
        # for video files, streams random T-frame windows via DataLoader
        # workers. Overfit wins over this; mix < 1.0 blends with
        # generator clips.
        aux_row = tk.Frame(cfg_left, bg=BG_PANEL)
        aux_row.pack(fill="x", pady=(3, 0))
        tk.Label(aux_row, text="Aux dataset dir:",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(
                     side="left", padx=(0, 4))
        self.aux_dir_var = tk.StringVar(
            value=os.path.join(PROJECT_ROOT, "trainingdata", "frames"))
        tk.Entry(aux_row, textvariable=self.aux_dir_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(aux_row, "Browse", self._pick_aux_dir, BLUE, 8).pack(
            side="left", padx=(0, 4))
        make_btn(aux_row, "Clear",
                 lambda: self.aux_dir_var.set(""), RED, 6).pack(
                     side="left", padx=(0, 8))
        f, self.aux_mix_var = make_float(
            aux_row, "mix", "1.0", width=5)
        f.pack(side="left", padx=(0, 8))
        f, self.aux_workers_var = make_spin(
            aux_row, "workers", default=4, width=4)
        f.pack(side="left", padx=(0, 8))
        self.aux_recursive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(aux_row, text="recursive",
                       variable=self.aux_recursive_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       font=FONT_SMALL).pack(side="left")

        # Arch (left column)
        arch_row = tk.Frame(cfg_left, bg=BG_PANEL)
        arch_row.pack(fill="x", pady=(3, 0))
        self.cfg_var = tk.StringVar(value="debug")
        tk.Label(arch_row, text="config", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        tk.OptionMenu(arch_row, self.cfg_var,
                      "debug", "tiny", "small", "base", "200m", "large"
                      ).pack(side="left", padx=(0, 10))
        self.bt_var = tk.StringVar(value="vae")
        tk.Label(arch_row, text="bottleneck", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        tk.OptionMenu(arch_row, self.bt_var, "vae", "fsq"
                      ).pack(side="left", padx=(0, 10))
        f, self.vae_bn_var = make_spin(
            arch_row, "vae_bn_dim", default=8, width=4)
        f.pack(side="left", padx=(0, 10))
        f, self.fsq_var = make_float(
            arch_row, "fsq_levels", "8,8,8,5,5,5", width=14)
        f.pack(side="left")

        # Sequence / mask / loss (left column)
        seq_row = tk.Frame(cfg_left, bg=BG_PANEL)
        seq_row.pack(fill="x", pady=(3, 0))
        f, self.max_seq_var = make_spin(
            seq_row, "max_sequence_length", default=4096, width=7)
        f.pack(side="left", padx=(0, 10))
        f, self.max_toks_var = make_spin(
            seq_row, "max_toks (block)", default=3680, width=7)
        f.pack(side="left", padx=(0, 10))
        f, self.min_toks_var = make_spin(
            seq_row, "min_toks", default=128, width=6)
        f.pack(side="left", padx=(0, 10))
        # keep_upper decouples the per-block keep upper bound from the
        # attention block size. 0 = use max_toks (backward compat).
        # Set equal to min_toks for fixed-K training at any K ≤ block.
        f, self.keep_upper_var = make_spin(
            seq_row, "keep_upper (0=max_toks)", default=0, width=7)
        f.pack(side="left", padx=(0, 10))
        f, self.fpb_var = make_spin(
            seq_row, "frames_per_block", default=4, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.w_lpips_var = make_float(
            seq_row, "lpips_loss_ratio", "0.1", width=6)
        f.pack(side="left")

        # Optim (left column)
        opt_row = tk.Frame(cfg_left, bg=BG_PANEL)
        opt_row.pack(fill="x", pady=(3, 0))
        f, self.lr_var = make_float(opt_row, "LR", "1e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.end_lr_var = make_float(opt_row, "end_LR", "1e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.lr_warmup_var = make_spin(
            opt_row, "LR warmup", default=2000, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.wd_var = make_float(opt_row, "weight_decay", "1e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.gc_var = make_float(opt_row, "grad_clip", "1.0")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(opt_row, "batch", default=2, width=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(
            opt_row, "total_steps", default=100000)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(opt_row, "precision", "bf16",
                                        width=5)
        f.pack(side="left")

        # --- Right column ---
        # Prefill bank (read-only here; build/grow in Data -> Prefill
        # Bank). Loads clips from the dir + mixes them in at the given
        # ratio. aux_mix + prefill_mix cap at 1.0; remainder = live gen.
        pf_row = tk.Frame(cfg_right, bg=BG_PANEL)
        pf_row.pack(fill="x", pady=(0, 0))
        tk.Label(pf_row, text="Prefill bank dir:",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(
                     side="left", padx=(0, 4))
        self.prefill_dir_var = tk.StringVar(
            value=os.path.join(
                PROJECT_ROOT, "prefill_banks", "default"))
        tk.Entry(pf_row, textvariable=self.prefill_dir_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(pf_row, "Browse", self._pick_prefill_dir,
                 BLUE, 8).pack(side="left", padx=(0, 4))
        make_btn(pf_row, "Clear",
                 lambda: self.prefill_dir_var.set(""), RED, 6).pack(
                     side="left", padx=(0, 8))
        f, self.prefill_mix_var = make_float(
            pf_row, "mix", "0.5", width=5)
        f.pack(side="left")

        # Data
        data_row = tk.Frame(cfg_right, bg=BG_PANEL)
        data_row.pack(fill="x", pady=(3, 0))
        f, self.bank_var = make_spin(data_row, "bank", default=5000, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.layers_var = make_spin(
            data_row, "layers", default=128, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.pool_var = make_spin(data_row, "pool", default=200, width=5)
        f.pack(side="left", padx=(0, 10))
        self.disco_var = tk.BooleanVar(value=False)
        tk.Checkbutton(data_row, text="Disco BG",
                       variable=self.disco_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT_SMALL).pack(
                           side="left")

        # I/O (right column)
        io_row = tk.Frame(cfg_right, bg=BG_PANEL)
        io_row.pack(fill="x", pady=(3, 0))
        f, self.log_every_var = make_spin(
            io_row, "log_every", default=1, width=4)
        f.pack(side="left", padx=(0, 10))
        f, self.save_every_var = make_spin(
            io_row, "save_every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every_var = make_spin(
            io_row, "preview_every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.keeps_var = make_float(
            io_row, "preview keeps", "128,512,2048", width=14)
        f.pack(side="left", padx=(0, 10))
        f, self.logdir_var = make_float(
            io_row, "logdir", "synthyper_elastictok_logs", width=30)
        f.pack(side="left")

        # Preview ref (right column)
        pimg_row = tk.Frame(cfg_right, bg=BG_PANEL)
        pimg_row.pack(fill="x", pady=(3, 0))
        tk.Label(pimg_row, text="Preview video (optional):",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(
                     side="left", padx=(0, 4))
        self.preview_img_var = tk.StringVar(value="")
        tk.Entry(pimg_row, textvariable=self.preview_img_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL, width=50).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(pimg_row, "Browse", self._pick_preview_vid, BLUE, 8).pack(
            side="left", padx=(0, 4))
        make_btn(pimg_row, "Clear",
                 lambda: self.preview_img_var.set(""), RED, 6).pack(
                     side="left", padx=(0, 8))
        f, self.preview_skip_var = make_spin(
            pimg_row, "Skip frames", default=0, width=5)
        f.pack(side="left", padx=(0, 8))
        f, self.preview_T_var = make_spin(
            pimg_row, "Total frames (0=use T)", default=0, width=5)
        f.pack(side="left")

        # Resume (right column)
        res_row = tk.Frame(cfg_right, bg=BG_PANEL)
        res_row.pack(fill="x", pady=(3, 0))
        f, self.resume_var = make_float(
            res_row, "Resume",
            os.path.join(PROJECT_ROOT, "synthyper_elastictok_logs",
                         "latest.pt"), width=60)
        f.pack(side="left")
        chk = tk.Frame(cfg_right, bg=BG_PANEL)
        chk.pack(fill="x", pady=(3, 0))
        self.use_latest_var = tk.BooleanVar(value=False)
        tk.Checkbutton(chk, text="Resume from latest.pt",
                       variable=self.use_latest_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.fresh_opt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(chk, text="Fresh optimizer",
                       variable=self.fresh_opt_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(
                           side="left", padx=(10, 0))

        # Buttons
        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Train", self.start, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Stop (save)", self.stop_save, BLUE).pack(
            side="left", padx=(0, 5))
        make_btn(btn, "Kill", self.kill, RED).pack(side="left")

        # Log pinned to bottom at a fixed height (10 rows). Preview
        # label takes ALL remaining vertical space, image centered via
        # label anchor. Now that the mp4 itself is geometrically
        # correct (no frame-misalignment scroll), fill=both on the
        # label is safe and gives the preview room to breathe.
        self.log = make_log(self)
        self.log.configure(height=8)
        self.log.pack(side="bottom", fill="x", padx=5, pady=5)
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)
        self.runner = ProcRunner(self.log)

        for v in (self.H_var, self.W_var, self.T_var, self.patch_var,
                  self.haar_var, self.max_seq_var,
                  self.max_toks_var, self.min_toks_var,
                  self.keep_upper_var, self.bt_var,
                  self.vae_bn_var, self.fsq_var, self.keeps_var,
                  self.cfg_var):
            try:
                v.trace_add("write", lambda *a: self._update_info())
            except Exception:
                pass
        self.init_mp4_preview(self.preview_label)
        self._update_info()

    def _update_info(self):
        try:
            H = int(self.H_var.get()); W = int(self.W_var.get())
            T = int(self.T_var.get())
            Tp, Hp, Wp = (int(x) for x in self.patch_var.get().split(","))
            # L is computed on the POST-Haar grid (what the model sees).
            # Without this, Haar-enabled runs wrongly triggered the L
            # warnings on working configs.
            try:
                haar = max(0, int(self.haar_var.get()))
            except Exception:
                haar = 0
            haar_scale = 2 ** haar
            H_post = H // haar_scale if haar_scale > 0 else H
            W_post = W // haar_scale if haar_scale > 0 else W
            nT = T // Tp
            nH = H_post // Hp
            nW = W_post // Wp
            L = nT * nH * nW
            max_toks = int(self.max_toks_var.get())
            min_toks = int(self.min_toks_var.get())
            try:
                keep_upper_raw = int(self.keep_upper_var.get())
            except Exception:
                keep_upper_raw = 0
            keep_upper = (keep_upper_raw if keep_upper_raw > 0
                          else max_toks)
            n_blocks = L // max_toks if max_toks > 0 else 0
            bt = self.bt_var.get()

            # Dim per latent token = what the downstream model ingests/
            # predicts per-token. VAE: bottleneck_dim. FSQ: len(levels).
            if bt == "vae":
                dim_per_tok = int(self.vae_bn_var.get())
                tok_tag = f"{dim_per_tok} floats/tok (vae)"
            else:
                try:
                    levels = [int(x) for x in self.fsq_var.get().split(",")
                              if x.strip()]
                except Exception:
                    levels = []
                dim_per_tok = len(levels)
                tok_tag = f"{dim_per_tok} ints/tok (fsq {levels})"

            raw_floats = T * H * W * 3
            in_channels = 3 * (4 ** haar)
            haar_tag = (f"  Haar {haar}x ({H}x{W}->{H_post}x{W_post}, "
                        f"3ch->{in_channels}ch)" if haar > 0 else "")

            # Analytic parameter count for the chosen preset + config.
            try:
                fsq_levels_parsed = [int(x) for x in
                                     self.fsq_var.get().split(",")
                                     if x.strip()]
            except Exception:
                fsq_levels_parsed = []
            try:
                n_params = _estimate_param_count(
                    self.cfg_var.get(),
                    (Tp, Hp, Wp), in_channels, bt,
                    int(self.vae_bn_var.get()), fsq_levels_parsed)
                p_tag = f"  | model ≈ {n_params/1e6:.1f}M params"
            except Exception:
                p_tag = ""

            keep_tag = (f"tail-drop keep in [{min_toks}, {keep_upper}]"
                        + (f" (< block {max_toks})"
                           if keep_upper != max_toks else ""))
            head = (f"[{self.cfg_var.get()}]  "
                    f"grid ({nT},{nH},{nW})  L={L}  "
                    f"max_toks={max_toks}  n_blocks={n_blocks}  "
                    f"{keep_tag}  "
                    f"bottleneck={bt} ({tok_tag})"
                    f"{haar_tag}"
                    f"  | raw clip = {raw_floats:,} floats"
                    f"{p_tag}")

            # Per-preview-keep ingest/predict footprint.
            try:
                keeps = [int(x) for x in self.keeps_var.get().split(",")
                         if x.strip()]
            except Exception:
                keeps = []
            # Keeps are PER BLOCK. Total clip ingest = keep × n_blocks ×
            # dim_per_tok. The raw-clip comparison below is per entire T
            # clip so units line up.
            if keeps and dim_per_tok > 0:
                lines = []
                for k in sorted(set(keeps)):
                    k_eff = min(k, max_toks if max_toks > 0 else k)
                    toks_per_clip = k_eff * max(1, n_blocks)
                    n_units = toks_per_clip * dim_per_tok
                    ratio = raw_floats / n_units if n_units else 0.0
                    suffix = (f" ({k_eff}/blk × {n_blocks} blk)"
                              if n_blocks > 1 else "")
                    lines.append(
                        f"keep={k_eff}: {toks_per_clip}*{dim_per_tok} = "
                        f"{n_units:,} units/clip{suffix}  "
                        f"({ratio:.0f}x vs raw)")
                msg = head + "   ||   " + "  |  ".join(lines)
            else:
                msg = head

            # Warnings — now computed against the POST-Haar L that the
            # model actually sees.
            if max_toks > 0 and L % max_toks != 0:
                msg += (f"  [WARN: L={L} not divisible by "
                        f"max_toks={max_toks}]")
            if L > int(self.max_seq_var.get()):
                msg += (f"  [WARN: L={L} > max_sequence_length="
                        f"{self.max_seq_var.get()}]")
            self.info_var.set(msg)
        except Exception as e:
            self.info_var.set(f"(info calc error: {e})")

    def _pick_preview_vid(self):
        path = filedialog.askopenfilename(
            title="Select preview reference video",
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm *.avi"),
                       ("All", "*.*")])
        if path:
            self.preview_img_var.set(path)

    def _pick_overfit_vid(self):
        path = filedialog.askopenfilename(
            title="Select video to overfit on",
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm *.avi"),
                       ("All", "*.*")])
        if path:
            self.overfit_vid_var.set(path)

    def _pick_aux_dir(self):
        path = filedialog.askdirectory(
            title="Select aux dataset directory")
        if path:
            self.aux_dir_var.set(path)

    def _pick_prefill_dir(self):
        path = filedialog.askdirectory(
            title="Select prefill bank directory")
        if path:
            self.prefill_dir_var.set(path)

    def start(self):
        cmd = [VENV_PYTHON, "-m",
               "training.train_elastictok",
               "--config-name", self.cfg_var.get(),
               "--patch-size", str(self.patch_var.get()),
               "--haar-levels", str(self.haar_var.get()),
               "--bottleneck-type", self.bt_var.get(),
               "--fsq-levels", str(self.fsq_var.get()),
               "--vae-bottleneck-dim", str(self.vae_bn_var.get()),
               "--max-sequence-length", str(self.max_seq_var.get()),
               "--max-toks", str(self.max_toks_var.get()),
               "--min-toks", str(self.min_toks_var.get()),
               "--keep-upper", str(self.keep_upper_var.get()),
               "--frames-per-block", str(self.fpb_var.get()),
               "--lpips-loss-ratio", str(self.w_lpips_var.get()),
               "--T", str(self.T_var.get()),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--pool-size", str(self.pool_var.get()),
               "--bank-size", str(self.bank_var.get()),
               "--n-layers", str(self.layers_var.get()),
               "--batch-size", str(self.batch_var.get()),
               "--lr", str(self.lr_var.get()),
               "--end-lr", str(self.end_lr_var.get()),
               "--lr-warmup", str(self.lr_warmup_var.get()),
               "--weight-decay", str(self.wd_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--grad-clip", str(self.gc_var.get()),
               "--precision", str(self.prec_var.get()),
               "--log-every", str(self.log_every_var.get()),
               "--save-every", str(self.save_every_var.get()),
               "--preview-every", str(self.preview_every_var.get()),
               "--keeps", str(self.keeps_var.get()),
               "--logdir", self.logdir_var.get()]
        # Overfit toggle overrides the generator path.
        overfit_path = self.overfit_vid_var.get().strip()
        if self.overfit_var.get() and overfit_path:
            cmd += ["--overfit-video", overfit_path,
                    "--overfit-frame-skip",
                    str(self.overfit_skip_var.get())]
        elif self.overfit_var.get() and not overfit_path:
            self.runner._append(
                "[Overfit is toggled on but no video picked; "
                "falling back to generator]\n")
        # Aux real-video dataset (empty string = disabled).
        aux_dir = self.aux_dir_var.get().strip()
        if aux_dir:
            cmd += ["--aux-data-dir", aux_dir,
                    "--aux-data-mix", str(self.aux_mix_var.get()),
                    "--aux-data-workers",
                    str(self.aux_workers_var.get()),
                    "--aux-data-recursive",
                    "1" if self.aux_recursive_var.get() else "0"]
        # Prefill bank (empty = off).
        pf_dir = self.prefill_dir_var.get().strip()
        if pf_dir:
            cmd += ["--prefill-dir", pf_dir,
                    "--prefill-mix", str(self.prefill_mix_var.get())]
        if self.disco_var.get():
            cmd.append("--disco")
        pvid = self.preview_img_var.get().strip()
        if pvid:
            cmd += ["--preview-image", pvid,
                    "--preview-frame-skip",
                    str(self.preview_skip_var.get())]
        try:
            pT = int(self.preview_T_var.get())
        except Exception:
            pT = 0
        if pT > 0:
            cmd += ["--preview-T", str(pT)]
        if self.use_latest_var.get():
            cmd += ["--resume", self.resume_var.get().strip()]
            if self.fresh_opt_var.get():
                cmd.append("--fresh-opt")
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop_save(self):
        logdir = self.logdir_var.get()
        if not os.path.isabs(logdir):
            logdir = os.path.join(PROJECT_ROOT, logdir)
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()


# =============================================================================
# Inference tab — load a trained checkpoint, reconstruct a video clip at
# multiple keep budgets. Runs model.eval() in-process (no subprocess).
# =============================================================================

class ElasticTokInfTab(tk.Frame, _Mp4PreviewMixin):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.model = None
        self._model_cfg = None
        self._model_haar_levels = 0
        self._model_H = 368
        self._model_W = 640
        self._busy = False
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="ElasticTok Inference",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        self.status_var = tk.StringVar(value="No checkpoint loaded.")
        tk.Label(top, textvariable=self.status_var, bg=BG_PANEL,
                 fg=ACCENT, font=FONT_SMALL, anchor="w").pack(
                     fill="x", pady=(6, 0))

        # Checkpoint row
        ckpt_row = tk.Frame(top, bg=BG_PANEL)
        ckpt_row.pack(fill="x", pady=(8, 0))
        tk.Label(ckpt_row, text="Checkpoint:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.ckpt_var = tk.StringVar(value=os.path.join(
            PROJECT_ROOT, "synthyper_elastictok_logs", "latest.pt"))
        tk.Entry(ckpt_row, textvariable=self.ckpt_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(ckpt_row, "Browse", self._pick_ckpt, BLUE, 8).pack(
            side="left", padx=(0, 4))
        make_btn(ckpt_row, "Load", self._load_ckpt, GREEN, 6).pack(
            side="left")

        # Input video row
        vid_row = tk.Frame(top, bg=BG_PANEL)
        vid_row.pack(fill="x", pady=(3, 0))
        tk.Label(vid_row, text="Input video:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.vid_var = tk.StringVar(value="")
        tk.Entry(vid_row, textvariable=self.vid_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(vid_row, "Browse", self._pick_video, BLUE, 8).pack(
            side="left", padx=(0, 4))
        make_btn(vid_row, "Clear",
                 lambda: self.vid_var.set(""), RED, 6).pack(side="left")

        # Params row
        par_row = tk.Frame(top, bg=BG_PANEL)
        par_row.pack(fill="x", pady=(3, 0))
        f, self.T_var = make_spin(par_row, "T (frames)", default=48, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.skip_var = make_spin(par_row, "Frame skip", default=0,
                                      width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.keeps_var = make_float(
            par_row, "keeps (CSV)", "128,256,512,2048", width=18)
        f.pack(side="left", padx=(0, 10))
        f, self.logdir_var = make_float(
            par_row, "Output logdir",
            "synthyper_elastictok_inf_logs", width=30)
        f.pack(side="left")

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(10, 0))
        make_btn(btn, "Reconstruct", self._reconstruct, ACCENT).pack(
            side="left")

        # Same layout as FlattenVideoTab / VideoTrainTab: preview at
        # natural image size (pady=5), log fills the remaining space.
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)
        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.init_mp4_preview(self.preview_label)

    def _append_log(self, msg):
        try:
            self.log.config(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.config(state="disabled")
        except Exception:
            pass

    def _pick_ckpt(self):
        p = filedialog.askopenfilename(
            title="Select ElasticTok checkpoint",
            filetypes=[("Checkpoint", "*.pt"), ("All", "*.*")],
            initialdir=os.path.join(PROJECT_ROOT,
                                     "synthyper_elastictok_logs"))
        if p:
            self.ckpt_var.set(p)

    def _pick_video(self):
        p = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm *.avi"),
                       ("All", "*.*")])
        if p:
            self.vid_var.set(p)

    def _load_ckpt(self):
        import threading
        self.status_var.set("Loading...")

        def _work():
            try:
                import torch
                from core.elastictok import ElasticTokConfig, ElasticTok
                path = self.ckpt_var.get().strip()
                if not os.path.isfile(path):
                    raise SystemExit(f"not found: {path}")
                ckpt = torch.load(
                    path, map_location="cpu", weights_only=False)
                cfg_dict = ckpt.get("config") or {}
                cfg = (ElasticTokConfig(**cfg_dict) if cfg_dict
                       else ElasticTokConfig.load_config("debug"))
                model = ElasticTok(cfg)
                model.load_state_dict(ckpt["model"], strict=False)
                device = ("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device).eval()
                # Derive haar_levels from in_channels (= 3 * 4**N)
                ic = int(getattr(cfg, "in_channels", 3))
                haar = 0
                while ic > 3 and haar < 4:
                    ic //= 4
                    haar += 1
                # Pull training H,W from saved args if available
                saved_args = ckpt.get("args") or {}
                H_saved = int(saved_args.get("H", 368))
                W_saved = int(saved_args.get("W", 640))
                self.model = model
                self._model_cfg = cfg
                self._model_haar_levels = haar
                self._model_H = H_saved
                self._model_W = W_saved
                n = sum(p.numel() for p in model.parameters())
                step = ckpt.get("step", "?")
                msg = (f"Loaded: {n/1e6:.1f}M params  step={step}  "
                       f"bottleneck={cfg.bottleneck_type}  "
                       f"haar={haar}  train H,W={H_saved},{W_saved}  "
                       f"device={device}")
                self.after(0, lambda: (self.status_var.set(msg),
                                        self._append_log(f"[load] {path}")))
            except Exception as e:
                import traceback; traceback.print_exc()
                self.after(0, lambda e=e: (
                    self.status_var.set(f"Load failed: {e}"),
                    self._append_log(f"[load] FAILED: {e}")))
        threading.Thread(target=_work, daemon=True).start()

    def _reconstruct(self):
        if self._busy:
            self._append_log("[inf] already running")
            return
        if self.model is None:
            self.status_var.set("Load a checkpoint first.")
            return
        vid = self.vid_var.get().strip()
        if not vid or not os.path.isfile(vid):
            self.status_var.set("Pick a valid input video.")
            return
        self._busy = True
        self.status_var.set("Reconstructing...")

        import threading

        def _work():
            import time as _t
            try:
                import torch
                # Reuse the training script's chunker + frame decoder so
                # layout + math match the Train preview exactly.
                from training.train_elastictok import (
                    _reconstruct_at_keep, _decode_video_frames,
                )
                cfg = self._model_cfg
                model = self.model
                device = next(model.parameters()).device
                T = int(self.T_var.get())
                skip = int(self.skip_var.get())
                H = int(self._model_H)
                W = int(self._model_W)
                haar = int(self._model_haar_levels)
                patch_size = tuple(cfg.patch_size)

                self.after(0, lambda: self._append_log(
                    f"[inf] decoding {T}f @ {H}x{W} (skip {skip})"))
                frames = _decode_video_frames(vid, skip, T, W, H)
                if not frames:
                    raise SystemExit("decoded 0 frames")
                while len(frames) < T:
                    frames.append(frames[-1])
                arr = np.stack(frames[:T]).astype(np.float32) / 255.0
                clip = torch.from_numpy(arr).permute(
                    0, 3, 1, 2).unsqueeze(0).to(device)
                clip_m11 = clip * 2 - 1

                try:
                    keeps = sorted({int(k) for k in
                                    self.keeps_var.get().split(",")
                                    if k.strip()})
                except Exception:
                    keeps = [int(cfg.max_toks)]
                keeps = [min(k, int(cfg.max_toks)) for k in keeps]

                recons = {}
                for k in keeps:
                    self.after(0, lambda k=k: self._append_log(
                        f"[inf] reconstructing keep={k}"))
                    with torch.no_grad():
                        r = _reconstruct_at_keep(
                            model, clip_m11, patch_size, keep=k,
                            max_toks=int(cfg.max_toks),
                            block_size=int(cfg.max_toks),
                            device=device, haar_levels=haar)
                    recons[k] = r.cpu().numpy()
                gt_np = clip.cpu().numpy()

                # Render MP4: GT | R@k1 | ... | R@kN (all at same H)
                logdir = self.logdir_var.get()
                if not os.path.isabs(logdir):
                    logdir = os.path.join(PROJECT_ROOT, logdir)
                os.makedirs(logdir, exist_ok=True)
                sep = 4
                sep_v = np.full((H, sep, 3), 14, dtype=np.uint8)
                n_cells = 1 + len(keeps)
                frame_w = W * n_cells + sep * (n_cells - 1)
                # libx264 wants even dims
                frame_w += frame_w % 2
                frame_h = H + (H % 2)

                ts = int(_t.time())
                stepped = os.path.join(logdir, f"inference_{ts}.mp4")
                latest = os.path.join(logdir, "preview_latest.mp4")
                for out_path in [stepped, latest]:
                    cmd = ["ffmpeg", "-y", "-v", "quiet",
                           "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{frame_w}x{frame_h}", "-r", "30",
                           "-i", "pipe:0",
                           "-c:v", "libx264", "-crf", "18",
                           "-pix_fmt", "yuv420p", out_path]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                    try:
                        for t in range(T):
                            g = (gt_np[0, t].transpose(1, 2, 0) * 255
                                 ).clip(0, 255).astype(np.uint8)
                            cells = [g]
                            for k in keeps:
                                rc = ((recons[k][0, t]
                                       .transpose(1, 2, 0) * 0.5 + 0.5)
                                      * 255).clip(0, 255).astype(np.uint8)
                                cells.append(rc)
                            row = cells[0]
                            for tile in cells[1:]:
                                row = np.concatenate(
                                    [row, sep_v, tile], axis=1)
                            # Pad to frame_w / frame_h if odd
                            if row.shape[1] != frame_w:
                                pad = np.full(
                                    (row.shape[0], frame_w - row.shape[1], 3),
                                    14, dtype=np.uint8)
                                row = np.concatenate([row, pad], axis=1)
                            if row.shape[0] != frame_h:
                                pad = np.full(
                                    (frame_h - row.shape[0], frame_w, 3),
                                    14, dtype=np.uint8)
                                row = np.concatenate([row, pad], axis=0)
                            proc.stdin.write(row.tobytes())
                        proc.stdin.close()
                        proc.wait()
                    except Exception:
                        try: proc.stdin.close()
                        except Exception: pass
                        proc.kill()
                        proc.wait()
                        raise

                self.after(0, lambda: (
                    self.status_var.set(
                        f"Saved: {os.path.basename(stepped)}  "
                        f"({T} frames, keeps={keeps})"),
                    self._append_log(f"[inf] wrote {stepped}"),
                ))
            except Exception as e:
                import traceback; traceback.print_exc()
                self.after(0, lambda e=e: (
                    self.status_var.set(f"Recon failed: {e}"),
                    self._append_log(f"[inf] FAILED: {e}")))
            finally:
                self._busy = False
        threading.Thread(target=_work, daemon=True).start()


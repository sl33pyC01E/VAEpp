#!/usr/bin/env python3
"""Standalone GUI for SR VAE experiment.

Downscale + upscale trained end-to-end on procedural data.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gui.common import (
    BG, BG_PANEL, BG_INPUT, BG_LOG, FG, FG_DIM, RED, GREEN, BLUE, ACCENT,
    FONT, FONT_BOLD, FONT_TITLE, FONT_SMALL, BILINEAR,
    ProcRunner, make_log, make_btn, make_spin, make_float,
    VENV_PYTHON,
)

from PIL import Image, ImageTk


class SRVAETab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._preview_photo = None
        self._last_mtime = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="SR VAE: Downscale + Upscale", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(top, text="Naive/learned downscale → ESPCN/simple upscale. "
                 "Latent is an RGB thumbnail.",
                 bg=BG_PANEL, fg=FG_DIM, font=FONT_SMALL).pack(anchor="w",
                                                                 pady=(2, 10))

        # Architecture
        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(5, 0))

        f, self.scale_var = make_spin(row1, "Scale", default=8)
        f.pack(side="left", padx=(0, 10))

        # Downscaler dropdown
        df = tk.Frame(row1, bg=BG_PANEL)
        tk.Label(df, text="Downscaler", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        self.down_var = tk.StringVar(value="area")
        down_menu = tk.OptionMenu(df, self.down_var,
                                  "area", "bilinear", "bicubic", "lanczos", "learned")
        down_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                         activebackground=BG_PANEL, activeforeground=FG,
                         highlightthickness=0, borderwidth=0)
        down_menu.pack(anchor="w")
        df.pack(side="left", padx=(0, 10))

        # Upscaler dropdown
        uf = tk.Frame(row1, bg=BG_PANEL)
        tk.Label(uf, text="Upscaler", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        self.up_var = tk.StringVar(value="espcn")
        up_menu = tk.OptionMenu(uf, self.up_var, "espcn", "simple")
        up_menu.config(bg=BG_INPUT, fg=FG, font=FONT_SMALL,
                       activebackground=BG_PANEL, activeforeground=FG,
                       highlightthickness=0, borderwidth=0)
        up_menu.pack(anchor="w")
        uf.pack(side="left", padx=(0, 10))

        f, self.up_hidden = make_spin(row1, "Up hidden", default=64)
        f.pack(side="left", padx=(0, 10))
        f, self.up_blocks = make_spin(row1, "Up blocks", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.down_hidden = make_spin(row1, "Down hidden", default=32)
        f.pack(side="left")

        # Training
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
        f, self.w_lpips = make_float(row2, "w_lpips", "0.0")
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(row2, "Precision", "bf16")
        f.pack(side="left")

        # Save/log
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        f, self.save_every = make_spin(row3, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every = make_spin(row3, "Preview every", default=100)
        f.pack(side="left", padx=(0, 10))
        f, self.freeze_up = make_spin(row3, "Freeze up steps", default=0)
        f.pack(side="left", padx=(0, 10))
        f, self.resume_var = make_float(row3, "Resume", "", width=20)
        f.pack(side="left", padx=(0, 10))
        f, self.load_down_var = make_float(row3, "Load down", "", width=20)
        f.pack(side="left", padx=(0, 10))
        f, self.load_up_var = make_float(row3, "Load up", "", width=20)
        f.pack(side="left")

        # Preview image
        row4 = tk.Frame(top, bg=BG_PANEL)
        row4.pack(fill="x", pady=(5, 0))
        self.preview_img_var = tk.StringVar(value="")
        f = tk.Frame(row4, bg=BG_PANEL)
        tk.Label(f, text="Preview image", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        ef = tk.Frame(f, bg=BG_PANEL)
        tk.Entry(ef, textvariable=self.preview_img_var, bg=BG_INPUT, fg=FG,
                 font=FONT, width=45, borderwidth=0,
                 insertbackground=FG).pack(side="left", fill="x", expand=True)
        make_btn(ef, "Browse",
                 lambda: self.preview_img_var.set(
                     filedialog.askopenfilename(
                         filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                                    ("All", "*.*")]) or self.preview_img_var.get()),
                 ACCENT, width=7).pack(side="left", padx=(5, 0))
        ef.pack(fill="x")
        f.pack(side="left", fill="x", expand=True)

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
        self._check_preview()

    def start(self):
        cmd = [VENV_PYTHON, "-m", "experiments.sr_vae",
               "--scale", str(self.scale_var.get()),
               "--downscaler", self.down_var.get(),
               "--upscaler", self.up_var.get(),
               "--up-hidden", str(self.up_hidden.get()),
               "--up-blocks", str(self.up_blocks.get()),
               "--down-hidden", str(self.down_hidden.get()),
               "--lr", self.lr_var.get(),
               "--batch-size", str(self.batch_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--w-l1", self.w_l1.get(),
               "--w-mse", self.w_mse.get(),
               "--w-lpips", self.w_lpips.get(),
               "--precision", self.prec_var.get(),
               "--freeze-up-steps", str(self.freeze_up.get()),
               "--save-every", str(self.save_every.get()),
               "--preview-every", str(self.preview_every.get())]
        resume = self.resume_var.get().strip()
        if resume:
            cmd.extend(["--resume", resume])
        load_down = self.load_down_var.get().strip()
        if load_down:
            cmd.extend(["--load-downscaler", load_down])
        load_up = self.load_up_var.get().strip()
        if load_up:
            cmd.extend(["--load-upscaler", load_up])
        prev = self.preview_img_var.get().strip()
        if prev:
            cmd.extend(["--preview-image", prev])
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop(self):
        Path(os.path.join(PROJECT_ROOT, "sr_vae_logs", ".stop")).touch()

    def kill(self):
        self.runner.kill()

    def _check_preview(self):
        try:
            path = os.path.join(PROJECT_ROOT, "sr_vae_logs", "preview_latest.png")
            if os.path.exists(path):
                mt = os.path.getmtime(path)
                if mt > self._last_mtime:
                    self._last_mtime = mt
                    pil = Image.open(path)
                    w, h = pil.size
                    try:
                        avail_w = self.preview_label.winfo_width()
                        avail_h = self.preview_label.winfo_height()
                        if avail_w < 100:
                            avail_w = self.winfo_width() - 20
                        if avail_h < 100:
                            avail_h = self.winfo_height() - 300
                    except:
                        avail_w, avail_h = 1000, 500
                    scale = min(avail_w / w, avail_h / h, 1.0)
                    if scale < 1.0:
                        pil = pil.resize((int(w * scale), int(h * scale)),
                                         BILINEAR)
                    self._preview_photo = ImageTk.PhotoImage(pil)
                    self.preview_label.config(image=self._preview_photo)
        except:
            pass
        self.after(2000, self._check_preview)


def main():
    root = tk.Tk()
    root.title("SR VAE Experiment")
    root.geometry("1100x800")
    root.configure(bg=BG)

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

    nb.add(SRVAETab(nb), text="SR VAE Train")

    root.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Data generation tabs -- Static Generator + Video Generator."""

import os
import sys
import tkinter as tk

import numpy as np
import torch
from PIL import Image, ImageTk

from gui.common import *

# -- Generator Tab -------------------------------------------------------------
class GeneratorTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.gen = None
        self._preview_photos = []
        self._bank_thumbs = []
        self.bank_dir = os.path.join(VAEPP_ROOT, "bank")
        self.build()

    def build(self):
        # Main horizontal split: controls left, preview right
        main = tk.PanedWindow(self, orient="horizontal", bg=BG,
                               sashwidth=4, sashrelief="flat")
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # ---- Left panel: controls (scrollable) ----
        left_outer = tk.Frame(main, bg=BG_PANEL)
        main.add(left_outer, width=380)

        canvas_scroll = tk.Canvas(left_outer, bg=BG_PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(left_outer, orient="vertical",
                                  command=canvas_scroll.yview)
        self.left = tk.Frame(canvas_scroll, bg=BG_PANEL, padx=10, pady=10)
        self.left.bind("<Configure>",
                        lambda e: canvas_scroll.configure(
                            scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=self.left, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)
        # Mouse wheel scroll (bound to canvas only, not globally)
        def _on_mousewheel(e):
            canvas_scroll.yview_scroll(-e.delta // 120, "units")
        canvas_scroll.bind("<MouseWheel>", _on_mousewheel)
        self.left.bind("<MouseWheel>", _on_mousewheel)
        # Ensure focus for scroll events
        canvas_scroll.bind("<Enter>", lambda e: canvas_scroll.focus_set())
        canvas_scroll.bind("<Leave>", lambda e: self.focus_set())

        self._build_controls()

        # ---- Right panel: preview ----
        right = tk.Frame(main, bg=BG)
        main.add(right)

        self.preview_label = tk.Label(right, bg=BG)
        self.preview_label.pack(fill="both", expand=True)

        # Log
        self.log = make_log(right)
        self.log.config(height=6)
        self.log.pack(fill="x", side="bottom", pady=(2, 0))

        # Bank browser at bottom
        self.bank_frame = tk.Frame(right, bg=BG_PANEL, height=140)
        self.bank_frame.pack(fill="x", side="bottom", pady=(5, 0))
        self.bank_frame.pack_propagate(False)

        tk.Label(self.bank_frame, text="Shape Bank Browser", bg=BG_PANEL,
                 fg=FG_DIM, font=FONT_SMALL).pack(anchor="w", padx=5)
        self.bank_canvas = tk.Canvas(self.bank_frame, bg=BG, height=110,
                                      highlightthickness=0)
        self.bank_scrollbar = tk.Scrollbar(self.bank_frame, orient="horizontal",
                                            command=self.bank_canvas.xview)
        self.bank_canvas.configure(xscrollcommand=self.bank_scrollbar.set)
        self.bank_scrollbar.pack(side="bottom", fill="x")
        self.bank_canvas.pack(fill="both", expand=True, padx=5)

    def _build_controls(self):
        L = self.left

        tk.Label(L, text="Generator Controls", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")

        # -- Actions (top of panel) --
        tk.Label(L, text="Actions", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        br_top = tk.Frame(L, bg=BG_PANEL)
        br_top.pack(fill="x", pady=2)
        make_btn(br_top, "Disco Quadrant", self.disco_quadrant, "#dd44dd", 12).pack(side="left", padx=(0, 3))
        make_btn(br_top, "Save Bank", self.save_bank, BLUE, 12).pack(side="left")
        tk.Label(L, text="^^^^^^^^", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        tk.Label(L, text="start here", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        br1 = tk.Frame(L, bg=BG_PANEL)
        br1.pack(fill="x", pady=2)
        make_btn(br1, "Build Banks", self.build_banks, GREEN, 12).pack(side="left", padx=(0, 3))
        make_btn(br1, "Refresh Layers", self.refresh_layers, BLUE, 12).pack(side="left")
        br2 = tk.Frame(L, bg=BG_PANEL)
        br2.pack(fill="x", pady=2)
        make_btn(br2, "Generate 1", self.gen_sample, ACCENT, 12).pack(side="left", padx=(0, 3))
        make_btn(br2, "Generate 8", self.gen_batch, ACCENT, 12).pack(side="left")
        br3 = tk.Frame(L, bg=BG_PANEL)
        br3.pack(fill="x", pady=2)
        make_btn(br3, "Load Bank", self.load_bank, BLUE, 12).pack(side="left", padx=(0, 3))
        make_btn(br3, "Accumulate", self.build_accumulate, GREEN, 12).pack(side="left")
        br4 = tk.Frame(L, bg=BG_PANEL)
        br4.pack(fill="x", pady=2)
        make_btn(br4, "Browse Bank", self.browse_bank, ACCENT, 12).pack(side="left", padx=(0, 3))
        make_btn(br4, "Disco Bank", self.disco_bank, "#dd44dd", 12).pack(side="left")
        br5 = tk.Frame(L, bg=BG_PANEL)
        br5.pack(fill="x", pady=2)
        make_btn(br5, "Empty Banks", self.empty_banks, RED, 12).pack(side="left")

        # Ripple (fluid-surface) controls for static snapshots
        tk.Label(L, text="Ripple (liquid surface)", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        br_rip = tk.Frame(L, bg=BG_PANEL)
        br_rip.pack(fill="x", pady=2)
        self.ripple_var = tk.BooleanVar(value=False)
        tk.Checkbutton(br_rip, text="Enable", variable=self.ripple_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       font=FONT).pack(side="left")
        br_rip2 = tk.Frame(L, bg=BG_PANEL)
        br_rip2.pack(fill="x", pady=2)
        f, self.ripple_warp = make_slider(br_rip2, "Warp", 0, 20, 8.0)
        f.pack(side="left", padx=(0, 4))
        f, self.ripple_drops = make_slider(br_rip2, "Drops", 0, 12, 3)
        f.pack(side="left")

        # Effects (Phase 2): shake + kaleidoscope for static
        tk.Label(L, text="Effects", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        br_eff = tk.Frame(L, bg=BG_PANEL)
        br_eff.pack(fill="x", pady=2)
        self.shake_var = tk.BooleanVar(value=False)
        tk.Checkbutton(br_eff, text="Shake", variable=self.shake_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       font=FONT).pack(side="left")
        self.kaleido_var = tk.BooleanVar(value=False)
        tk.Checkbutton(br_eff, text="Kaleidoscope", variable=self.kaleido_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       font=FONT).pack(side="left")
        br_eff2 = tk.Frame(L, bg=BG_PANEL)
        br_eff2.pack(fill="x", pady=2)
        f, self.shake_amp = make_slider(br_eff2, "Shake amp", 0, 0.15, 0.02)
        f.pack(side="left", padx=(0, 4))
        f, self.kaleido_slices = make_slider(br_eff2, "Slices", 2, 16, 6)
        f.pack(side="left")

        # -- Bank settings --
        tk.Label(L, text="Bank", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        r = tk.Frame(L, bg=BG_PANEL)
        r.pack(fill="x")
        f, self.bank_size_var = make_spin(r, "Bank size", default=5000, width=6)
        f.pack(side="left", padx=(0, 5))
        f, self.layer_count_var = make_spin(r, "Layers", default=128, width=6)
        f.pack(side="left", padx=(0, 5))
        f, self.shapes_per_layer_var = make_spin(r, "Shapes/layer", default=30, width=6)
        f.pack(side="left")
        r2 = tk.Frame(L, bg=BG_PANEL)
        r2.pack(fill="x", pady=(2, 0))
        f, self.shape_res_var = make_spin(r2, "Shape res", default=128, width=6)
        f.pack(side="left", padx=(0, 5))
        f, self.alpha_var = make_float(r2, "Alpha", 3.0, width=6)
        f.pack(side="left")

        # -- Shape type weights --
        tk.Label(L, text="Shape Types", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        f, self.w_circle = make_slider(L, "Circle", 0, 1, 0.14)
        f.pack(fill="x")
        f, self.w_rect = make_slider(L, "Rectangle", 0, 1, 0.14)
        f.pack(fill="x")
        f, self.w_triangle = make_slider(L, "Triangle", 0, 1, 0.10)
        f.pack(fill="x")
        f, self.w_ellipse = make_slider(L, "Ellipse", 0, 1, 0.10)
        f.pack(fill="x")
        f, self.w_blob = make_slider(L, "Blob", 0, 1, 0.12)
        f.pack(fill="x")
        f, self.w_line = make_slider(L, "Line", 0, 1, 0.12)
        f.pack(fill="x")
        f, self.w_stroke = make_slider(L, "Stroke", 0, 1, 0.10)
        f.pack(fill="x")
        f, self.w_hatch = make_slider(L, "Hatch", 0, 1, 0.10)
        f.pack(fill="x")
        f, self.w_stipple = make_slider(L, "Stipple", 0, 1, 0.07)
        f.pack(fill="x")
        f, self.w_fractal = make_slider(L, "Fractal", 0, 1, 0.12)
        f.pack(fill="x")

        # -- Texture weights --
        tk.Label(L, text="Textures", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        f, self.w_flat = make_slider(L, "Flat", 0, 1, 0.30)
        f.pack(fill="x")
        f, self.w_perlin = make_slider(L, "Perlin", 0, 1, 0.35)
        f.pack(fill="x")
        f, self.w_gradient = make_slider(L, "Gradient", 0, 1, 0.15)
        f.pack(fill="x")
        f, self.w_voronoi = make_slider(L, "Voronoi", 0, 1, 0.20)
        f.pack(fill="x")

        # -- Edge weights --
        tk.Label(L, text="Edges", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        f, self.w_hard = make_slider(L, "Hard", 0, 1, 0.45)
        f.pack(fill="x")
        f, self.w_soft = make_slider(L, "Soft", 0, 1, 0.35)
        f.pack(fill="x")
        f, self.w_textured = make_slider(L, "Textured", 0, 1, 0.20)
        f.pack(fill="x")

        # -- Scene templates --
        tk.Label(L, text="Scene Templates", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        self.tmpl_sliders = {}
        tmpl_defaults = {
            "random": 0.20, "horizon": 0.06, "v_stripes": 0.04,
            "h_stripes": 0.04, "d_stripes": 0.04, "grid": 0.05,
            "radial": 0.03, "perspective": 0.05, "depth_layers": 0.05,
            "symmetry": 0.03, "border": 0.03, "clusters": 0.03,
            "gradient": 0.03,
            "block_city": 0.05, "landscape": 0.05, "interior": 0.04,
            "road": 0.04, "water": 0.04, "forest": 0.04,
        }
        for name, default in tmpl_defaults.items():
            f, var = make_slider(L, name.replace("_", " ").title(),
                                 0, 1, default)
            f.pack(fill="x")
            self.tmpl_sliders[name] = var

        # -- Color controls --
        tk.Label(L, text="Color", bg=BG_PANEL, fg=ACCENT,
                 font=FONT_BOLD).pack(anchor="w", pady=(10, 0))
        f, self.sat_lo = make_slider(L, "Sat min", 0, 1, 0.3)
        f.pack(fill="x")
        f, self.sat_hi = make_slider(L, "Sat max", 0, 1, 0.95)
        f.pack(fill="x")
        f, self.val_lo = make_slider(L, "Val min", 0, 1, 0.25)
        f.pack(fill="x")
        f, self.val_hi = make_slider(L, "Val max", 0, 1, 1.0)
        f.pack(fill="x")


        # Stats
        self.stats_label = tk.Label(L, text="Banks: not built", bg=BG_PANEL,
                                     fg=FG_DIM, font=FONT_SMALL, anchor="w",
                                     wraplength=350)
        self.stats_label.pack(fill="x", pady=(5, 0))


    def _get_slider_weights(self):
        """Read current slider values and normalize to probabilities."""
        sw = [self.w_circle.get(), self.w_rect.get(), self.w_triangle.get(),
              self.w_ellipse.get(), self.w_blob.get(),
              self.w_line.get(), self.w_stroke.get(),
              self.w_hatch.get(), self.w_stipple.get(),
              self.w_fractal.get()]
        tw = [self.w_flat.get(), self.w_perlin.get(),
              self.w_gradient.get(), self.w_voronoi.get()]
        ew = [self.w_hard.get(), self.w_soft.get(), self.w_textured.get()]
        # Normalize
        def norm(w):
            s = sum(w)
            return tuple(x / s if s > 0 else 1.0 / len(w) for x in w)
        return norm(sw), norm(tw), norm(ew)

    def _get_gen(self):
        sw, tw, ew = self._get_slider_weights()
        if self.gen is None:
            sys.path.insert(0, PROJECT_ROOT)
            from core.generator import VAEpp0rGenerator
            self.gen = VAEpp0rGenerator(
                360, 640, device="cuda",
                bank_size=self.bank_size_var.get(),
                n_base_layers=self.layer_count_var.get(),
                shapes_per_layer=self.shapes_per_layer_var.get(),
                shape_res=self.shape_res_var.get(),
                alpha=float(self.alpha_var.get()),
                shape_weights=sw,
                texture_weights=tw,
                edge_weights=ew,
                saturation_range=(self.sat_lo.get(), self.sat_hi.get()),
                value_range=(self.val_lo.get(), self.val_hi.get()),
            )
        else:
            # Update weights on existing generator
            self.gen.shape_probs = torch.tensor(sw, device=self.gen.device)
            self.gen.texture_probs = torch.tensor(tw, device=self.gen.device)
            self.gen.edge_probs = torch.tensor(ew, device=self.gen.device)
            self.gen.sat_range = (self.sat_lo.get(), self.sat_hi.get())
            self.gen.val_range = (self.val_lo.get(), self.val_hi.get())
        # Update template weights
        tw_vals = [self.tmpl_sliders[n].get() for n in self.gen.template_names]
        tw_sum = sum(tw_vals)
        if tw_sum > 0:
            tw_vals = [v / tw_sum for v in tw_vals]
        self.gen.template_probs = torch.tensor(tw_vals, device=self.gen.device)
        return self.gen

    def build_banks(self):
        self.gen = None
        gen = self._get_gen()
        self.stats_label.config(text="Building banks...")
        self.update()
        def _done():
            self._update_stats()
            self._update_bank_browser()
        run_with_log(self, gen.build_banks, on_done=_done)

    def refresh_layers(self):
        gen = self._get_gen()
        if gen.shape_bank is None:
            self.stats_label.config(text="Build banks first!")
            return
        self.stats_label.config(text="Refreshing layers...")
        self.update()
        run_with_log(self, gen.refresh_base_layers, on_done=self._update_stats)

    def _apply_ripple_settings(self, gen):
        """Push GUI ripple/shake/kaleido toggles onto the generator instance."""
        gen.static_ripple = bool(self.ripple_var.get())
        gen.static_ripple_warp_strength = float(self.ripple_warp.get())
        gen.static_ripple_n_drops = int(self.ripple_drops.get())
        gen.static_shake = bool(self.shake_var.get())
        gen.static_shake_amp_xy = float(self.shake_amp.get())
        gen.static_shake_amp_rot = float(self.shake_amp.get())
        gen.static_shake_mode = "vibrate"
        gen.static_kaleido = bool(self.kaleido_var.get())
        gen.static_kaleido_slices = int(self.kaleido_slices.get())

    def gen_sample(self):
        gen = self._get_gen()
        if gen.base_layers is None:
            self.build_banks()
            return
        self._apply_ripple_settings(gen)
        with torch.no_grad():
            batch = gen.generate(1)
        img = (batch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        self._show_image(Image.fromarray(img))

    def gen_batch(self):
        gen = self._get_gen()
        if gen.base_layers is None:
            self.build_banks()
            return
        self._apply_ripple_settings(gen)
        with torch.no_grad():
            batch = gen.generate(8)
        imgs = [(batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                for i in range(8)]
        H, W = imgs[0].shape[:2]
        gap = 4
        grid = np.full((H * 2 + gap, W * 4 + gap * 3, 3), 14, dtype=np.uint8)
        for i, img in enumerate(imgs):
            r, c = i // 4, i % 4
            grid[r * (H + gap):r * (H + gap) + H,
                 c * (W + gap):c * (W + gap) + W] = img
        self._show_image(Image.fromarray(grid))

    def save_bank(self):
        gen = self._get_gen()
        if gen.shape_bank is None:
            self.stats_label.config(text="Nothing to save")
            return
        gen.save_to_bank_dir(self.bank_dir)
        self._update_stats()

    def load_bank(self):
        gen = self._get_gen()
        gen.load_bank_dir(self.bank_dir)
        self._update_stats()
        self._update_bank_browser()

    def build_accumulate(self):
        gen = self._get_gen()
        n = self.bank_size_var.get()
        self.stats_label.config(text="Building + accumulating...")
        self.update()
        def _do():
            new_shapes = []
            for i in range(n):
                new_shapes.append(gen._render_one_shape())
                if (i + 1) % 500 == 0:
                    print(f"  [{i+1}/{n}] shapes rendered", flush=True)
            new_bank = torch.stack(new_shapes)
            if gen.shape_bank is not None:
                gen.shape_bank = torch.cat([gen.shape_bank, new_bank], dim=0)
            else:
                gen.shape_bank = new_bank
            gen.bank_size = gen.shape_bank.shape[0]
            gen.build_base_layers()
            print(f"Accumulated: now {gen.bank_size} shapes total", flush=True)
        def _done():
            self._update_stats()
            self._update_bank_browser()
        run_with_log(self, _do, on_done=_done)

    def empty_banks(self):
        """Delete all bank files from disk and clear memory."""
        bank_dir = self.bank_dir
        if os.path.isdir(bank_dir):
            for f in os.listdir(bank_dir):
                fp = os.path.join(bank_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        if self.gen is not None:
            self.gen.shape_bank = None
            self.gen.base_layers = None
            self.gen.bank_size = 0
            self.gen._recipe_pool = []
        self._update_stats()
        self._bank_thumbs = []
        self.bank_canvas.delete("all")
        self.stats_label.config(text="Banks emptied")

    def disco_quadrant(self):
        """Activate quadrant mode, build bank, deactivate quadrant mode."""
        self.gen = None
        gen = self._get_gen()
        gen.disco_quadrant = True
        self.stats_label.config(text="Disco quadrant: building banks...")
        self.update()
        def _fn():
            gen.build_banks()
            gen.disco_quadrant = False
        def _done():
            self._update_stats()
            self._update_bank_browser()
        run_with_log(self, _fn, on_done=_done)

    def disco_bank(self):
        """Build a bank where every shape has completely random parameters."""
        gen = self._get_gen()
        n = self.bank_size_var.get()
        self.stats_label.config(text=f"Disco mode: building {n} shapes with random params...")
        self.update()

        def _do():
            # Save original params to restore after disco
            orig_shape_probs = gen.shape_probs.clone()
            orig_texture_probs = gen.texture_probs.clone()
            orig_edge_probs = gen.edge_probs.clone()
            orig_sat_range = gen.sat_range
            orig_val_range = gen.val_range
            orig_min_r_frac = gen.min_r_frac
            orig_max_r_frac = gen.max_r_frac
            orig_soft_range = gen.soft_range
            orig_template_probs = gen.template_probs.clone()

            shapes = []
            for i in range(n):
                gen.shape_probs = torch.rand(len(orig_shape_probs), device=gen.device)
                gen.shape_probs = gen.shape_probs / gen.shape_probs.sum()
                gen.texture_probs = torch.rand(len(orig_texture_probs), device=gen.device)
                gen.texture_probs = gen.texture_probs / gen.texture_probs.sum()
                gen.edge_probs = torch.rand(len(orig_edge_probs), device=gen.device)
                gen.edge_probs = gen.edge_probs / gen.edge_probs.sum()
                gen.sat_range = (torch.rand(1).item() * 0.5,
                                  torch.rand(1).item() * 0.5 + 0.5)
                gen.val_range = (torch.rand(1).item() * 0.4,
                                  torch.rand(1).item() * 0.4 + 0.6)
                gen.min_r_frac = torch.rand(1).item() * 0.3 + 0.05
                gen.max_r_frac = torch.rand(1).item() * 0.4 + 0.5
                gen.soft_range = (torch.rand(1).item() * 2 + 0.5,
                                   torch.rand(1).item() * 6 + 2)
                tw = torch.rand(len(gen.template_names), device=gen.device)
                tw[0] = tw[0] * 0.1
                gen.template_probs = tw / tw.sum()

                shapes.append(gen._render_one_shape())
                if (i + 1) % 100 == 0:
                    print(f"  disco [{i+1}/{n}] {gen._last_shape_log}", flush=True)

            gen.shape_probs = orig_shape_probs
            gen.texture_probs = orig_texture_probs
            gen.edge_probs = orig_edge_probs
            gen.sat_range = orig_sat_range
            gen.val_range = orig_val_range
            gen.min_r_frac = orig_min_r_frac
            gen.max_r_frac = orig_max_r_frac
            gen.soft_range = orig_soft_range
            gen.template_probs = orig_template_probs

            new_bank = torch.stack(shapes)
            if gen.shape_bank is not None:
                gen.shape_bank = torch.cat([gen.shape_bank, new_bank], dim=0)
            else:
                gen.shape_bank = new_bank
            gen.bank_size = gen.shape_bank.shape[0]
            gen.build_base_layers()
            print(f"Disco bank done: {gen.bank_size} shapes", flush=True)

        def _done():
            self._update_stats()
            self._update_bank_browser()
        run_with_log(self, _do, on_done=_done)

    def browse_bank(self):
        """Refresh the bank browser strip."""
        self._update_bank_browser()

    def _update_bank_browser(self):
        """Show shape bank as scrollable thumbnail strip."""
        gen = self._get_gen()
        if gen.shape_bank is None:
            return
        self.bank_canvas.delete("all")
        self._bank_thumbs = []
        N = min(gen.shape_bank.shape[0], 200)  # limit displayed
        thumb_size = 96
        for i in range(N):
            rgba = gen.shape_bank[i].cpu().numpy()  # (4, S, S)
            rgb = rgba[:3].transpose(1, 2, 0)
            alpha = rgba[3]
            # Composite on dark background
            bg = np.full_like(rgb, 0.1)
            a = alpha[:, :, None]
            comp = bg * (1 - a) + rgb * a
            comp = (comp * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(comp).resize((thumb_size, thumb_size), BILINEAR)
            photo = ImageTk.PhotoImage(pil)
            self._bank_thumbs.append(photo)
            x = i * (thumb_size + 4) + 2
            self.bank_canvas.create_image(x, 2, image=photo, anchor="nw")
        self.bank_canvas.configure(scrollregion=(0, 0,
            N * (thumb_size + 4), thumb_size + 4))

    def _update_stats(self):
        gen = self._get_gen()
        stats = gen.bank_stats()
        n_files = 0
        if os.path.isdir(self.bank_dir):
            n_files = len([f for f in os.listdir(self.bank_dir) if f.endswith(".pt")])
        s = " | ".join(f"{k}: {v['count']} ({v['mb']:.0f} MB)" for k, v in stats.items())
        s += f" | disk: {n_files} files"
        self.stats_label.config(text=s)

    def _show_image(self, pil_img):
        # Scale to fit preview area
        max_w = self.preview_label.winfo_width() - 20
        max_h = self.preview_label.winfo_height() - 20
        if max_w < 100:
            max_w = 700
        if max_h < 100:
            max_h = 400
        scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
        if scale < 1.0:
            new_w = int(pil_img.width * scale)
            new_h = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), BILINEAR)

        photo = ImageTk.PhotoImage(pil_img)
        self._preview_photos = [photo]  # keep reference
        self.preview_label.config(image=photo)


# -- Training Tab --------------------------------------------------------------

class VideoGenTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.gen = None
        self._preview_photos = []
        self._video_frames = []
        self._video_playing = False
        self._video_idx = 0
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Video Generator (Stage 2)", bg=BG_PANEL, fg=FG,
                 font=FONT_TITLE).pack(anchor="w")

        row1 = tk.Frame(top, bg=BG_PANEL)
        row1.pack(fill="x", pady=(10, 0))
        f, self.T_var = make_spin(row1, "Frames (T)", default=24, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.bank_var = make_spin(row1, "Bank size", default=5000, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.layers_var = make_spin(row1, "Layers", default=128, width=6)
        f.pack(side="left")

        # Motion controls
        row2 = tk.Frame(top, bg=BG_PANEL)
        row2.pack(fill="x", pady=(5, 0))
        f, self.pan_str = make_slider(row2, "Pan", 0, 1, 0.5)
        f.pack(side="left", padx=(0, 5))
        f, self.motion_str = make_slider(row2, "Motion", 0, 1, 0.4)
        f.pack(side="left", padx=(0, 5))

        # Checkboxes
        row2b = tk.Frame(top, bg=BG_PANEL)
        row2b.pack(fill="x", pady=(2, 0))
        self.physics_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2b, text="Physics",
                       variable=self.physics_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.rotation_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2b, text="Rotation",
                       variable=self.rotation_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.zoom_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2b, text="Zoom",
                       variable=self.zoom_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.fade_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2b, text="Fade",
                       variable=self.fade_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.viewport_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2b, text="Viewport",
                       variable=self.viewport_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.fluid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2b, text="Fluid",
                       variable=self.fluid_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.ripple_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2b, text="Ripple",
                       variable=self.ripple_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")

        # Viewport + fluid sliders
        row2c = tk.Frame(top, bg=BG_PANEL)
        row2c.pack(fill="x", pady=(2, 0))
        f, self.vp_pan_str = make_slider(row2c, "VP Pan", 0, 1, 0.3)
        f.pack(side="left", padx=(0, 5))
        f, self.vp_zoom_str = make_slider(row2c, "VP Zoom", 0, 0.5, 0.15)
        f.pack(side="left", padx=(0, 5))
        f, self.vp_rot_str = make_slider(row2c, "VP Rot", 0, 1, 0.2)
        f.pack(side="left", padx=(0, 5))
        f, self.fluid_str = make_slider(row2c, "Fluid", 0, 3, 1.0)
        f.pack(side="left")

        # Ripple (liquid-surface) sliders
        row2d = tk.Frame(top, bg=BG_PANEL)
        row2d.pack(fill="x", pady=(2, 0))
        f, self.ripple_warp = make_slider(row2d, "Ripple warp", 0, 20, 8.0)
        f.pack(side="left", padx=(0, 5))
        f, self.ripple_drops = make_slider(row2d, "Raindrops", 0, 12, 3)
        f.pack(side="left")

        # Effects row (Phase 2): shake, kaleidoscope, fast transforms
        row2e = tk.Frame(top, bg=BG_PANEL)
        row2e.pack(fill="x", pady=(2, 0))
        self.shake_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2e, text="Shake",
                       variable=self.shake_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.shake_mode_var = tk.StringVar(value="vibrate")
        tk.OptionMenu(row2e, self.shake_mode_var,
                      "vibrate", "earthquake", "handheld").pack(side="left", padx=(0, 6))
        self.kaleido_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2e, text="Kaleidoscope",
                       variable=self.kaleido_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")
        self.fast_tx_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2e, text="Fast transforms",
                       variable=self.fast_tx_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, font=FONT).pack(side="left")

        # Effects sliders row
        row2f = tk.Frame(top, bg=BG_PANEL)
        row2f.pack(fill="x", pady=(2, 0))
        f, self.shake_amp = make_slider(row2f, "Shake amp", 0, 0.15, 0.02)
        f.pack(side="left", padx=(0, 5))
        f, self.kaleido_slices = make_slider(row2f, "Kaleido slices", 2, 16, 6)
        f.pack(side="left", padx=(0, 5))
        f, self.kaleido_rot = make_slider(row2f, "Kaleido rot/frame", 0, 0.2, 0.03)
        f.pack(side="left", padx=(0, 5))
        f, self.fast_scale = make_slider(row2f, "Fast scale", 1, 10, 4.0)
        f.pack(side="left")

        # Buttons
        row3 = tk.Frame(top, bg=BG_PANEL)
        row3.pack(fill="x", pady=(5, 0))
        make_btn(row3, "Generate 1", self.gen_video, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(row3, "Generate 8 Grid", self.gen_grid, ACCENT).pack(
            side="left", padx=(0, 5))
        make_btn(row3, "Build Banks", self.build_banks, BLUE).pack(
            side="left", padx=(0, 5))

        row3b = tk.Frame(top, bg=BG_PANEL)
        row3b.pack(fill="x", pady=(2, 0))
        make_btn(row3b, "Build Pool", self.build_pool, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(row3b, "Save Pool", self.save_pool, BLUE).pack(
            side="left", padx=(0, 5))
        make_btn(row3b, "Load Pool", self.load_pool, BLUE).pack(
            side="left", padx=(0, 5))

        row3c = tk.Frame(top, bg=BG_PANEL)
        row3c.pack(fill="x", pady=(2, 0))
        make_btn(row3c, "Empty Pool", self.empty_pool, RED).pack(
            side="left", padx=(0, 5))
        make_btn(row3c, "Disco Pool", self.disco_pool, "#dd44dd").pack(
            side="left")

        self.status = tk.Label(top, text="Ready", bg=BG_PANEL, fg=FG_DIM,
                                font=FONT_SMALL)
        self.status.pack(fill="x", pady=(5, 0))

        self.log = make_log(self)
        self.log.config(height=8)
        self.log.pack(fill="x", padx=5, pady=(0, 5), side="bottom")

        # Preview — shows first frame of generated video
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, pady=5)

    def _get_gen(self):
        if self.gen is None:
            sys.path.insert(0, PROJECT_ROOT)
            from core.generator import VAEpp0rGenerator
            self.gen = VAEpp0rGenerator(
                360, 640, device="cuda",
                bank_size=self.bank_var.get(),
                n_base_layers=self.layers_var.get(),
            )
            # Try loading from bank dir
            bank_dir = os.path.join(VAEPP_ROOT, "bank")
            if os.path.isdir(bank_dir):
                bank_files = [f for f in os.listdir(bank_dir)
                              if f.startswith("shapes_") and f.endswith(".pt")]
                if bank_files:
                    self.gen.setup_dynamic_bank(bank_dir,
                        working_size=self.bank_var.get())
        return self.gen

    def build_banks(self):
        self.gen = None
        gen = self._get_gen()
        self.status.config(text="Building banks...")
        self.update()
        def _done():
            self.status.config(text=f"Banks ready: {gen.bank_size} shapes")
        run_with_log(self, gen.build_banks, on_done=_done)

    def _get_seq_kwargs(self):
        return dict(
            use_physics=self.physics_var.get(),
            use_rotation=self.rotation_var.get(),
            use_zoom=self.zoom_var.get(),
            use_fade=self.fade_var.get(),
            use_viewport=self.viewport_var.get(),
            use_fluid=self.fluid_var.get(),
            pan_strength=self.pan_str.get(),
            motion_strength=self.motion_str.get(),
            viewport_pan=self.vp_pan_str.get(),
            viewport_zoom=self.vp_zoom_str.get(),
            viewport_rotation=self.vp_rot_str.get(),
            fluid_strength=self.fluid_str.get(),
            use_ripple=self.ripple_var.get(),
            ripple_warp_strength=self.ripple_warp.get(),
            ripple_n_drops=int(self.ripple_drops.get()),
            use_shake=self.shake_var.get(),
            shake_mode=self.shake_mode_var.get(),
            shake_amp_xy=self.shake_amp.get(),
            shake_amp_rot=self.shake_amp.get(),
            use_kaleido=self.kaleido_var.get(),
            kaleido_slices=int(self.kaleido_slices.get()),
            kaleido_rot_per_frame=self.kaleido_rot.get(),
            fast_transform=self.fast_tx_var.get(),
            fast_scale=self.fast_scale.get(),
        )

    def build_pool(self):
        gen = self._get_gen()
        T = self.T_var.get()
        self.status.config(text=f"Building motion pool T={T}...")
        self.update()
        seq_kw = self._get_seq_kwargs()
        def _fn():
            if gen.base_layers is None:
                gen.build_base_layers()
            gen.build_motion_pool(n_clips=200, T=T, **seq_kw)
        def _done():
            stats = gen.motion_pool_stats()
            self.status.config(text=f"Pool ready: {stats}")
        run_with_log(self, _fn, on_done=_done)

    def save_pool(self):
        gen = self._get_gen()
        pool_path = os.path.join(VAEPP_ROOT, "bank", "motion_pool.json")
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        gen.save_motion_pool(pool_path)
        self.status.config(text=f"Saved pool to {pool_path}")

    def load_pool(self):
        gen = self._get_gen()
        bank_dir = os.path.join(VAEPP_ROOT, "bank")
        gen.load_motion_pool(bank_dir)
        stats = gen.motion_pool_stats()
        if stats:
            self.status.config(text=f"Loaded pool: {stats['count']} recipes")
        else:
            self.status.config(text="No recipes found in bank/")

    def empty_pool(self):
        """Delete all recipe files and clear pool from memory."""
        bank_dir = os.path.join(VAEPP_ROOT, "bank")
        if os.path.isdir(bank_dir):
            for f in os.listdir(bank_dir):
                if f.startswith("recipes_") and f.endswith(".json"):
                    os.remove(os.path.join(bank_dir, f))
        gen = self._get_gen()
        gen._recipe_pool = []
        self.status.config(text="Recipe pool emptied")

    def disco_pool(self):
        """Build recipes with randomized parameters for each recipe."""
        gen = self._get_gen()
        n = 200
        self.status.config(text=f"Disco: building {n} randomized recipes...")
        self.update()

        def _do():
            if gen.base_layers is None:
                gen.build_base_layers()
            orig_shape_probs = gen.shape_probs.clone()
            orig_texture_probs = gen.texture_probs.clone()
            orig_template_probs = gen.template_probs.clone()

            T = self.T_var.get()
            for i in range(n):
                kw = {
                    "use_physics": torch.rand(1).item() > 0.3,
                    "use_rotation": torch.rand(1).item() > 0.3,
                    "use_zoom": torch.rand(1).item() > 0.3,
                    "use_fade": torch.rand(1).item() > 0.3,
                    "use_viewport": torch.rand(1).item() > 0.3,
                    "use_fluid": torch.rand(1).item() > 0.7,
                    "pan_strength": torch.rand(1).item() * 0.8,
                    "motion_strength": torch.rand(1).item() * 0.6,
                    "viewport_pan": torch.rand(1).item() * 0.5,
                    "viewport_zoom": torch.rand(1).item() * 0.3,
                    "viewport_rotation": torch.rand(1).item() * 0.4,
                    "fluid_strength": torch.rand(1).item() * 2.0,
                }
                gen.shape_probs = torch.rand(len(orig_shape_probs), device=gen.device)
                gen.shape_probs = gen.shape_probs / gen.shape_probs.sum()
                gen.texture_probs = torch.rand(len(orig_texture_probs), device=gen.device)
                gen.texture_probs = gen.texture_probs / gen.texture_probs.sum()
                tw = torch.rand(len(gen.template_names), device=gen.device)
                tw[0] = tw[0] * 0.1
                gen.template_probs = tw / tw.sum()

                recipe = gen._generate_recipe(T=T, **kw)
                gen._recipe_pool.append(recipe)

                if (i + 1) % 50 == 0:
                    print(f"  disco recipes [{i+1}/{n}]", flush=True)

            gen.shape_probs = orig_shape_probs
            gen.texture_probs = orig_texture_probs
            gen.template_probs = orig_template_probs
            gen._motion_pool_T = T
            print(f"Disco pool ready: {len(gen._recipe_pool)} recipes", flush=True)

        def _done():
            self.status.config(text=f"Disco pool: {len(gen._recipe_pool)} recipes")
        run_with_log(self, _do, on_done=_done)

    def gen_video(self):
        self._video_playing = False
        gen = self._get_gen()
        T = self.T_var.get()
        seq_kw = self._get_seq_kwargs()
        self.status.config(text=f"Generating T={T} clip...")
        self.update()

        def _gen():
            if gen.base_layers is None:
                gen.build_base_layers()
            with torch.no_grad():
                clip = gen.generate_sequence(1, T=T, **seq_kw)
            # Convert to numpy on bg thread (no tkinter calls)
            frames_np = []
            scale = min(700 / gen.W, 400 / gen.H, 1.0)
            for t in range(T):
                arr = (clip[0, t].permute(1, 2, 0).cpu().numpy() * 255
                       ).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                if scale < 1.0:
                    pil = pil.resize((int(gen.W * scale), int(gen.H * scale)),
                                     BILINEAR)
                frames_np.append(pil)
            # Marshal to main thread for PhotoImage creation
            self.after(0, lambda: self._show_frames(frames_np, T))

        run_with_log(self, _gen)

    def gen_grid(self):
        """Generate 8 clips, tile into a 4x2 grid, play inline."""
        self._video_playing = False
        gen = self._get_gen()
        T = self.T_var.get()
        seq_kw = self._get_seq_kwargs()
        self.status.config(text=f"Generating 8 clips T={T}...")
        self.update()

        def _gen():
            if gen.base_layers is None:
                gen.build_base_layers()
            with torch.no_grad():
                clips = gen.generate_sequence(8, T=T, **seq_kw)
            H, W = gen.H, gen.W
            sh, sw = H // 2, W // 2
            gap = 2
            grid_w = sw * 4 + gap * 3
            grid_h = sh * 2 + gap
            frames_pil = []
            for ti in range(T):
                grid = np.full((grid_h, grid_w, 3), 14, dtype=np.uint8)
                for ci in range(8):
                    frame = (clips[ci, ti].permute(1, 2, 0).cpu().numpy() * 255
                             ).clip(0, 255).astype(np.uint8)
                    small = np.array(Image.fromarray(frame).resize((sw, sh),
                                     BILINEAR))
                    r, c = ci // 4, ci % 4
                    y = r * (sh + gap)
                    x = c * (sw + gap)
                    grid[y:y+sh, x:x+sw] = small
                pil = Image.fromarray(grid)
                scale = min(700 / grid_w, 400 / grid_h, 1.0)
                if scale < 1.0:
                    pil = pil.resize((int(grid_w * scale), int(grid_h * scale)),
                                     BILINEAR)
                frames_pil.append(pil)
            self.after(0, lambda: self._show_frames(frames_pil, T))

        run_with_log(self, _gen)

    def _show_frames(self, pil_frames, T):
        """Create PhotoImages on main thread and start playback."""
        self._video_frames = [ImageTk.PhotoImage(p) for p in pil_frames]
        self._video_idx = 0
        self._play_gen = getattr(self, '_play_gen', 0) + 1
        self.status.config(text=f"Playing T={T} (looping)")
        self._play_video_loop(self._play_gen)

    def _play_video_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        self._video_idx = self._video_idx % len(self._video_frames)
        self.preview_label.config(image=self._video_frames[self._video_idx])
        self._video_idx += 1
        self.after(33, self._play_video_loop, self._play_gen)


# -- Video Training Tab --------------------------------------------------------

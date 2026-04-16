#!/usr/bin/env python3
"""Motion recipe pool system for VAEpp0r generator.

Lightweight motion parameter storage and rendering for fast temporal sampling.
"""

import math
import os
import torch


class RecipesMixin:
    """Mixin providing motion recipe/pool methods for VAEpp0rGenerator."""

    def _generate_recipe(self, T=8, **seq_kwargs):
        """Generate a single motion recipe — lightweight parameter dict.

        A recipe stores which shapes/layers to use and how they move,
        NOT the rendered pixels. ~1KB per recipe vs ~9MB per rendered clip.
        """
        # Fast-transform preset: scale motion knobs up at sample time so
        # viewport/layer/stamp motion is all larger/faster. Pure multiplier,
        # no runtime branch needed elsewhere.
        seq_kwargs = self._fast_transform_scale(seq_kwargs)
        H, W = self.H, self.W
        n_layers = torch.randint(
            self.layers_per_image[0], self.layers_per_image[1] + 1, (1,)).item()
        n_stamps = torch.randint(
            self.stamps_per_image[0], self.stamps_per_image[1] + 1, (1,)).item()

        n_micro = torch.randint(20, 60, (1,)).item() if torch.rand(1).item() < 0.5 else 0
        n_micro_alloc = max(n_micro, 1)

        recipe = {
            "T": T,
            "bg_color": torch.rand(3).tolist(),
            # Layers
            "layer_idx": torch.randint(0, self.n_base_layers, (n_layers,)).tolist(),
            "pan_dx": ((torch.rand(n_layers) - 0.5) * W * seq_kwargs.get("pan_strength", 0.5)).tolist(),
            "pan_dy": ((torch.rand(n_layers) - 0.5) * H * seq_kwargs.get("pan_strength", 0.5)).tolist(),
            "pan_speed": (torch.rand(n_layers) * 0.8 + 0.2).tolist(),
            "opacity_start": (torch.rand(n_layers) * 0.5 + 0.3).tolist(),
            "opacity_end": (torch.rand(n_layers) * 0.5 + 0.3).tolist(),
            "color_shift": ((torch.rand(n_layers, 3) - 0.5) * 0.3).tolist(),
            "zoom_start": (torch.rand(n_layers) * 0.3 + 0.85).tolist(),
            "zoom_end": (torch.rand(n_layers) * 0.3 + 0.85).tolist(),
            # Stamps
            "stamp_idx": torch.randint(0, max(self.bank_size, 1), (n_stamps,)).tolist(),
            "stamp_x": (torch.rand(n_stamps) * W).tolist(),
            "stamp_y": (torch.rand(n_stamps) * H * 0.5).tolist(),
            "stamp_vx": ((torch.rand(n_stamps) - 0.5) * 15).tolist(),
            "stamp_vy": ((torch.rand(n_stamps) - 0.5) * 5).tolist(),
            "stamp_gravity": (torch.rand(n_stamps) * 3 + 1).tolist(),
            "stamp_scale": (torch.rand(n_stamps) * 1.0 + 0.3).tolist(),
            "stamp_rot": (torch.rand(n_stamps) * 2 * math.pi).tolist(),
            "stamp_rot_speed": ((torch.rand(n_stamps) - 0.5) * 0.3).tolist(),
            # Viewport
            "vp_pan": ((torch.rand(2) - 0.5) * torch.tensor([W, H]).float() *
                        seq_kwargs.get("viewport_pan", 0.3)).tolist(),
            "vp_zoom": [1.0 - seq_kwargs.get("viewport_zoom", 0.15) / 2,
                         1.0 + seq_kwargs.get("viewport_zoom", 0.15) / 2],
            "vp_rot": ((torch.rand(2) - 0.5) * seq_kwargs.get("viewport_rotation", 0.2)).tolist(),
            # Fine grain
            "fine_layer_idx": torch.randint(0, self.n_base_layers, (1,)).item(),
            "fine_tile": torch.randint(3, 7, (1,)).item(),
            "fine_opacity": torch.rand(1).item() * 0.4 + 0.15,
            "use_fine": torch.rand(1).item() < 0.5,
            # Micro stamps
            "n_micro": n_micro,
            "micro_idx": torch.randint(0, max(self.bank_size, 1), (n_micro_alloc,)).tolist(),
            "micro_x": (torch.rand(n_micro_alloc) * W).tolist(),
            "micro_y": (torch.rand(n_micro_alloc) * H).tolist(),
            "micro_dx": ((torch.rand(n_micro_alloc) - 0.5) * W * 0.15).tolist(),
            "micro_dy": ((torch.rand(n_micro_alloc) - 0.5) * H * 0.15).tolist(),
            # Settings
            "gamma": torch.rand(1).item() * 0.7 + 0.7,
            "seq_kwargs": seq_kwargs,
        }
        # Optional fluid (ripple + raindrops) block — additive, off by default
        if seq_kwargs.get("use_ripple", False):
            recipe["fluid"] = self._sample_fluid_recipe(
                T=T,
                n_drops=int(seq_kwargs.get("ripple_n_drops", 3)),
                warp_strength=float(seq_kwargs.get("ripple_warp_strength", 8.0)),
            )
        # Optional camera shake / vibration
        if seq_kwargs.get("use_shake", False):
            recipe["shake"] = self._sample_shake_recipe(
                T=T,
                amp_xy=float(seq_kwargs.get("shake_amp_xy", 0.02)),
                amp_rot=float(seq_kwargs.get("shake_amp_rot", 0.02)),
                mode=str(seq_kwargs.get("shake_mode", "vibrate")),
            )
        # Optional whole-image kaleidoscope
        if seq_kwargs.get("use_kaleido", False):
            recipe["kaleido"] = self._sample_kaleido_recipe(
                n_slices=int(seq_kwargs.get("kaleido_slices", 6)),
                rot_per_frame=float(seq_kwargs.get("kaleido_rot_per_frame", 0.03)),
            )
        # Optional flash frames + strobe
        if seq_kwargs.get("use_flash", False) or seq_kwargs.get("strobe_rate", 0.0) > 0:
            recipe["flash"] = self._sample_flash_recipe(
                T=T,
                n_flashes=int(seq_kwargs.get("flash_n", 2))
                    if seq_kwargs.get("use_flash", False) else 0,
                strobe_rate=float(seq_kwargs.get("strobe_rate", 0.0)),
                strobe_strength=float(seq_kwargs.get("strobe_strength", 0.3)),
            )
        # Optional palette cycle
        if seq_kwargs.get("use_palette_cycle", False):
            speed = float(seq_kwargs.get("palette_speed", 0.05))
            recipe["palette"] = self._sample_palette_recipe(
                T=T, speed_range=(speed, speed),
                sat_boost=float(seq_kwargs.get("palette_sat_boost", 1.0)),
            )
        # Optional text overlay (typing / scrolling)
        if seq_kwargs.get("use_text", False):
            recipe["text"] = self._sample_text_recipe(
                T=T,
                mode=str(seq_kwargs.get("text_mode", "typing")),
                language=str(seq_kwargs.get("text_language", "mixed")),
                font_size=int(seq_kwargs.get("text_font_size", 24)),
                cps=float(seq_kwargs.get("text_cps", 12.0)),
                scroll_pxpf=float(seq_kwargs.get("text_scroll_pxpf", 8.0)),
            )
        # Optional signage overlay
        if seq_kwargs.get("use_signage", False):
            recipe["signage"] = self._sample_signage_recipe(
                T=T,
                mode=str(seq_kwargs.get("signage_mode", "auto")),
                font_size=int(seq_kwargs.get("signage_font_size", 32)),
            )
        return recipe

    def build_motion_pool(self, n_clips=200, T=8, **seq_kwargs):
        """Generate motion recipes (not pixels). Lightweight, ~1KB each.

        Args:
            n_clips: number of recipes
            T: frames per clip
            **seq_kwargs: motion settings
        """
        print(f"Building motion recipe pool ({n_clips} recipes, T={T})...",
              flush=True)
        recipes = []
        for i in range(n_clips):
            recipes.append(self._generate_recipe(T=T, **seq_kwargs))
            if (i + 1) % 100 == 0 or i == n_clips - 1:
                print(f"  [{i+1}/{n_clips}]", flush=True)
        self._recipe_pool = recipes
        self._motion_pool_T = T
        self._motion_pool_kwargs = seq_kwargs
        self._motion_pool_call_count = 0
        print(f"Recipe pool done: {n_clips} recipes", flush=True)

    def _refresh_motion_pool(self, swap_frac=0.2, **seq_kwargs):
        """Swap a fraction of recipes with fresh ones."""
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return
        N = len(self._recipe_pool)
        T = self._motion_pool_T
        n_swap = max(1, int(N * swap_frac))
        for _ in range(n_swap):
            idx = torch.randint(0, N, (1,)).item()
            self._recipe_pool[idx] = self._generate_recipe(
                T=T, **seq_kwargs)

    def _render_recipe(self, recipe):
        """Render a single recipe into a (T, 3, H, W) clip on GPU.

        Uses stored recipe parameters for deterministic, reproducible rendering.
        """
        import torch.nn.functional as F

        T = recipe["T"]
        H, W = self.H, self.W
        dev = self.device

        bg = torch.tensor(recipe["bg_color"], device=dev).view(1, 3, 1, 1)

        # Disco background (consistent across all frames in clip)
        disco_bg = None
        if self.disco_quadrant:
            disco_bg = self._generate_disco(1)  # (1, 3, H, W)

        layer_idx = recipe["layer_idx"]
        n_layers = len(layer_idx)
        pan_dx = torch.tensor(recipe["pan_dx"], device=dev)
        pan_dy = torch.tensor(recipe["pan_dy"], device=dev)
        pan_speed = torch.tensor(recipe["pan_speed"], device=dev)
        opacity_start = torch.tensor(recipe["opacity_start"], device=dev)
        opacity_end = torch.tensor(recipe["opacity_end"], device=dev)
        color_shift = torch.tensor(recipe["color_shift"], device=dev)  # (n_layers, 3)
        zoom_start = torch.tensor(recipe["zoom_start"], device=dev)
        zoom_end = torch.tensor(recipe["zoom_end"], device=dev)

        stamp_idx = recipe["stamp_idx"]
        n_stamps = len(stamp_idx)
        stamp_x = torch.tensor(recipe["stamp_x"], device=dev)
        stamp_y = torch.tensor(recipe["stamp_y"], device=dev)
        stamp_vx = torch.tensor(recipe["stamp_vx"], device=dev)
        stamp_vy = torch.tensor(recipe["stamp_vy"], device=dev)
        stamp_gravity = torch.tensor(recipe["stamp_gravity"], device=dev)
        stamp_scale = torch.tensor(recipe["stamp_scale"], device=dev)
        stamp_rot = torch.tensor(recipe["stamp_rot"], device=dev)
        stamp_rot_speed = torch.tensor(recipe["stamp_rot_speed"], device=dev)

        vp_pan = torch.tensor(recipe["vp_pan"], device=dev)
        vp_zoom = recipe["vp_zoom"]
        vp_rot = torch.tensor(recipe["vp_rot"], device=dev)

        gamma = recipe.get("gamma", 1.0)

        # Fine grain layer
        use_fine = recipe.get("use_fine", False)
        fine_layer = None
        fine_opacity = 0
        if use_fine and self.base_layers is not None:
            fidx = recipe.get("fine_layer_idx", 0) % self.n_base_layers
            tile_f = recipe.get("fine_tile", 4)
            fine_src = self.base_layers[fidx:fidx+1]
            sh = (H + tile_f - 1) // tile_f
            sw = (W + tile_f - 1) // tile_f
            small = F.interpolate(fine_src, (sh, sw),
                                  mode="bilinear", align_corners=False)
            fine_layer = small.repeat(1, 1, tile_f, tile_f)[:, :, :H, :W]
            fine_opacity = recipe.get("fine_opacity", 0.2)

        # Micro stamp params
        n_micro = recipe.get("n_micro", 0)
        micro_idx = recipe.get("micro_idx", [])
        micro_x = torch.tensor(recipe.get("micro_x", []), device=dev)
        micro_y = torch.tensor(recipe.get("micro_y", []), device=dev)
        micro_dx_t = torch.tensor(recipe.get("micro_dx", []), device=dev)
        micro_dy_t = torch.tensor(recipe.get("micro_dy", []), device=dev)

        # Pre-transform stamps
        stamp_shapes = []
        if self.shape_bank is not None and n_stamps > 0:
            for si in range(n_stamps):
                sidx = stamp_idx[si] % self.shape_bank.shape[0]
                rgba = self._transform_bank_shape(self.shape_bank[sidx].clone())
                stamp_shapes.append(rgba)

        # Pre-transform micro stamps
        micro_shapes = []
        if self.shape_bank is not None and n_micro > 0:
            for mi in range(min(n_micro, len(micro_idx))):
                midx = micro_idx[mi] % self.shape_bank.shape[0]
                rgba = self._transform_bank_shape(self.shape_bank[midx].clone())
                micro_shapes.append(rgba)

        # Simulate stamp physics
        sx = stamp_x.clone()
        sy = stamp_y.clone()
        svx = stamp_vx.clone()
        svy = stamp_vy.clone()
        s_rot = stamp_rot.clone()

        stamp_trajectories = torch.zeros(n_stamps, T, 4, device=dev)
        for ti in range(T):
            stamp_trajectories[:, ti, 0] = sx
            stamp_trajectories[:, ti, 1] = sy
            stamp_trajectories[:, ti, 2] = stamp_scale
            stamp_trajectories[:, ti, 3] = s_rot

            svy = svy + stamp_gravity
            sx = sx + svx
            sy = sy + svy
            s_rot = s_rot + stamp_rot_speed

            # Bounce (with energy loss matching _simulate_physics)
            bounce_x = (sx < 0) | (sx > W)
            bounce_y = (sy < 0) | (sy > H)
            svx[bounce_x] = -svx[bounce_x] * 0.8
            svy[bounce_y] = -svy[bounce_y] * 0.7
            sx.clamp_(0, W)
            sy.clamp_(0, H)

        # Render frames
        frames = []
        for ti in range(T):
            t_frac = ti / max(T - 1, 1)
            if disco_bg is not None:
                canvas = disco_bg.clone()
            else:
                canvas = bg.expand(1, 3, H, W).clone()

            # Layers
            for li in range(n_layers):
                if self.base_layers is None:
                    break
                lidx = layer_idx[li] % self.n_base_layers
                layer = self.base_layers[lidx:lidx+1].clone()
                cs = color_shift[li].view(1, 3, 1, 1)
                layer = (layer + cs).clamp(0, 1)

                dx = pan_dx[li] * pan_speed[li] * t_frac
                dy = pan_dy[li] * pan_speed[li] * t_frac
                zoom = zoom_start[li] * (1 - t_frac) + zoom_end[li] * t_frac

                theta = torch.zeros(1, 2, 3, device=dev)
                theta[0, 0, 0] = 1.0 / zoom
                theta[0, 1, 1] = 1.0 / zoom
                theta[0, 0, 2] = -dx / (W / 2)
                theta[0, 1, 2] = -dy / (H / 2)
                grid = F.affine_grid(theta, (1, 3, H, W), align_corners=False)
                layer = F.grid_sample(layer, grid, mode="bilinear",
                                       padding_mode="reflection", align_corners=False)

                opacity = opacity_start[li] * (1 - t_frac) + opacity_end[li] * t_frac
                canvas = canvas * (1 - opacity) + layer * opacity

            # Stamps
            for si in range(n_stamps):
                if not stamp_shapes:
                    break
                rgba = stamp_shapes[si]
                px_f = stamp_trajectories[si, ti, 0].item()
                py_f = stamp_trajectories[si, ti, 1].item()
                sc = stamp_trajectories[si, ti, 2].item()
                rot_angle = stamp_trajectories[si, ti, 3].item()

                if abs(rot_angle) > 0.01:
                    cos_r = math.cos(rot_angle)
                    sin_r = math.sin(rot_angle)
                    rot_theta = torch.tensor(
                        [[cos_r, -sin_r, 0], [sin_r, cos_r, 0]],
                        device=dev).unsqueeze(0)
                    rot_grid = F.affine_grid(
                        rot_theta, (1, 4, self.shape_res, self.shape_res),
                        align_corners=False)
                    rgba = F.grid_sample(
                        rgba.unsqueeze(0), rot_grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)[0]

                rgb = rgba[:3]
                alpha_s = rgba[3:4]

                th = max(3, int(self.shape_res * sc * H / 360))
                tw = max(3, int(self.shape_res * sc * W / 640))
                th, tw = min(th, H // 2), min(tw, W // 2)

                rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                      mode="bilinear", align_corners=False)[0]
                a_r = F.interpolate(alpha_s.unsqueeze(0), (th, tw),
                                    mode="bilinear", align_corners=False)[0]

                px = int(px_f) - tw // 2
                py = int(py_f) - th // 2

                sx_c = max(0, -px)
                sy_c = max(0, -py)
                ex = min(tw, W - px)
                ey = min(th, H - py)
                if ex <= sx_c or ey <= sy_c:
                    continue
                cx, cy = max(0, px), max(0, py)
                rh, rw = ey - sy_c, ex - sx_c

                a = a_r[:, sy_c:ey, sx_c:ex]
                canvas[0, :, cy:cy+rh, cx:cx+rw] = \
                    canvas[0, :, cy:cy+rh, cx:cx+rw] * (1 - a) + \
                    rgb_r[:, sy_c:ey, sx_c:ex] * a

            # Fine grain
            if fine_layer is not None:
                canvas = canvas * (1 - fine_opacity) + fine_layer * fine_opacity

            # Micro stamps
            for mi in range(min(n_micro, len(micro_shapes))):
                rgba = micro_shapes[mi]
                rgb = rgba[:3]
                alpha_m = rgba[3:4]
                sc = 0.12
                th = max(3, int(self.shape_res * sc * H / 360))
                tw = max(3, int(self.shape_res * sc * W / 640))
                th, tw = min(th, H // 3), min(tw, W // 3)
                rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                      mode="bilinear", align_corners=False)[0]
                a_r = F.interpolate(alpha_m.unsqueeze(0), (th, tw),
                                    mode="bilinear", align_corners=False)[0]
                mpx = int(micro_x[mi].item() + micro_dx_t[mi].item() * t_frac) - tw // 2
                mpy = int(micro_y[mi].item() + micro_dy_t[mi].item() * t_frac) - th // 2
                if 0 <= mpx < W - tw and 0 <= mpy < H - th:
                    canvas[0, :, mpy:mpy+th, mpx:mpx+tw] = \
                        canvas[0, :, mpy:mpy+th, mpx:mpx+tw] * (1 - a_r) + rgb_r * a_r

            # Viewport
            canvas = self._apply_viewport(canvas, ti, T,
                                          vp_pan.unsqueeze(0),
                                          torch.tensor([vp_zoom], device=dev),
                                          vp_rot.unsqueeze(0))

            # Ripple-surface warp (optional, from recipe["fluid"])
            fp = recipe.get("fluid")
            if fp is not None and fp.get("enable", False):
                canvas = self._apply_ripples(canvas, ti, T, fp)

            # Camera shake (optional, from recipe["shake"])
            sp = recipe.get("shake")
            if sp is not None and sp.get("enable", False):
                canvas = self._apply_camera_shake(canvas, ti, sp)

            # Whole-image kaleidoscope (optional, from recipe["kaleido"])
            kp = recipe.get("kaleido")
            if kp is not None and kp.get("enable", False):
                canvas = self._apply_kaleidoscope(canvas, ti, kp)

            # Palette hue rotation
            pp = recipe.get("palette")
            if pp is not None and pp.get("enable", False):
                canvas = self._apply_palette_cycle(canvas, ti, pp)

            # Flash / strobe
            fl = recipe.get("flash")
            if fl is not None and fl.get("enable", False):
                canvas = self._apply_flash(canvas, ti, fl)

            # Text overlay (typing / scrolling)
            tx = recipe.get("text")
            if tx is not None and tx.get("enable", False):
                canvas = self._apply_text(canvas, ti, tx)

            # Signage overlay
            sg = recipe.get("signage")
            if sg is not None and sg.get("enable", False):
                canvas = self._apply_signage(canvas, ti, sg)

            # Post-processing
            canvas = canvas.clamp(1e-6, 1).pow(gamma)
            frames.append(canvas.clamp(0, 1))

        return torch.cat(frames, dim=0)  # (T, 3, H, W) — squeezed from (T, 1, 3, H, W)

    def generate_from_pool(self, batch_size, refresh_interval=100):
        """Render B clips from recipe pool.

        Picks random recipes, renders them live from the shape bank.
        Applies anti-memorization transforms after rendering.

        Returns: (B, T, 3, H, W) tensor in [0, 1] on self.device.
        """
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            raise RuntimeError("Call build_motion_pool() first")

        B = batch_size
        N = len(self._recipe_pool)

        # Periodic refresh
        self._motion_pool_call_count += 1
        if self._motion_pool_call_count % refresh_interval == 0:
            self._refresh_motion_pool(**getattr(self, '_motion_pool_kwargs', {}))

        # Render B recipes
        T_expected = self._motion_pool_T
        clips = []
        for bi in range(B):
            idx = torch.randint(0, N, (1,)).item()
            clip = self._render_recipe(self._recipe_pool[idx])  # (T_i, 3, H, W)
            # Pad or trim to consistent T for stacking
            T_i = clip.shape[0]
            if T_i < T_expected:
                pad = clip[-1:].expand(T_expected - T_i, -1, -1, -1)
                clip = torch.cat([clip, pad], dim=0)
            elif T_i > T_expected:
                clip = clip[:T_expected]
            clips.append(clip)
        clips = torch.stack(clips)  # (B, T, 3, H, W)

        # Anti-memorization transforms (fast, on GPU)
        for bi in range(B):
            if torch.rand(1).item() < 0.5:
                clips[bi] = clips[bi].flip(-1)
            if torch.rand(1).item() < 0.2:
                clips[bi] = clips[bi].flip(-2)
            if torch.rand(1).item() < 0.3:
                clips[bi] = clips[bi].flip(0)  # temporal reverse
            shift = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
            clips[bi] = (clips[bi] + shift).clamp(0, 1)
            if torch.rand(1).item() < 0.3:
                clips[bi] = clips[bi].roll(torch.randint(1, 3, (1,)).item(), dims=0)

        return clips.clamp(0, 1)

    def save_motion_pool(self, path):
        """Save recipe pool to the given path (or timestamped file in same dir)."""
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return
        import json, time as _time
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        # Save to the requested path
        with open(path, 'w') as f:
            json.dump(self._recipe_pool, f)
        # Also save a timestamped backup
        ts = _time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(d, f"recipes_{ts}.json")
        with open(backup, 'w') as f:
            json.dump(self._recipe_pool, f)
        kb = os.path.getsize(path) / 1e3
        print(f"Saved {len(self._recipe_pool)} recipes to {path} "
              f"({kb:.0f} KB)", flush=True)

    def load_motion_pool(self, path):
        """Load recipe pool from disk. If path is a directory, loads all recipe files."""
        import json
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path)
                           if f.startswith("recipes_") and f.endswith(".json"))
            if not files:
                print(f"No recipe files in {path}", flush=True)
                return
            for f in files:
                fp = os.path.join(path, f)
                with open(fp) as fh:
                    recipes = json.load(fh)
                self._recipe_pool.extend(recipes)
                print(f"  Loaded {len(recipes)} recipes from {f}", flush=True)
        else:
            with open(path) as f:
                recipes = json.load(f)
            self._recipe_pool.extend(recipes)
        if self._recipe_pool:
            t_values = set(r["T"] for r in self._recipe_pool)
            self._motion_pool_T = max(t_values)
            if len(t_values) > 1:
                print(f"  WARNING: mixed T values in pool: {t_values}, using T={self._motion_pool_T}", flush=True)
        else:
            self._motion_pool_T = 8
        self._motion_pool_call_count = 0
        if not hasattr(self, '_motion_pool_kwargs'):
            self._motion_pool_kwargs = {}
        print(f"Total recipes: {len(self._recipe_pool)}", flush=True)

    def motion_pool_stats(self):
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return None
        return {
            "count": len(self._recipe_pool),
            "T": self._motion_pool_T,
            "approx_kb": len(self._recipe_pool),  # ~1KB each
        }

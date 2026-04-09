#!/usr/bin/env python3
"""Temporal motion and sequence generation for VAEpp0r generator.

Physics simulation, viewport transforms, fluid advection, and
multi-frame sequence rendering (3ch and 9ch).
"""

import math
import torch
import torch.nn.functional as F


class MotionMixin:
    """Mixin providing temporal/motion methods for VAEpp0rGenerator."""

    # Semantic role colors (approximate C-RADIO PCA mapping)
    _SEM_BG = torch.tensor([0.3, 0.2, 0.8])      # background: blue-pink
    _SEM_LAYER = torch.tensor([0.8, 0.5, 0.3])    # static layers: orange
    _SEM_STAMP = torch.tensor([0.2, 0.8, 0.3])    # moving stamps: green
    _SEM_MICRO = torch.tensor([0.3, 0.6, 0.6])    # micro detail: cyan

    def _simulate_physics(self, B, M, T):
        """Pre-compute stamp positions for T frames with physics.

        Returns: (B, M, T, 4) tensor — [x, y, scale, rotation] per stamp per frame.
        """
        H, W = self.H, self.W

        # Initial positions
        x = torch.rand(B, M, device=self.device) * W
        y = torch.rand(B, M, device=self.device) * H * 0.5  # start upper half

        # Initial velocities (pixels per frame)
        vx = (torch.rand(B, M, device=self.device) - 0.5) * 15
        vy = (torch.rand(B, M, device=self.device) - 0.5) * 5

        # Gravity
        gravity = torch.rand(B, M, device=self.device) * 3 + 1  # pixels/frame^2

        # Scale: start + oscillation
        scale_base = torch.rand(B, M, device=self.device) * 1.0 + 0.3
        scale_freq = torch.rand(B, M, device=self.device) * 2  # oscillation speed
        scale_amp = torch.rand(B, M, device=self.device) * 0.3  # oscillation amplitude

        # Rotation
        rot = torch.rand(B, M, device=self.device) * 2 * math.pi
        rot_speed = (torch.rand(B, M, device=self.device) - 0.5) * 0.3  # rad/frame

        trajectories = torch.zeros(B, M, T, 4, device=self.device)

        for ti in range(T):
            # Record state
            scale_t = scale_base + scale_amp * torch.sin(
                scale_freq * ti * 0.5)
            trajectories[:, :, ti, 0] = x
            trajectories[:, :, ti, 1] = y
            trajectories[:, :, ti, 2] = scale_t
            trajectories[:, :, ti, 3] = rot

            # Update physics
            vy = vy + gravity  # gravity pulls down
            x = x + vx
            y = y + vy
            rot = rot + rot_speed

            # Bounce off walls
            hit_left = x < 0
            hit_right = x > W
            hit_top = y < 0
            hit_bottom = y > H

            vx = torch.where(hit_left | hit_right, -vx * 0.8, vx)
            vy = torch.where(hit_top | hit_bottom, -vy * 0.7, vy)
            x = x.clamp(0, W)
            y = y.clamp(0, H)

        return trajectories

    def _generate_flow_field(self, B, T):
        """Generate a smooth 2D velocity field for fluid-like advection.

        Returns: (B, T, 2, H, W) displacement field — how much each pixel
        moves per frame. Based on time-varying Perlin noise.
        """
        H, W = self.H, self.W
        # Coarse velocity field (faster than full-res)
        ch, cw = H // 8, W // 8

        # Base noise fields (persistent across frames for temporal coherence)
        base_vx = torch.randn(B, 1, ch, cw, device=self.device)
        base_vy = torch.randn(B, 1, ch, cw, device=self.device)
        # Random phase for temporal variation
        phase = torch.rand(B, 1, 1, 1, device=self.device) * 2 * math.pi

        # Smoothing kernel
        k = 3
        g = torch.tensor([0.25, 0.5, 0.25], device=self.device).view(1, 1, k, 1)

        fields = []
        for ti in range(T):
            t_val = ti / max(T - 1, 1)
            # Blend base noise with per-frame noise for smooth temporal variation
            t_phase = phase + t_val * 2 * math.pi
            blend = 0.7  # weight of persistent base field
            vx_noise = base_vx * blend + torch.randn(B, 1, ch, cw, device=self.device) * (1 - blend)
            vy_noise = base_vy * blend + torch.randn(B, 1, ch, cw, device=self.device) * (1 - blend)
            # Rotate flow direction over time using phase
            cos_p = torch.cos(t_phase)
            sin_p = torch.sin(t_phase)
            vx_rot = vx_noise * cos_p - vy_noise * sin_p
            vy_rot = vx_noise * sin_p + vy_noise * cos_p
            # Smooth
            vx_rot = F.conv2d(vx_rot, g, padding=(1, 0))
            vx_rot = F.conv2d(vx_rot, g.permute(0, 1, 3, 2), padding=(0, 1))
            vy_rot = F.conv2d(vy_rot, g, padding=(1, 0))
            vy_rot = F.conv2d(vy_rot, g.permute(0, 1, 3, 2), padding=(0, 1))
            # Upsample to full res
            vx = F.interpolate(vx_rot, (H, W), mode="bilinear",
                               align_corners=False)
            vy = F.interpolate(vy_rot, (H, W), mode="bilinear",
                               align_corners=False)
            # Scale displacement (pixels per frame)
            strength = 3.0
            field = torch.cat([vx * strength, vy * strength], dim=1)  # (B, 2, H, W)
            fields.append(field)

        return torch.stack(fields, dim=1)  # (B, T, 2, H, W)

    def _apply_viewport(self, canvas, ti, T, vp_pan, vp_zoom, vp_rot):
        """Apply viewport-level transform to entire canvas.

        Args:
            canvas: (B, 3, H, W)
            ti: current frame index
            T: total frames
            vp_pan: (B, 2) total pan displacement [dx, dy]
            vp_zoom: (B, 2) start/end zoom [start, end]
            vp_rot: (B, 2) start/end rotation [start, end] in radians
        """
        B = canvas.shape[0]
        H, W = self.H, self.W
        t_frac = ti / max(T - 1, 1)

        dx = vp_pan[:, 0] * t_frac
        dy = vp_pan[:, 1] * t_frac
        zoom = vp_zoom[:, 0] * (1 - t_frac) + vp_zoom[:, 1] * t_frac
        angle = vp_rot[:, 0] * (1 - t_frac) + vp_rot[:, 1] * t_frac

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = cos_a / zoom
        theta[:, 0, 1] = -sin_a / zoom
        theta[:, 1, 0] = sin_a / zoom
        theta[:, 1, 1] = cos_a / zoom
        theta[:, 0, 2] = -dx / (W / 2)
        theta[:, 1, 2] = -dy / (H / 2)

        grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    def _apply_fluid(self, canvas, flow_field):
        """Advect canvas pixels through a velocity field.

        Args:
            canvas: (B, 3, H, W)
            flow_field: (B, 2, H, W) displacement in pixels
        """
        B, _, H, W = canvas.shape
        # Build sample grid: identity + displacement
        y = torch.linspace(-1, 1, H, device=self.device)
        x = torch.linspace(-1, 1, W, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Convert pixel displacement to normalized coords
        dx = flow_field[:, 0:1].permute(0, 2, 3, 1) / (W / 2)  # (B, H, W, 1)
        dy = flow_field[:, 1:2].permute(0, 2, 3, 1) / (H / 2)
        offset = torch.cat([dx, dy], dim=-1)  # (B, H, W, 2)

        grid = base_grid + offset
        return F.grid_sample(canvas, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    @torch.no_grad()
    def generate_sequence(self, batch_size, T=8,
                           use_physics=True, use_rotation=True,
                           use_zoom=True, use_fade=True,
                           use_viewport=True, use_fluid=False,
                           pan_strength=0.5, motion_strength=0.4,
                           viewport_pan=0.3, viewport_zoom=0.15,
                           viewport_rotation=0.2, fluid_strength=1.0):
        """Generate animated clips with physics, viewport effects, and fluid motion.

        Returns: (B, T, 3, H, W) tensor in [0, 1] on self.device.
        """
        if self.base_layers is None:
            self.build_base_layers()
        self._maybe_refresh_dynamic()

        B = batch_size
        H, W = self.H, self.W

        # --- Background ---
        bg = self._sample_colors(B).view(B, 3, 1, 1)

        # Disco quadrant: use disco output as animated background
        disco_bg = None
        if self.disco_quadrant:
            disco_bg = self._generate_disco(B)  # (B, 3, H, W)

        # Scene template (consistent across all frames in clip)
        template = self._pick_template()
        has_template = template != "random"

        # Template-reduced overlay counts (match static pipeline)
        if has_template:
            n_layers = torch.randint(1, 3, (B,), device=self.device)
            n_stamps = torch.randint(2, 6, (B,), device=self.device)
            overlay_opacity_scale = 0.3
        else:
            n_layers = torch.randint(
                self.layers_per_image[0], self.layers_per_image[1] + 1,
                (B,), device=self.device)
            n_stamps = torch.randint(
                self.stamps_per_image[0], self.stamps_per_image[1] + 1,
                (B,), device=self.device)
            overlay_opacity_scale = 1.0

        # --- Layer setup ---
        max_layers = n_layers.max().item()

        layer_idx = torch.randint(0, self.n_base_layers, (B, max_layers),
                                   device=self.device)
        pan_dx = (torch.rand(B, max_layers, device=self.device) - 0.5) * W * pan_strength
        pan_dy = (torch.rand(B, max_layers, device=self.device) - 0.5) * H * pan_strength
        pan_speed = torch.rand(B, max_layers, device=self.device) * 0.8 + 0.2

        opacity_start = torch.rand(B, max_layers, device=self.device) * 0.5 + 0.3
        opacity_end = opacity_start.clone() if not use_fade else \
            torch.rand(B, max_layers, device=self.device) * 0.5 + 0.3

        color_shift = (torch.rand(B, max_layers, 3, device=self.device) - 0.5) * 0.3

        # Layer zoom
        if use_zoom:
            zoom_start = torch.rand(B, max_layers, device=self.device) * 0.3 + 0.85
            zoom_end = torch.rand(B, max_layers, device=self.device) * 0.3 + 0.85
        else:
            zoom_start = torch.ones(B, max_layers, device=self.device)
            zoom_end = zoom_start

        # Shape masks for layers (40% per-image chance, consistent across frames)
        layer_masks = None
        use_layer_mask = torch.rand(B, device=self.device) < 0.4
        if use_layer_mask.any() and self.shape_bank is not None:
            midx = torch.randint(0, self.bank_size, (B,), device=self.device)
            raw_masks = self.shape_bank[midx, 3:4]  # (B, 1, S, S)
            raw_masks = F.interpolate(raw_masks, (H, W),
                                       mode="bilinear", align_corners=False)
            layer_masks = torch.ones(B, 1, H, W, device=self.device)
            layer_masks[use_layer_mask] = raw_masks[use_layer_mask]

        # --- Stamp setup with physics ---
        max_stamps = n_stamps.max().item()

        if self.shape_bank is None:
            max_stamps = 0
            n_stamps = torch.zeros(B, device=self.device, dtype=torch.long)

        stamp_idx = torch.randint(0, max(1, self.shape_bank.shape[0] if self.shape_bank is not None else 1),
                                   (B, max(max_stamps, 1)), device=self.device)

        if use_physics:
            trajectories = self._simulate_physics(B, max_stamps, T)
        else:
            # Simple linear motion
            trajectories = torch.zeros(B, max_stamps, T, 4, device=self.device)
            sx = torch.rand(B, max_stamps, device=self.device) * W
            sy = torch.rand(B, max_stamps, device=self.device) * H
            dx = (torch.rand(B, max_stamps, device=self.device) - 0.5) * W * motion_strength
            dy = (torch.rand(B, max_stamps, device=self.device) - 0.5) * H * motion_strength
            sc = torch.rand(B, max_stamps, device=self.device) * 1.2 + 0.3
            for ti in range(T):
                t_frac = ti / max(T - 1, 1)
                trajectories[:, :, ti, 0] = sx + dx * t_frac
                trajectories[:, :, ti, 1] = sy + dy * t_frac
                trajectories[:, :, ti, 2] = sc
                trajectories[:, :, ti, 3] = 0

        # Pre-transform stamps (consistent across frames for temporal coherence)
        stamp_shapes = []
        for si in range(max_stamps):
            per_batch = []
            for bi in range(B):
                rgba = self._transform_bank_shape(
                    self.shape_bank[stamp_idx[bi, si]].clone())
                per_batch.append(rgba)
            stamp_shapes.append(torch.stack(per_batch))  # (B, 4, S, S)

        # Pre-generate fine-grain texture layer (consistent across frames)
        fine_layer = None
        fine_opacity = 0
        if torch.rand(1).item() < 0.5:
            fidx = torch.randint(0, self.n_base_layers, (B,), device=self.device)
            fine_src = self.base_layers[fidx].clone()
            tile_f = torch.randint(3, 7, (1,)).item()
            fh = (H + tile_f - 1) // tile_f
            fw = (W + tile_f - 1) // tile_f
            small = F.interpolate(fine_src, (fh, fw),
                                  mode="bilinear", align_corners=False)
            fine_layer = small.repeat(1, 1, tile_f, tile_f)[:, :, :H, :W]
            cs = (torch.rand(B, 3, 1, 1, device=self.device) - 0.5) * 0.4
            fine_layer = (fine_layer + cs).clamp(0, 1)
            fine_opacity = torch.rand(1, device=self.device).item() * 0.4 + 0.15

        # Pre-generate micro-stamp positions (consistent shapes, moving positions)
        n_micro = 0
        micro_shapes = []
        micro_start_x = None
        micro_start_y = None
        micro_dx = None
        micro_dy = None
        micro_scales = None
        if torch.rand(1).item() < 0.5 and self.shape_bank is not None:
            n_micro = torch.randint(20, 60, (1,)).item()
            for _ in range(n_micro):
                sidx = torch.randint(0, self.shape_bank.shape[0], (B,),
                                      device=self.device)
                s = self.shape_bank[sidx].clone()
                for bi in range(B):
                    s[bi] = self._transform_bank_shape(s[bi])
                micro_shapes.append(s)
            micro_start_x = torch.rand(B, n_micro, device=self.device) * W
            micro_start_y = torch.rand(B, n_micro, device=self.device) * H
            micro_dx = (torch.rand(B, n_micro, device=self.device) - 0.5) * W * 0.15
            micro_dy = (torch.rand(B, n_micro, device=self.device) - 0.5) * H * 0.15
            # Pre-compute scales for temporal coherence (no per-frame flickering)
            micro_scales = 0.1 + torch.rand(B, n_micro, device=self.device) * 0.15

        # --- Tessellation tiling (30% chance, consistent across frames) ---
        tess_layer = None
        tess_opacity = None
        if torch.rand(1).item() < 0.3:
            tidx = torch.randint(0, self.n_base_layers, (B,), device=self.device)
            tess_src = self.base_layers[tidx].clone()
            tile_f = torch.randint(2, 6, (1,)).item()
            sh = (H + tile_f - 1) // tile_f
            sw = (W + tile_f - 1) // tile_f
            small = F.interpolate(tess_src, (sh, sw),
                                  mode="bilinear", align_corners=False)
            tess_layer = small.repeat(1, 1, tile_f, tile_f)[:, :, :H, :W]
            tshift = (torch.rand(B, 3, 1, 1, device=self.device) - 0.5) * 0.4
            tess_layer = (tess_layer + tshift).clamp(0, 1)
            tess_opacity = torch.rand(B, 1, 1, 1, device=self.device) * 0.5 + 0.2

        # --- Multi-tile grid (25% chance, consistent across frames) ---
        grid_layer = None
        grid_opacity = None
        if torch.rand(1).item() < 0.25:
            grid_n = torch.randint(2, 5, (1,)).item()
            cell_h, cell_w = H // grid_n, W // grid_n
            grid_layer = torch.zeros(B, 3, H, W, device=self.device)
            for bi in range(B):
                for gy in range(grid_n):
                    for gx in range(grid_n):
                        lidx = torch.randint(0, self.n_base_layers, (1,)).item()
                        cell = F.interpolate(
                            self.base_layers[lidx:lidx+1], (cell_h, cell_w),
                            mode="bilinear", align_corners=False)[0]
                        cs = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
                        cell = (cell + cs).clamp(0, 1)
                        y0, x0 = gy * cell_h, gx * cell_w
                        eh = min(cell_h, H - y0)
                        ew = min(cell_w, W - x0)
                        grid_layer[bi, :, y0:y0+eh, x0:x0+ew] = cell[:, :eh, :ew]
            grid_opacity = torch.rand(B, 1, 1, 1, device=self.device) * 0.5 + 0.3

        # --- Fractal layout (30% chance, pre-rendered for temporal coherence) ---
        use_fractal = torch.rand(1).item() < 0.3 and self.shape_bank is not None
        n_frac = torch.randint(8, 20, (1,)).item() if use_fractal else 0
        fractal_overlay = None
        if use_fractal:
            # Pre-render fractal layout onto a black canvas once,
            # then blend it each frame to avoid per-frame re-randomization
            frac_canvas = torch.zeros(B, 3, H, W, device=self.device)
            frac_canvas = self._render_fractal_layout(frac_canvas, n_shapes=n_frac)
            fractal_overlay = frac_canvas  # (B, 3, H, W), black where no shapes

        # --- Post-processing params (consistent across frames) ---
        pp_gamma = torch.rand(B, 1, 1, 1, device=self.device) * 0.7 + 0.7
        pp_hue_shift = torch.randint(1, 3, (1,)).item() if torch.rand(1).item() < 0.4 else 0
        pp_wave_freq = 0
        pp_wave = None
        if torch.rand(1).item() < 0.25:
            pp_wave_freq = torch.rand(1, device=self.device).item() * 4 + 1
            pp_wave_phase = torch.rand(1, device=self.device).item() * 2 * math.pi
            x_lin = torch.linspace(0, 1, W, device=self.device)
            pp_wave = (torch.sin(x_lin * pp_wave_freq * 2 * math.pi + pp_wave_phase) * 0.1
                       ).view(1, 1, 1, W)
        pp_vignette = None
        if torch.rand(1).item() < 0.15:
            vy = torch.linspace(-1, 1, H, device=self.device)
            vx = torch.linspace(-1, 1, W, device=self.device)
            vyy, vxx = torch.meshgrid(vy, vx, indexing="ij")
            vdist = torch.sqrt(vxx ** 2 + vyy ** 2)
            vstrength = torch.rand(1, device=self.device).item() * 0.5 + 0.2
            pp_vignette = (1.0 - vdist * vstrength).clamp(0.3, 1.0).view(1, 1, H, W)

        # Viewport effects
        if use_viewport:
            vp_pan = (torch.rand(B, 2, device=self.device) - 0.5) * \
                torch.tensor([W, H], device=self.device).float() * viewport_pan
            vp_zoom = torch.ones(B, 2, device=self.device)
            vp_zoom[:, 0] = 1.0 - viewport_zoom / 2
            vp_zoom[:, 1] = 1.0 + viewport_zoom / 2
            vp_rot = (torch.rand(B, 2, device=self.device) - 0.5) * viewport_rotation
        else:
            vp_pan = torch.zeros(B, 2, device=self.device)
            vp_zoom = torch.ones(B, 2, device=self.device)
            vp_rot = torch.zeros(B, 2, device=self.device)

        # Fluid flow field
        flow_fields = None
        if use_fluid:
            flow_fields = self._generate_flow_field(B, T) * fluid_strength

        # Pre-render scene template once (templates use random values internally,
        # so calling per-frame would cause flickering)
        template_canvas = None
        if has_template:
            base_canvas = disco_bg.clone() if disco_bg is not None else bg.expand(B, 3, H, W).clone()
            template_canvas = self._apply_scene_template(base_canvas, template, B)

        # --- Render T frames ---
        frames = []
        for ti in range(T):
            t_frac = ti / max(T - 1, 1)

            # Background: disco or solid color, or cached template
            if template_canvas is not None:
                canvas = template_canvas.clone()
            elif disco_bg is not None:
                canvas = disco_bg.clone()
            else:
                canvas = bg.expand(B, 3, H, W).clone()

            # Tessellation overlay (consistent across frames)
            if tess_layer is not None:
                canvas = canvas * (1 - tess_opacity) + tess_layer * tess_opacity

            # Grid overlay (consistent across frames)
            if grid_layer is not None:
                canvas = canvas * (1 - grid_opacity) + grid_layer * grid_opacity

            # --- Layers with pan + zoom ---
            for li in range(max_layers):
                active = (li < n_layers).float().view(B, 1, 1, 1)
                layer = self.base_layers[layer_idx[:, li]].clone()
                cs = color_shift[:, li].view(B, 3, 1, 1)
                layer = (layer + cs).clamp(0, 1)

                dx = pan_dx[:, li] * pan_speed[:, li] * t_frac
                dy = pan_dy[:, li] * pan_speed[:, li] * t_frac
                zoom = zoom_start[:, li] * (1 - t_frac) + zoom_end[:, li] * t_frac

                theta = torch.zeros(B, 2, 3, device=self.device)
                theta[:, 0, 0] = 1.0 / zoom
                theta[:, 1, 1] = 1.0 / zoom
                theta[:, 0, 2] = -dx / (W / 2)
                theta[:, 1, 2] = -dy / (H / 2)
                grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
                layer = F.grid_sample(layer, grid, mode="bilinear",
                                       padding_mode="reflection", align_corners=False)

                opacity = (opacity_start[:, li] * (1 - t_frac) +
                           opacity_end[:, li] * t_frac).view(B, 1, 1, 1)
                alpha = opacity * active * overlay_opacity_scale
                if layer_masks is not None:
                    alpha = alpha * layer_masks
                canvas = canvas * (1 - alpha) + layer * alpha

            # --- Stamps with physics trajectories ---
            for si in range(max_stamps):
                active_s = (si < n_stamps).float()
                shapes = stamp_shapes[si]  # (B, 4, S, S)

                for bi in range(B):
                    if active_s[bi] < 0.5:
                        continue

                    px_f = trajectories[bi, si, ti, 0].item()
                    py_f = trajectories[bi, si, ti, 1].item()
                    sc = trajectories[bi, si, ti, 2].item()
                    rot_angle = trajectories[bi, si, ti, 3].item()

                    rgba = shapes[bi]  # (4, S, S)

                    # Apply rotation if enabled
                    if use_rotation and abs(rot_angle) > 0.01:
                        cos_r = math.cos(rot_angle)
                        sin_r = math.sin(rot_angle)
                        rot_theta = torch.tensor(
                            [[cos_r, -sin_r, 0], [sin_r, cos_r, 0]],
                            device=self.device).unsqueeze(0)
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

                    sx = max(0, -px)
                    sy = max(0, -py)
                    ex = min(tw, W - px)
                    ey = min(th, H - py)
                    if ex <= sx or ey <= sy:
                        continue
                    cx, cy = max(0, px), max(0, py)
                    rh, rw = ey - sy, ex - sx

                    a = a_r[:, sy:ey, sx:ex]
                    canvas[bi, :, cy:cy+rh, cx:cx+rw] = \
                        canvas[bi, :, cy:cy+rh, cx:cx+rw] * (1 - a) + \
                        rgb_r[:, sy:ey, sx:ex] * a

            # Fine-grain texture overlay (moves with viewport)
            if fine_layer is not None:
                canvas = canvas * (1 - fine_opacity) + fine_layer * fine_opacity

            # Micro-stamps with linear motion
            if n_micro > 0:
                for mi in range(n_micro):
                    shapes_m = micro_shapes[mi]  # (B, 4, S, S)
                    for bi in range(B):
                        rgba = shapes_m[bi]
                        rgb = rgba[:3]
                        alpha_m = rgba[3:4]
                        sc = micro_scales[bi, mi].item()
                        th = max(3, int(self.shape_res * sc * H / 360))
                        tw = max(3, int(self.shape_res * sc * W / 640))
                        th, tw = min(th, H // 3), min(tw, W // 3)
                        rgb_r = F.interpolate(rgb.unsqueeze(0), (th, tw),
                                              mode="bilinear", align_corners=False)[0]
                        a_r = F.interpolate(alpha_m.unsqueeze(0), (th, tw),
                                            mode="bilinear", align_corners=False)[0]
                        px = int(micro_start_x[bi, mi].item() +
                                 micro_dx[bi, mi].item() * t_frac) - tw // 2
                        py = int(micro_start_y[bi, mi].item() +
                                 micro_dy[bi, mi].item() * t_frac) - th // 2
                        if 0 <= px < W - tw and 0 <= py < H - th:
                            canvas[bi, :, py:py+th, px:px+tw] = \
                                canvas[bi, :, py:py+th, px:px+tw] * (1 - a_r) + \
                                rgb_r * a_r

            # Fractal layout (pre-rendered overlay for temporal coherence)
            if fractal_overlay is not None:
                # Additive blend: fractal shapes rendered on black, so
                # non-zero pixels are the shapes to composite
                frac_mask = (fractal_overlay.sum(dim=1, keepdim=True) > 0.01).float()
                canvas = canvas * (1 - frac_mask) + fractal_overlay * frac_mask

            # Viewport transform (pan + zoom + rotation on entire canvas)
            if use_viewport:
                canvas = self._apply_viewport(canvas, ti, T, vp_pan, vp_zoom, vp_rot)

            # Fluid advection
            if flow_fields is not None:
                canvas = self._apply_fluid(canvas, flow_fields[:, ti])

            # Post-processing (consistent params across frames)
            canvas = canvas.clamp(1e-6, 1).pow(pp_gamma)
            if pp_hue_shift > 0:
                canvas = canvas.roll(pp_hue_shift, dims=1)
            if pp_wave is not None:
                canvas = canvas + pp_wave
            if pp_vignette is not None:
                canvas = canvas * pp_vignette
            frames.append(canvas.clamp(0, 1))

        return torch.stack(frames, dim=1)  # (B, T, 3, H, W)

    @torch.no_grad()
    def generate_sequence_9ch(self, batch_size, T=8, **kwargs):
        """Generate animated clips with all 9 channels.

        Returns: (B, T, 9, H, W) tensor in [0, 1] on self.device.
        Channels: RGB(3) + depth(1) + flow(2) + semantic(3)
        """
        if self.base_layers is None:
            self.build_base_layers()
        self._maybe_refresh_dynamic()

        B = batch_size
        H, W = self.H, self.W
        dev = self.device

        # Semantic role colors on device
        sem_bg = self._SEM_BG.to(dev).view(3, 1, 1)
        sem_layer = self._SEM_LAYER.to(dev).view(3, 1, 1)
        sem_stamp = self._SEM_STAMP.to(dev).view(3, 1, 1)
        sem_micro = self._SEM_MICRO.to(dev).view(3, 1, 1)

        # --- Scene setup (same as generate_sequence) ---
        bg_rgb = self._sample_colors(B).view(B, 3, 1, 1)
        bg_depth = torch.ones(B, 1, 1, 1, device=dev)  # far
        bg_sem = sem_bg.unsqueeze(0).expand(B, 3, 1, 1)

        # Layer params
        n_layers = torch.randint(
            self.layers_per_image[0], self.layers_per_image[1] + 1,
            (B,), device=dev)
        max_layers = n_layers.max().item()
        layer_idx = torch.randint(0, self.n_base_layers, (B, max_layers), device=dev)
        pan_str = kwargs.get("pan_strength", 0.5)
        pan_dx = (torch.rand(B, max_layers, device=dev) - 0.5) * W * pan_str
        pan_dy = (torch.rand(B, max_layers, device=dev) - 0.5) * H * pan_str
        pan_speed = torch.rand(B, max_layers, device=dev) * 0.8 + 0.2
        opacity_start = torch.rand(B, max_layers, device=dev) * 0.5 + 0.3
        opacity_end = torch.rand(B, max_layers, device=dev) * 0.5 + 0.3
        color_shift = (torch.rand(B, max_layers, 3, device=dev) - 0.5) * 0.3
        zoom_start = torch.rand(B, max_layers, device=dev) * 0.3 + 0.85
        zoom_end = torch.rand(B, max_layers, device=dev) * 0.3 + 0.85

        # Stamp params with physics
        n_stamps = torch.randint(
            self.stamps_per_image[0], self.stamps_per_image[1] + 1,
            (B,), device=dev)
        max_stamps = n_stamps.max().item()

        if self.shape_bank is None:
            max_stamps = 0
            n_stamps = torch.zeros(B, device=dev, dtype=torch.long)

        stamp_idx = torch.randint(0, max(1, self.shape_bank.shape[0] if self.shape_bank is not None else 1),
                                   (B, max(max_stamps, 1)), device=dev)
        trajectories = self._simulate_physics(B, max(max_stamps, 1), T)

        # Pre-transform stamps
        stamp_shapes = []
        for si in range(max_stamps):
            per_batch = []
            for bi in range(B):
                rgba = self._transform_bank_shape(
                    self.shape_bank[stamp_idx[bi, si]].clone())
                per_batch.append(rgba)
            stamp_shapes.append(torch.stack(per_batch))

        # Viewport params
        use_vp = kwargs.get("use_viewport", True)
        if use_vp:
            vp_pan_s = kwargs.get("viewport_pan", 0.3)
            vp_pan = (torch.rand(B, 2, device=dev) - 0.5) * \
                torch.tensor([W, H], device=dev).float() * vp_pan_s
            vp_zoom_s = kwargs.get("viewport_zoom", 0.15)
            vp_zoom = torch.ones(B, 2, device=dev)
            vp_zoom[:, 0] = 1.0 - vp_zoom_s / 2
            vp_zoom[:, 1] = 1.0 + vp_zoom_s / 2
            vp_rot_s = kwargs.get("viewport_rotation", 0.2)
            vp_rot = (torch.rand(B, 2, device=dev) - 0.5) * vp_rot_s

        pp_gamma = torch.rand(B, 1, 1, 1, device=dev) * 0.7 + 0.7

        # Depth values per layer/stamp
        layer_depths = torch.linspace(0.9, 0.6, max_layers, device=dev)
        stamp_depths = torch.linspace(0.5, 0.2, max_stamps, device=dev)

        # Scene template (consistent across all frames)
        template = self._pick_template()
        has_template = template != "random"

        # Pre-render template once to avoid per-frame flickering
        template_rgb = None
        if has_template:
            template_rgb = self._apply_scene_template(
                bg_rgb.expand(B, 3, H, W).clone(), template, B)

        # --- Render T frames with all channels ---
        all_frames = []  # list of (B, 9, H, W)
        prev_rgb = None

        for ti in range(T):
            t_frac = ti / max(T - 1, 1)

            # Init canvases
            if template_rgb is not None:
                rgb = template_rgb.clone()
            else:
                rgb = bg_rgb.expand(B, 3, H, W).clone()
            depth = bg_depth.expand(B, 1, H, W).clone()
            sem = bg_sem.expand(B, 3, H, W).clone()

            # --- Layer compositing ---
            for li in range(max_layers):
                active = (li < n_layers).float().view(B, 1, 1, 1)
                layer = self.base_layers[layer_idx[:, li]].clone()
                cs = color_shift[:, li].view(B, 3, 1, 1)
                layer = (layer + cs).clamp(0, 1)

                dx = pan_dx[:, li] * pan_speed[:, li] * t_frac
                dy = pan_dy[:, li] * pan_speed[:, li] * t_frac
                zoom = zoom_start[:, li] * (1 - t_frac) + zoom_end[:, li] * t_frac

                theta = torch.zeros(B, 2, 3, device=dev)
                theta[:, 0, 0] = 1.0 / zoom
                theta[:, 1, 1] = 1.0 / zoom
                theta[:, 0, 2] = -dx / (W / 2)
                theta[:, 1, 2] = -dy / (H / 2)
                grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
                layer = F.grid_sample(layer, grid, mode="bilinear",
                                       padding_mode="reflection", align_corners=False)

                opacity = (opacity_start[:, li] * (1 - t_frac) +
                           opacity_end[:, li] * t_frac).view(B, 1, 1, 1)
                alpha = opacity * active

                # Parallel compositing: RGB + depth + semantic
                rgb = rgb * (1 - alpha) + layer * alpha
                d_val = layer_depths[li].view(1, 1, 1, 1)
                depth = depth * (1 - alpha) + d_val * alpha
                sem_c = (sem_layer + torch.rand(3, 1, 1, device=dev) * 0.1
                         ).unsqueeze(0)
                sem = sem * (1 - alpha) + sem_c * alpha

            # --- Stamp compositing ---
            for si in range(max_stamps):
                active_s = (si < n_stamps).float()
                shapes = stamp_shapes[si]

                for bi in range(B):
                    if active_s[bi] < 0.5:
                        continue

                    px_f = trajectories[bi, si, ti, 0].item()
                    py_f = trajectories[bi, si, ti, 1].item()
                    sc = trajectories[bi, si, ti, 2].item()
                    rot_angle = trajectories[bi, si, ti, 3].item()

                    rgba = shapes[bi]
                    if kwargs.get("use_rotation", True) and abs(rot_angle) > 0.01:
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

                    s_rgb = rgba[:3]
                    s_alpha = rgba[3:4]

                    th = max(3, int(self.shape_res * sc * H / 360))
                    tw = max(3, int(self.shape_res * sc * W / 640))
                    th, tw = min(th, H // 2), min(tw, W // 2)

                    rgb_r = F.interpolate(s_rgb.unsqueeze(0), (th, tw),
                                          mode="bilinear", align_corners=False)[0]
                    a_r = F.interpolate(s_alpha.unsqueeze(0), (th, tw),
                                        mode="bilinear", align_corners=False)[0]

                    px = int(px_f) - tw // 2
                    py = int(py_f) - th // 2
                    if 0 <= px < W - tw and 0 <= py < H - th:
                        a = a_r
                        rgb[bi, :, py:py+th, px:px+tw] = \
                            rgb[bi, :, py:py+th, px:px+tw] * (1 - a) + rgb_r * a
                        d_val = stamp_depths[min(si, len(stamp_depths) - 1)]
                        depth[bi, :, py:py+th, px:px+tw] = \
                            depth[bi, :, py:py+th, px:px+tw] * (1 - a) + d_val * a
                        sem_c = sem_stamp + torch.rand(3, 1, 1, device=dev) * 0.1
                        sem[bi, :, py:py+th, px:px+tw] = \
                            sem[bi, :, py:py+th, px:px+tw] * (1 - a) + sem_c * a

            # Viewport transform (applied to all channels)
            if use_vp:
                rgb = self._apply_viewport(rgb, ti, T, vp_pan, vp_zoom, vp_rot)
                depth = self._apply_viewport(depth, ti, T, vp_pan, vp_zoom, vp_rot)
                sem = self._apply_viewport(sem, ti, T, vp_pan, vp_zoom, vp_rot)

            # Gamma on RGB only
            rgb = rgb.clamp(1e-6, 1).pow(pp_gamma)

            # --- Flow computation ---
            if prev_rgb is not None:
                # Approximate flow from pixel difference
                # Real flow comes from motion params but this is a simpler start
                flow = torch.zeros(B, 2, H, W, device=dev)

                # Global flow from viewport
                if use_vp:
                    per_frame_dx = vp_pan[:, 0] / max(T - 1, 1)
                    per_frame_dy = vp_pan[:, 1] / max(T - 1, 1)
                    flow[:, 0] = (per_frame_dx / (W / 2)).view(B, 1, 1).expand(B, H, W)
                    flow[:, 1] = (per_frame_dy / (H / 2)).view(B, 1, 1).expand(B, H, W)

                # Layer pan flow (additive, weighted by layer alpha)
                for li in range(max_layers):
                    active_l = (li < n_layers).float().view(B, 1, 1)
                    per_frame_dx = pan_dx[:, li] * pan_speed[:, li] / max(T - 1, 1)
                    per_frame_dy = pan_dy[:, li] * pan_speed[:, li] / max(T - 1, 1)
                    w = opacity_start[:, li].view(B, 1, 1) * active_l * 0.3
                    flow[:, 0] += (per_frame_dx / (W / 2)).view(B, 1, 1) * w
                    flow[:, 1] += (per_frame_dy / (H / 2)).view(B, 1, 1) * w

                # Stamp flow (local, at stamp positions)
                for si in range(max_stamps):
                    for bi in range(B):
                        if (si < n_stamps[bi]).item():
                            dx_s = (trajectories[bi, si, ti, 0] -
                                    trajectories[bi, si, ti-1, 0]).item()
                            dy_s = (trajectories[bi, si, ti, 1] -
                                    trajectories[bi, si, ti-1, 1]).item()
                            sc_s = trajectories[bi, si, ti, 2].item()
                            sx = int(trajectories[bi, si, ti, 0].item())
                            sy = int(trajectories[bi, si, ti, 1].item())
                            r = max(3, int(self.shape_res * sc_s * H / 720))
                            x0 = max(0, sx - r)
                            y0 = max(0, sy - r)
                            x1 = min(W, sx + r)
                            y1 = min(H, sy + r)
                            if x1 > x0 and y1 > y0:
                                flow[bi, 0, y0:y1, x0:x1] += dx_s / (W / 2)
                                flow[bi, 1, y0:y1, x0:x1] += dy_s / (H / 2)

                # Normalize to [0, 1] from [-1, 1]
                flow = (flow.clamp(-1, 1) + 1) / 2
            else:
                flow = torch.full((B, 2, H, W), 0.5, device=dev)  # neutral

            prev_rgb = rgb.clone()

            # Combine all channels
            frame_9ch = torch.cat([
                rgb.clamp(0, 1),
                depth.clamp(0, 1),
                flow.clamp(0, 1),
                sem.clamp(0, 1),
            ], dim=1)  # (B, 9, H, W)
            all_frames.append(frame_9ch)

        return torch.stack(all_frames, dim=1)  # (B, T, 9, H, W)

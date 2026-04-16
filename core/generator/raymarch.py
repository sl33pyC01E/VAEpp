#!/usr/bin/env python3
"""SDF ray marching for VAEpp0r generator.

Fully-vectorized 3D primitive rendering on GPU:
  - Sphere SDF
  - Rounded-box SDF
  - Torus SDF

For each pixel, cast a ray from the camera through the image plane and
step along it, computing the minimum SDF distance to any primitive in
the scene. On hit, apply Lambert shading with a single directional light.
Depth composite onto the underlying canvas.

Also: "sphere dip" scene — a sphere with a downward Z velocity that
crosses the fluid plane, triggering a ripple impact on the existing
Phase-1 fluid system.
"""

import math
import torch


class RaymarchMixin:
    """Mixin providing SDF ray-marched 3D primitives."""

    # ------------------------------------------------------------------
    # Ray grid cache
    # ------------------------------------------------------------------
    def _ensure_ray_grid(self, fov_deg=60.0):
        H, W = self.H, self.W
        key = (H, W, float(fov_deg), str(self.device))
        if getattr(self, "_ray_grid_key", None) == key:
            return
        dev = self.device
        aspect = W / H
        fov = math.radians(fov_deg)
        tan_half = math.tan(fov / 2)
        # Normalized screen coords in [-1, 1]
        y = torch.linspace(-1, 1, H, device=dev)
        x = torch.linspace(-1, 1, W, device=dev) * aspect
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # Camera at (0, 0, -2), looking along +Z
        rd = torch.stack([xx * tan_half, -yy * tan_half,
                          torch.ones_like(xx)], dim=0)
        rd = rd / rd.norm(dim=0, keepdim=True)
        self._ray_dir = rd          # (3, H, W)
        self._ray_origin = torch.tensor([0.0, 0.0, -2.0], device=dev)
        self._ray_grid_key = key

    # ------------------------------------------------------------------
    # SDF primitives (vectorized over pixels)
    # ------------------------------------------------------------------
    @staticmethod
    def _sdf_sphere(p, center, radius):
        """p: (3, H, W) positions; center: (3,); radius: scalar.
        Returns (H, W) signed distance."""
        d = p - center.view(3, 1, 1)
        return d.norm(dim=0) - radius

    @staticmethod
    def _sdf_box(p, center, half_extents, radius=0.05):
        """Rounded box: p (3,H,W), center (3,), half_extents (3,)."""
        d = (p - center.view(3, 1, 1)).abs() - half_extents.view(3, 1, 1)
        # Exterior part
        ext = d.clamp_min(0.0).norm(dim=0)
        # Interior negative
        interior = d.max(dim=0).values.clamp_max(0.0)
        return ext + interior - radius

    @staticmethod
    def _sdf_torus(p, center, R, r):
        """Torus with major radius R, minor radius r."""
        d = p - center.view(3, 1, 1)
        q_x = torch.sqrt(d[0] * d[0] + d[2] * d[2]) - R
        q_y = d[1]
        return torch.sqrt(q_x * q_x + q_y * q_y) - r

    # ------------------------------------------------------------------
    # Scene SDF dispatch
    # ------------------------------------------------------------------
    def _scene_sdf(self, p, primitives):
        """Return (dist, color) where dist is (H, W) min over all primitives
        and color is (3, H, W) taken from the closest primitive."""
        dev = self.device
        H, W = p.shape[-2], p.shape[-1]
        min_d = torch.full((H, W), 1e9, device=dev)
        col = torch.zeros(3, H, W, device=dev)
        for prim in primitives:
            kind = prim["kind"]
            center = torch.tensor(prim["center"], device=dev, dtype=torch.float32)
            c = torch.tensor(prim["color"], device=dev, dtype=torch.float32)
            if kind == "sphere":
                d = self._sdf_sphere(p, center, float(prim["radius"]))
            elif kind == "box":
                he = torch.tensor(prim["half_extents"], device=dev, dtype=torch.float32)
                d = self._sdf_box(p, center, he, float(prim.get("r", 0.05)))
            elif kind == "torus":
                d = self._sdf_torus(p, center, float(prim["R"]), float(prim["r"]))
            else:
                continue
            mask = d < min_d
            min_d = torch.where(mask, d, min_d)
            # Broadcast color into (3, H, W) over `mask`
            col = torch.where(mask.unsqueeze(0), c.view(3, 1, 1).expand_as(col), col)
        return min_d, col

    # ------------------------------------------------------------------
    # Recipe sampling
    # ------------------------------------------------------------------
    def _sample_raymarch_recipe(self, T, n_spheres=2, n_boxes=0, n_tori=0,
                                march_steps=24, sphere_dip=False):
        """Sample primitive positions + velocities + colors.
        Each primitive has (x0, y0, z0, vx, vy, vz, gravity, color).

        Primitives spawn close to the camera with large radii so they
        cover a visually-obvious portion of the frame. At z=0.3 with
        radius 0.5, a sphere projects to roughly 60% of the canvas
        height — unmistakable."""
        rng_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = __import__("random").Random(rng_seed)
        prims = []
        for _ in range(n_spheres):
            prims.append({
                "kind": "sphere",
                "radius": rng.uniform(0.35, 0.75),
                "x0": rng.uniform(-0.5, 0.5),
                "y0": rng.uniform(-0.4, 0.4),
                "z0": rng.uniform(-0.2, 0.6),
                "vx": rng.uniform(-0.02, 0.02),
                "vy": rng.uniform(-0.02, 0.02),
                "vz": rng.uniform(-0.03, 0.03),
                "gravity": 0.0,
                "color": [rng.uniform(0.4, 1), rng.uniform(0.4, 1), rng.uniform(0.4, 1)],
            })
        for _ in range(n_boxes):
            prims.append({
                "kind": "box",
                "half_extents": [rng.uniform(0.1, 0.3)] * 3,
                "x0": rng.uniform(-0.8, 0.8),
                "y0": rng.uniform(-0.6, 0.6),
                "z0": rng.uniform(0.5, 2.5),
                "vx": rng.uniform(-0.02, 0.02),
                "vy": rng.uniform(-0.02, 0.02),
                "vz": rng.uniform(-0.02, 0.02),
                "gravity": 0.0,
                "color": [rng.uniform(0.3, 1), rng.uniform(0.3, 1), rng.uniform(0.3, 1)],
                "r": rng.uniform(0.03, 0.08),
            })
        for _ in range(n_tori):
            prims.append({
                "kind": "torus",
                "R": rng.uniform(0.2, 0.4), "r": rng.uniform(0.05, 0.12),
                "x0": rng.uniform(-0.6, 0.6),
                "y0": rng.uniform(-0.4, 0.4),
                "z0": rng.uniform(0.7, 2.2),
                "vx": rng.uniform(-0.015, 0.015),
                "vy": rng.uniform(-0.015, 0.015),
                "vz": rng.uniform(-0.02, 0.02),
                "gravity": 0.0,
                "color": [rng.uniform(0.3, 1), rng.uniform(0.3, 1), rng.uniform(0.3, 1)],
            })

        # Sphere-dip scene: one sphere with strong downward z velocity, plus
        # a flagged impact time when it crosses z=0.
        dip_events = []
        if sphere_dip:
            sph = {
                "kind": "sphere",
                "radius": rng.uniform(0.25, 0.4),
                "x0": rng.uniform(-0.3, 0.3),
                "y0": rng.uniform(-0.6, -0.3),
                "z0": rng.uniform(0.8, 1.5),
                "vx": 0.0, "vy": rng.uniform(0.03, 0.06),  # falling in y
                "vz": 0.0, "gravity": 0.05,
                "color": [rng.uniform(0.5, 1), rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.5)],
            }
            prims.append(sph)
            # Detect when sphere y crosses the "plane" (y=0): y(t) = y0 + vy*t + 0.5*g*t^2 = 0
            a = 0.5 * sph["gravity"]
            b = sph["vy"]
            c = sph["y0"]
            if abs(a) > 1e-6:
                disc = b * b - 4 * a * c
                if disc >= 0:
                    t_imp = (-b + math.sqrt(disc)) / (2 * a)
                    if 0 < t_imp < T:
                        dip_events.append({"t": int(t_imp)})
            else:
                if abs(b) > 1e-6:
                    t_imp = -c / b
                    if 0 < t_imp < T:
                        dip_events.append({"t": int(t_imp)})

        return {
            "enable": True,
            "march_steps": int(march_steps),
            "primitives": prims,
            "sphere_dip": bool(sphere_dip),
            "dip_events": dip_events,
            "light_dir": [0.4, -0.7, -0.6],
            "bg_fog": 0.2,
        }

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def _apply_raymarch(self, canvas, ti, rm):
        if rm is None or not rm.get("enable", False):
            return canvas
        self._ensure_ray_grid(fov_deg=60.0)
        dev = self.device
        H, W = self.H, self.W
        # Advance primitives by ti
        dt = float(ti)
        prims = []
        for p in rm["primitives"]:
            pp = dict(p)
            pp["center"] = [
                p["x0"] + p.get("vx", 0) * dt,
                p["y0"] + p.get("vy", 0) * dt + 0.5 * p.get("gravity", 0) * dt * dt,
                p["z0"] + p.get("vz", 0) * dt,
            ]
            prims.append(pp)

        # Ray march
        ro = self._ray_origin.view(3, 1, 1).expand(3, H, W)
        rd = self._ray_dir  # (3, H, W)
        t = torch.zeros(H, W, device=dev)
        hit_mask = torch.zeros(H, W, device=dev, dtype=torch.bool)
        hit_color = torch.zeros(3, H, W, device=dev)
        for _ in range(int(rm["march_steps"])):
            p = ro + rd * t.unsqueeze(0)
            d, col = self._scene_sdf(p, prims)
            # Record hits
            new_hits = (d < 0.005) & (~hit_mask)
            if new_hits.any():
                hit_color = torch.where(new_hits.unsqueeze(0), col, hit_color)
                hit_mask = hit_mask | new_hits
            # Advance rays that haven't hit yet; clamp min step to avoid zero advance
            step = d.clamp(min=0.005)
            t = torch.where(hit_mask, t, t + step)
            # Cut out rays that drifted too far
            if (t > 8.0).all():
                break

        # Lambert shading on hit surface using gradient normal
        # Cheap estimate: finite-diff normal on SDF
        eps = 0.005
        p = ro + rd * t.unsqueeze(0)
        # Only normal where hit
        e = torch.tensor([eps, 0, 0], device=dev).view(3, 1, 1)
        dx = self._scene_sdf(p + e, prims)[0] - self._scene_sdf(p - e, prims)[0]
        e = torch.tensor([0, eps, 0], device=dev).view(3, 1, 1)
        dy = self._scene_sdf(p + e, prims)[0] - self._scene_sdf(p - e, prims)[0]
        e = torch.tensor([0, 0, eps], device=dev).view(3, 1, 1)
        dz = self._scene_sdf(p + e, prims)[0] - self._scene_sdf(p - e, prims)[0]
        normal = torch.stack([dx, dy, dz], dim=0)
        normal = normal / normal.norm(dim=0, keepdim=True).clamp_min(1e-6)
        light = torch.tensor(rm["light_dir"], device=dev, dtype=torch.float32)
        light = light / light.norm().clamp_min(1e-6)
        lambert = (-normal * light.view(3, 1, 1)).sum(dim=0).clamp_min(0.1)

        shaded = hit_color * lambert.unsqueeze(0)
        # Composite onto canvas
        hit_mask_b = hit_mask.unsqueeze(0).unsqueeze(0).float()
        return canvas * (1 - hit_mask_b) + shaded.unsqueeze(0) * hit_mask_b

    def _dip_impact_to_fluid(self, rm, fluid_params):
        """If sphere_dip is set, append impacts to fluid_params based on
        dip_events. Call AFTER fluid sampling but BEFORE rendering."""
        if rm is None or not rm.get("sphere_dip", False):
            return fluid_params
        if fluid_params is None:
            # Create a minimal fluid dict with the dip impacts
            fluid_params = {
                "enable": True,
                "gerstner": [],
                "impacts": [],
                "warp_strength": 10.0,
                "border_atten": 0.15,
            }
        for ev in rm.get("dip_events", []):
            fluid_params["impacts"].append({
                "t": int(ev["t"]),
                "x": 0.0, "y": 0.0,  # center for simple demo
                "amp": 0.04,
                "lambda": 0.1,
                "damp": 0.1,
                "speed": 1.5,
            })
        return fluid_params

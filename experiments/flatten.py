#!/usr/bin/env python3
"""Flatten/Deflatten experiment.

Freezes a trained VAE encoder+decoder, inserts a 1D kernel-1 conv
flatten/deflatten bottleneck in latent space, trains only the
bottleneck to reconstruct through the frozen VAE.

Tests whether a flat 1D representation can preserve the spatial
information in the latent space.

Usage:
    python -m experiments.flatten --vae-ckpt synthyper_logs/latest.pt
"""

import argparse
import math
import os
import pathlib
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MiniVAE
from core.generator import SynthyperGenerator


class FlattenDeflatten(nn.Module):
    """1D kernel-1 conv flatten/deflatten bottleneck.

    Takes a spatial latent (B, C, H, W), serializes to a 1D sequence
    along a walk order, projects channels via 1x1 conv (bottleneck),
    then projects back and reshapes to spatial.

    Args:
        latent_channels: input/output channel count (e.g. 32)
        bottleneck_channels: compressed channel count (e.g. 6 for 6:1)
        spatial_h, spatial_w: latent spatial dims (e.g. 45, 80)
        walk_order: "raster" or "hilbert" (how to serialize 2D to 1D)
    """

    def __init__(self, latent_channels=32, bottleneck_channels=6,
                 spatial_h=45, spatial_w=80, walk_order="raster"):
        super().__init__()
        self.C = latent_channels
        self.B_ch = bottleneck_channels
        self.H = spatial_h
        self.W = spatial_w
        self.n_positions = spatial_h * spatial_w
        self.walk_order = walk_order

        # Flatten: per-position channel projection (1x1 conv = Linear per position)
        self.flatten_conv = nn.Conv1d(latent_channels, bottleneck_channels, 1)

        # Deflatten: project back + learned spatial embedding
        self.deflatten_conv = nn.Conv1d(bottleneck_channels, latent_channels, 1)
        self.spatial_embed = nn.Parameter(
            torch.randn(1, latent_channels, self.n_positions) * 0.02)

        # Walk order index (precomputed)
        self.register_buffer("walk_idx", self._build_walk_index())
        self.register_buffer("unwalk_idx", self._build_unwalk_index())

    def _build_walk_index(self):
        """Build serialization index for the chosen walk order."""
        if self.walk_order == "raster":
            return torch.arange(self.n_positions)
        elif self.walk_order == "hilbert":
            return torch.tensor(self._hilbert_curve(self.H, self.W),
                                dtype=torch.long)
        return torch.arange(self.n_positions)

    def _build_unwalk_index(self):
        """Build inverse index to restore spatial order."""
        idx = self.walk_idx
        inv = torch.zeros_like(idx)
        inv[idx] = torch.arange(len(idx))
        return inv

    def _hilbert_curve(self, h, w):
        """Approximate Hilbert curve for non-power-of-2 grids.
        Returns list of flat indices in Hilbert order."""
        # Use simple Z-order (Morton code) as approximation
        # True Hilbert would need a proper implementation
        n = h * w
        coords = []
        for i in range(h):
            for j in range(w):
                # Interleave bits of i and j for Z-order
                z = 0
                for bit in range(16):
                    z |= ((i >> bit) & 1) << (2 * bit + 1)
                    z |= ((j >> bit) & 1) << (2 * bit)
                coords.append((z, i * w + j))
        coords.sort()
        return [c[1] for c in coords]

    def flatten(self, latent):
        """Flatten spatial latent to 1D sequence with channel compression.

        Args:
            latent: (B, C, H, W)

        Returns:
            flat: (B, B_ch, N) where N = H*W, B_ch = bottleneck_channels
        """
        B, C, H, W = latent.shape
        # Reshape to sequence: (B, C, H*W)
        seq = latent.reshape(B, C, H * W)
        # Reorder along walk path
        seq = seq[:, :, self.walk_idx]
        # Channel compression: (B, C, N) -> (B, B_ch, N)
        flat = self.flatten_conv(seq)
        return flat

    def deflatten(self, flat):
        """Deflatten 1D sequence back to spatial latent.

        Args:
            flat: (B, B_ch, N)

        Returns:
            latent: (B, C, H, W)
        """
        B = flat.shape[0]
        # Channel expansion: (B, B_ch, N) -> (B, C, N)
        seq = self.deflatten_conv(flat)
        # Add spatial embedding
        seq = seq + self.spatial_embed
        # Undo walk order
        seq = seq[:, :, self.unwalk_idx]
        # Reshape to spatial
        return seq.reshape(B, self.C, self.H, self.W)

    def forward(self, latent):
        """Full round trip: flatten then deflatten."""
        flat = self.flatten(latent)
        recon = self.deflatten(flat)
        return recon, flat

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# -- Preview -------------------------------------------------------------------

@torch.no_grad()
def save_preview(vae, bottleneck, gen, logdir, step, device, amp_dtype):
    """Save GT | VAE-only | VAE+Flatten reconstruction comparison."""
    try:
        vae.eval()
        bottleneck.eval()
        images = gen.generate(8)  # (8, 3, H, W)
        x = images.unsqueeze(1).to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # VAE only (no bottleneck)
            recon_vae, latent = vae(x)
            # VAE + bottleneck
            lat = vae.encode_video(x)
            lat_recon, flat = bottleneck(lat.squeeze(1))
            recon_flat = vae.decode_video(lat_recon.unsqueeze(1))

        T_r = recon_vae.shape[1]
        gt = images.cpu().numpy()
        rc_vae = recon_vae[:, -1, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[:, -1, :3].clamp(0, 1).float().cpu().numpy()

        del recon_vae, recon_flat, latent, lat, lat_recon
        vae.train()
        bottleneck.train()

        H, W = gen.H, gen.W
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)

        from PIL import Image
        rows = []
        for i in range(min(4, len(gt))):
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            v = (rc_vae[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            f = (rc_flat[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            row = np.concatenate([g, sep, v, sep, f], axis=1)
            rows.append(row)

        gap = np.full((4, rows[0].shape[1], 3), 14, dtype=np.uint8)
        grid = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], axis=0)

        stepped = os.path.join(logdir, f"preview_{step:06d}.png")
        latest = os.path.join(logdir, "preview_latest.png")
        Image.fromarray(grid).save(stepped)
        Image.fromarray(grid).save(latest)
        print(f"  preview: {stepped} (GT | VAE | Flatten)", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()


# -- Training ------------------------------------------------------------------

_stop_requested = False

def _handle_stop(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[Stop requested]", flush=True)

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, _handle_stop)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Load frozen VAE --
    print(f"Loading VAE from {args.vae_ckpt}...")
    ckpt = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    vae_config = ckpt.get("config", {})
    ch = vae_config.get("image_channels", 3)
    lat_ch = vae_config.get("latent_channels", 32)

    vae = MiniVAE(
        latent_channels=lat_ch, image_channels=ch, output_channels=ch,
        encoder_channels=64, decoder_channels=(256, 128, 64),
        encoder_time_downscale=(False, False, False),
        decoder_time_upscale=(False, False, False),
    ).to(device)

    src_sd = ckpt["model"] if "model" in ckpt else ckpt
    target_sd = vae.state_dict()
    for k, v in src_sd.items():
        if k in target_sd and v.shape == target_sd[k].shape:
            target_sd[k] = v
    vae.load_state_dict(target_sd)
    vae.eval()
    vae.requires_grad_(False)
    print(f"  VAE: {ch}ch, {lat_ch} latent, frozen")

    # Probe latent spatial dims
    with torch.no_grad():
        dummy = torch.randn(1, 1, ch, args.H, args.W, device=device)
        lat_dummy = vae.encode_video(dummy)
        _, _, lat_C, lat_H, lat_W = lat_dummy.shape
    print(f"  Latent: ({lat_C}, {lat_H}, {lat_W}) = {lat_C * lat_H * lat_W} values")
    print(f"  Bottleneck: {args.bottleneck_ch}ch × {lat_H * lat_W} positions "
          f"= {args.bottleneck_ch * lat_H * lat_W} flat values")
    print(f"  Compression: {lat_C / args.bottleneck_ch:.1f}:1 channel, "
          f"total {lat_C * lat_H * lat_W} -> {args.bottleneck_ch * lat_H * lat_W}")

    # -- Flatten/Deflatten bottleneck --
    bottleneck = FlattenDeflatten(
        latent_channels=lat_C,
        bottleneck_channels=args.bottleneck_ch,
        spatial_h=lat_H, spatial_w=lat_W,
        walk_order=args.walk_order,
    ).to(device)
    print(f"  Bottleneck: {bottleneck.param_count():,} params, "
          f"walk={args.walk_order}")

    # -- Generator --
    gen = SynthyperGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=500, n_base_layers=64,
    )
    bank_dir = os.path.join(os.path.dirname(__file__), "bank")
    if os.path.isdir(bank_dir):
        bank_files = [f for f in os.listdir(bank_dir)
                      if f.startswith("shapes_") and f.endswith(".pt")]
        if bank_files:
            gen.setup_dynamic_bank(bank_dir, working_size=500)
    gen.build_base_layers()

    # -- Optimizer (bottleneck only) --
    opt = torch.optim.AdamW(bottleneck.parameters(), lr=float(args.lr),
                            weight_decay=0.01)

    # -- Resume --
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "bottleneck" in rk:
            bottleneck.load_state_dict(rk["bottleneck"])
            start_step = rk.get("step", 0)
            if rk.get("optimizer") and not args.fresh_opt:
                try:
                    opt.load_state_dict(rk["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
            print(f"  Resumed bottleneck from {args.resume} at step {start_step}")
        else:
            print(f"  WARNING: no 'bottleneck' key in {args.resume}")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}")
    print(flush=True)

    # Initial preview
    save_preview(vae, bottleneck, gen, str(logdir), start_step, device, amp_dtype)

    # -- Loop --
    t0 = time.time()

    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    step = start_step
    for step in range(start_step + 1, args.total_steps + 1):
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            break

        images = gen.generate(args.batch_size)
        x = images.unsqueeze(1).to(device)

        bottleneck.train()
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Encode through frozen VAE
            with torch.no_grad():
                latent = vae.encode_video(x).squeeze(1)  # (B, C, H, W)

            # Flatten + deflatten
            lat_recon, flat = bottleneck(latent)

            # Latent reconstruction loss
            lat_loss = F.mse_loss(lat_recon, latent)

            # Decode through frozen VAE for pixel loss
            with torch.no_grad():
                gt_recon = vae.decode_video(latent.unsqueeze(1))
            flat_recon = vae.decode_video(lat_recon.unsqueeze(1))
            T_r = flat_recon.shape[1]
            pixel_loss = F.mse_loss(flat_recon[:, -T_r:], gt_recon[:, -T_r:])

            total = args.w_latent * lat_loss + args.w_pixel * pixel_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), 1.0)
        opt.step()

        del latent, lat_recon, flat, gt_recon, flat_recon, images, x

        if step % args.log_every == 0:
            el = time.time() - t0
            sps = step / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"[{step}/{args.total_steps}] lat={lat_loss.item():.6f} "
                  f"pix={pixel_loss.item():.6f} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            save_preview(vae, bottleneck, gen, str(logdir), step,
                         device, amp_dtype)

        if step % args.save_every == 0:
            d = {
                "bottleneck": bottleneck.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "config": {
                    "latent_channels": lat_C,
                    "bottleneck_channels": args.bottleneck_ch,
                    "spatial_h": lat_H,
                    "spatial_w": lat_W,
                    "walk_order": args.walk_order,
                    "vae_ckpt": args.vae_ckpt,
                },
            }
            torch.save(d, logdir / f"step_{step:06d}.pt")
            torch.save(d, logdir / "latest.pt")
            print(f"  saved step {step}", flush=True)

            # Keep last 10
            ckpts = sorted(logdir.glob("step_*.pt"),
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if step > start_step:
        d = {
            "bottleneck": bottleneck.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": {
                "latent_channels": lat_C,
                "bottleneck_channels": args.bottleneck_ch,
                "spatial_h": lat_H,
                "spatial_w": lat_W,
                "walk_order": args.walk_order,
                "vae_ckpt": args.vae_ckpt,
            },
        }
        torch.save(d, logdir / f"step_{step:06d}.pt")
        torch.save(d, logdir / "latest.pt")
        print(f"  saved step {step}", flush=True)

    print(f"\nDone. {step - start_step} steps in {(time.time() - t0) / 60:.1f}min")


def main():
    p = argparse.ArgumentParser(description="Flatten/Deflatten experiment")
    p.add_argument("--vae-ckpt", required=True,
                   help="Path to trained static VAE checkpoint")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--bottleneck-ch", type=int, default=6,
                   help="Channels after flatten (6 = ~5:1 compression)")
    p.add_argument("--walk-order", default="raster",
                   choices=["raster", "hilbert"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="1e-3")
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--w-latent", type=float, default=1.0)
    p.add_argument("--w-pixel", type=float, default=0.5)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--logdir", default="flatten_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--preview-every", type=int, default=200)
    p.add_argument("--resume", default=None,
                   help="Resume from bottleneck checkpoint")
    p.add_argument("--fresh-opt", action="store_true",
                   help="Fresh optimizer on resume")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

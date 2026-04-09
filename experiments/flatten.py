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
from core.generator import VAEpp0rGenerator


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
        elif self.walk_order == "morton":
            return torch.tensor(self._morton_curve(self.H, self.W),
                                dtype=torch.long)
        return torch.arange(self.n_positions)

    def _build_unwalk_index(self):
        """Build inverse index to restore spatial order."""
        idx = self.walk_idx
        inv = torch.zeros_like(idx)
        inv[idx] = torch.arange(len(idx))
        return inv

    def _hilbert_curve(self, h, w):
        """Hilbert curve for arbitrary grids.
        Returns list of flat indices in Hilbert order."""
        # Compute on a power-of-2 grid, then filter to actual dims
        order = max(h, w).bit_length()
        n = 1 << order  # next power of 2

        def d2xy(n, d):
            """Convert Hilbert index d to (x, y) on n×n grid."""
            x = y = 0
            s = 1
            while s < n:
                rx = 1 if (d & 2) else 0
                ry = 1 if ((d & 1) ^ rx) else 0
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                d >>= 2
                s <<= 1
            return x, y

        coords = []
        for d in range(n * n):
            x, y = d2xy(n, d)
            if y < h and x < w:
                coords.append(y * w + x)
        return coords

    def _morton_curve(self, h, w):
        """Z-order (Morton code) curve for arbitrary grids.
        Returns list of flat indices in Morton order."""
        coords = []
        for i in range(h):
            for j in range(w):
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
def save_preview(vae, bottleneck, gen, logdir, step, device, amp_dtype,
                  encode_fn=None, decode_fn=None):
    """Save GT | VAE | Flatten reconstruction comparison.
    encode_fn/decode_fn override vae paths when provided (for FSQ)."""
    _encode = encode_fn or (lambda x: vae.encode_video(x))
    _decode = decode_fn or (lambda z: vae.decode_video(z))
    try:
        vae.eval()
        bottleneck.eval()
        images = gen.generate(8)  # (8, 3, H, W)
        x = images.unsqueeze(1).to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Encode (includes FSQ if present)
            lat = _encode(x)

            # VAE-only decode (no bottleneck)
            recon_vae = _decode(lat)

            # VAE + bottleneck
            lat_recon, flat = bottleneck(lat.squeeze(1))
            recon_flat = _decode(lat_recon.unsqueeze(1))

        gt = images.cpu().numpy()
        rc_vae = recon_vae[:, -1, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[:, -1, :3].clamp(0, 1).float().cpu().numpy()

        del recon_vae, recon_flat, lat, lat_recon
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
        bottleneck.train()


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
    enc_ch = vae_config.get("encoder_channels", 64)
    dec_ch_str = vae_config.get("decoder_channels", "256,128,64")
    if isinstance(dec_ch_str, str):
        dec_ch = tuple(int(x) for x in dec_ch_str.split(","))
    elif isinstance(dec_ch_str, (list, tuple)):
        dec_ch = tuple(dec_ch_str)
    else:
        dec_ch = (256, 128, 64)

    vae = MiniVAE(
        latent_channels=lat_ch, image_channels=ch, output_channels=ch,
        encoder_channels=enc_ch, decoder_channels=dec_ch,
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

    # Load FSQ projections if present in checkpoint
    fsq_cfg = vae_config.get("fsq", {})
    fsq_layer = None
    pre_quant = None
    post_quant = None
    if fsq_cfg and fsq_cfg.get("levels"):
        from core.fsq import FSQ
        levels = fsq_cfg["levels"]
        if isinstance(levels, str):
            levels = [int(x) for x in levels.split(",")]
        fsq_dims = len(levels)
        fsq_layer = FSQ(levels=levels).to(device)
        pre_quant = nn.Conv2d(lat_ch, fsq_dims, 1).to(device)
        post_quant = nn.Conv2d(fsq_dims, lat_ch, 1).to(device)
        if ckpt.get("pre_quant"):
            pre_quant.load_state_dict(ckpt["pre_quant"])
        if ckpt.get("post_quant"):
            post_quant.load_state_dict(ckpt["post_quant"])
        pre_quant.eval()
        post_quant.eval()
        pre_quant.requires_grad_(False)
        post_quant.requires_grad_(False)
        print(f"  FSQ: levels={levels}, {fsq_dims} dims, frozen")

    has_fsq = fsq_layer is not None

    def encode_latent(x_in):
        """Encode through VAE, and FSQ projections if present."""
        lat = vae.encode_video(x_in)
        if has_fsq:
            B, Tp, C, Hl, Wl = lat.shape
            lf = lat.reshape(B * Tp, C, Hl, Wl)
            z_proj = pre_quant(lf)
            z_q, _ = fsq_layer(z_proj)
            lat = post_quant(z_q).reshape(B, Tp, C, Hl, Wl)
        return lat

    def decode_latent(lat_in):
        """Decode through VAE."""
        return vae.decode_video(lat_in)

    # Probe latent spatial dims
    with torch.no_grad():
        dummy = torch.randn(1, 1, ch, args.H, args.W, device=device)
        lat_dummy = encode_latent(dummy)
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
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=args.bank_size, n_base_layers=args.n_layers,
    )
    bank_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bank")
    bank_files = [f for f in os.listdir(bank_dir)
                  if f.startswith("shapes_") and f.endswith(".pt")] \
        if os.path.isdir(bank_dir) else []
    if bank_files:
        gen.setup_dynamic_bank(bank_dir, working_size=args.bank_size,
                                refresh_interval=50)
        gen.build_base_layers()
    else:
        gen.build_banks()
    if args.disco:
        gen.disco_quadrant = True
    print(f"  Generator: bank={gen.bank_size}, disco={getattr(gen, 'disco_quadrant', False)}")

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
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size
    accum = args.grad_accum

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}"
          f"{f', gen-batch={gen_bs}' if gen_bs != args.batch_size else ''}")
    print(flush=True)

    def _make_checkpoint():
        return {
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

    # Initial preview
    save_preview(vae, bottleneck, gen, str(logdir), start_step, device, amp_dtype,
                 encode_fn=encode_latent, decode_fn=decode_latent)

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

        bottleneck.train()
        opt.zero_grad(set_to_none=True)

        for _ai in range(accum):
            if gen_bs < args.batch_size:
                chunks = []
                rem = args.batch_size
                while rem > 0:
                    n = min(gen_bs, rem)
                    chunks.append(gen.generate(n))
                    rem -= n
                images = torch.cat(chunks)
            else:
                images = gen.generate(args.batch_size)
            x = images.unsqueeze(1).to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                # Encode through frozen VAE (+FSQ if present)
                with torch.no_grad():
                    latent = encode_latent(x).squeeze(1)  # (B, C, H, W)

                # Flatten + deflatten
                lat_recon, flat = bottleneck(latent)

                # Latent reconstruction loss
                lat_loss = F.mse_loss(lat_recon, latent)

                # Decode through frozen VAE for pixel loss
                with torch.no_grad():
                    gt_recon = decode_latent(latent.unsqueeze(1))
                flat_recon = decode_latent(lat_recon.unsqueeze(1))
                T_r = flat_recon.shape[1]
                pixel_loss = F.mse_loss(flat_recon[:, -T_r:], gt_recon[:, -T_r:])

                total = args.w_latent * lat_loss + args.w_pixel * pixel_loss

            scaler.scale(total / accum).backward()
            del latent, lat_recon, flat, gt_recon, flat_recon, images, x

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            el = time.time() - t0
            sps = (step - start_step) / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"[{step}/{args.total_steps}] lat={lat_loss.item():.6f} "
                  f"pix={pixel_loss.item():.6f} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            save_preview(vae, bottleneck, gen, str(logdir), step,
                         device, amp_dtype,
                         encode_fn=encode_latent, decode_fn=decode_latent)

        if step % args.save_every == 0:
            d = _make_checkpoint()
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
        d = _make_checkpoint()
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
                   choices=["raster", "hilbert", "morton"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="1e-3")
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--w-latent", type=float, default=1.0)
    p.add_argument("--w-pixel", type=float, default=0.5)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--gen-batch", type=int, default=0,
                   help="Generator batch size (0 = same as batch-size). "
                        "Lower to reduce generator VRAM.")
    p.add_argument("--logdir", default="flatten_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--preview-every", type=int, default=200)
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode")
    p.add_argument("--resume", default=None,
                   help="Resume from bottleneck checkpoint")
    p.add_argument("--fresh-opt", action="store_true",
                   help="Fresh optimizer on resume")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

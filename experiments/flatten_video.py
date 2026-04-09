#!/usr/bin/env python3
"""Flatten/Deflatten experiment — TEMPORAL variant.

Freezes a trained temporal VAE encoder+decoder, inserts a 1D kernel-1
conv flatten/deflatten bottleneck in latent space (per-frame, no temporal
info), trains only the bottleneck to reconstruct through the frozen VAE.

Tests whether a flat 1D representation can preserve the spatial AND
temporal information after the VAE's TPool/TGrow compression.

Usage:
    python -m experiments.flatten_video \
        --vae-ckpt synthyper_video_logs/latest.pt
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
from experiments.flatten import FlattenDeflatten


# -- Preview (MP4 video) ------------------------------------------------------

@torch.no_grad()
def save_preview(vae, bottleneck, gen, logdir, step, device, amp_dtype, T=8,
                  encode_fn=None, decode_fn=None):
    """Save GT | VAE | Flatten reconstruction as MP4.
    encode_fn/decode_fn override vae.encode_video/decode_video when provided
    (e.g. to include FSQ projections in the pipeline)."""
    _encode = encode_fn or (lambda x: vae.encode_video(x))
    _decode = decode_fn or (lambda z: vae.decode_video(z))
    try:
        vae.eval()
        bottleneck.eval()
        clips = gen.generate_sequence(2, T=T)  # (2, T, 3, H, W)
        x = clips.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Encode (includes FSQ if present)
            lat = _encode(x)  # (B, T', C, H, W)

            # VAE-only decode (no bottleneck)
            recon_vae = _decode(lat)

            # VAE + bottleneck (per-frame)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            lat_recon_flat, _ = bottleneck(lat_flat)
            lat_recon = lat_recon_flat.reshape(B, Tp, C, Hl, Wl)
            recon_flat = _decode(lat_recon)

        T_vae = recon_vae.shape[1]
        T_flat = recon_flat.shape[1]
        T_in = x.shape[1]
        T_show = min(T_vae, T_flat, T_in)

        gt = x[:, T_in - T_show:, :3].float().cpu().numpy()
        rc_vae = recon_vae[:, T_vae - T_show:, :3].clamp(0, 1).float().cpu().numpy()
        rc_flat = recon_flat[:, T_flat - T_show:, :3].clamp(0, 1).float().cpu().numpy()

        del recon_vae, recon_flat, lat, lat_recon
        bottleneck.train()

        H, W = gen.H, gen.W
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        hsep_w = W * 3 + 8
        hsep = np.full((4, hsep_w, 3), 14, dtype=np.uint8)

        stepped = os.path.join(logdir, f"preview_{step:06d}.mp4")
        latest = os.path.join(logdir, "preview_latest.mp4")

        for out_path in [stepped, latest]:
            frame_w = hsep_w
            frame_h = H * 2 + 4
            cmd = ["ffmpeg", "-y", "-v", "quiet",
                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{frame_w}x{frame_h}", "-r", "30",
                   "-i", "pipe:0",
                   "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", out_path]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for t in range(T_show):
                    rows = []
                    for b in range(min(2, B)):
                        g = (gt[b, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        v = (rc_vae[b, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        f_ = (rc_flat[b, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        row = np.concatenate([g, sep, v, sep, f_], axis=1)
                        rows.append(row)

                    if len(rows) == 2:
                        frame = np.concatenate([rows[0], hsep, rows[1]], axis=0)
                    else:
                        frame = rows[0]
                    proc.stdin.write(frame.tobytes())
            finally:
                proc.stdin.close()
                proc.wait()

        print(f"  preview: {stepped} (GT | VAE | Flatten, {T_show} frames)",
              flush=True)
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

    # -- Load frozen temporal VAE --
    print(f"Loading temporal VAE from {args.vae_ckpt}...")
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
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=(False, True, True),
    ).to(device)

    src_sd = ckpt["model"] if "model" in ckpt else ckpt
    target_sd = vae.state_dict()
    loaded = 0
    for k, v in src_sd.items():
        if k in target_sd and v.shape == target_sd[k].shape:
            target_sd[k] = v
            loaded += 1
    vae.load_state_dict(target_sd)
    vae.eval()
    vae.requires_grad_(False)
    print(f"  VAE: {ch}ch, {lat_ch} latent, temporal 4x, frozen ({loaded} weights)")

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
        dummy = torch.randn(1, args.T, ch, args.H, args.W, device=device)
        lat_dummy = encode_latent(dummy)
        _, Tp, lat_C, lat_H, lat_W = lat_dummy.shape
    print(f"  Latent: T'={Tp} (from T={args.T}), spatial ({lat_C}, {lat_H}, {lat_W}) "
          f"= {lat_C * lat_H * lat_W} values/frame")
    print(f"  Bottleneck: {args.bottleneck_ch}ch × {lat_H * lat_W} positions "
          f"= {args.bottleneck_ch * lat_H * lat_W} flat values/frame")
    print(f"  Compression: {lat_C / args.bottleneck_ch:.1f}:1 channel, "
          f"total {lat_C * lat_H * lat_W} -> {args.bottleneck_ch * lat_H * lat_W}")
    del dummy, lat_dummy

    # -- Flatten/Deflatten bottleneck --
    bottleneck = FlattenDeflatten(
        latent_channels=lat_C,
        bottleneck_channels=args.bottleneck_ch,
        spatial_h=lat_H, spatial_w=lat_W,
        walk_order=args.walk_order,
    ).to(device)
    print(f"  Bottleneck: {bottleneck.param_count():,} params, "
          f"walk={args.walk_order}")

    # -- Generator with motion pool --
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

    # Motion pool for fast temporal sampling
    pool_path = os.path.join(bank_dir, "motion_pool.json")
    if os.path.exists(pool_path):
        gen.load_motion_pool(pool_path)
    else:
        gen.build_motion_pool(n_clips=args.pool_size, T=args.T)
    if args.disco:
        gen.disco_quadrant = True
    print(f"  Generator: bank={gen.bank_size}, pool={gen.motion_pool_stats()}, "
          f"disco={getattr(gen, 'disco_quadrant', False)}")

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

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}, "
          f"T: {args.T}"
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
                "temporal": True,
                "T": args.T,
            },
        }

    # Initial preview
    save_preview(vae, bottleneck, gen, str(logdir), start_step, device,
                 amp_dtype, args.T, encode_fn=encode_latent, decode_fn=decode_latent)

    # -- Loop --
    t0 = time.time()

    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    step = start_step
    while step < args.total_steps:
        step += 1
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
                    chunks.append(gen.generate_from_pool(n))
                    rem -= n
                clips = torch.cat(chunks)
            else:
                clips = gen.generate_from_pool(args.batch_size)  # (B, T, 3, H, W)
            x = clips.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                # Encode through frozen VAE (+FSQ if present)
                with torch.no_grad():
                    latent = encode_latent(x)  # (B, T', C, H, W)

                B, Tp, C, Hl, Wl = latent.shape

                # Flatten + deflatten per-frame
                lat_flat = latent.reshape(B * Tp, C, Hl, Wl)
                lat_recon_flat, flat = bottleneck(lat_flat)
                lat_recon = lat_recon_flat.reshape(B, Tp, C, Hl, Wl)

                # Latent reconstruction loss (per-frame)
                lat_loss = F.mse_loss(lat_recon, latent)

                # Decode through frozen VAE for pixel loss
                with torch.no_grad():
                    gt_recon = decode_latent(latent)
                flat_recon = decode_latent(lat_recon)

                T_gt = gt_recon.shape[1]
                T_fl = flat_recon.shape[1]
                T_match = min(T_gt, T_fl)
                pixel_loss = F.mse_loss(
                    flat_recon[:, T_fl - T_match:],
                    gt_recon[:, T_gt - T_match:])

                # Temporal consistency in latent space
                temp_loss = torch.tensor(0.0, device=device)
                if Tp >= 2:
                    gt_diff = latent[:, 1:] - latent[:, :-1]
                    rc_diff = lat_recon[:, 1:] - lat_recon[:, :-1]
                    temp_loss = F.l1_loss(rc_diff, gt_diff)

                total = (args.w_latent * lat_loss
                         + args.w_pixel * pixel_loss
                         + args.w_temporal * temp_loss)

            scaler.scale(total / accum).backward()
            del latent, lat_recon, flat, gt_recon, flat_recon, clips, x

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            el = time.time() - t0
            steps_run = step - start_step
            sps = steps_run / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            parts = [f"lat={lat_loss.item():.6f}",
                     f"pix={pixel_loss.item():.6f}"]
            if Tp >= 2:
                parts.append(f"temp={temp_loss.item():.6f}")
            print(f"[{step}/{args.total_steps}] {' '.join(parts)} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            save_preview(vae, bottleneck, gen, str(logdir), step,
                         device, amp_dtype, args.T,
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
    p = argparse.ArgumentParser(
        description="Flatten/Deflatten experiment — temporal")
    p.add_argument("--vae-ckpt", required=True,
                   help="Path to trained temporal VAE checkpoint")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--bottleneck-ch", type=int, default=6,
                   help="Channels after flatten (6 = ~5:1 compression)")
    p.add_argument("--walk-order", default="raster",
                   choices=["raster", "hilbert", "morton"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", default="1e-3")
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--w-latent", type=float, default=1.0)
    p.add_argument("--w-pixel", type=float, default=0.5)
    p.add_argument("--w-temporal", type=float, default=1.0)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--gen-batch", type=int, default=0,
                   help="Generator batch size (0 = same as batch-size). "
                        "Lower to reduce generator VRAM.")
    p.add_argument("--logdir", default="flatten_video_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--resume", default=None,
                   help="Resume from bottleneck checkpoint")
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode")
    p.add_argument("--fresh-opt", action="store_true",
                   help="Fresh optimizer on resume")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

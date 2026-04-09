#!/usr/bin/env python3
"""FSQ fine-tuning experiment.

Loads a trained continuous VAE, inserts FSQ quantization layer between
encoder and decoder, and fine-tunes the full model with straight-through
estimator gradients flowing through the quantizer.

Preview: GT | Continuous VAE | FSQ-quantized VAE side-by-side.

Usage:
    python -m experiments.fsq --vae-ckpt synthyper_logs/latest.pt
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
from core.fsq import FSQ
from core.generator import VAEpp0rGenerator


# -- Preview -------------------------------------------------------------------

@torch.no_grad()
def save_preview(vae, vae_ref, fsq_layer, pre_quant, post_quant, gen, logdir,
                  step, device, amp_dtype):
    """Save Original VAE | FSQ VAE side-by-side PNG."""
    try:
        vae.eval()
        pre_quant.eval()
        post_quant.eval()
        images = gen.generate(8)  # (8, 3, H, W)
        x = images.unsqueeze(1).to(device)  # (8, 1, 3, H, W)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Original VAE (frozen reference)
            recon_ref, _ = vae_ref(x)

            # FSQ path
            lat = vae.encode_video(x).squeeze(1)  # (B, C, H, W)
            z_proj = pre_quant(lat)
            z_q, indices = fsq_layer(z_proj)
            lat_q = post_quant(z_q)
            recon_fsq = vae.decode_video(lat_q.unsqueeze(1))

        rc_ref = recon_ref[:, -1, :3].clamp(0, 1).float().cpu().numpy()
        rc_fsq = recon_fsq[:, -1, :3].clamp(0, 1).float().cpu().numpy()

        H, W = gen.H, gen.W
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        gap = np.full((4, W * 2 + 4, 3), 14, dtype=np.uint8)
        rows = []
        for i in range(min(8, len(rc_ref))):
            r = (rc_ref[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            q = (rc_fsq[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            row = np.concatenate([r, sep, q], axis=1)
            rows.append(row)

        grid = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], axis=0)
        from PIL import Image
        pil = Image.fromarray(grid)

        stepped = os.path.join(logdir, f"preview_{step:06d}.png")
        latest = os.path.join(logdir, "preview_latest.png")
        pil.save(stepped)
        pil.save(latest)

        # Codebook utilization stats
        all_idx = indices.reshape(-1)
        unique_codes = all_idx.unique().numel()
        total_codes = fsq_layer.codebook_size
        util = 100 * unique_codes / total_codes
        print(f"  preview: {stepped} (VAE | FSQ, "
              f"codebook: {unique_codes}/{total_codes} = {util:.1f}%)",
              flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        vae.train()
        pre_quant.train()
        post_quant.train()


@torch.no_grad()
def save_preview_video(vae, vae_ref, fsq_layer, pre_quant, post_quant, gen,
                       logdir, step, device, amp_dtype, T=8):
    """Save Original VAE | FSQ VAE side-by-side MP4 video."""
    try:
        vae.eval()
        pre_quant.eval()
        post_quant.eval()
        clips = gen.generate_sequence(2, T=T)  # (2, T, 3, H, W)
        x = clips.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Original VAE (frozen reference)
            recon_ref, _ = vae_ref(x)

            # FSQ path — quantize per-frame in latent space
            lat = vae.encode_video(x)  # (B, T', C, H, W)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            z_proj = pre_quant(lat_flat)
            z_q, indices = fsq_layer(z_proj)
            lat_q = post_quant(z_q)
            lat_q = lat_q.reshape(B, Tp, C, Hl, Wl)
            recon_fsq = vae.decode_video(lat_q)

        T_ref = recon_ref.shape[1]
        T_fsq = recon_fsq.shape[1]
        T_show = min(T_ref, T_fsq)

        rc_ref = recon_ref[:, T_ref - T_show:, :3].clamp(0, 1).float().cpu().numpy()
        rc_fsq = recon_fsq[:, T_fsq - T_show:, :3].clamp(0, 1).float().cpu().numpy()

        del recon_ref, recon_fsq, lat, lat_q, x
        vae.train()
        pre_quant.train()
        post_quant.train()

        H, W = gen.H, gen.W
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        hsep_w = W * 2 + 4
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
                        r = (rc_ref[b, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        q = (rc_fsq[b, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        row = np.concatenate([r, sep, q], axis=1)
                        rows.append(row)
                    if len(rows) == 2:
                        frame = np.concatenate([rows[0], hsep, rows[1]], axis=0)
                    else:
                        frame = rows[0]
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
                proc.wait()
            except Exception:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.kill()
                proc.wait()
                raise

        # Codebook utilization stats
        all_idx = indices.reshape(-1)
        unique_codes = all_idx.unique().numel()
        total_codes = fsq_layer.codebook_size
        util = 100 * unique_codes / total_codes
        print(f"  preview: {stepped} (VAE | FSQ, {T_show} frames, "
              f"codebook: {unique_codes}/{total_codes} = {util:.1f}%)",
              flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        vae.train()
        pre_quant.train()
        post_quant.train()


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

    # -- Load VAE --
    print(f"Loading VAE from {args.vae_ckpt}...", flush=True)
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

    temporal = vae_config.get("temporal", False)
    if temporal:
        etd = (True, True, False)
        dtu = (False, True, True)
    else:
        etd = (False, False, False)
        dtu = (False, False, False)

    vae = MiniVAE(
        latent_channels=lat_ch, image_channels=ch, output_channels=ch,
        encoder_channels=enc_ch, decoder_channels=dec_ch,
        encoder_time_downscale=etd, decoder_time_upscale=dtu,
    ).to(device)
    if args.grad_checkpoint:
        vae.use_checkpoint = True

    src_sd = ckpt["model"] if "model" in ckpt else ckpt
    target_sd = vae.state_dict()
    loaded = 0
    for k, v in src_sd.items():
        if k in target_sd and v.shape == target_sd[k].shape:
            target_sd[k] = v
            loaded += 1
    vae.load_state_dict(target_sd)
    pc = vae.param_count()
    vae_step = ckpt.get("global_step", ckpt.get("step", 0))
    print(f"  VAE: {ch}ch, lat={lat_ch}, enc={enc_ch}, dec={dec_ch}, "
          f"temporal={temporal}, {pc['total']:,} params, "
          f"step {vae_step}, {loaded} weights loaded", flush=True)

    # -- Reference VAE (frozen, for preview comparison) --
    vae_ref = MiniVAE(
        latent_channels=lat_ch, image_channels=ch, output_channels=ch,
        encoder_channels=enc_ch, decoder_channels=dec_ch,
        encoder_time_downscale=etd, decoder_time_upscale=dtu,
    ).to(device)
    ref_sd = vae_ref.state_dict()
    for k, v in (ckpt["model"] if "model" in ckpt else ckpt).items():
        if k in ref_sd and v.shape == ref_sd[k].shape:
            ref_sd[k] = v
    vae_ref.load_state_dict(ref_sd)
    vae_ref.eval()
    vae_ref.requires_grad_(False)
    del ckpt

    # -- FSQ layer --
    levels = [int(x) for x in args.levels.split(",")]
    fsq_dims = len(levels)
    fsq_layer = FSQ(levels=levels).to(device)
    codebook = fsq_layer.codebook_size
    print(f"  FSQ: levels={levels}, {fsq_dims} dims, "
          f"{codebook:,} codes", flush=True)

    # 1x1 conv projections: latent_channels <-> FSQ dims
    pre_quant = nn.Conv2d(lat_ch, fsq_dims, 1).to(device)
    post_quant = nn.Conv2d(fsq_dims, lat_ch, 1).to(device)
    print(f"  Projections: {lat_ch} -> {fsq_dims} -> {lat_ch}", flush=True)

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
    if temporal:
        pool_path = os.path.join(bank_dir, "motion_pool.json")
        if os.path.exists(pool_path):
            gen.load_motion_pool(pool_path)
        else:
            gen.build_motion_pool(n_clips=args.pool_size, T=args.T)
    if args.disco:
        gen.disco_quadrant = True
    print(f"  Generator: bank={gen.bank_size}, T={args.T}, "
          f"disco={getattr(gen, 'disco_quadrant', False)}", flush=True)

    # -- Optimizer (VAE + projections end-to-end, FSQ has no params) --
    params = list(vae.parameters()) + list(pre_quant.parameters()) + list(post_quant.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr),
                            weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    # -- Resume --
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in rk:
            src = rk["model"]
            tgt = vae.state_dict()
            for k, v in src.items():
                if k in tgt and v.shape == tgt[k].shape:
                    tgt[k] = v
            vae.load_state_dict(tgt)
            if rk.get("pre_quant"):
                try:
                    pre_quant.load_state_dict(rk["pre_quant"])
                except Exception:
                    print("  Fresh pre_quant (mismatch)", flush=True)
            if rk.get("post_quant"):
                try:
                    post_quant.load_state_dict(rk["post_quant"])
                except Exception:
                    print("  Fresh post_quant (mismatch)", flush=True)
            start_step = rk.get("global_step", rk.get("step", 0))
            if rk.get("optimizer") and not args.fresh_opt:
                try:
                    opt.load_state_dict(rk["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)", flush=True)
            if rk.get("scheduler") and not args.fresh_opt:
                try:
                    sched.load_state_dict(rk["scheduler"])
                except Exception:
                    print("  Fresh scheduler (mismatch)", flush=True)
            elif start_step > 0:
                # Rebuild scheduler at correct position for old checkpoints
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.total_steps,
                    eta_min=float(args.lr) * 0.01,
                    last_epoch=start_step)
            print(f"  Resumed from {args.resume} at step {start_step}", flush=True)

    # -- Precision --
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    if start_step > 0 and not args.fresh_opt and 'rk' in locals():
        if rk.get("scaler") and args.precision == "fp16":
            try:
                scaler.load_state_dict(rk["scaler"])
            except Exception:
                pass

    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size
    accum = args.grad_accum

    print(f"Steps: {args.total_steps}, LR: {args.lr}, "
          f"Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}"
          f"{f', gen-batch={gen_bs}' if gen_bs != args.batch_size else ''}",
          flush=True)
    print(f"Weights: w_mse={args.w_mse}", flush=True)
    print(flush=True)

    def _make_checkpoint():
        return {
            "model": vae.state_dict(),
            "pre_quant": pre_quant.state_dict(),
            "post_quant": post_quant.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": step,
            "config": {
                "image_channels": ch,
                "latent_channels": lat_ch,
                "encoder_channels": enc_ch,
                "decoder_channels": ",".join(str(x) for x in dec_ch),
                "temporal": temporal,
                "fsq": {
                    "levels": levels,
                },
            },
        }

    # Preview T for temporal models
    preview_T = args.T

    # Initial preview
    if temporal:
        save_preview_video(vae, vae_ref, fsq_layer, pre_quant, post_quant, gen,
                           str(logdir), start_step, device, amp_dtype, preview_T)
    else:
        save_preview(vae, vae_ref, fsq_layer, pre_quant, post_quant, gen,
                     str(logdir), start_step, device, amp_dtype)

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

        vae.train()
        pre_quant.train()
        post_quant.train()
        opt.zero_grad(set_to_none=True)

        for _ai in range(accum):
            if temporal:
                # Generate video clips
                if gen_bs < args.batch_size:
                    chunks = []
                    rem = args.batch_size
                    while rem > 0:
                        n = min(gen_bs, rem)
                        chunks.append(gen.generate_sequence(n, T=args.T))
                        rem -= n
                    x = torch.cat(chunks).to(device)
                else:
                    x = gen.generate_sequence(args.batch_size, T=args.T).to(device)
            else:
                # Generate single frames
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
                x = images.unsqueeze(1).to(device)  # (B, 1, C, H, W)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                # Encode → project → quantize → unproject → decode
                lat = vae.encode_video(x)  # (B, T', C, H, W)
                B_lat, Tp, C_lat, Hl, Wl = lat.shape
                lat_flat = lat.reshape(B_lat * Tp, C_lat, Hl, Wl)
                z_proj = pre_quant(lat_flat)
                z_q, indices = fsq_layer(z_proj)
                lat_q = post_quant(z_q)
                lat_q = lat_q.reshape(B_lat, Tp, C_lat, Hl, Wl)
                recon = vae.decode_video(lat_q)

                # Align temporal
                T_out = recon.shape[1]
                gt = x[:, -T_out:]

                # Reconstruction loss
                mse = F.mse_loss(recon, gt)
                total = args.w_mse * mse

            scaler.scale(total / accum).backward()
            del lat, lat_flat, z_proj, z_q, lat_q, recon, x

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_every == 0:
            el = time.time() - t0
            steps_run = step - start_step
            sps = steps_run / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            lr = opt.param_groups[0]["lr"]
            print(f"[{step}/{args.total_steps}] mse={mse.item():.6f} "
                  f"lr={lr:.1e} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            if temporal:
                save_preview_video(vae, vae_ref, fsq_layer, pre_quant, post_quant,
                                   gen, str(logdir), step, device, amp_dtype,
                                   preview_T)
            else:
                save_preview(vae, vae_ref, fsq_layer, pre_quant, post_quant, gen,
                             str(logdir), step, device, amp_dtype)

        if step % args.save_every == 0:
            d = _make_checkpoint()
            torch.save(d, logdir / f"step_{step:06d}.pt")
            torch.save(d, logdir / "latest.pt")
            print(f"  saved step {step}", flush=True)

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

    print(f"\nDone. {step - start_step} steps in {(time.time() - t0) / 60:.1f}min",
          flush=True)


def main():
    p = argparse.ArgumentParser(description="FSQ fine-tuning experiment")
    p.add_argument("--vae-ckpt", required=True,
                   help="Path to trained continuous VAE checkpoint")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--levels", type=str, default="8,8,8,8,8,8",
                   help="Comma-separated FSQ levels per dimension, "
                        "e.g. '8,8,8,8,8,8' or '3,5,7,9,11'")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="5e-4")
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--logdir", default="fsq_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--gen-batch", type=int, default=0,
                   help="Generator batch size (0 = same as batch-size). "
                        "Lower to reduce generator VRAM.")
    p.add_argument("--T", type=int, default=24,
                   help="Temporal frame count for video training")
    p.add_argument("--bank-size", type=int, default=5000,
                   help="Generator shape bank size")
    p.add_argument("--n-layers", type=int, default=128,
                   help="Generator base layers")
    p.add_argument("--pool-size", type=int, default=200,
                   help="Motion pool size for temporal training")
    p.add_argument("--alpha", type=float, default=3.0,
                   help="Generator alpha")
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Gradient checkpointing (trade compute for memory)")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

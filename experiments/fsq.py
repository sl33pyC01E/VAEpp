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
from core.generator import SynthyperGenerator


# -- Preview -------------------------------------------------------------------

@torch.no_grad()
def save_preview(vae, fsq_layer, gen, logdir, step, device, amp_dtype):
    """Save GT | Continuous | FSQ side-by-side PNG."""
    try:
        vae.eval()
        images = gen.generate(8)  # (8, 3, H, W)
        x = images.unsqueeze(1).to(device)  # (8, 1, 3, H, W)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Continuous path
            recon_cont, _ = vae(x)

            # FSQ path
            lat = vae.encode_video(x).squeeze(1)  # (B, C, H, W)
            lat_q, indices = fsq_layer(lat)
            recon_fsq = vae.decode_video(lat_q.unsqueeze(1))

        gt = images[:, :3].cpu().numpy()
        rc_cont = recon_cont[:, -1, :3].clamp(0, 1).float().cpu().numpy()
        rc_fsq = recon_fsq[:, -1, :3].clamp(0, 1).float().cpu().numpy()

        H, W = gen.H, gen.W
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        gap = np.full((4, W * 3 + 8, 3), 14, dtype=np.uint8)
        rows = []
        for i in range(min(8, len(gt))):
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            c = (rc_cont[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            q = (rc_fsq[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            row = np.concatenate([g, sep, c, sep, q], axis=1)
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
        print(f"  preview: {stepped} (GT | Cont | FSQ, "
              f"codebook: {unique_codes}/{total_codes} = {util:.1f}%)",
              flush=True)

        vae.train()
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
    else:
        dec_ch = tuple(dec_ch_str)

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

    # -- FSQ layer --
    n_groups = args.n_groups
    cpg = args.ch_per_group if args.ch_per_group > 0 else lat_ch // n_groups
    assert cpg * n_groups == lat_ch, (
        f"channels_per_group ({cpg}) * n_groups ({n_groups}) = {cpg * n_groups} "
        f"!= latent_channels ({lat_ch})")

    fsq_layer = FSQ(levels=args.levels, n_groups=n_groups,
                    channels_per_group=cpg).to(device)
    codebook = fsq_layer.codebook_size
    print(f"  FSQ: {args.levels} levels, {n_groups} groups, {cpg} ch/group, "
          f"{codebook:,} codes/group", flush=True)

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
        else:
            gen.build_banks()
    else:
        gen.build_banks()
    print(f"  Generator: bank={gen.bank_size}", flush=True)

    # -- Optimizer (full VAE + FSQ, FSQ has no learnable params but VAE adapts) --
    opt = torch.optim.AdamW(vae.parameters(), lr=float(args.lr),
                            weight_decay=0.01)
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
            start_step = rk.get("global_step", rk.get("step", 0))
            if rk.get("optimizer") and not args.fresh_opt:
                try:
                    opt.load_state_dict(rk["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)", flush=True)
            print(f"  Resumed from {args.resume} at step {start_step}", flush=True)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}",
          flush=True)
    print(f"Weights: w_mse={args.w_mse}, w_commit={args.w_commit}", flush=True)
    print(flush=True)

    # Initial preview
    save_preview(vae, fsq_layer, gen, str(logdir), start_step, device, amp_dtype)

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
        x = images.unsqueeze(1).to(device)  # (B, 1, C, H, W)

        vae.train()
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            # Encode → quantize → decode
            lat = vae.encode_video(x).squeeze(1)  # (B, C, H, W)
            lat_q, indices = fsq_layer(lat)
            recon = vae.decode_video(lat_q.unsqueeze(1))

            # Align temporal
            T_out = recon.shape[1]
            gt = x[:, -T_out:]

            # Reconstruction loss
            mse = F.mse_loss(recon, gt)

            # Commitment loss (push encoder output toward quantized)
            commit = F.mse_loss(lat, lat_q.detach())

            total = args.w_mse * mse + args.w_commit * commit

        total.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        sched.step()

        del lat, lat_q, recon, images, x

        if step % args.log_every == 0:
            el = time.time() - t0
            steps_run = step - start_step
            sps = steps_run / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            lr = opt.param_groups[0]["lr"]
            print(f"[{step}/{args.total_steps}] mse={mse.item():.6f} "
                  f"commit={commit.item():.6f} lr={lr:.1e} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            save_preview(vae, fsq_layer, gen, str(logdir), step,
                         device, amp_dtype)

        if step % args.save_every == 0:
            d = {
                "model": vae.state_dict(),
                "optimizer": opt.state_dict(),
                "global_step": step,
                "config": {
                    "image_channels": ch,
                    "latent_channels": lat_ch,
                    "encoder_channels": enc_ch,
                    "decoder_channels": ",".join(str(x) for x in dec_ch),
                    "temporal": temporal,
                    "fsq": {
                        "levels": args.levels,
                        "n_groups": n_groups,
                        "channels_per_group": cpg,
                    },
                },
            }
            torch.save(d, logdir / f"step_{step:06d}.pt")
            torch.save(d, logdir / "latest.pt")
            print(f"  saved step {step}", flush=True)

            ckpts = sorted(logdir.glob("step_*.pt"),
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if step > start_step:
        d = {
            "model": vae.state_dict(),
            "optimizer": opt.state_dict(),
            "global_step": step,
            "config": {
                "image_channels": ch,
                "latent_channels": lat_ch,
                "encoder_channels": enc_ch,
                "decoder_channels": ",".join(str(x) for x in dec_ch),
                "temporal": temporal,
                "fsq": {
                    "levels": args.levels,
                    "n_groups": n_groups,
                    "channels_per_group": cpg,
                },
            },
        }
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
    p.add_argument("--levels", type=int, default=8,
                   help="FSQ quantization levels per channel")
    p.add_argument("--n-groups", type=int, default=1,
                   help="Number of FSQ groups")
    p.add_argument("--ch-per-group", type=int, default=0,
                   help="Channels per group (0 = auto from latent_ch / n_groups)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="5e-4")
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--w-commit", type=float, default=0.25,
                   help="Commitment loss weight (push encoder toward quantized)")
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--logdir", default="fsq_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VAEpp Stage 2 — temporal training on animated synthetic data.

Enables VAE temporal compression (TPool/TGrow/MemBlock).
Generates T-frame clips with smooth motion, trains with temporal consistency loss.

Usage:
    python -m training.train_video
    python -m training.train_video --resume synthyper_logs/latest.pt --fresh-opt
"""

import argparse
import os
import pathlib
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MiniVAE
from core.generator import VAEppGenerator


# -- Preview -------------------------------------------------------------------

@torch.no_grad()
def save_preview(model, gen, logdir, step, device, amp_dtype, T=8):
    """Save GT | Recon side-by-side mp4 video."""
    try:
        model.eval()
        clips = gen.generate_sequence(2, T=T)  # (2, T, 3, H, W)
        x = clips.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon, _ = model(x)

        T_out = recon.shape[1]
        T_in = x.shape[1]
        T_show = min(T_out, T_in)
        gt = x[:, T_in - T_show:].float().cpu().numpy()
        rc = recon[:, T_out - T_show:].clamp(0, 1).float().cpu().numpy()

        del recon, x
        model.train()

        H, W = gen.H, gen.W
        sep = np.full((H, 4, 3), 14, dtype=np.uint8)
        hsep = np.full((4, W * 2 + 4, 3), 14, dtype=np.uint8)

        stepped = os.path.join(logdir, f"preview_{step:06d}.mp4")
        latest = os.path.join(logdir, "preview_latest.mp4")

        for out_path in [stepped, latest]:
            frame_w = W * 2 + 4
            frame_h = H * 2 + 4
            cmd = ["ffmpeg", "-y", "-v", "quiet",
                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{frame_w}x{frame_h}", "-r", "30",
                   "-i", "pipe:0",
                   "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", out_path]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            for t in range(T_show):
                # Clip 0: GT | Recon (top row)
                g0 = (gt[0, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                r0 = (rc[0, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                top = np.concatenate([g0, sep, r0], axis=1)

                # Clip 1: GT | Recon (bottom row)
                g1 = (gt[1, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                r1 = (rc[1, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                bot = np.concatenate([g1, sep, r1], axis=1)

                frame = np.concatenate([top, hsep, bot], axis=0)
                proc.stdin.write(frame.tobytes())

            proc.stdin.close()
            proc.wait()

        print(f"  preview: {stepped} ({T_show} frames)", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        model.train()


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

    # -- Model (3ch RGB, temporal ENABLED) --
    model = MiniVAE(
        latent_channels=args.latent_ch,
        image_channels=3,
        output_channels=3,
        encoder_channels=64,
        decoder_channels=(256, 128, 64),
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=(False, True, True),
    ).to(device)
    if args.grad_checkpoint:
        model.use_checkpoint = True
    pc = model.param_count()
    print(f"MiniVAE (3ch, temporal 4x): {pc['total']:,} params"
          f"{', grad-checkpoint' if args.grad_checkpoint else ''}")
    print(f"  t_downscale={model.t_downscale}, t_upscale={model.t_upscale}, "
          f"frames_to_trim={model.frames_to_trim}")

    # -- Generator --
    gen = VAEppGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=args.bank_size,
        n_base_layers=args.n_layers,
        alpha=args.alpha,
    )
    bank_dir = os.path.join(os.path.dirname(__file__), "bank")
    bank_files = [f for f in os.listdir(bank_dir)
                  if f.startswith("shapes_") and f.endswith(".pt")] \
        if os.path.isdir(bank_dir) else []
    if bank_files:
        gen.setup_dynamic_bank(bank_dir, working_size=args.bank_size,
                                refresh_interval=50)
        gen.build_base_layers()
    else:
        gen.build_banks()
    # Build motion clip pool
    pool_path = os.path.join(os.path.dirname(__file__), "bank", "motion_pool.json")
    if os.path.exists(pool_path):
        gen.load_motion_pool(pool_path)
    else:
        gen.build_motion_pool(n_clips=args.pool_size, T=args.T)
    if args.disco:
        gen.disco_quadrant = True
    print(f"Generator: bank={gen.bank_size}, layers={args.n_layers}, "
          f"T={args.T}, pool={gen.motion_pool_stats()}, "
          f"disco={gen.disco_quadrant}")

    # -- LPIPS --
    lpips_fn = None
    if args.w_lpips > 0:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="squeeze").to(device)
            lpips_fn.eval()
            lpips_fn.requires_grad_(False)
            print(f"LPIPS (squeeze) on {device}")
        except ImportError:
            print("WARNING: pip install lpips")

    # -- Optimizer --
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    # -- Resume --
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            src_sd = ckpt["model"]
            target_sd = model.state_dict()
            loaded, skipped = 0, 0
            for k, v in src_sd.items():
                if k in target_sd and v.shape == target_sd[k].shape:
                    target_sd[k] = v
                    loaded += 1
                else:
                    skipped += 1
            model.load_state_dict(target_sd)
            print(f"  Loaded {loaded} layers, {skipped} skipped "
                  f"(size mismatch or new)")
            global_step = ckpt.get("global_step", 0)
            if not args.fresh_opt and ckpt.get("optimizer"):
                try:
                    opt.load_state_dict(ckpt["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Resumed from {args.resume} at step {global_step}")

    if args.fresh_opt and global_step > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps - global_step,
            eta_min=float(args.lr) * 0.01)
        print(f"  Fresh optimizer from step {global_step}")

    # -- Precision --
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    if global_step > 0 and not args.fresh_opt:
        if ckpt.get("scheduler"):
            sched.load_state_dict(ckpt["scheduler"])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps - global_step,
                eta_min=float(args.lr) * 0.01, last_epoch=global_step)
        if ckpt.get("scaler") and args.precision == "fp16":
            scaler.load_state_dict(ckpt["scaler"])

    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size
    accum = args.grad_accum

    print(f"Steps: {args.total_steps}, LR: {args.lr}, "
          f"Batch: {args.batch_size}, T: {args.T}"
          f"{f', accum={accum}' if accum > 1 else ''}"
          f"{f', gen-batch={gen_bs}' if gen_bs != args.batch_size else ''}")
    print(f"Weights: mse={args.w_mse} lpips={args.w_lpips} "
          f"temporal={args.w_temporal}")
    print(f"Precision: {args.precision}, Device: {device}")
    print(flush=True)

    # -- Initial preview --
    save_preview(model, gen, str(logdir), global_step, device, amp_dtype, args.T)

    # -- Loop --
    t0 = time.time()
    start_step = global_step

    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    while global_step < args.total_steps:
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            print("[Stop detected, saving...]", flush=True)
            break

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = {}

        for _ai in range(accum):
            # Sample from motion pool (fast) or generate fresh (slow)
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
                recon, latent = model(x)

                # Align temporal
                T_out = recon.shape[1]
                T_in = x.shape[1]
                T_match = min(T_out, T_in)
                gt = x[:, T_in - T_match:]
                rc = recon[:, T_out - T_match:]

                # MSE loss
                mse = F.mse_loss(rc, gt)
                losses["mse"] = mse.item()

                total = args.w_mse * mse

                # Temporal consistency loss
                if T_match >= 2:
                    gt_diff = gt[:, 1:] - gt[:, :-1]
                    rc_diff = rc[:, 1:] - rc[:, :-1]
                    temp = F.l1_loss(rc_diff, gt_diff)
                    total = total + args.w_temporal * temp
                    losses["temp"] = temp.item()

                # LPIPS (per-frame, on RGB)
                if lpips_fn is not None:
                    BT = rc.shape[0] * T_match
                    rc_lp = rc.reshape(BT, 3, args.H, args.W) * 2 - 1
                    gt_lp = gt.reshape(BT, 3, args.H, args.W) * 2 - 1
                    lp_chunks = []
                    for ci in range(0, BT, 4):
                        lp_chunks.append(lpips_fn(rc_lp[ci:ci+4], gt_lp[ci:ci+4]))
                    lp = torch.cat(lp_chunks, 0).mean()
                    total = total + args.w_lpips * lp
                    losses["lpips"] = lp.item()

            if total.dim() > 0:
                total = total.mean()

            scaler.scale(total / accum).backward()
            del clips, x, recon, latent, rc, gt

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        global_step += 1

        # -- Log --
        if global_step % args.log_every == 0:
            el = time.time() - t0
            steps_run = global_step - start_step
            sps = steps_run / max(el, 1)
            eta = (args.total_steps - global_step) / max(sps, 1e-6)
            lr = opt.param_groups[0]["lr"]
            vm = torch.cuda.memory_allocated() / 1e9
            ls = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
            print(f"[{global_step}/{args.total_steps}] {ls} "
                  f"lr={lr:.1e} vram={vm:.1f}G "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        # -- Preview --
        if global_step % args.preview_every == 0:
            save_preview(model, gen, str(logdir), global_step,
                         device, amp_dtype, args.T)

        # -- Checkpoint --
        if global_step % args.save_every == 0:
            d = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
                "config": {
                    "latent_channels": args.latent_ch,
                    "image_channels": 3,
                    "output_channels": 3,
                    "encoder_channels": 64,
                    "decoder_channels": "256,128,64",
                    "temporal": True,
                    "T": args.T,
                    "synthyper_stage": 2,
                },
            }
            p = logdir / f"step_{global_step:06d}.pt"
            torch.save(d, p)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {p}", flush=True)

            ckpts = sorted(logdir.glob("step_*.pt"),
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if global_step > start_step:
        d = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "config": {
                "latent_channels": args.latent_ch,
                "image_channels": 3,
                "output_channels": 3,
                "encoder_channels": 64,
                "decoder_channels": "256,128,64",
                "temporal": True,
                "T": args.T,
                "synthyper_stage": 2,
            },
        }
        torch.save(d, logdir / f"step_{global_step:06d}.pt")
        torch.save(d, logdir / "latest.pt")
        print(f"  saved step {global_step}", flush=True)

    print(f"\nDone. {global_step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min", flush=True)


def main():
    p = argparse.ArgumentParser(description="VAEpp Stage 2 — temporal")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--latent-ch", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", default="2e-4")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--w-lpips", type=float, default=0.5)
    p.add_argument("--w-temporal", type=float, default=2.0)
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--gen-batch", type=int, default=0,
                   help="Generator batch size (0 = same as batch-size). "
                        "Lower to reduce generator VRAM.")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Gradient checkpointing (trade compute for memory)")
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    p.add_argument("--logdir", default="synthyper_video_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=100)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

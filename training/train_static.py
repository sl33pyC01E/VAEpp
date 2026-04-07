#!/usr/bin/env python3
"""VAEpp training — pretrain VAE on procedural images.

Stage 1: 2D static, 3ch RGB, single frame (T=1).
No DataLoader — generator produces data directly on GPU.

Usage:
    python -m training.train_static
    python -m training.train_static --total-steps 50000 --batch-size 4
"""

import argparse
import os
import pathlib
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MiniVAE
from core.generator import VAEppGenerator


# -- Preview -------------------------------------------------------------------

@torch.no_grad()
def save_preview(model, gen, logdir, step, device, amp_dtype):
    """Save GT | Recon grid as PNG."""
    try:
        model.eval()
        images = gen.generate(8)  # (8, 3, H, W)
        x = images.unsqueeze(1).to(device)  # (8, 1, 3, H, W)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon, _ = model(x)

        T_r = recon.shape[1]
        rc = recon[:, -1].clamp(0, 1).float().cpu().numpy()  # (8, 3, H, W)
        gt = images.cpu().numpy()  # (8, 3, H, W)

        del recon, x
        model.train()

        from PIL import Image
        H, W = gen.H, gen.W
        # 4x2 grid: 4 columns of GT|Recon pairs
        cols = 4
        rows = 2
        grid_w = cols * (W * 2 + 4) + (cols - 1) * 2
        grid_h = rows * H + (rows - 1) * 2

        grid = np.full((grid_h, grid_w, 3), 14, dtype=np.uint8)
        for i in range(min(8, len(gt))):
            r, c = i // cols, i % cols
            gy = r * (H + 2)
            gx = c * (W * 2 + 4 + 2)
            g_img = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            r_img = (rc[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            grid[gy:gy+H, gx:gx+W] = g_img
            grid[gy:gy+H, gx+W+2:gx+W*2+2] = r_img

        stepped = os.path.join(logdir, f"preview_{step:06d}.png")
        latest = os.path.join(logdir, "preview_latest.png")
        Image.fromarray(grid).save(stepped)
        Image.fromarray(grid).save(latest)
        print(f"  preview: {stepped}", flush=True)
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

    # -- Model --
    dec_ch = tuple(int(x) for x in args.dec_ch.split(","))
    model = MiniVAE(
        latent_channels=args.latent_ch,
        image_channels=args.image_ch,
        output_channels=args.image_ch,
        encoder_channels=args.enc_ch,
        decoder_channels=dec_ch,
        encoder_time_downscale=(False, False, False),
        decoder_time_upscale=(False, False, False),
    ).to(device)
    if args.grad_checkpoint:
        model.use_checkpoint = True
    pc = model.param_count()
    mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    print(f"MiniVAE ({args.image_ch}ch, no temporal): {pc['total']:,} params, "
          f"{mb:.1f}MB"
          f"{', grad-checkpoint' if args.grad_checkpoint else ''}")

    # -- Generator --
    gen = VAEppGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=args.bank_size,
        n_base_layers=args.n_layers,
        alpha=args.alpha,
    )
    # Use dynamic bank if bank dir exists with saved shapes
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
    print(f"Generator: bank={gen.bank_size}, layers={args.n_layers}, "
          f"alpha={args.alpha}, disco={gen.disco_quadrant}")

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
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            global_step = ckpt.get("global_step", 0)
            if not args.fresh_opt and ckpt.get("optimizer"):
                try:
                    opt.load_state_dict(ckpt["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Resumed from {args.resume} at step {global_step}")

    # -- Precision --
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    if global_step > 0 and not args.fresh_opt and ckpt is not None:
        if ckpt.get("scheduler"):
            sched.load_state_dict(ckpt["scheduler"])
        else:
            # Fallback for old checkpoints: rebuild scheduler at correct position
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01,
                last_epoch=global_step)
        if ckpt.get("scaler") and args.precision == "fp16":
            scaler.load_state_dict(ckpt["scaler"])

    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size
    accum = args.grad_accum

    print(f"Steps: {args.total_steps}, LR: {args.lr}, "
          f"Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}"
          f"{f', gen-batch={gen_bs}' if gen_bs != args.batch_size else ''}")
    print(f"Weights: mse={args.w_mse} lpips={args.w_lpips}")
    print(f"Precision: {args.precision}, Device: {device}")
    print(flush=True)

    # -- Initial preview --
    save_preview(model, gen, str(logdir), global_step, device, amp_dtype)

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
            # Generate batch on GPU — no DataLoader
            if gen_bs < args.batch_size:
                chunks = []
                rem = args.batch_size
                while rem > 0:
                    n = min(gen_bs, rem)
                    chunks.append(gen.generate(n))
                    rem -= n
                images = torch.cat(chunks)
            else:
                images = gen.generate(args.batch_size)  # (B, 3, H, W)
            x = images.unsqueeze(1)  # (B, 1, 3, H, W)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                recon, latent = model(x)

                T_out = recon.shape[1]
                gt = x[:, x.shape[1] - T_out:]
                rc = recon

                mse = F.mse_loss(rc, gt)
                total = args.w_mse * mse
                losses["mse"] = mse.item()

                if lpips_fn is not None:
                    BT = rc.shape[0] * T_out
                    rc_lp = rc[:, :, :3].reshape(BT, 3, args.H, args.W) * 2 - 1
                    gt_lp = gt[:, :, :3].reshape(BT, 3, args.H, args.W) * 2 - 1
                    lp = lpips_fn(rc_lp, gt_lp).mean()
                    total = total + args.w_lpips * lp
                    losses["lpips"] = lp.item()

            if total.dim() > 0:
                total = total.mean()

            scaler.scale(total / accum).backward()
            del recon, latent, rc, gt, images, x

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
                         device, amp_dtype)

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
                    "image_channels": args.image_ch,
                    "output_channels": args.image_ch,
                    "encoder_channels": args.enc_ch,
                    "decoder_channels": args.dec_ch,
                    "synthyper": True,
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
                "image_channels": args.image_ch,
                "output_channels": args.image_ch,
                "encoder_channels": args.enc_ch,
                "decoder_channels": args.dec_ch,
                "synthyper": True,
            },
        }
        torch.save(d, logdir / f"step_{global_step:06d}.pt")
        torch.save(d, logdir / "latest.pt")
        print(f"  saved step {global_step}", flush=True)

    print(f"\nDone. {global_step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min", flush=True)


def main():
    p = argparse.ArgumentParser(description="VAEpp training")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--image-ch", type=int, default=3,
                   help="Input/output channels (3=RGB, 9=RGB+depth+flow+semantic)")
    p.add_argument("--latent-ch", type=int, default=32)
    p.add_argument("--enc-ch", type=int, default=64,
                   help="Encoder channel width")
    p.add_argument("--dec-ch", default="256,128,64",
                   help="Decoder channel widths (comma-separated, 3 stages)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="2e-4")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--w-lpips", type=float, default=0.5)
    p.add_argument("--bank-size", type=int, default=500)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--gen-batch", type=int, default=0,
                   help="Generator batch size (0 = same as batch-size). "
                        "Lower to reduce generator VRAM.")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Gradient checkpointing (trade compute for memory)")
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode (25%% pattern / 25%% collage / "
                        "25%% dense random / 25%% structured)")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    p.add_argument("--logdir", default="synthyper_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=100)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

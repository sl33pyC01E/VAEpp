#!/usr/bin/env python3
"""Fusion training — CPU VAE preprocessor + MiniVAE.

Frozen CPU VAE encodes images to a multi-channel latent grid,
MiniVAE trains on that latent as if it were a multi-channel image.

Usage:
    python -m training.train_fusion \
        --cpu-vae-ckpt cpu_vae_logs/latest.pt \
        --dec-ch 256,128,64 --latent-ch 4
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MiniVAE
from core.generator import VAEpp0rGenerator
from experiments.cpu_vae import _load_pipeline


# -- Preview -------------------------------------------------------------------

@torch.no_grad()
def save_preview(mini_vae, cpu_encode, cpu_decode, gen, logdir, step,
                 device, amp_dtype, preview_image=None):
    """Save GT | Full pipeline reconstruction preview."""
    try:
        mini_vae.eval()
        from PIL import Image

        H, W = gen.H, gen.W
        sections = []

        # -- Reference image --
        if preview_image and os.path.exists(preview_image):
            ref_pil = Image.open(preview_image).convert("RGB")
            ref_pil = ref_pil.resize((W, H), Image.BILINEAR)
            ref_arr = np.array(ref_pil, dtype=np.float32) / 255.0
            ref_t = torch.from_numpy(ref_arr).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.amp.autocast(device.type, dtype=amp_dtype):
                cpu_lat = normalize_latent(cpu_encode(ref_t))
                cpu_lat_5d = cpu_lat.unsqueeze(1)  # (B, 1, C, H, W)
                recon_lat, _ = mini_vae(cpu_lat_5d)
                recon_lat = recon_lat[:, -1]  # (B, C, H, W)
                # Crop to match
                recon_lat = recon_lat[:, :, :cpu_lat.shape[2], :cpu_lat.shape[3]]
                recon_rgb = cpu_decode(denormalize_latent(recon_lat))

            ref_gt = (ref_arr * 255).clip(0, 255).astype(np.uint8)
            ref_rc = recon_rgb[0, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
            ref_rc = (ref_rc.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            sep = np.full((H, 4, 3), 14, dtype=np.uint8)
            sections.append(np.concatenate([ref_gt, sep, ref_rc], axis=1))
            del ref_t, cpu_lat, recon_lat, recon_rgb

        # -- Synthetic strip --
        images = gen.generate(8)
        x = images.to(device)

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            cpu_lat = cpu_encode(x)
            cpu_lat_5d = cpu_lat.unsqueeze(1)
            recon_lat, _ = mini_vae(cpu_lat_5d)
            recon_lat = recon_lat[:, -1]
            recon_lat = recon_lat[:, :, :cpu_lat.shape[2], :cpu_lat.shape[3]]
            recon_rgb = cpu_decode(denormalize_latent(recon_lat))

        gt = images.cpu().numpy()
        rc = recon_rgb[:, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
        del recon_lat, recon_rgb, cpu_lat

        cols, rows = 4, 2
        sep_w = 4
        grid_w = cols * (W * 2 + sep_w) + (cols - 1) * 2
        grid_h = rows * H + (rows - 1) * 2
        synth_grid = np.full((grid_h, grid_w, 3), 14, dtype=np.uint8)
        for i in range(min(8, len(gt))):
            r, c = i // cols, i % cols
            gy = r * (H + 2)
            gx = c * (W * 2 + sep_w + 2)
            g_img = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            r_img = (rc[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            synth_grid[gy:gy+H, gx:gx+W] = g_img
            synth_grid[gy:gy+H, gx+W+2:gx+W*2+2] = r_img
        sections.append(synth_grid)

        # Combine
        if len(sections) > 1:
            from PIL import Image as _PILImg
            syn_w = sections[1].shape[1]
            ref_pil2 = _PILImg.fromarray(sections[0])
            scale = syn_w / sections[0].shape[1]
            ref_pil2 = ref_pil2.resize((syn_w, int(sections[0].shape[0] * scale)),
                                        _PILImg.BILINEAR)
            sections[0] = np.array(ref_pil2)
            gap = np.full((6, syn_w, 3), 14, dtype=np.uint8)
            grid = np.concatenate([sections[0], gap, sections[1]], axis=0)
        else:
            grid = sections[0]

        mini_vae.train()
        stepped = os.path.join(logdir, f"preview_{step:06d}.png")
        latest = os.path.join(logdir, "preview_latest.png")
        Image.fromarray(grid).save(stepped)
        Image.fromarray(grid).save(latest)
        print(f"  preview: {stepped}", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        mini_vae.train()


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

    # -- Load frozen CPU VAE --
    print(f"Loading CPU VAE from {args.cpu_vae_ckpt}...")
    models_list, cpu_encode, cpu_decode, spatial_sizes, cpu_ckpt = \
        _load_pipeline(args.cpu_vae_ckpt, device)

    # Freeze CPU VAE
    for mt, m, c in models_list:
        m.eval()
        m.requires_grad_(False)

    # Determine CPU VAE output shape
    cpu_out_H, cpu_out_W = spatial_sizes[-1]
    last_mt, last_m, last_c = models_list[-1]
    if last_mt in ("unrolled", "patch"):
        cpu_out_ch = last_m.latent_channels
    else:
        cpu_out_ch = last_c.get("latent_channels", 3)

    print(f"  CPU VAE output: ({cpu_out_ch}, {cpu_out_H}, {cpu_out_W})")
    print(f"  Pipeline: {len(models_list)} stages, "
          f"{sum(sum(p.numel() for p in m.parameters()) for _, m, _ in models_list):,} params (frozen)")

    # -- Probe CPU VAE output stats for normalization --
    print("  Probing latent distribution...")
    _probe_gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=500, n_base_layers=64,
    )
    _probe_gen.build_banks()
    with torch.no_grad():
        _probe_imgs = _probe_gen.generate(16)
        _probe_lat = cpu_encode(_probe_imgs)
        lat_mean = _probe_lat.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        lat_std = _probe_lat.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    del _probe_gen, _probe_imgs, _probe_lat
    print(f"  Latent stats per channel:")
    for c in range(cpu_out_ch):
        print(f"    ch{c}: mean={lat_mean[0,c,0,0]:.3f}, std={lat_std[0,c,0,0]:.3f}")

    def normalize_latent(lat):
        return (lat - lat_mean) / lat_std

    def denormalize_latent(lat):
        return lat * lat_std + lat_mean

    # -- MiniVAE --
    enc_ch_str = args.enc_ch
    dec_ch = tuple(int(x) for x in args.dec_ch.split(","))
    latent_ch = args.latent_ch

    if isinstance(enc_ch_str, str) and "," in enc_ch_str:
        enc_ch = tuple(int(x) for x in enc_ch_str.split(","))
    else:
        enc_ch = int(enc_ch_str)

    n_stages = len(dec_ch)

    if args.resume:
        _peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        _cfg = _peek.get("config", {})
        enc_ch_str = _cfg.get("encoder_channels", enc_ch_str)
        latent_ch = _cfg.get("latent_channels", latent_ch)
        dec_ch_str = _cfg.get("decoder_channels", args.dec_ch)
        if isinstance(dec_ch_str, str):
            dec_ch = tuple(int(x) for x in dec_ch_str.split(","))
        elif isinstance(dec_ch_str, (list, tuple)):
            dec_ch = tuple(dec_ch_str)
        if isinstance(enc_ch_str, str) and "," in enc_ch_str:
            enc_ch = tuple(int(x) for x in enc_ch_str.split(","))
        elif isinstance(enc_ch_str, (list, tuple)):
            enc_ch = tuple(enc_ch_str)
        else:
            enc_ch = int(enc_ch_str)
        n_stages = len(dec_ch)
        del _peek

    model = MiniVAE(
        latent_channels=latent_ch,
        image_channels=cpu_out_ch,
        output_channels=cpu_out_ch,
        encoder_channels=enc_ch,
        decoder_channels=dec_ch,
        encoder_time_downscale=tuple([False] * n_stages),
        decoder_time_upscale=tuple([False] * n_stages),
    ).to(device)

    if args.grad_checkpoint:
        model.use_checkpoint = True

    pc = model.param_count()
    spatial_comp = 2 ** n_stages
    final_H = cpu_out_H // spatial_comp
    final_W = cpu_out_W // spatial_comp
    total_spatial = (args.H * args.W) // (final_H * final_W)
    final_dims = latent_ch * final_H * final_W
    total_comp = (args.H * args.W * 3) / final_dims

    print(f"MiniVAE: {cpu_out_ch}ch input, {latent_ch}ch latent, "
          f"{n_stages}-stage ({spatial_comp}x spatial)")
    print(f"  enc={enc_ch}, dec={dec_ch}")
    print(f"  {pc['total']:,} params")
    print(f"  Latent: ({latent_ch}, {final_H}, {final_W}) = {final_dims} dims")
    print(f"  Total compression: {args.H}x{args.W}x3 -> {final_dims} dims "
          f"= {total_comp:.0f}:1")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=args.bank_size, n_base_layers=args.n_layers,
    )
    bank_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "bank")
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
          f"disco={getattr(gen, 'disco_quadrant', False)}")

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
            print("WARNING: pip install lpips for perceptual loss")

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
            model.load_state_dict(ckpt["model"], strict=False)
            global_step = ckpt.get("global_step", 0)
            if not args.fresh_opt and ckpt.get("optimizer"):
                try:
                    opt.load_state_dict(ckpt["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
            if ckpt.get("scheduler") and not args.fresh_opt:
                sched.load_state_dict(ckpt["scheduler"])
            else:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01,
                    last_epoch=global_step)
        print(f"Resumed from {args.resume} at step {global_step}")

    if args.fresh_opt and global_step > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps - global_step,
            eta_min=float(args.lr) * 0.01)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    if global_step > 0 and not args.fresh_opt:
        if args.resume:
            ckpt2 = torch.load(args.resume, map_location="cpu", weights_only=False)
            if ckpt2.get("scaler") and args.precision == "fp16":
                scaler.load_state_dict(ckpt2["scaler"])
            del ckpt2

    accum = args.grad_accum
    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}"
          f"{f', accum={accum}' if accum > 1 else ''}")
    print(f"Weights: l1={args.w_l1} mse={args.w_mse} lpips={args.w_lpips} "
          f"pixel={args.w_pixel}")
    print(f"Precision: {args.precision}, Device: {device}")
    print(flush=True)

    def _ckpt_name(step):
        ec_str = ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else str(enc_ch)
        dc_str = ",".join(str(x) for x in dec_ch)
        steps_k = f"{step // 1000}k" if step >= 1000 else str(step)
        return f"fusion-{cpu_out_ch}ch-lc{latent_ch}-ec{ec_str}-dc{dc_str}-{spatial_comp}x-{steps_k}.pt"

    def _make_checkpoint():
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "config": {
                "latent_channels": latent_ch,
                "image_channels": cpu_out_ch,
                "output_channels": cpu_out_ch,
                "encoder_channels": ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else enc_ch,
                "decoder_channels": ",".join(str(x) for x in dec_ch),
                "cpu_vae_ckpt": args.cpu_vae_ckpt,
                "fusion": True,
            },
            "lat_mean": lat_mean.cpu(),
            "lat_std": lat_std.cpu(),
        }

    # -- Wrap encode/decode with normalization for preview --
    def norm_cpu_encode(x):
        return normalize_latent(cpu_encode(x))

    def norm_cpu_decode(x):
        return cpu_decode(denormalize_latent(x))

    # -- Initial preview --
    preview_image = getattr(args, 'preview_image', None)
    save_preview(model, norm_cpu_encode, norm_cpu_decode, gen, str(logdir),
                 global_step, device, amp_dtype, preview_image=preview_image)

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
            # Generate batch on GPU
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

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                # Encode through frozen CPU VAE
                with torch.no_grad():
                    cpu_lat = normalize_latent(cpu_encode(images))  # normalized

                # MiniVAE forward (static: T=1)
                cpu_lat_5d = cpu_lat.unsqueeze(1)  # (B, 1, C, H, W)
                recon_5d, latent = model(cpu_lat_5d)
                recon = recon_5d[:, -1]  # (B, C, H, W)

                # Crop to match (model may pad)
                recon = recon[:, :, :cpu_lat.shape[2], :cpu_lat.shape[3]]

                # Latent-space loss (MiniVAE recon vs CPU VAE output)
                total = torch.tensor(0.0, device=device)
                if args.w_l1 > 0:
                    l1 = F.l1_loss(recon, cpu_lat)
                    total = total + args.w_l1 * l1
                    losses["l1"] = losses.get("l1", 0) + l1.item() / accum
                if args.w_mse > 0:
                    mse = F.mse_loss(recon, cpu_lat)
                    total = total + args.w_mse * mse
                    losses["mse"] = losses.get("mse", 0) + mse.item() / accum

                # Pixel loss (full pipeline decode vs original image)
                if args.w_pixel > 0:
                    with torch.no_grad():
                        pass  # images already on device
                    recon_rgb = cpu_decode(denormalize_latent(recon))
                    recon_rgb = recon_rgb[:, :3, :args.H, :args.W]
                    pixel_loss = F.l1_loss(recon_rgb, images[:, :3])
                    total = total + args.w_pixel * pixel_loss
                    losses["pix"] = losses.get("pix", 0) + pixel_loss.item() / accum

                # LPIPS (on RGB pixels)
                if lpips_fn is not None:
                    if args.w_pixel <= 0:
                        recon_rgb = cpu_decode(denormalize_latent(recon))
                        recon_rgb = recon_rgb[:, :3, :args.H, :args.W]
                    rc_lp = recon_rgb * 2 - 1
                    gt_lp = images[:, :3] * 2 - 1
                    lp = lpips_fn(rc_lp, gt_lp).mean()
                    total = total + args.w_lpips * lp
                    losses["lpips"] = losses.get("lpips", 0) + lp.item() / accum

            if total.dim() > 0:
                total = total.mean()

            scaler.scale(total / accum).backward()
            del recon, latent, cpu_lat, images
            if 'recon_rgb' in dir():
                del recon_rgb

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
            save_preview(model, norm_cpu_encode, norm_cpu_decode, gen, str(logdir),
                         global_step, device, amp_dtype,
                         preview_image=preview_image)

        # -- Checkpoint --
        if global_step % args.save_every == 0:
            d = _make_checkpoint()
            named = _ckpt_name(global_step)
            torch.save(d, logdir / named)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {named}", flush=True)

            ckpts = sorted([f for f in logdir.glob("*.pt")
                           if f.name != "latest.pt"],
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if global_step > start_step:
        d = _make_checkpoint()
        named = _ckpt_name(global_step)
        torch.save(d, logdir / named)
        torch.save(d, logdir / "latest.pt")
        print(f"  saved {named}", flush=True)

    print(f"\nDone. {global_step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min", flush=True)


def main():
    p = argparse.ArgumentParser(description="Fusion training: CPU VAE + MiniVAE")
    p.add_argument("--cpu-vae-ckpt", required=True,
                   help="Path to trained CPU VAE pipeline checkpoint")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--latent-ch", type=int, default=4)
    p.add_argument("--enc-ch", default="64",
                   help="MiniVAE encoder channels (int or comma-separated)")
    p.add_argument("--dec-ch", default="256,128,64",
                   help="MiniVAE decoder channels (comma-separated)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="2e-4")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--w-l1", type=float, default=1.0,
                   help="L1 loss on MiniVAE latent reconstruction")
    p.add_argument("--w-mse", type=float, default=0.0)
    p.add_argument("--w-lpips", type=float, default=0.0,
                   help="LPIPS on full pipeline pixel output")
    p.add_argument("--w-pixel", type=float, default=0.5,
                   help="L1 pixel loss through full decode pipeline")
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--gen-batch", type=int, default=0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--disco", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    p.add_argument("--logdir", default="fusion_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--preview-image", default=None,
                   help="Reference image for tracking progress")


    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

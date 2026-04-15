#!/usr/bin/env python3
"""VAEpp0r training — pretrain VAE on procedural images.

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
from core.generator import VAEpp0rGenerator
from core.discriminator import PatchDiscriminator, hinge_d_loss, hinge_g_loss


# -- Haar wavelet (lossless 2x down/up) ----------------------------------------

def haar_down(x):
    """2x spatial downscale: (B, C, H, W) -> (B, 4C, H/2, W/2). Lossless."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)

def haar_up(x):
    """2x spatial upscale: (B, 4C, H, W) -> (B, C, H*2, W*2). Lossless inverse."""
    C = x.shape[1] // 4
    ll, lh, hl, hh = x[:, 0*C:1*C], x[:, 1*C:2*C], x[:, 2*C:3*C], x[:, 3*C:4*C]
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    B, Ch, H, W = a.shape
    out = torch.zeros(B, Ch, H * 2, W * 2, device=x.device, dtype=x.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out


def haar_down_n(x, n):
    """Apply haar_down n times. 0=identity, 1=2x, 2=4x."""
    for _ in range(n):
        x = haar_down(x)
    return x


def haar_up_n(x, n):
    """Apply haar_up n times. 0=identity, 1=2x, 2=4x."""
    for _ in range(n):
        x = haar_up(x)
    return x


# -- Preview -------------------------------------------------------------------

def _load_preview_image(path, H, W, device):
    """Load a real image for preview target."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)


@torch.no_grad()
def save_preview(model, gen, logdir, step, device, amp_dtype,
                 preview_image=None, haar_rounds=0):
    """Save GT | Recon grid as PNG with optional reference image."""
    try:
        model.eval()
        from PIL import Image

        H, W = gen.H, gen.W
        sections = []

        # -- Reference image (large, top) --
        if preview_image and os.path.exists(preview_image):
            ref = _load_preview_image(preview_image, H, W, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                if haar_rounds > 0:
                    ref_haar = haar_down_n(ref[:, 0], haar_rounds)
                    ref_haar_5d = ref_haar.unsqueeze(1)
                    ref_recon, _ = model(ref_haar_5d)
                    hH, hW = H // (2 ** haar_rounds), W // (2 ** haar_rounds)
                    ref_recon_haar = ref_recon[:, -1, :, :hH, :hW]
                    ref_recon_rgb = haar_up_n(ref_recon_haar, haar_rounds)
                else:
                    ref_recon, _ = model(ref)
                    ref_recon_rgb = ref_recon[:, -1:].squeeze(1)
            ref_gt = ref[0, 0, :3].cpu().numpy().transpose(1, 2, 0)
            ref_rc = ref_recon_rgb[0, :3, :H, :W].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            ref_gt = (ref_gt * 255).clip(0, 255).astype(np.uint8)
            ref_rc = (ref_rc * 255).clip(0, 255).astype(np.uint8)
            sep_v = np.full((H, 4, 3), 14, dtype=np.uint8)
            sections.append(np.concatenate([ref_gt, sep_v, ref_rc], axis=1))
            del ref, ref_recon

        # -- Synthetic strip --
        images = gen.generate(8)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            if haar_rounds > 0:
                x_haar = haar_down_n(images, haar_rounds).unsqueeze(1)
                recon, _ = model(x_haar)
                hH, hW = H // (2 ** haar_rounds), W // (2 ** haar_rounds)
                recon_haar = recon[:, -1, :, :hH, :hW]
                recon_rgb = haar_up_n(recon_haar, haar_rounds)
            else:
                x = images.unsqueeze(1).to(device)
                recon, _ = model(x)
                recon_rgb = recon[:, -1]

        rc = recon_rgb[:, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
        gt = images.cpu().numpy()

        del recon

        cols, rows = 4, 2
        sep = 4
        grid_w = cols * (W * 2 + sep) + (cols - 1) * 2
        grid_h = rows * H + (rows - 1) * 2

        synth_grid = np.full((grid_h, grid_w, 3), 14, dtype=np.uint8)
        for i in range(min(8, len(gt))):
            r, c = i // cols, i % cols
            gy = r * (H + 2)
            gx = c * (W * 2 + sep + 2)
            g_img = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            r_img = (rc[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            synth_grid[gy:gy+H, gx:gx+W] = g_img
            synth_grid[gy:gy+H, gx+W+2:gx+W*2+2] = r_img
        sections.append(synth_grid)

        # -- Combine --
        if len(sections) > 1:
            from PIL import Image as _PILImg
            syn_w = sections[1].shape[1]
            ref_pil = _PILImg.fromarray(sections[0])
            scale = syn_w / sections[0].shape[1]
            ref_pil = ref_pil.resize((syn_w, int(sections[0].shape[0] * scale)),
                                      _PILImg.BILINEAR)
            sections[0] = np.array(ref_pil)
            gap = np.full((6, syn_w, 3), 14, dtype=np.uint8)
            grid = np.concatenate([sections[0], gap, sections[1]], axis=0)
        else:
            grid = sections[0]

        model.train()

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

    # -- Model (read arch from checkpoint if resuming) --
    enc_ch_str = args.enc_ch
    dec_ch = tuple(int(x) for x in args.dec_ch.split(","))
    latent_ch = args.latent_ch
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
        del _peek
    # Parse encoder channels: int or comma-separated
    if isinstance(enc_ch_str, int):
        enc_ch = enc_ch_str
    elif isinstance(enc_ch_str, str) and "," in enc_ch_str:
        enc_ch = tuple(int(x) for x in enc_ch_str.split(","))
    elif isinstance(enc_ch_str, (list, tuple)):
        enc_ch = tuple(enc_ch_str)
    else:
        enc_ch = int(enc_ch_str)
    n_stages = len(dec_ch)
    haar_mode = getattr(args, 'haar', 'none')
    # Backward compat: old checkpoints stored haar as bool
    if haar_mode is True:
        haar_mode = '2x'
    elif haar_mode is False or haar_mode is None:
        haar_mode = 'none'
    haar_rounds = {'none': 0, '2x': 1, '4x': 2}[haar_mode]
    haar_ch_mult = 4 ** haar_rounds  # 1, 4, or 16
    haar_spatial_mult = 2 ** haar_rounds  # 1, 2, or 4
    if haar_rounds > 0:
        vae_in_ch = args.image_ch * haar_ch_mult
        print(f"Haar {haar_mode}: {args.image_ch}ch -> {vae_in_ch}ch ({haar_spatial_mult}x spatial pre-compression)")
    else:
        vae_in_ch = args.image_ch
    model = MiniVAE(
        latent_channels=latent_ch,
        image_channels=vae_in_ch,
        output_channels=vae_in_ch,
        encoder_channels=enc_ch,
        decoder_channels=dec_ch,
        encoder_time_downscale=tuple([False] * n_stages),
        decoder_time_upscale=tuple([False] * n_stages),
        residual_shortcut=getattr(args, 'residual_shortcut', False),
        use_attention=getattr(args, 'use_attention', False),
        use_groupnorm=getattr(args, 'use_groupnorm', False),
    ).to(device)
    if args.grad_checkpoint:
        model.use_checkpoint = True
    pc = model.param_count()
    mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    spatial = 2 ** n_stages * haar_spatial_mult
    print(f"MiniVAE ({vae_in_ch}ch in, {latent_ch}ch latent, "
          f"{spatial}x spatial): {pc['total']:,} params, {mb:.1f}MB"
          f"{', grad-checkpoint' if args.grad_checkpoint else ''}")

    # -- Generator --
    gen = VAEpp0rGenerator(
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

    # -- PatchGAN discriminator --
    disc = None
    opt_disc = None
    if args.w_gan > 0:
        disc = PatchDiscriminator(in_ch=3, nf=args.disc_nf).to(device)
        _disc_lr_str = args.disc_lr.strip() if args.disc_lr else "none"
        _disc_lr = float(args.lr) * 0.5 if _disc_lr_str in ("none", "") else float(_disc_lr_str)
        opt_disc = torch.optim.Adam(disc.parameters(), lr=_disc_lr,
                                    betas=(0.5, 0.999))
        pc_d = sum(p.numel() for p in disc.parameters())
        print(f"PatchGAN: nf={args.disc_nf}, {pc_d:,} params, "
              f"w_gan={args.w_gan}, start_step={args.gan_start}, "
              f"warmup={args.gan_warmup}, disc_lr={_disc_lr:.2e}")

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
            if disc is not None and ckpt.get("discriminator"):
                try:
                    disc.load_state_dict(ckpt["discriminator"])
                    if not args.fresh_opt and ckpt.get("optimizer_disc"):
                        opt_disc.load_state_dict(ckpt["optimizer_disc"])
                except Exception:
                    print("  Fresh discriminator (mismatch)")
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
    print(f"Weights: l1={args.w_l1} mse={args.w_mse} lpips={args.w_lpips} gan={args.w_gan}")
    print(f"Precision: {args.precision}, Device: {device}")
    print(flush=True)

    def _ckpt_name(step):
        ec_str = ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else str(enc_ch)
        dc_str = ",".join(str(x) for x in dec_ch)
        haar_str = f"-haar{haar_mode}" if haar_rounds > 0 else ""
        steps_k = f"{step // 1000}k" if step >= 1000 else str(step)
        return f"vae-{args.image_ch}ch-lc{latent_ch}-ec{ec_str}-dc{dc_str}-{spatial}x{haar_str}-{steps_k}.pt"

    # Glob pattern scoped to THIS run only — never touch other runs' checkpoints
    _ec_str = ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else str(enc_ch)
    _dc_str = ",".join(str(x) for x in dec_ch)
    _haar_str = f"-haar{haar_mode}" if haar_rounds > 0 else ""
    _run_glob = f"vae-{args.image_ch}ch-lc{latent_ch}-ec{_ec_str}-dc{_dc_str}-{spatial}x{_haar_str}-*.pt"

    def _make_checkpoint():
        d = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "config": {
                "latent_channels": latent_ch,
                "image_channels": args.image_ch,
                "output_channels": args.image_ch,
                "encoder_channels": ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else enc_ch,
                "decoder_channels": ",".join(str(x) for x in dec_ch),
                "haar": haar_mode,
                "residual_shortcut": getattr(args, 'residual_shortcut', False),
                "use_attention": getattr(args, 'use_attention', False),
                "use_groupnorm": getattr(args, 'use_groupnorm', False),
                "synthyper": True,
            },
        }
        if disc is not None:
            d["discriminator"] = disc.state_dict()
            d["optimizer_disc"] = opt_disc.state_dict()
        return d

    # -- Initial preview --
    preview_image = getattr(args, 'preview_image', None)
    save_preview(model, gen, str(logdir), global_step, device, amp_dtype,
                 preview_image=preview_image, haar_rounds=haar_rounds)

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
        if disc is not None:
            disc.train()
        opt.zero_grad(set_to_none=True)
        losses = {}

        _d_real = None
        _d_fake = None

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
            if haar_rounds > 0:
                haar_input = haar_down_n(images, haar_rounds)
                x = haar_input.unsqueeze(1)
            else:
                x = images.unsqueeze(1)  # (B, 1, 3, H, W)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                recon, latent = model(x)

                T_out = recon.shape[1]
                T_in = x.shape[1]
                T_match = min(T_out, T_in)
                gt = x[:, T_in - T_match:]
                rc = recon[:, T_out - T_match:]
                # Crop to input spatial size (model may pad for divisibility)
                if haar_rounds > 0:
                    hH = args.H // (2 ** haar_rounds)
                    hW = args.W // (2 ** haar_rounds)
                    rc = rc[:, :, :, :hH, :hW]
                else:
                    rc = rc[:, :, :, :args.H, :args.W]

                total = torch.tensor(0.0, device=device)
                if args.w_l1 > 0:
                    l1 = F.l1_loss(rc, gt)
                    total = total + args.w_l1 * l1
                    losses["l1"] = losses.get("l1", 0) + l1.item() / accum
                if args.w_mse > 0:
                    mse = F.mse_loss(rc, gt)
                    total = total + args.w_mse * mse
                    losses["mse"] = losses.get("mse", 0) + mse.item() / accum

                if lpips_fn is not None:
                    BT = rc.shape[0] * T_match
                    lpH = rc.shape[3]
                    lpW = rc.shape[4]
                    rc_lp = rc[:, :, :3].reshape(BT, 3, lpH, lpW) * 2 - 1
                    gt_lp = gt[:, :, :3].reshape(BT, 3, lpH, lpW) * 2 - 1
                    lp = lpips_fn(rc_lp, gt_lp).mean()
                    total = total + args.w_lpips * lp
                    losses["lpips"] = losses.get("lpips", 0) + lp.item() / accum

                # GAN generator loss
                if disc is not None and global_step >= args.gan_start:
                    _steps_since_start = global_step - args.gan_start
                    if args.gan_warmup > 0:
                        _eff_w_gan = args.w_gan * min(1.0, _steps_since_start / args.gan_warmup)
                    else:
                        _eff_w_gan = args.w_gan
                    if _eff_w_gan > 0:
                        if haar_rounds > 0:
                            rc_2d = rc[:, 0]  # (B, vae_ch, hH, hW)
                            fake_rgb = haar_up_n(rc_2d, haar_rounds)[:, :3, :args.H, :args.W]
                        else:
                            fake_rgb = rc[:, 0, :3, :args.H, :args.W]
                        g_loss = hinge_g_loss(disc(fake_rgb * 2 - 1))
                        total = total + _eff_w_gan * g_loss
                        losses["g"] = losses.get("g", 0) + g_loss.item() / accum
                        # Save detached copies for D update (last accum iter wins)
                        _d_real = images[:, :3].detach()
                        _d_fake = fake_rgb.detach()

            if total.dim() > 0:
                total = total.mean()

            scaler.scale(total / accum).backward()
            del recon, latent, rc, gt, images, x

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        # D update (once per step, after G)
        if disc is not None and global_step >= args.gan_start and _d_real is not None:
            opt_disc.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                d_loss = hinge_d_loss(disc(_d_real * 2 - 1), disc(_d_fake * 2 - 1))
            d_loss.backward()
            opt_disc.step()
            losses["d"] = d_loss.item()

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
                         device, amp_dtype, preview_image=preview_image,
                         haar_rounds=haar_rounds)

        # -- Checkpoint --
        if global_step % args.save_every == 0:
            d = _make_checkpoint()
            named = _ckpt_name(global_step)
            torch.save(d, logdir / named)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {named}", flush=True)

            ckpts = sorted([f for f in logdir.glob(_run_glob)
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
    p = argparse.ArgumentParser(description="VAEpp0r training")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--image-ch", type=int, default=3,
                   help="Input/output channels (3=RGB, 9=RGB+depth+flow+semantic)")
    p.add_argument("--latent-ch", type=int, default=32)
    p.add_argument("--enc-ch", default="64",
                   help="Encoder channel width (int or comma-separated per stage)")
    p.add_argument("--dec-ch", default="256,128,64",
                   help="Decoder channel widths (comma-separated, 3 stages)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="2e-4")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--w-l1", type=float, default=1.0)
    p.add_argument("--w-mse", type=float, default=0.0)
    p.add_argument("--w-lpips", type=float, default=0.5)
    p.add_argument("--w-gan", type=float, default=0.1,
                   help="PatchGAN adversarial loss weight (0=disabled)")
    p.add_argument("--gan-start", type=int, default=1000,
                   help="Step to start GAN training (let recon stabilize first)")
    p.add_argument("--gan-warmup", type=int, default=2000,
                   help="Steps to linearly ramp gan weight from 0 to w_gan after gan_start (0=hard switch)")
    p.add_argument("--disc-lr", type=str, default="none",
                   help="Discriminator Adam LR. 'none' or empty = args.lr * 0.5")
    p.add_argument("--disc-nf", type=int, default=64,
                   help="PatchGAN base channel count")
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
    p.add_argument("--haar", default="none", choices=["none", "2x", "4x"],
                   help="Haar wavelet pre-compression (none/2x/4x). "
                        "2x: 3ch->12ch at half res. 4x: 3ch->48ch at quarter res.")
    p.add_argument("--residual-shortcut", action="store_true",
                   help="DC-AE style residual shortcuts (pixel_unshuffle/shuffle bypasses)")
    p.add_argument("--use-attention", action="store_true",
                   help="Add linear attention at deepest encoder/decoder stage")
    p.add_argument("--use-groupnorm", action="store_true",
                   help="Add GroupNorm inside MemBlock conv layers")
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode (25%% pattern / 25%% collage / "
                        "25%% dense random / 25%% structured)")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    p.add_argument("--logdir", default="synthyper_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--preview-image", default=None,
                   help="Path to reference image for tracking progress")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

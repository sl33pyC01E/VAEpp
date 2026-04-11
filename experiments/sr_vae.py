#!/usr/bin/env python3
"""Super-Resolution VAE — naive downscale + learned upscale.

Pairs a downscale method (lanczos, area, bilinear, or learned) with
a lightweight upscale network (ESPCN-style) trained end-to-end.

The "latent" is a small RGB thumbnail. Human-readable, 3 channels.

Usage:
    python -m experiments.sr_vae --scale 8 --upscaler espcn
    python -m experiments.sr_vae_gui
"""

import argparse
import math
import os
import pathlib
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generator import VAEpp0rGenerator


# =============================================================================
# Downscalers
# =============================================================================

class FixedDownscaler(nn.Module):
    """Non-trainable downscaler using interpolation."""
    def __init__(self, scale, mode="area"):
        super().__init__()
        self.scale = scale
        self.mode = mode  # area, bilinear, bicubic, lanczos

    def forward(self, x):
        if self.mode == "lanczos":
            # PIL lanczos — process per image on CPU, return to device
            from PIL import Image as PILImg
            device = x.device
            B, C, H, W = x.shape
            tH, tW = H // self.scale, W // self.scale
            out = []
            for i in range(B):
                img = x[i].float().cpu().clamp(0, 1)
                # CHW -> HWC -> PIL
                pil = PILImg.fromarray(
                    (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                pil = pil.resize((tW, tH), PILImg.LANCZOS)
                arr = torch.from_numpy(
                    np.array(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
                out.append(arr)
            return torch.stack(out).to(device)
        return F.interpolate(x, scale_factor=1.0/self.scale, mode=self.mode)

    def param_count(self):
        return 0


class LearnedDownscaler(nn.Module):
    """Lightweight trainable downscaler using strided convolutions.

    Uses depthwise separable convs (ollin's suggestion) for efficiency.
    """
    def __init__(self, scale, channels=3, hidden=32):
        super().__init__()
        self.scale = scale
        n_down = int(math.log2(scale))
        assert 2 ** n_down == scale, f"scale must be power of 2, got {scale}"

        layers = []
        in_ch = channels
        for i in range(n_down):
            out_ch = hidden if i < n_down - 1 else channels
            # Depthwise separable: depthwise conv + pointwise conv
            layers.extend([
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
                nn.GELU() if i < n_down - 1 else nn.Identity(),
            ])
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Upscalers
# =============================================================================

class ESPCNUpscaler(nn.Module):
    """ESPCN-style upscaler using sub-pixel convolution (PixelShuffle).

    Operates entirely in LR space, then PixelShuffle at the end.
    Very efficient — all conv layers process the small image.

    Args:
        scale: upscale factor (must be power of 2)
        channels: output channels (3 for RGB)
        hidden: internal feature channels
        n_blocks: number of residual blocks
    """
    def __init__(self, scale, channels=3, hidden=64, n_blocks=4):
        super().__init__()
        self.scale = scale
        n_up = int(math.log2(scale))

        # Feature extraction in LR space
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
        )

        # Residual blocks in LR space
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
            ))
        self.blocks = nn.ModuleList(blocks)

        # Upscale stages: conv + PixelShuffle per 2x
        upscale = []
        for i in range(n_up):
            out_ch = hidden if i < n_up - 1 else channels
            upscale.extend([
                nn.Conv2d(hidden, out_ch * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU() if i < n_up - 1 else nn.Identity(),
            ])
            hidden = out_ch
        self.upscale = nn.Sequential(*upscale)

    def forward(self, x):
        h = self.head(x)
        for block in self.blocks:
            h = h + block(h)
        return self.upscale(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class SimpleUpscaler(nn.Module):
    """Simple upscaler using bilinear upsample + conv refinement."""
    def __init__(self, scale, channels=3, hidden=64, n_blocks=4):
        super().__init__()
        self.scale = scale

        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.tail = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                              align_corners=False)
        h = self.head(x_up)
        for block in self.blocks:
            h = h + block(h)
        return self.tail(h) + x_up

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# SR VAE (downscaler + upscaler as a single module)
# =============================================================================

class SRVAE(nn.Module):
    """Super-Resolution VAE: downscale encoder + upscale decoder.

    The latent is a small RGB thumbnail.
    """
    def __init__(self, downscaler, upscaler, scale):
        super().__init__()
        self.downscaler = downscaler
        self.upscaler = upscaler
        self.scale = scale

    def encode(self, x):
        return self.downscaler(x)

    def decode(self, z):
        return self.upscaler(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        # Crop to input size
        recon = recon[:, :, :x.shape[2], :x.shape[3]]
        return recon, z

    def param_count(self):
        down_p = self.downscaler.param_count()
        up_p = self.upscaler.param_count()
        return {"downscaler": down_p, "upscaler": up_p,
                "total": down_p + up_p}


# =============================================================================
# Preview
# =============================================================================

@torch.no_grad()
def save_preview(model, gen, logdir, step, device, amp_dtype,
                 preview_image=None):
    """Save GT | Recon preview with optional reference image."""
    try:
        model.eval()
        from PIL import Image

        H, W = gen.H, gen.W
        sections = []

        # Reference image
        if preview_image and os.path.exists(preview_image):
            pil = Image.open(preview_image).convert("RGB")
            pil = pil.resize((W, H), Image.BILINEAR)
            arr = np.array(pil, dtype=np.float32) / 255.0
            ref = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                recon, z = model(ref)
            ref_gt = (arr * 255).clip(0, 255).astype(np.uint8)
            ref_rc = recon[0, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
            ref_rc = (ref_rc.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            # Also show the thumbnail
            z_up = F.interpolate(z, size=(H, W), mode="nearest")
            z_img = z_up[0, :3].clamp(0, 1).float().cpu().numpy()
            z_img = (z_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            sep = np.full((H, 4, 3), 14, dtype=np.uint8)
            sections.append(np.concatenate([ref_gt, sep, z_img, sep, ref_rc], axis=1))

        # Synthetic
        images = gen.generate(8)
        x = images.to(device)
        with torch.amp.autocast(device.type, dtype=amp_dtype):
            recon, z = model(x)
        gt = images.cpu().numpy()
        rc = recon[:, :3, :H, :W].clamp(0, 1).float().cpu().numpy()

        cols, rows = 4, 2
        grid_w = cols * (W * 2 + 4) + (cols - 1) * 2
        grid_h = rows * H + (rows - 1) * 2
        synth = np.full((grid_h, grid_w, 3), 14, dtype=np.uint8)
        for i in range(min(8, len(gt))):
            r, c = i // cols, i % cols
            gy = r * (H + 2)
            gx = c * (W * 2 + 6)
            g = (gt[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            r_img = (rc[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            synth[gy:gy+H, gx:gx+W] = g
            synth[gy:gy+H, gx+W+2:gx+W*2+2] = r_img
        sections.append(synth)

        if len(sections) > 1:
            from PIL import Image as _PIL
            syn_w = sections[1].shape[1]
            ref_pil = _PIL.fromarray(sections[0])
            scale = syn_w / sections[0].shape[1]
            ref_pil = ref_pil.resize((syn_w, int(sections[0].shape[0] * scale)),
                                      _PIL.BILINEAR)
            sections[0] = np.array(ref_pil)
            gap = np.full((6, syn_w, 3), 14, dtype=np.uint8)
            grid = np.concatenate([sections[0], gap, sections[1]], axis=0)
        else:
            grid = sections[0]

        model.train()
        Image.fromarray(grid).save(os.path.join(logdir, f"preview_{step:06d}.png"))
        Image.fromarray(grid).save(os.path.join(logdir, "preview_latest.png"))
        print(f"  preview saved", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        model.train()


# =============================================================================
# Training
# =============================================================================

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

    # -- Build downscaler --
    if args.downscaler == "learned":
        down = LearnedDownscaler(args.scale, channels=3,
                                 hidden=args.down_hidden)
    else:
        down = FixedDownscaler(args.scale, mode=args.downscaler)

    # -- Build upscaler --
    if args.upscaler == "espcn":
        up = ESPCNUpscaler(args.scale, channels=3, hidden=args.up_hidden,
                           n_blocks=args.up_blocks)
    elif args.upscaler == "simple":
        up = SimpleUpscaler(args.scale, channels=3, hidden=args.up_hidden,
                            n_blocks=args.up_blocks)

    model = SRVAE(down, up, args.scale).to(device)
    pc = model.param_count()
    print(f"SR VAE: scale={args.scale}x")
    print(f"  Downscaler: {args.downscaler}, {pc['downscaler']:,} params")
    print(f"  Upscaler: {args.upscaler}, {pc['upscaler']:,} params")
    print(f"  Total: {pc['total']:,} params")

    # Latent dims
    lH, lW = args.H // args.scale, args.W // args.scale
    print(f"  Latent: ({3}, {lH}, {lW}) = {3 * lH * lW} dims "
          f"({args.H * args.W * 3 / (3 * lH * lW):.0f}:1)")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=5000, n_base_layers=128,
    )
    gen.build_banks()
    gen.disco_quadrant = True

    # -- LPIPS --
    lpips_fn = None
    if args.w_lpips > 0:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="squeeze").to(device)
            lpips_fn.eval()
            lpips_fn.requires_grad_(False)
        except ImportError:
            print("WARNING: pip install lpips")

    # -- Freeze upscaler during warmup --
    freeze_up_steps = getattr(args, 'freeze_up_steps', 0)
    if freeze_up_steps > 0:
        model.upscaler.requires_grad_(False)
        print(f"Upscaler frozen for first {freeze_up_steps} steps (downscaler warmup)")

    # -- Optimizer --
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(args.lr), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    # -- Resume --
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {args.resume} at step {global_step}")

    # -- Load independent components --
    if args.load_downscaler:
        ckpt_d = torch.load(args.load_downscaler, map_location="cpu",
                            weights_only=False)
        sd = ckpt_d.get("downscaler", ckpt_d.get("model", ckpt_d))
        model.downscaler.load_state_dict(sd, strict=False)
        print(f"Loaded downscaler from {args.load_downscaler}")

    if args.load_upscaler:
        ckpt_u = torch.load(args.load_upscaler, map_location="cpu",
                            weights_only=False)
        sd = ckpt_u.get("upscaler", ckpt_u.get("model", ckpt_u))
        model.upscaler.load_state_dict(sd, strict=False)
        print(f"Loaded upscaler from {args.load_upscaler}")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}")
    print(flush=True)

    def _ckpt_name(step):
        steps_k = f"{step // 1000}k" if step >= 1000 else str(step)
        return f"srvae-{args.downscaler}-{args.upscaler}-{args.scale}x-h{args.up_hidden}-b{args.up_blocks}-{steps_k}.pt"

    # -- Lanczos warmup target --
    lanczos_target = None
    if freeze_up_steps > 0 and args.downscaler == "learned":
        lanczos_target = FixedDownscaler(args.scale, mode="lanczos")
        print(f"Lanczos warmup: downscaler trained against lanczos targets "
              f"for {freeze_up_steps} steps")

    def _make_ckpt():
        return {
            "model": model.state_dict(),
            "downscaler": model.downscaler.state_dict(),
            "upscaler": model.upscaler.state_dict(),
            "optimizer": opt.state_dict(),
            "global_step": global_step,
            "config": {
                "scale": args.scale,
                "downscaler": args.downscaler,
                "upscaler": args.upscaler,
                "up_hidden": args.up_hidden,
                "up_blocks": args.up_blocks,
                "down_hidden": args.down_hidden,
            },
        }

    # Initial preview
    preview_image = getattr(args, 'preview_image', None)
    save_preview(model, gen, str(logdir), global_step, device, amp_dtype,
                 preview_image=preview_image)

    # -- Loop --
    t0 = time.time()
    start_step = global_step
    stop_file = logdir / ".stop"

    while global_step < args.total_steps:
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            break

        # Unfreeze upscaler after warmup
        if freeze_up_steps > 0 and global_step == freeze_up_steps:
            model.upscaler.requires_grad_(True)
            # Rebuild optimizer with all params
            opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                                    weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps - global_step,
                eta_min=float(args.lr) * 0.01)
            print(f"[step {global_step}] Upscaler unfrozen, optimizer rebuilt")

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = {}

        images = gen.generate(args.batch_size)
        x = images.to(device)

        with torch.amp.autocast(device.type, dtype=amp_dtype):
            recon, z = model(x)

            total = torch.tensor(0.0, device=device)

            # Lanczos warmup: train downscaler to match lanczos output
            if lanczos_target is not None and global_step < freeze_up_steps:
                with torch.no_grad():
                    lanczos_out = lanczos_target(x)
                down_loss = F.l1_loss(z, lanczos_out)
                total = total + down_loss
                losses["down"] = down_loss.item()
            else:
                # Normal end-to-end losses
                if args.w_l1 > 0:
                    l1 = F.l1_loss(recon, x)
                    total = total + args.w_l1 * l1
                    losses["l1"] = l1.item()
                if args.w_mse > 0:
                    mse = F.mse_loss(recon, x)
                    total = total + args.w_mse * mse
                    losses["mse"] = mse.item()

            if lpips_fn is not None and global_step >= freeze_up_steps:
                lp = lpips_fn(recon * 2 - 1, x * 2 - 1).mean()
                total = total + args.w_lpips * lp
                losses["lpips"] = lp.item()

        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        global_step += 1

        if global_step % args.log_every == 0:
            el = time.time() - t0
            sps = (global_step - start_step) / max(el, 1)
            eta = (args.total_steps - global_step) / max(sps, 1e-6)
            ls = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
            print(f"[{global_step}/{args.total_steps}] {ls} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if global_step % args.preview_every == 0:
            save_preview(model, gen, str(logdir), global_step, device,
                         amp_dtype, preview_image=preview_image)

        if global_step % args.save_every == 0:
            d = _make_ckpt()
            named = _ckpt_name(global_step)
            torch.save(d, logdir / named)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {named}", flush=True)

    if global_step > start_step:
        d = _make_ckpt()
        named = _ckpt_name(global_step)
        torch.save(d, logdir / named)
        torch.save(d, logdir / "latest.pt")
        print(f"  saved {named}", flush=True)

    print(f"\nDone. {global_step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min")


def main():
    p = argparse.ArgumentParser(description="SR VAE: downscale + upscale")
    p.add_argument("--scale", type=int, default=8,
                   help="Spatial scale factor (must be power of 2)")
    p.add_argument("--downscaler", default="area",
                   choices=["area", "bilinear", "bicubic", "lanczos", "learned"],
                   help="Downscale method")
    p.add_argument("--upscaler", default="espcn",
                   choices=["espcn", "simple"],
                   help="Upscale network")
    p.add_argument("--up-hidden", type=int, default=64,
                   help="Upscaler hidden channels")
    p.add_argument("--up-blocks", type=int, default=4,
                   help="Upscaler residual blocks")
    p.add_argument("--down-hidden", type=int, default=32,
                   help="Learned downscaler hidden channels")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="2e-4")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--w-l1", type=float, default=1.0)
    p.add_argument("--w-mse", type=float, default=0.0)
    p.add_argument("--w-lpips", type=float, default=0.0)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--freeze-up-steps", type=int, default=0,
                   help="Freeze upscaler for N steps (learned downscaler warmup)")
    p.add_argument("--resume", default=None)
    p.add_argument("--load-downscaler", default=None,
                   help="Load downscaler weights independently")
    p.add_argument("--load-upscaler", default=None,
                   help="Load upscaler weights independently")
    p.add_argument("--logdir", default="sr_vae_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--preview-image", default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train ElasticVideoTokenizer on top of a frozen MiniVAE3D.

The tokenizer learns to compress the frozen VAE's continuous latent
grid (N, T', C_lat, H', W') to a 1D sequence of N_q query tokens, with
random tail-drop during training so any suffix budget works at
inference.

Loss: pure MSE in VAE latent space. The frozen VAE decoder is used
only for saved preview mp4s (no gradient flows through it).

Usage:
    python -m training.train_tokenizer --vae-ckpt PATH [options]
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MiniVAE, MiniVAE3D
from core.generator import VAEpp0rGenerator
from core.tokenizer import ElasticVideoTokenizer


# -- Frozen-VAE loading --------------------------------------------------------

def _parse_bool_list(s, default):
    return tuple(x.strip().lower() in ("true", "1", "yes")
                 for x in str(s if s is not None else default).split(","))


def _reconstruct_vae_from_ckpt(ckpt_path):
    """Load a 2D MiniVAE or 3D MiniVAE3D checkpoint from disk and rebuild
    the model. Dispatches on config['model_class']; defaults to 2D
    MiniVAE when absent (older 2D checkpoints have no model_class key).
    Returns (model, ckpt_dict). Model is eval() + requires_grad_(False).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_class = config.get("model_class", None)

    # Heuristic for old 2D checkpoints that don't set model_class
    if model_class is None:
        if "encoder_time_downscale" in config or "residual_shortcut" in config:
            model_class = "MiniVAE"
        else:
            model_class = "MiniVAE3D"

    if model_class == "MiniVAE":
        vae = _build_minivae_from_config(config)
    elif model_class == "MiniVAE3D":
        vae = _build_minivae3d_from_config(config, stored=ckpt.get("args"))
    else:
        raise RuntimeError(f"Unknown model_class={model_class!r} in {ckpt_path}")

    state = ckpt.get("model", ckpt.get("state_dict"))
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if missing:
        print(f"  [vae] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [vae] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    vae.eval().requires_grad_(False)
    return vae, ckpt


def _build_minivae_from_config(config):
    """Reconstruct a 2D MiniVAE from its stored config dict."""
    from gui.common import parse_arch_config
    enc_ch, dec_ch = parse_arch_config(config)
    n_stages = len(dec_ch) if isinstance(dec_ch, tuple) else 3

    def _ptl(key, default):
        s = config.get(key, default)
        return tuple(x.strip().lower() in ("true", "1", "yes")
                     for x in str(s).split(","))

    enc_t = _ptl("encoder_time_downscale", ",".join(["true"] * n_stages))
    dec_t = _ptl("decoder_time_upscale", ",".join(["true"] * n_stages))
    enc_s = _ptl("encoder_spatial_downscale", ",".join(["true"] * n_stages))
    dec_s = _ptl("decoder_spatial_upscale", ",".join(["true"] * n_stages))

    haar = config.get("haar", "none")
    if haar is True: haar = "2x"
    if not haar or haar is False: haar = "none"
    haar_rounds = {"none": 0, "2x": 1, "4x": 2}.get(haar, 0)
    in_ch = int(config.get("image_channels", 3))
    # When haar is on, MiniVAE expects image_channels = 3 * 4**haar_rounds
    vae_in = in_ch * (4 ** haar_rounds) if in_ch == 3 and haar_rounds > 0 else in_ch

    return MiniVAE(
        latent_channels=int(config.get("latent_channels", 32)),
        image_channels=vae_in, output_channels=vae_in,
        encoder_channels=enc_ch, decoder_channels=dec_ch,
        encoder_time_downscale=enc_t, decoder_time_upscale=dec_t,
        encoder_spatial_downscale=enc_s, decoder_spatial_upscale=dec_s,
        residual_shortcut=bool(config.get("residual_shortcut", False)),
        use_attention=bool(config.get("use_attention", False)),
        use_groupnorm=bool(config.get("use_groupnorm", False)),
    )


def _build_minivae3d_from_config(config, stored=None):
    """Reconstruct a 3D MiniVAE3D. Prefers new enc/dec schema; falls
    back to base_channels + channel_mult for older checkpoints."""
    def _get(obj, k, default=None):
        if isinstance(obj, dict):
            return obj.get(k, default)
        return getattr(obj, k, default) if obj is not None else default

    enc_ch_cfg = config.get("encoder_channels") or _get(stored, "enc_ch", "")
    dec_ch_cfg = config.get("decoder_channels") or _get(stored, "dec_ch", "")
    if enc_ch_cfg and dec_ch_cfg:
        enc_channels = tuple(int(x) for x in str(enc_ch_cfg).split(","))
        dec_channels = tuple(int(x) for x in str(dec_ch_cfg).split(","))
    else:
        ch_mult_str = config.get("channel_mult") or _get(stored, "ch_mult", "1,2,4,4")
        base_ch = int(config.get("base_channels") or _get(stored, "base_ch", 64))
        ch_mult = tuple(int(x) for x in str(ch_mult_str).split(","))
        enc_channels = tuple(base_ch * m for m in ch_mult)
        dec_channels = tuple(reversed(enc_channels))

    t_down = _parse_bool_list(
        config.get("temporal_downsample") or _get(stored, "temporal_down"),
        "true,true,true,false")
    s_down = _parse_bool_list(
        config.get("spatial_downsample") or _get(stored, "spatial_down"),
        "true,true,true,true")
    fsq_levels = tuple(int(x) for x in str(
        config.get("fsq_levels") or _get(stored, "fsq_levels", "8,8,8,5,5,5")
    ).split(","))
    return MiniVAE3D(
        latent_channels=int(config.get("latent_channels")
                            or _get(stored, "latent_ch", 16)),
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        num_res_blocks=int(config.get("num_res_blocks")
                           or _get(stored, "num_res_blocks", 2)),
        temporal_downsample=t_down,
        spatial_downsample=s_down,
        haar_levels=int(config.get("haar_levels")
                        or _get(stored, "haar_levels", 0)),
        fsq=bool(config.get("fsq", _get(stored, "fsq", False))),
        fsq_levels=fsq_levels,
        fsq_stages=int(config.get("fsq_stages")
                       or _get(stored, "fsq_stages", 4)),
        use_attention=bool(config.get("use_attention", True)),
        use_groupnorm=bool(config.get("use_groupnorm", True)),
        residual_shortcut=bool(config.get("residual_shortcut", False)),
        attn_heads=int(config.get("attn_heads", 8)),
        gn_groups=int(config.get("gn_groups", 1)),
    )

    vae.eval().requires_grad_(False)
    return vae, ckpt


# -- Preview helpers -----------------------------------------------------------

@torch.no_grad()
def save_preview(tok, vae, gen, logdir, step, device, pT, keeps):
    """Render side-by-side: GT | keep=32 | keep=64 | keep=N_q (pixels).
    Uses two clips from the generator, two rows in the grid.
    """
    try:
        tok.eval()
        H, W = gen.H, gen.W

        # Two distinct pool recipes (same guard as train_video3d.save_preview)
        N = len(getattr(gen, "_recipe_pool", []) or [])
        if N >= 2:
            idxs = random.sample(range(N), 2)
            clips = torch.stack([
                gen._render_recipe(gen._recipe_pool[i]) for i in idxs
            ], dim=0)                                        # (2, T, 3, H, W)
        else:
            # Fallback: live render
            clips = gen.generate_sequence(2, T=pT)
        clips = clips.to(device)

        # Reconstruct at each keep budget
        recons = tok.reconstruct(clips, keeps=keeps)         # {keep: (2,T,3,H,W)}

        # Layout: for each clip row, concat [GT | keep_min | ... | keep_max]
        # Cells are WxH each, gap 4px between cells, 4px gap between rows.
        gap = 4
        n_cols = 1 + len(keeps)
        cell_w, cell_h = W, H
        row_w = cell_w * n_cols + gap * (n_cols - 1)
        frame_h = cell_h * 2 + gap
        frame_w = row_w

        T_show = clips.shape[1]

        stepped = os.path.join(logdir, f"preview_{step:06d}.mp4")
        latest = os.path.join(logdir, "preview_latest.mp4")
        for out_path in [stepped, latest]:
            cmd = ["ffmpeg", "-y", "-v", "quiet",
                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{frame_w}x{frame_h}", "-r", "30",
                   "-i", "pipe:0",
                   "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", out_path]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                gap_v = np.full((cell_h, gap, 3), 14, dtype=np.uint8)
                gap_h = np.full((gap, frame_w, 3), 14, dtype=np.uint8)
                for t in range(T_show):
                    rows_np = []
                    for b in range(2):
                        gt = (clips[b, t].permute(1, 2, 0).float().cpu()
                              .numpy() * 255).clip(0, 255).astype(np.uint8)
                        tiles = [gt]
                        for k in sorted(recons.keys()):
                            rc = (recons[k][b, t].permute(1, 2, 0).float()
                                  .cpu().clamp(0, 1).numpy() * 255
                                  ).astype(np.uint8)
                            tiles.append(rc)
                        row = tiles[0]
                        for tile in tiles[1:]:
                            row = np.concatenate([row, gap_v, tile], axis=1)
                        rows_np.append(row)
                    frame = np.concatenate(
                        [rows_np[0], gap_h, rows_np[1]], axis=0)
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
                proc.wait()
            except Exception:
                try: proc.stdin.close()
                except Exception: pass
                proc.kill()
                proc.wait()
                raise

        print(f"  preview: {stepped}  ({T_show} frames, keeps={sorted(recons.keys())})",
              flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        tok.train()


# -- Training ------------------------------------------------------------------

_stop_requested = False

def _handle_stop(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[Stop requested]", flush=True)


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


def train(args):
    # Loud banner so you can tell the new script is running
    print("=" * 60)
    print("  VAEpp0r train_tokenizer — ElasticVideoTokenizer on frozen VAE")
    print("=" * 60, flush=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Stem resolution (three modes, mutually exclusive) --
    # 1) --vae-ckpt only              -> stem = frozen MiniVAE/MiniVAE3D
    # 2) --vae-ckpt + --flatten-ckpt   -> stem = StemChain(vae, flattener)
    # 3) --latent-cache DIR            -> stem = None, latents loaded from disk
    stem = None
    vae = None
    flattener = None
    latent_cache = getattr(args, "latent_cache", "") or ""
    if latent_cache:
        if args.vae_ckpt:
            raise SystemExit(
                "--latent-cache and --vae-ckpt are mutually exclusive.")
        if not (args.latent_ch and args.latent_t_ds and args.latent_s_ds):
            raise SystemExit(
                "--latent-cache mode requires --latent-ch, --latent-t-ds, "
                "--latent-s-ds so the tokenizer knows the latent shape.")
        print(f"Latent-cache mode: loading pre-computed latents from "
              f"{latent_cache}", flush=True)
        C_lat = int(args.latent_ch)
        t_ds = int(args.latent_t_ds)
        s_ds = int(args.latent_s_ds)
    else:
        if not args.vae_ckpt:
            raise SystemExit(
                "Need --vae-ckpt (optionally with --flatten-ckpt) or "
                "--latent-cache DIR.")
        print(f"Loading VAE from {args.vae_ckpt} ...", flush=True)
        vae, vae_ckpt = _reconstruct_vae_from_ckpt(args.vae_ckpt)
        vae = vae.to(device)
        print(f"  VAE t_downscale={vae.t_downscale}, s_downscale={vae.s_downscale}, "
              f"latent_channels={vae.latent_channels}", flush=True)
        if args.flatten_ckpt:
            from core.tokenizer import StemChain
            from experiments.flatten import FlattenDeflatten
            print(f"Loading flattener from {args.flatten_ckpt} ...", flush=True)
            fckpt = torch.load(args.flatten_ckpt, map_location="cpu",
                               weights_only=False)
            fcfg = fckpt.get("config", {})
            flattener = FlattenDeflatten(
                latent_channels=int(fcfg.get("latent_channels",
                                             vae.latent_channels)),
                bottleneck_channels=int(fcfg["bottleneck_channels"]),
                spatial_h=int(fcfg["spatial_h"]),
                spatial_w=int(fcfg["spatial_w"]),
                walk_order=fcfg.get("walk_order", "raster"),
                kernel_size=int(fcfg.get("kernel_size", 1)),
                deflatten_hidden=int(fcfg.get("deflatten_hidden", 0)),
            )
            flattener.load_state_dict(fckpt["model"], strict=False)
            flattener.eval().requires_grad_(False)
            flattener = flattener.to(device)
            stem = StemChain(vae, flattener)
            print(f"  Flattener: {vae.latent_channels}ch -> "
                  f"{flattener.B_ch}ch @ {flattener.H}x{flattener.W}, "
                  f"walk={flattener.walk_order}", flush=True)
        else:
            stem = vae
        C_lat = int(stem.latent_channels)
        t_ds = int(stem.t_downscale)
        s_ds = int(stem.s_downscale)

    # -- Tokenizer --
    # Pos-emb caps sized from the latent grid at the configured (T, H, W).
    t_lat_cap = max(4, -(-args.T // t_ds) + 2)
    h_lat_cap = max(12, -(-args.H // s_ds) + 4)
    w_lat_cap = max(16, -(-args.W // s_ds) + 4)

    tok = ElasticVideoTokenizer(
        stem=stem,
        C_lat=C_lat, t_downscale=t_ds, s_downscale=s_ds,
        n_queries=args.n_queries,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_mult=args.mlp_mult,
        d_bottleneck=args.d_bottleneck,
        min_keep=args.min_keep,
        dropout=args.dropout,
        pos_emb_max_t=t_lat_cap,
        pos_emb_max_h=h_lat_cap,
        pos_emb_max_w=w_lat_cap,
    ).to(device)

    n_trainable = sum(p.numel() for p in tok.parameters() if p.requires_grad)
    print(f"  Tokenizer trainable params: {n_trainable/1e6:.2f}M  "
          f"(N_q={args.n_queries}, dim={args.dim}, depth={args.depth}, "
          f"d_bottleneck={args.d_bottleneck})", flush=True)

    # -- Data source --
    # Two paths: (a) live generator + stem.encode_video on the fly, or
    # (b) pre-computed latent cache on disk (no generator, no stem).
    gen = None
    latent_files = []
    if latent_cache:
        if not os.path.isdir(latent_cache):
            raise SystemExit(f"--latent-cache {latent_cache} not a directory")
        latent_files = sorted(
            os.path.join(latent_cache, f)
            for f in os.listdir(latent_cache)
            if f.endswith(".pt"))
        if not latent_files:
            raise SystemExit(f"No .pt files found in {latent_cache}")
        print(f"  Latent cache: {len(latent_files)} files from {latent_cache}",
              flush=True)
    else:
        gen = VAEpp0rGenerator(
            height=args.H, width=args.W, device=str(device),
            bank_size=args.bank_size, n_base_layers=args.n_layers,
        )
        bank_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bank")
        bank_files = [f for f in os.listdir(bank_dir)
                      if f.startswith("shapes_") and f.endswith(".pt")] \
            if os.path.isdir(bank_dir) else []
        if bank_files:
            gen.setup_dynamic_bank(bank_dir, working_size=args.bank_size,
                                    refresh_interval=50)
            gen.build_base_layers()
        else:
            gen.build_banks()

        pool_kwargs = dict(
            use_fluid=True, use_ripple=True, use_shake=True, use_kaleido=True,
            fast_transform=True, use_flash=True, use_palette_cycle=True,
            use_text=True, use_signage=True, use_particles=True,
            use_raymarch=True, sphere_dip=True, use_arcade=True,
            use_glitch=True, use_chromatic=True, use_scanlines=True,
            use_fire=True, use_vortex=True, use_starfield=True, use_eq=True,
        )
        gen.build_motion_pool(
            n_clips=args.pool_size, T=args.T, random_mix=True, **pool_kwargs)
        if args.disco:
            gen.disco_quadrant = True
        print(f"Generator ready: pool={len(gen._recipe_pool)} recipes, "
              f"disco={gen.disco_quadrant}", flush=True)

    # -- Optim --
    opt = torch.optim.AdamW(tok.parameters(), lr=float(args.lr),
                            weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    def _next_batch():
        """Return (z_latent, clips_or_None) for one training step.

        Routes through the active data mode: either live-generate clips
        and push through the stem, or sample from the latent cache.
        """
        if latent_cache:
            import random as _rnd
            n_want = args.batch_size
            if len(latent_files) >= n_want:
                paths = _rnd.sample(latent_files, n_want)
            else:
                paths = [_rnd.choice(latent_files) for _ in range(n_want)]
            lats = [torch.load(p, map_location="cpu", weights_only=False)
                    for p in paths]
            lats = [(l["latent"] if isinstance(l, dict) else l) for l in lats]
            batch = torch.stack(lats, dim=0).to(device)
            return batch, None
        clips = gen.generate_from_pool(args.batch_size).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                z = stem.encode_video(clips)
        return z, clips

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        tok.load_state_dict(ckpt["model"])
        if not args.fresh_opt and "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
                sched.load_state_dict(ckpt["scheduler"])
            except Exception:
                print("  fresh optimizer (state mismatch on resume)")
        start_step = int(ckpt.get("step", 0))
        print(f"  resumed at step {start_step}", flush=True)

    # -- Preview keep budgets --
    keeps = sorted({min(int(k), args.n_queries) for k in args.keeps.split(",")})
    print(f"  preview keeps = {keeps}", flush=True)

    # Arch-tagged checkpoint filenames. Same convention as
    # train_video3d / train_elastictok: name carries enough info to
    # identify the arch, and a run-scoped glob lets multiple runs in
    # the same logdir live side-by-side without stepping on each
    # other's prune.
    _arch_stem = (f"tokenizer-q{args.n_queries}-kmin{args.min_keep}"
                  f"-d{args.dim}x{args.depth}h{args.heads}"
                  f"-bn{args.d_bottleneck}")

    def _ckpt_name(step):
        steps_k = f"{step // 1000}k" if step >= 1000 else str(step)
        return f"{_arch_stem}-{steps_k}.pt"

    _run_glob = f"{_arch_stem}-*.pt"

    # -- Training loop --
    tok.train()
    t0 = time.time()
    loss_ema = None
    stop_sentinel = logdir / ".stop"
    # Clear any stale .stop from a prior run so we don't exit immediately.
    try:
        if stop_sentinel.exists():
            stop_sentinel.unlink()
    except Exception:
        pass

    for step in range(start_step, args.total_steps):
        if _stop_requested or stop_sentinel.exists():
            print(f"[Stopping gracefully at step {step}]", flush=True)
            try:
                if stop_sentinel.exists():
                    stop_sentinel.unlink()
            except Exception:
                pass
            break

        # One batch through the active data mode.
        z_target, clips = _next_batch()

        # Random per-batch `keep` ∈ [min_keep, n_queries].
        keep = random.randint(args.min_keep, args.n_queries)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            tokens = tok.encode_latent(z_target.to(amp_dtype))
            Tp = z_target.shape[1]
            Hp = z_target.shape[3]
            Wp = z_target.shape[4]
            z_hat = tok.decode_tokens(tokens, Tp, Hp, Wp, keep=keep)
            loss = F.mse_loss(z_hat.float(), z_target.float())

        if amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(tok.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), args.grad_clip)
            opt.step()
        sched.step()

        loss_v = float(loss.item())
        loss_ema = loss_v if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_v

        if step % args.log_every == 0:
            lr = opt.param_groups[0]["lr"]
            dt = time.time() - t0
            print(f"  [{step:6d}/{args.total_steps}]  "
                  f"loss={loss_v:.4f}  ema={loss_ema:.4f}  "
                  f"keep={keep}/{args.n_queries}  "
                  f"lr={lr:.2e}  elapsed={dt:.0f}s",
                  flush=True)

        if args.preview_every > 0 and step % args.preview_every == 0 \
                and step != start_step:
            if stem is not None and gen is not None:
                # Full pixel preview (stem available, generator available)
                save_preview(tok, stem, gen, str(logdir), step, device,
                             pT=args.T, keeps=keeps)
            else:
                # Latent-cache mode: no pixels available. Log per-keep
                # latent MSE across the current batch instead.
                with torch.no_grad():
                    recs = tok.reconstruct_latent(z_target, keeps=keeps)
                    msgs = []
                    for k in sorted(recs.keys()):
                        mse = F.mse_loss(recs[k].float(),
                                         z_target.float()).item()
                        msgs.append(f"keep={k}:{mse:.4f}")
                print(f"  [preview step={step}] latent MSE — "
                      + "  ".join(msgs), flush=True)

        if args.save_every > 0 and step % args.save_every == 0 \
                and step != start_step:
            d = _make_ckpt(tok, opt, sched, step, args)
            named = _ckpt_name(step)
            torch.save(d, str(logdir / named))
            torch.save(d, str(logdir / "latest.pt"))
            print(f"  saved {named}", flush=True)
            # Prune older stepped ckpts matching THIS run's arch glob.
            ckpts = sorted(
                [f for f in logdir.glob(_run_glob)
                 if f.name != "latest.pt"],
                key=lambda p: p.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink(missing_ok=True)

    # Final save
    d = _make_ckpt(tok, opt, sched, args.total_steps, args)
    named_final = _ckpt_name(args.total_steps)
    torch.save(d, str(logdir / named_final))
    torch.save(d, str(logdir / "latest.pt"))
    print(f"Done. saved {named_final}  final loss_ema={loss_ema}", flush=True)


def _make_ckpt(tok, opt, sched, step, args):
    return {
        "model": tok.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "args": vars(args),
        "vae_ckpt_path": args.vae_ckpt,
    }


# -- CLI -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train ElasticVideoTokenizer on a frozen MiniVAE3D")

    # Stem / data source (pick one mode):
    #   1) --vae-ckpt PATH                            (VAE stem)
    #   2) --vae-ckpt PATH --flatten-ckpt PATH         (VAE + flattener stem)
    #   3) --latent-cache DIR  (+ --latent-ch/--latent-t-ds/--latent-s-ds)
    #      pre-computed latents, no stem, no generator
    p.add_argument("--vae-ckpt", default="",
                   help="Path to frozen MiniVAE or MiniVAE3D checkpoint. "
                        "Optional if --latent-cache is set.")
    p.add_argument("--flatten-ckpt", default="",
                   help="Optional flattener checkpoint. When set with "
                        "--vae-ckpt, the tokenizer sees the post-flatten "
                        "latent instead of the raw VAE latent.")
    p.add_argument("--latent-cache", default="",
                   help="Directory of pre-computed .pt latent files. "
                        "Mutually exclusive with --vae-ckpt. Each .pt is "
                        "either a (T',C,H',W') tensor or a dict with "
                        "key 'latent'.")
    p.add_argument("--latent-ch", type=int, default=0,
                   help="Latent channel count (required with --latent-cache)")
    p.add_argument("--latent-t-ds", type=int, default=0,
                   help="Latent temporal downscale (required with --latent-cache)")
    p.add_argument("--latent-s-ds", type=int, default=0,
                   help="Latent spatial downscale (required with --latent-cache)")

    # Tokenizer arch
    p.add_argument("--n-queries", type=int, default=128,
                   help="Max queries per clip (upper bound on token budget)")
    p.add_argument("--min-keep", type=int, default=32,
                   help="Lower bound for random tail-drop during training")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--d-bottleneck", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    # Data
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--T", type=int, default=17,
                   help="Frames per clip (Cosmos-convention 1 + k*t_downscale)")
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--disco", action="store_true")

    # Optim
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", default="1e-4")
    p.add_argument("--weight-decay", default="1e-4")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")

    # I/O
    p.add_argument("--logdir", default="synthyper_tokenizer_logs")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=500)
    p.add_argument("--keeps", default="32,64,128",
                   help="Comma-separated keep budgets to render in previews")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

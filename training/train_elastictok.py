#!/usr/bin/env python3
"""Training script for the PyTorch port of ElasticTok, using clips from
VAEpp0rGenerator as data.

The training recipe is a 1:1 port of LargeWorldModel/ElasticTok
(scripts/train.py + scripts/run_train.sh):

  - L2 (MSE) reconstruction loss on the patch tensor (normalized to [-1,1])
  - LPIPS perceptual loss on ONE randomly sampled frame from ONE
    randomly sampled block, weighted by `lpips_loss_ratio` (default 0.1)
  - `aux_loss` = VAE KL * 1e-8 (auto-zero if FSQ bottleneck)
  - total = recon_loss + aux_loss
  - AdamW optimizer, lr=1e-4, weight_decay=1e-4, 2000-step linear warmup,
    cosine decay to 1e-4 (the ref sets end_lr = lr, so effectively no
    decay in the shell script — we preserve that default)
  - mask_type='elastic': per-block encoding_mask with keep count
    uniformly drawn from [min_toks, max_toks]
  - Preview: MP4 with ref clip (optional) + 2 synth clips. Encoded at
    multiple `keep` budgets to visualize variable-length behavior.

No creative liberties taken with the training recipe; numeric defaults
match `scripts/run_train.sh` verbatim where applicable.
"""

from __future__ import annotations

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


PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.generator import VAEpp0rGenerator
from core.elastictok import ElasticTokConfig, ElasticTok


# ---------------- Haar wavelet (lossless 2x spatial, 4x channel) ------
# Optional per-frame pre-compression. With haar_levels=N:
#   (B, T, C, H, W) -> (B, T, C*4**N, H/2**N, W/2**N)
# Encoder sees the downsampled tensor with more channels; decoder output
# is inverted back to pixels. Purely mathematical, no learned params.

def _haar_down_2d(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, 4C, H/2, W/2). Orthonormal 2D Haar."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)


def _haar_up_2d(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _haar_down_2d."""
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


def haar_down_video(x: torch.Tensor, n: int) -> torch.Tensor:
    """Per-frame Haar-down n times. (B, T, C, H, W) -> (B, T, C*4^n, H/2^n, W/2^n)."""
    if n == 0:
        return x
    B, T, C, H, W = x.shape
    y = x.reshape(B * T, C, H, W)
    for _ in range(n):
        y = _haar_down_2d(y)
    return y.reshape(B, T, y.shape[1], y.shape[2], y.shape[3])


def haar_up_video(x: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of haar_down_video."""
    if n == 0:
        return x
    B, T, C, H, W = x.shape
    y = x.reshape(B * T, C, H, W)
    for _ in range(n):
        y = _haar_up_2d(y)
    return y.reshape(B, T, y.shape[1], y.shape[2], y.shape[3])


# ---------------- Patchify / unpatchify -------------------------------
# Reference uses `dataset.patchify(...)` outside train.py; here we inline
# the same transform. A clip (B, T, 3, H, W) becomes (B, L, patch_dim)
# with L = (T//Tp)*(H//Hp)*(W//Wp) and patch_dim = Tp*Hp*Wp*3.


def patchify(clip: torch.Tensor, patch_size) -> torch.Tensor:
    """(B, T, 3, H, W) -> (B, L, Tp*Hp*Wp*3)."""
    Tp, Hp, Wp = patch_size
    B, T, C, H, W = clip.shape
    assert T % Tp == 0, f"T={T} not divisible by Tp={Tp}"
    assert H % Hp == 0, f"H={H} not divisible by Hp={Hp}"
    assert W % Wp == 0, f"W={W} not divisible by Wp={Wp}"
    nT, nH, nW = T // Tp, H // Hp, W // Wp
    # (B, T, C, H, W) -> (B, nT, Tp, C, nH, Hp, nW, Wp)
    x = clip.reshape(B, nT, Tp, C, nH, Hp, nW, Wp)
    # Permute to (B, nT, nH, nW, Tp, Hp, Wp, C)
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    # Flatten patch & seq axes: (B, L, patch_dim)
    L = nT * nH * nW
    patch_dim = Tp * Hp * Wp * C
    return x.reshape(B, L, patch_dim)


def unpatchify(patches: torch.Tensor, patch_size, grid) -> torch.Tensor:
    """Inverse of `patchify`. Returns (B, T, 3, H, W)."""
    Tp, Hp, Wp = patch_size
    nT, nH, nW = grid
    B, L, patch_dim = patches.shape
    assert L == nT * nH * nW
    C = patch_dim // (Tp * Hp * Wp)
    x = patches.reshape(B, nT, nH, nW, Tp, Hp, Wp, C)
    # Inverse permute: (B, nT, Tp, C, nH, Hp, nW, Wp)
    x = x.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
    T, H, W = nT * Tp, nH * Hp, nW * Wp
    return x.reshape(B, T, C, H, W)


# ---------------- Elastic tail-drop mask ------------------------------

def elastic_mask(B: int, L: int, block_size: int,
                  min_toks: int, max_toks: int,
                  device) -> torch.Tensor:
    """Per-block mask with the first K tokens kept, K~U[min_toks, max_toks].

    Mirrors ElasticTok's `mask_mode='elastic'`. `block_size` should equal
    `max_toks` per the training recipe (one block of `max_toks` each).
    For a single-clip / single-block setup (common here), `L == block_size`.
    """
    assert L % block_size == 0, (L, block_size)
    n_blocks = L // block_size
    mask = torch.zeros(B, n_blocks, block_size, dtype=torch.bool,
                        device=device)
    for b in range(B):
        for i in range(n_blocks):
            k = random.randint(min_toks, max_toks)
            mask[b, i, :k] = True
    return mask.reshape(B, L)


# ---------------- LPIPS loss (on one random frame of one block) -------

def _build_lpips(device):
    try:
        import lpips as lpips_mod
        net = lpips_mod.LPIPS(net="vgg").to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net
    except Exception as e:
        print(f"  LPIPS disabled ({e})", flush=True)
        return None


def lpips_on_block_frame(recon_patches: torch.Tensor,
                          real_patches: torch.Tensor,
                          patch_size, frames_per_block: int,
                          resolution_h: int, resolution_w: int,
                          lpips_fn,
                          haar_levels: int = 0,
                          out_H: int = None,
                          out_W: int = None) -> torch.Tensor:
    """Mirrors ref train.py `compute_lpips`, with optional Haar-up to
    return to pixel-space RGB before LPIPS.

    `resolution_h`/`resolution_w` are the POST-Haar dims the model saw.
    `out_H`/`out_W` are the original RGB dims (=resolution_* if haar=0).
    """
    if lpips_fn is None:
        return torch.zeros((), device=recon_patches.device,
                            dtype=recon_patches.dtype)
    Tp, Hp, Wp = patch_size
    T = frames_per_block
    B = recon_patches.shape[0]
    nT = T // Tp
    nH = resolution_h // Hp
    nW = resolution_w // Wp

    def _to_video(x):
        # (B, L, patch_dim) -> (B, T, C, H, W) in post-haar space
        return unpatchify(x, patch_size, (nT, nH, nW))

    fake = _to_video(recon_patches)
    real = _to_video(real_patches)
    # Haar-up back to RGB pixel space so LPIPS sees 3-channel images.
    if haar_levels > 0:
        fake = haar_up_video(fake, haar_levels)
        real = haar_up_video(real, haar_levels)
    # pick random block (we only have one) and one frame
    block_idx = 0
    frame_idx = random.randint(0, T - 1)
    fake_f = fake[:, frame_idx]         # (B, 3, H, W)
    real_f = real[:, frame_idx]
    fake_f = F.interpolate(fake_f, size=(224, 224),
                           mode="bilinear", align_corners=False)
    real_f = F.interpolate(real_f, size=(224, 224),
                           mode="bilinear", align_corners=False)
    return lpips_fn(real_f, fake_f).mean()


# ---------------- MP4 preview ------------------------------------------

def _probe_fps(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s = r.stdout.decode().strip()
        num, den = s.split("/")
        return float(num) / float(den)
    except Exception:
        return 30.0


def _decode_video_frames(path, frame_skip, n_frames, W, H):
    fps = _probe_fps(path)
    seek_s = frame_skip / max(fps, 1e-6)
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ss", str(seek_s),
         "-vf",
         (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"),
         "-vframes", str(n_frames),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = result.stdout
    fb = H * W * 3
    return [np.frombuffer(raw[i*fb:(i+1)*fb], dtype=np.uint8).reshape(H, W, 3)
            for i in range(len(raw) // fb)]


@torch.no_grad()
def _model_recon_single(model, clip_m11, patch_size, keep,
                         block_size, device):
    """Run the model on one clip that already fits in max_sequence_length.
    Used as the per-chunk op by `_reconstruct_at_keep`."""
    B, T, C, H, W = clip_m11.shape
    vision = patchify(clip_m11, patch_size)
    L = vision.shape[1]
    att_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    if L % block_size == 0:
        n_blocks = L // block_size
        enc_mask = torch.zeros(B, n_blocks, block_size, dtype=torch.bool,
                                device=device)
        enc_mask[:, :, :min(keep, block_size)] = True
        enc_mask = enc_mask.reshape(B, L)
    else:
        # Non-training chunk size: fall back to one block spanning L,
        # apply the same keep fraction.
        enc_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        enc_mask[:, :min(keep, L)] = True
    recon_patches, _ = model(vision, enc_mask, att_mask, seg_ids, pos_ids,
                              training=False)
    nT, nH, nW = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    return unpatchify(recon_patches, patch_size, (nT, nH, nW)).clamp(-1, 1)


@torch.no_grad()
def _reconstruct_at_keep(model, clip_m11, patch_size, keep,
                          max_toks, block_size, device, haar_levels=0):
    """Encode+decode one RGB clip in [-1,1] with `keep` tokens per block.

    If `haar_levels>0`, per-frame 2D Haar-down is applied before patchify
    and the inverse is applied after unpatchify, so the caller always
    receives RGB pixels at the original resolution.

    Chunks along T when the post-haar clip's L exceeds max_sequence_length.
    Same pattern as gui/common.py `chunked_vae_inference`."""
    # Haar-down to the effective resolution the model operates on.
    clip_h = (haar_down_video(clip_m11, haar_levels)
              if haar_levels > 0 else clip_m11)
    B, T, C, Hh, Wh = clip_h.shape
    Tp, Hp, Wp = patch_size
    nH = Hh // Hp
    nW = Wh // Wp
    max_seq = int(getattr(model.config, "max_sequence_length", 4096))
    nT_max = max_seq // max(nH * nW, 1)
    T_chunk = nT_max * Tp
    if T_chunk < Tp:
        raise RuntimeError(
            f"post-haar spatial tokens per frame ({nH * nW}) exceed "
            f"max_sequence_length ({max_seq}). Increase "
            f"max_sequence_length, reduce resolution, or raise haar_levels.")

    if T <= T_chunk:
        recon_h = _model_recon_single(
            model, clip_h, patch_size, keep, block_size, device)
    else:
        pieces = []
        t_start = 0
        while t_start < T:
            t_end = min(t_start + T_chunk, T)
            t_len = ((t_end - t_start) // Tp) * Tp
            if t_len < Tp:
                break
            sub = clip_h[:, t_start:t_start + t_len]
            pieces.append(
                _model_recon_single(model, sub, patch_size, keep,
                                     block_size, device).float().cpu())
            t_start += t_len
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        recon_h = torch.cat(pieces, dim=1).to(device)

    # Invert Haar back to original RGB resolution.
    recon = (haar_up_video(recon_h, haar_levels)
             if haar_levels > 0 else recon_h)
    return recon.clamp(-1, 1)


@torch.no_grad()
def save_preview(model, gen, logdir, step, device, pT, keeps, patch_size,
                  max_toks, block_size,
                  preview_image=None, preview_frame_skip=0,
                  overfit_clip=None, H=None, W=None,
                  haar_levels: int = 0,
                  overfit_video_path: str = None,
                  overfit_frame_skip: int = 0):
    """Preview MP4 with optional ref video row + 2 synth clips row.

    If `overfit_clip` (1, T, 3, H, W in [0,1]) is provided, it replaces
    both synth clips and the ref (you're training on ONE clip and want
    to see how the recon of that clip converges). Otherwise it uses
    the generator like before.
    """
    from PIL import Image as _PIL
    try:
        model.eval()
        if overfit_clip is not None:
            H, W = overfit_clip.shape[-2], overfit_clip.shape[-1]
        elif gen is not None:
            H, W = gen.H, gen.W

        # ---- Synth: render 2 distinct clips (or duplicate overfit) ---
        # Same logic as train_video.py save_preview: use the pool when
        # its T matches, else fall back to generate_sequence BUT pass
        # the training pool's kwargs through so preview sees the same
        # effect distribution (disco, ripple, shake, kaleido, text,
        # signage, particles, raymarch, arcade, fire, vortex, star-
        # field, eq, ...) as the training step. Calling
        # generate_sequence with no kwargs — as this file did before —
        # defaults every effect flag to False, producing vanilla
        # clips and the "preview sees disco-off while training sees
        # disco" mismatch.
        if overfit_clip is not None:
            # Use the same clip twice; the recon difference will come
            # from stochastic tail-drop at preview time (if keeps differ).
            # If the cached training tensor is shorter than pT (training
            # uses args.T frames, preview wants pT), re-decode pT frames
            # from the source path — same pattern the ref-video branch
            # uses. The model recon chunks via _reconstruct_at_keep so
            # pT > max_sequence_length is already handled.
            if (overfit_video_path and
                    overfit_clip.shape[1] < pT and
                    os.path.exists(overfit_video_path)):
                frames = _decode_video_frames(
                    overfit_video_path, int(overfit_frame_skip or 0),
                    pT, W, H)
                if frames:
                    while len(frames) < pT:
                        frames.append(frames[-1])
                    arr = (np.stack(frames[:pT]).astype(np.float32)
                           / 255.0)  # (pT, H, W, 3)
                    big = torch.from_numpy(arr).permute(
                        0, 3, 1, 2).unsqueeze(0).to(device)  # (1, pT, 3, H, W)
                    clips = big.repeat(2, 1, 1, 1, 1)
                else:
                    clips = overfit_clip[:1].repeat(2, 1, 1, 1, 1)
            else:
                clips = overfit_clip[:1].repeat(2, 1, 1, 1, 1)
        else:
            has_pool = (hasattr(gen, "_recipe_pool") and gen._recipe_pool
                        and getattr(gen, "_motion_pool_T", None) == pT)
            if has_pool:
                N = len(gen._recipe_pool)
                idxs = random.sample(range(N), k=min(2, N))
                if len(idxs) < 2:
                    idxs = idxs + [idxs[0]]
                clips = torch.stack(
                    [gen._render_recipe(gen._recipe_pool[i])
                     for i in idxs], dim=0)
            else:
                kw_raw = getattr(gen, "_train_pool_kwargs", {}) or {}
                rmix = bool(getattr(gen, "_train_random_mix", True))

                def _roll(kw):
                    if not rmix:
                        return kw
                    import random as _rnd
                    drops = getattr(gen, "_RANDOM_MIX_DROPS", {}) or {}
                    out = dict(kw)
                    for k, p in drops.items():
                        if out.get(k) and _rnd.random() < p:
                            out[k] = False
                    return out

                parts = []
                for _ in range(2):
                    parts.append(gen.generate_sequence(
                        1, T=pT, **_roll(kw_raw)))
                clips = torch.cat(parts, dim=0)
            clips = clips.to(device)
        clip_m11 = clips * 2 - 1   # [0,1] -> [-1,1]
        synth_recons = {
            int(k): _reconstruct_at_keep(
                model, clip_m11, patch_size, int(k), max_toks, block_size,
                device, haar_levels=haar_levels).cpu().numpy()
            for k in keeps}
        synth_gt_np = clips.cpu().numpy()  # already [0,1]
        del clips, clip_m11

        # ---- Ref clip (optional) -----------------------------------
        ref_gt_np = None
        ref_recons = None
        if preview_image and os.path.exists(preview_image):
            frames = _decode_video_frames(
                preview_image, preview_frame_skip or 0, pT, W, H)
            if len(frames) >= 1:
                while len(frames) < pT:
                    frames.append(frames[-1])
                ref_arr = np.stack(frames[:pT]).astype(np.float32) / 255.0
                ref_t = torch.from_numpy(ref_arr).permute(
                    0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, 3, H, W)
                ref_m11 = ref_t * 2 - 1
                ref_recons = {
                    int(k): _reconstruct_at_keep(
                        model, ref_m11, patch_size, int(k), max_toks,
                        block_size, device,
                        haar_levels=haar_levels).cpu().numpy()
                    for k in keeps}
                ref_gt_np = ref_t.cpu().numpy()
                del ref_t, ref_m11

        # ---- Layout ------------------------------------------------
        # Overfit mode: GT | Recon@k1 | Recon@k2 | ... | Recon@kN  —
        #   one row showing the SAME clip reconstructed at every
        #   budget in `keeps`. Seeing the budget sweep is the whole
        #   point of a variable-length tokenizer.
        # Non-overfit: GT | Recon@disp_keep for two distinct synth
        #   clips, side by side (original layout).
        keep_keys = sorted(synth_recons.keys())
        disp_keep = keep_keys[-1]
        sep_v = np.full((H, 4, 3), 14, dtype=np.uint8)
        gap_v = np.full((H, 2, 3), 14, dtype=np.uint8)
        overfit_mode = (overfit_clip is not None)
        # Layout:
        #   overfit: single row GT | R@k1 | ... | R@kN at full H.
        #   non-overfit (ref):
        #     ROW 1 (hero): GT | R@k1 | ... | R@kN — built at natural
        #            (ref_natural_w, H), PIL-resized uniformly to
        #            (canvas_w, ref_h). Aspect preserved.
        #     ROW 2 (synth, smaller): 2 distinct clips side-by-side
        #            (GT|R@max each) at HALF scale (W/2, H/2 per cell)
        #            so the ref dominates visually even when ref natural
        #            is already wide from many keeps.
        has_ref = ref_gt_np is not None and not overfit_mode
        ref_n_cells = 1 + len(keep_keys)
        ref_natural_w = W * ref_n_cells + 4 * (ref_n_cells - 1)
        # Synth at half scale (both axes) below the ref.
        W_h = W // 2
        H_h = H // 2
        syn_half_cell_w = W_h * 2 + 4            # GT | R half-size
        syn_half_row_w = syn_half_cell_w * 2 + 2  # 2 clips
        if overfit_mode:
            # Overfit keeps full-scale single row.
            n_cells = 1 + len(keep_keys)
            syn_w = W * n_cells + 4 * (n_cells - 1)
            canvas_w = syn_w
            ref_h = H
            bottom_h = 0
            gap = 0
        else:
            # Canvas width = wider of ref_natural or synth_half_row.
            canvas_w = max(ref_natural_w, syn_half_row_w)
            # Scale ref proportionally to canvas_w (aspect preserved).
            ref_scale = canvas_w / ref_natural_w if has_ref else 1.0
            ref_h = int(round(H * ref_scale)) if has_ref else H
            bottom_h = H_h if has_ref else 0
            gap = 6 if has_ref else 0
        # libx264 / yuv420p require even dims.
        if ref_h % 2 == 1:
            ref_h += 1
        frame_w = canvas_w + (canvas_w % 2)
        frame_h = ref_h + gap + bottom_h
        frame_h += frame_h % 2
        T_show = pT

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
                for t in range(T_show):
                    rows = []
                    if has_ref:
                        # Build GT | R@k1 | ... | R@kN at natural (H,
                        # ref_natural_w), then PIL-resize UNIFORMLY to
                        # (syn_w, ref_h) — same scale factor both axes.
                        # This is what train_video3d.py does; face is
                        # scaled but not distorted.
                        g = (ref_gt_np[0, t].transpose(1, 2, 0) * 255
                             ).clip(0, 255).astype(np.uint8)
                        ref_tiles = [g]
                        for k in keep_keys:
                            rk = ((ref_recons[k][0, t]
                                   .transpose(1, 2, 0) * 0.5 + 0.5) * 255
                                  ).clip(0, 255).astype(np.uint8)
                            ref_tiles.append(rk)
                        ref_cell = ref_tiles[0]
                        for tile in ref_tiles[1:]:
                            ref_cell = np.concatenate(
                                [ref_cell, sep_v, tile], axis=1)
                        ref_scaled = np.array(
                            _PIL.fromarray(ref_cell).resize(
                                (frame_w, ref_h), _PIL.BILINEAR))
                        rows.append(ref_scaled)
                        rows.append(np.full((6, frame_w, 3), 14,
                                             dtype=np.uint8))
                    if overfit_mode:
                        # GT | Recon@k1 | ... | Recon@kN for the SAME clip
                        gt_img = (synth_gt_np[0, t].transpose(1, 2, 0) * 255
                                  ).clip(0, 255).astype(np.uint8)
                        cells = [gt_img]
                        for k in keep_keys:
                            rc = ((synth_recons[k][0, t].transpose(1, 2, 0)
                                   * 0.5 + 0.5) * 255
                                  ).clip(0, 255).astype(np.uint8)
                            cells.append(rc)
                        synth_row = cells[0]
                        for tile in cells[1:]:
                            synth_row = np.concatenate(
                                [synth_row, sep_v, tile], axis=1)
                    else:
                        g0 = (synth_gt_np[0, t].transpose(1, 2, 0) * 255
                              ).clip(0, 255).astype(np.uint8)
                        r0 = ((synth_recons[disp_keep][0, t]
                               .transpose(1, 2, 0) * 0.5 + 0.5) * 255
                              ).clip(0, 255).astype(np.uint8)
                        g1 = (synth_gt_np[1, t].transpose(1, 2, 0) * 255
                              ).clip(0, 255).astype(np.uint8)
                        r1 = ((synth_recons[disp_keep][1, t]
                               .transpose(1, 2, 0) * 0.5 + 0.5) * 255
                              ).clip(0, 255).astype(np.uint8)
                        synth_row = np.concatenate(
                            [g0, sep_v, r0, gap_v, g1, sep_v, r1], axis=1)
                        # With has_ref, the layout budgeted bottom_h
                        # (= H/2) rows for the synth row. Previously
                        # this never actually got resized — the row was
                        # written at full H, so each rawvideo frame
                        # overflowed frame_h by H/2 rows, causing the
                        # next frame to start mid-content (visible as a
                        # vertical scroll/wrap in the mp4). Resize here.
                        if has_ref:
                            new_w = int(round(
                                synth_row.shape[1] *
                                (bottom_h / synth_row.shape[0])))
                            synth_row = np.array(
                                _PIL.fromarray(synth_row).resize(
                                    (new_w, bottom_h), _PIL.BILINEAR))
                    # Center synth_row within frame_w if ref was wider
                    # (pad evenly on left/right instead of dumping all
                    # slack on the right edge).
                    if synth_row.shape[1] < frame_w:
                        total_pad = frame_w - synth_row.shape[1]
                        left = total_pad // 2
                        right = total_pad - left
                        pad_L = np.full(
                            (synth_row.shape[0], left, 3),
                            14, dtype=np.uint8)
                        pad_R = np.full(
                            (synth_row.shape[0], right, 3),
                            14, dtype=np.uint8)
                        synth_row = np.concatenate(
                            [pad_L, synth_row, pad_R], axis=1)
                    rows.append(synth_row)
                    proc.stdin.write(np.concatenate(rows, axis=0).tobytes())
                proc.stdin.close()
                proc.wait()
            except Exception:
                try: proc.stdin.close()
                except Exception: pass
                proc.kill()
                proc.wait()
                raise
        ref_note = " +ref" if has_ref else ""
        print(f"  preview: {stepped} ({T_show} frames{ref_note}, "
              f"disp keep={disp_keep} of {keep_keys})", flush=True)
    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        model.train()


# ---------------- Training loop ---------------------------------------

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
    print("=" * 60)
    print("  ElasticTok (PyTorch port) — training")
    print("=" * 60, flush=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Config --
    # If resuming, load the checkpoint's saved config FIRST so the model
    # is built with the exact architecture that produced the weights.
    # Otherwise CLI args build a fresh config. (Previously the CLI always
    # won and resume mismatches produced size_mismatch errors when the
    # ckpt used a bigger preset than args.config_name.)
    resume_cfg = None
    if args.resume and os.path.exists(args.resume):
        try:
            _peek = torch.load(args.resume, map_location="cpu",
                                weights_only=False)
            resume_cfg = _peek.get("config") or None
            del _peek
            if resume_cfg:
                print(f"  [cfg] using saved config from {args.resume}",
                      flush=True)
        except Exception as e:
            print(f"  [cfg] could not peek resume ckpt: {e}", flush=True)

    if resume_cfg:
        cfg = ElasticTokConfig(**resume_cfg)
        # patch_size may arrive as list from JSON-style dicts; normalize
        patch_size = tuple(int(x) for x in cfg.patch_size)
    else:
        cfg = ElasticTokConfig.load_config(args.config_name)
        patch_size = tuple(int(x) for x in args.patch_size.split(","))
        assert len(patch_size) == 3
        cfg.update(dict(
            patch_size=patch_size,
            in_channels=3 * (4 ** int(getattr(args, "haar_levels", 0)
                                       or 0)),
            bottleneck_type=args.bottleneck_type,
            fsq_quant_levels=tuple(int(x) for x in
                                    args.fsq_levels.split(",")
                                    if x.strip()),
            vae_bottleneck_dim=int(args.vae_bottleneck_dim),
            max_sequence_length=int(args.max_sequence_length),
            max_toks=int(args.max_toks),
            min_toks=int(args.min_toks),
            frames_per_block=int(args.frames_per_block),
            lpips_loss_ratio=float(args.lpips_loss_ratio),
        ))

    # Validate TRAINING clip fits max_sequence_length. The preview is
    # chunked along T by `_reconstruct_at_keep` so it can handle any
    # length regardless of max_sequence_length.
    # Post-Haar spatial dims are reduced by 2**haar_levels per axis.
    haar_levels = int(getattr(args, "haar_levels", 0) or 0)
    haar_scale = 2 ** haar_levels
    H_post = args.H // haar_scale
    W_post = args.W // haar_scale
    if args.H % haar_scale != 0 or args.W % haar_scale != 0:
        raise SystemExit(
            f"H,W ({args.H},{args.W}) must be divisible by "
            f"2**haar_levels = {haar_scale}.")
    nT_train = int(args.T) // patch_size[0]
    nH = H_post // patch_size[1]
    nW = W_post // patch_size[2]
    L_train = nT_train * nH * nW
    if L_train > cfg.max_sequence_length:
        raise SystemExit(
            f"Training L={L_train} > max_sequence_length="
            f"{cfg.max_sequence_length}. Increase --max-sequence-length, "
            f"reduce clip/patch size, or raise --haar-levels.")
    if L_train % cfg.max_toks != 0:
        raise SystemExit(
            f"L={L_train} not divisible by max_toks={cfg.max_toks}. "
            f"Pick H/W/patch_size/haar so the token grid is a multiple of "
            f"max_toks, or set --max-toks={L_train}.")
    n_blocks = L_train // cfg.max_toks
    # Effective per-block keep upper bound. Defaults to block size
    # (cfg.max_toks) for backward-compatible behaviour; --keep-upper
    # lets you pin a smaller value (e.g. fixed keep=256 with block=3680
    # when L=3680 isn't divisible by 256).
    keep_upper_eff = (int(args.keep_upper)
                      if int(args.keep_upper) > 0
                      else cfg.max_toks)
    if not (cfg.min_toks <= keep_upper_eff <= cfg.max_toks):
        raise SystemExit(
            f"keep_upper={keep_upper_eff} must satisfy "
            f"min_toks ({cfg.min_toks}) <= keep_upper <= "
            f"max_toks ({cfg.max_toks}).")
    haar_tag = (f" [Haar {haar_levels}x: {args.H}x{args.W} "
                f"-> {H_post}x{W_post}, 3ch -> "
                f"{3 * (4**haar_levels)}ch]") if haar_levels > 0 else ""
    keep_tag = (f"  keep/block in [{cfg.min_toks}, {keep_upper_eff}]"
                + ("" if keep_upper_eff == cfg.max_toks
                   else f" (< block_size={cfg.max_toks})"))
    print(f"  grid (train T={args.T}): ({nT_train}, {nH}, {nW})  "
          f"L={L_train}  n_blocks={n_blocks}  "
          f"block_size={cfg.max_toks}{haar_tag}{keep_tag}", flush=True)

    # Arch-tagged checkpoint filenames, same convention as
    # train_video3d: the name carries enough arch info that you can
    # eyeball it and know what config produced the weights. Plus a
    # matching glob so we only prune checkpoints FROM THIS RUN's arch
    # tag — runs with different patch/bottleneck/etc. settings kept
    # side-by-side in the same logdir survive.
    Tp_tag, Hp_tag, Wp_tag = patch_size
    _patch_tag = f"p{Tp_tag}x{Hp_tag}x{Wp_tag}"
    _mt_tag = f"mt{cfg.max_toks}"
    _min_tag = f"kmin{cfg.min_toks}"
    _ku_tag = (f"-ku{keep_upper_eff}"
               if keep_upper_eff != cfg.max_toks else "")
    if args.bottleneck_type == "vae":
        _bn_tag = f"bn{int(args.vae_bottleneck_dim)}"
    else:
        _bn_tag = f"fsq{len(args.fsq_levels.split(','))}"
    _haar_tag_f = f"-h{haar_levels}" if haar_levels > 0 else ""
    _arch_stem = (f"elastictok-{args.config_name}-{_patch_tag}-"
                  f"{_mt_tag}-{_min_tag}{_ku_tag}-{_bn_tag}"
                  f"{_haar_tag_f}")

    def _ckpt_name(step):
        steps_k = f"{step // 1000}k" if step >= 1000 else str(step)
        return f"{_arch_stem}-{steps_k}.pt"

    _run_glob = f"{_arch_stem}-*.pt"
    preview_T_eff = int(args.preview_T) if args.preview_T else args.T
    if preview_T_eff != args.T:
        L_prev = (preview_T_eff // patch_size[0]) * nH * nW
        print(f"  grid (preview T={preview_T_eff}): L={L_prev}"
              + (" — CHUNKED (> max_sequence_length)"
                 if L_prev > cfg.max_sequence_length else " — single pass"),
              flush=True)

    model = ElasticTok(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ElasticTok: {n_params/1e6:.2f}M params  "
          f"(config={args.config_name}, bottleneck={cfg.bottleneck_type})",
          flush=True)

    lpips_fn = _build_lpips(device) if cfg.lpips_loss_ratio > 0 else None

    # -- Overfit mode: load a single clip from video, no generator --
    overfit_clip = None  # (1, T, 3, H, W) in [0,1]
    if args.overfit_video:
        if not os.path.isfile(args.overfit_video):
            raise SystemExit(f"overfit-video not found: {args.overfit_video}")
        print(f"  [overfit] loading {args.overfit_video}  "
              f"T={args.T} skip={args.overfit_frame_skip}", flush=True)
        frames = _decode_video_frames(
            args.overfit_video, int(args.overfit_frame_skip),
            args.T, args.W, args.H)
        if len(frames) < args.T:
            if not frames:
                raise SystemExit("overfit-video decoded 0 frames")
            while len(frames) < args.T:
                frames.append(frames[-1])
        arr = np.stack(frames[:args.T]).astype(np.float32) / 255.0  # (T, H, W, 3)
        overfit_clip = torch.from_numpy(arr).permute(
            0, 3, 1, 2).unsqueeze(0).to(device).contiguous()   # (1, T, 3, H, W)
        # Expand to batch size if batch > 1 — identical clips, but the
        # encoder's random mask still differs per-sample.
        if args.batch_size > 1:
            overfit_clip = overfit_clip.expand(
                args.batch_size, -1, -1, -1, -1).contiguous()
        print(f"  [overfit] cached clip {tuple(overfit_clip.shape)}",
              flush=True)

    # -- Generator (skipped in overfit mode) --
    gen = None
    pool_kwargs = None
    if overfit_clip is None:
        gen = VAEpp0rGenerator(
            height=args.H, width=args.W, device=str(device),
            bank_size=args.bank_size, n_base_layers=args.n_layers)
        bank_dir = os.path.join(PROJECT_ROOT, "bank")
        root_shapes = [f for f in os.listdir(bank_dir)
                        if f.startswith("shapes_") and f.endswith(".pt")] \
            if os.path.isdir(bank_dir) else []
        if root_shapes:
            print(f"  [gen] using root bank {bank_dir}", flush=True)
            gen.setup_dynamic_bank(bank_dir, working_size=args.bank_size,
                                    refresh_interval=50)
            gen.build_base_layers()
        else:
            print(f"  [gen] no shapes in {bank_dir} — building fresh",
                  flush=True)
            os.makedirs(bank_dir, exist_ok=True)
            gen.build_banks()
            try:
                gen.save_to_bank_dir(bank_dir)
            except Exception as e:
                print(f"  [gen] could not save bank: {e}", flush=True)
        pool_kwargs = dict(
            use_fluid=True, use_ripple=True, use_shake=True,
            use_kaleido=True, fast_transform=True, use_flash=True,
            use_palette_cycle=True, use_text=True, use_signage=True,
            use_particles=True, use_raymarch=True, sphere_dip=True,
            use_arcade=True, use_glitch=True, use_chromatic=True,
            use_scanlines=True, use_fire=True, use_vortex=True,
            use_starfield=True, use_eq=True,
        )
        gen.build_motion_pool(
            n_clips=args.pool_size, T=args.T, random_mix=True,
            **pool_kwargs)
        gen._train_pool_kwargs = pool_kwargs
        gen._train_random_mix = True
        if args.disco:
            gen.disco_quadrant = True
        print(f"  [gen] pool={len(gen._recipe_pool)} "
              f"disco={gen.disco_quadrant}", flush=True)

    # -- Optimizer (ref: AdamW, lr=1e-4, end_lr=1e-4, warmup=2000) --
    opt = torch.optim.AdamW(model.parameters(),
                             lr=float(args.lr),
                             weight_decay=float(args.weight_decay),
                             betas=(0.9, 0.999))
    if args.lr_warmup > 0:
        w = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0,
            total_iters=int(args.lr_warmup))
        c = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.total_steps - int(args.lr_warmup)),
            eta_min=float(args.end_lr))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[w, c], milestones=[int(args.lr_warmup)])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps, eta_min=float(args.end_lr))

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(amp_dtype == torch.float16))

    # -- Resume --
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if not args.fresh_opt and "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
                sched.load_state_dict(ckpt["scheduler"])
            except Exception:
                print("  fresh optimizer (state mismatch)", flush=True)
        start_step = int(ckpt.get("step", 0))
        print(f"  resumed from {args.resume} at step {start_step}",
              flush=True)

    keeps = sorted({min(int(k), cfg.max_toks)
                    for k in args.keeps.split(",")})
    print(f"  preview keeps = {keeps}", flush=True)

    # -- Initial preview (step 0) --
    pT = int(args.preview_T) if args.preview_T else args.T
    if args.preview_every > 0:
        save_preview(model, gen, str(logdir), start_step, device,
                     pT=pT, keeps=keeps, patch_size=patch_size,
                     max_toks=cfg.max_toks, block_size=cfg.max_toks,
                     preview_image=args.preview_image,
                     preview_frame_skip=int(args.preview_frame_skip),
                     overfit_clip=overfit_clip,
                     H=args.H, W=args.W,
                     haar_levels=haar_levels,
                     overfit_video_path=args.overfit_video,
                     overfit_frame_skip=int(args.overfit_frame_skip))

    # -- Auxiliary real-video dataset (optional) --
    # If --aux-data-dir is set, build an iterable DataLoader that
    # streams random T-frame windows in parallel via ffmpeg. One
    # single-clip stream that training consumes `batch_size` times per
    # step; DataLoader(num_workers=K) parallelizes decode.
    # Skipped entirely in overfit mode — the overfit clip overrides
    # all other data sources, so there's no point probing / loading
    # the aux dataset.
    aux_iter = None
    aux_frac = 0.0
    if overfit_clip is not None and args.aux_data_dir:
        print(f"  [aux-data] skipped: overfit mode active",
              flush=True)
    elif args.aux_data_dir and os.path.isdir(args.aux_data_dir):
        from training.aux_data import (
            AuxVideoDataset, aux_single_collate)
        aux_ds = AuxVideoDataset(
            args.aux_data_dir, H=args.H, W=args.W, T=args.T,
            recursive=bool(int(args.aux_data_recursive)))
        aux_loader = torch.utils.data.DataLoader(
            aux_ds, batch_size=1,
            num_workers=int(args.aux_data_workers),
            persistent_workers=(int(args.aux_data_workers) > 0),
            collate_fn=aux_single_collate)
        aux_iter = iter(aux_loader)
        aux_frac = max(0.0, min(1.0, float(args.aux_data_mix)))
        n_aux = int(args.batch_size * aux_frac)
        print(f"  [aux-data] enabled: mix={aux_frac:.2f} "
              f"-> {n_aux}/{args.batch_size} clips/step from aux, "
              f"workers={args.aux_data_workers}", flush=True)
    elif args.aux_data_dir:
        print(f"  [aux-data] WARNING: --aux-data-dir="
              f"{args.aux_data_dir} not a directory — ignored",
              flush=True)

    # -- Prefill bank (read-only) --
    # The training tab only CONSUMES a pre-built bank. Building /
    # growing the bank happens in the Data -> Prefill Bank tab
    # (subprocess: training.build_prefill). If the dir has no clips,
    # we warn and disable. Skipped in overfit mode for the same
    # reason the aux path is — overfit overrides all batch sources.
    prefill_bank = None
    prefill_frac = 0.0
    if overfit_clip is not None and args.prefill_dir:
        print(f"  [prefill] skipped: overfit mode active",
              flush=True)
    elif args.prefill_dir:
        pfdir = args.prefill_dir
        if not os.path.isabs(pfdir):
            pfdir = str(logdir / pfdir)
        if os.path.isdir(pfdir):
            from training.prefill_bank import PrefillBank
            prefill_bank = PrefillBank(
                pfdir, H=args.H, W=args.W, T=args.T, verbose=False)
            n_in_bank = prefill_bank.count()
            if n_in_bank == 0:
                print(f"  [prefill] WARNING: dir {pfdir} is empty. "
                      f"Build the bank via the Prefill Bank tab first.",
                      flush=True)
                prefill_bank = None
            else:
                prefill_frac = max(
                    0.0, min(1.0, float(args.prefill_mix)))
                if aux_frac + prefill_frac > 1.0:
                    over = aux_frac + prefill_frac
                    print(f"  [prefill] WARN: aux_mix({aux_frac:.2f}) + "
                          f"prefill_mix({prefill_frac:.2f}) = "
                          f"{over:.2f} > 1.0. Clamping prefill to "
                          f"{1.0 - aux_frac:.2f}.", flush=True)
                    prefill_frac = max(0.0, 1.0 - aux_frac)
                n_prefill = int(args.batch_size * prefill_frac)
                print(f"  [prefill] enabled: dir={pfdir}  "
                      f"mix={prefill_frac:.2f} -> {n_prefill}/"
                      f"{args.batch_size} clips/step  "
                      f"(bank has {n_in_bank} clips)", flush=True)
        else:
            print(f"  [prefill] WARNING: --prefill-dir={pfdir} is not "
                  f"a directory — prefill disabled.", flush=True)

    # -- Training loop --
    model.train()
    t0 = time.time()
    loop_start_step = start_step
    loss_ema = None
    stop_sentinel = logdir / ".stop"
    try:
        if stop_sentinel.exists():
            stop_sentinel.unlink()
    except Exception:
        pass

    for step in range(start_step, args.total_steps):
        if _stop_requested or stop_sentinel.exists():
            print(f"[Stopping at step {step}]", flush=True)
            try:
                if stop_sentinel.exists():
                    stop_sentinel.unlink()
            except Exception:
                pass
            break

        # -- Data: overfit > aux + prefill + generator pool mix --
        # Overfit wins unconditionally. Otherwise the batch is a blend:
        #   n_aux     clips from aux-data loader
        #   n_prefill clips from prefill bank
        #   n_gen     clips from gen.generate_from_pool (honors disco)
        # Shares come from --aux-data-mix / --prefill-mix; generator
        # gets whatever's left. Allocation is integer so we use floor
        # and give the remainder to the generator.
        if overfit_clip is not None:
            clips = overfit_clip
        else:
            B = args.batch_size
            n_aux = int(B * aux_frac) if aux_iter is not None else 0
            n_prefill = (int(B * prefill_frac)
                         if prefill_bank is not None else 0)
            # Clamp if mixes overallocate (shouldn't happen post-
            # clamping earlier, but be safe).
            if n_aux + n_prefill > B:
                n_prefill = max(0, B - n_aux)
            n_gen = B - n_aux - n_prefill
            parts = []
            if n_aux > 0:
                aux_parts = [next(aux_iter) for _ in range(n_aux)]
                parts.append(
                    (torch.stack(aux_parts).float() / 255.0).to(device))
            if n_prefill > 0:
                pf = prefill_bank.sample_batch(n_prefill)
                if pf is None:
                    # Bank emptied mid-run (unlikely); fall back to gen.
                    n_gen += n_prefill
                else:
                    parts.append(pf.to(device))
            if n_gen > 0:
                parts.append(gen.generate_from_pool(n_gen).to(device))
            clips = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        # [0,1] -> [-1,1]  (ref train.py: batch/127.5 - 1 from uint8)
        clip_m11 = clips * 2 - 1

        # Optional Haar pre-compression (per-frame 2D).
        # Model operates on the post-Haar tensor; recon comes out in
        # post-Haar space and MSE is computed there (still a pixel-space
        # loss — Haar is orthonormal, so MSE on post-Haar == MSE on RGB
        # up to a known constant).
        if haar_levels > 0:
            clip_h = haar_down_video(clip_m11, haar_levels)
        else:
            clip_h = clip_m11

        # Patchify -> (B, L, patch_dim)
        vision = patchify(clip_h, patch_size)
        B = vision.shape[0]
        L = vision.shape[1]

        # Elastic tail-drop mask. block_size = cfg.max_toks (required by
        # the model's block-causal attention); keep upper bound is the
        # separately-configurable keep_upper_eff (defaults to block size
        # unless --keep-upper was set).
        enc_mask = elastic_mask(B, L, cfg.max_toks,
                                 cfg.min_toks, keep_upper_eff, device)
        att_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon, stats = model(
                vision, enc_mask, att_mask, seg_ids, pos_ids,
                training=True)
            # ref: recon_loss = (recon - vision)**2 .mean()
            recon_loss = F.mse_loss(recon.float(), vision.float())

            losses = {"recon": recon_loss.item()}

            # LPIPS on one random frame — needs pixel-space RGB, so
            # Haar-up both recon and vision patches first.
            lpips_loss = None
            if cfg.lpips_loss_ratio > 0 and lpips_fn is not None:
                lp = lpips_on_block_frame(
                    recon.float(), vision.float(),
                    patch_size=patch_size,
                    frames_per_block=args.T,  # 1 block per clip here
                    resolution_h=args.H // (2 ** haar_levels),
                    resolution_w=args.W // (2 ** haar_levels),
                    lpips_fn=lpips_fn,
                    haar_levels=haar_levels,
                    out_H=args.H, out_W=args.W)
                lpips_loss = lp
                losses["lpips"] = lp.item()

            total_recon = recon_loss
            if lpips_loss is not None:
                total_recon = total_recon + cfg.lpips_loss_ratio * lpips_loss

            aux = stats.get("aux_loss",
                             torch.zeros((), device=device))
            total = total_recon + aux
            losses["aux"] = float(aux.item()) if aux.ndim == 0 else 0.0

        if amp_dtype == torch.float16:
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.grad_clip))
            opt.step()
        sched.step()

        tv = float(total.item())
        loss_ema = tv if loss_ema is None else 0.98 * loss_ema + 0.02 * tv

        if step % args.log_every == 0:
            lr = opt.param_groups[0]["lr"]
            dt = time.time() - t0
            steps_run = max(1, step - loop_start_step)
            sps = steps_run / max(dt, 1e-6)
            eta_s = (args.total_steps - step) / max(sps, 1e-6)
            eta = (f"{eta_s/3600:.1f}h" if eta_s > 3600
                   else f"{eta_s/60:.0f}m")
            parts = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
            print(f"  [{step:6d}/{args.total_steps}]  total={tv:.4f} "
                  f"ema={loss_ema:.4f}  {parts}  lr={lr:.2e}  "
                  f"{sps:.2f} sps  ETA {eta}", flush=True)

        if (args.preview_every > 0 and step % args.preview_every == 0
                and step != start_step):
            save_preview(model, gen, str(logdir), step, device,
                         pT=pT, keeps=keeps, patch_size=patch_size,
                         max_toks=cfg.max_toks, block_size=cfg.max_toks,
                         preview_image=args.preview_image,
                         preview_frame_skip=int(args.preview_frame_skip),
                         overfit_clip=overfit_clip,
                         H=args.H, W=args.W,
                         haar_levels=haar_levels,
                         overfit_video_path=args.overfit_video,
                         overfit_frame_skip=int(args.overfit_frame_skip))

        if (args.save_every > 0 and step % args.save_every == 0
                and step != start_step):
            d = dict(model=model.state_dict(),
                      optimizer=opt.state_dict(),
                      scheduler=sched.state_dict(),
                      step=step + 1, args=vars(args),
                      config=cfg.to_dict())
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

        # Track resume point AFTER the iteration body completes.
        # If the loop is interrupted, `next_step` holds the last finished
        # step + 1 (= where to resume). If the loop never entered, it
        # stays at start_step. The final save below uses this instead of
        # the old `step=args.total_steps`, which previously stamped
        # latest.pt as "done" even on a killed mid-run session.
        next_step = step + 1

    # Save final — uses the actual last-completed step for resume.
    if 'next_step' not in locals():
        next_step = start_step
    d = dict(model=model.state_dict(),
              optimizer=opt.state_dict(),
              scheduler=sched.state_dict(),
              step=next_step, args=vars(args),
              config=cfg.to_dict())
    named_final = _ckpt_name(next_step)
    torch.save(d, str(logdir / named_final))
    torch.save(d, str(logdir / "latest.pt"))
    print(f"Done at step {next_step}/{args.total_steps}.  "
          f"saved {named_final}  final ema={loss_ema}", flush=True)


def main():
    p = argparse.ArgumentParser()
    # Arch
    # Import CONFIGS keys dynamically so this list can't drift from the
    # actual CONFIGS dict in core/elastictok/model.py.
    from core.elastictok.model import CONFIGS as _ET_CONFIGS
    p.add_argument("--config-name", default="debug",
                   choices=list(_ET_CONFIGS.keys()))
    p.add_argument("--haar-levels", type=int, default=0,
                   help="Per-frame 2D Haar pre-compression levels. Each "
                        "level halves H,W and quadruples channels (purely "
                        "lossless + invertible, no params). Reduces the "
                        "patch token count by 4**N. H,W must be divisible "
                        "by 2**N.")
    p.add_argument("--patch-size", default="1,16,16",
                   help="Tp,Hp,Wp (ElasticTok default (1,8,8)).")
    p.add_argument("--bottleneck-type", default="vae",
                   choices=["vae", "fsq"])
    p.add_argument("--fsq-levels", default="8,8,8,5,5,5")
    p.add_argument("--vae-bottleneck-dim", type=int, default=8)
    p.add_argument("--max-sequence-length", type=int, default=4096)
    p.add_argument("--max-toks", type=int, default=3680)
    p.add_argument("--min-toks", type=int, default=128)
    p.add_argument("--keep-upper", type=int, default=0,
                   help="Per-block keep upper bound. 0 (default) uses "
                        "max_toks. Decouples elastic-mask range from the "
                        "attention block size so you can train with a "
                        "fixed small keep (min_toks=keep_upper) even when "
                        "L is not divisible by that keep value.")
    p.add_argument("--frames-per-block", type=int, default=4)
    p.add_argument("--lpips-loss-ratio", type=float, default=0.1)
    # Clip geometry
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--H", type=int, default=368)
    p.add_argument("--W", type=int, default=640)
    # Data
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--disco", action="store_true")
    # Optim (ref: lr=1e-4, end_lr=1e-4, warmup=2000, wd=1e-4, clip 1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", default="1e-4")
    p.add_argument("--end-lr", default="1e-4")
    p.add_argument("--lr-warmup", type=int, default=2000)
    p.add_argument("--weight-decay", default="1e-4")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    # I/O
    p.add_argument("--logdir", default="synthyper_elastictok_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--keeps", default="128,512,2048")
    p.add_argument("--preview-image", default=None)
    p.add_argument("--preview-frame-skip", type=int, default=0)
    p.add_argument("--preview-T", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    # -- Overfit mode --
    # Bypass the generator entirely and train on a single cached
    # T-frame segment extracted from the given video. Preview shows
    # the same clip's reconstruction at multiple keep budgets.
    p.add_argument("--overfit-video", default=None,
                   help="If set, train on a single T-frame segment of "
                        "this video indefinitely. Generator / bank / "
                        "pool setup is skipped.")
    p.add_argument("--overfit-frame-skip", type=int, default=0,
                   help="Starting frame offset in --overfit-video.")
    # Auxiliary real-video dataset. Walks a directory for video files,
    # samples random T-frame windows stretched to (W, H). Fully
    # replaces the generator when --aux-data-mix == 1.0; partially
    # mixes otherwise (first floor(B*mix) of each batch from aux,
    # remainder from generator).
    p.add_argument("--aux-data-dir", default="",
                   help="Directory of real video files to train on. "
                        "Empty string disables.")
    p.add_argument("--aux-data-mix", type=float, default=1.0,
                   help="Fraction of each batch drawn from aux-data "
                        "vs generator. 1.0 = all aux, 0.0 = none.")
    p.add_argument("--aux-data-workers", type=int, default=4,
                   help="DataLoader num_workers for parallel ffmpeg "
                        "decode of aux clips.")
    p.add_argument("--aux-data-recursive", type=int, default=1,
                   help="If 1, recurse subdirectories of --aux-data-dir.")
    # Prefill bank: READ-ONLY from the training script's POV.
    # Build/grow/trim the bank via the Prefill Bank tab (Data ->
    # Prefill Bank), which subprocesses training.build_prefill.
    p.add_argument("--prefill-dir", default="",
                   help="Directory containing a pre-built clip bank. "
                        "Empty = off.")
    p.add_argument("--prefill-mix", type=float, default=0.5,
                   help="Fraction of each batch drawn from the prefill "
                        "bank. 1.0 = all prefill, 0.0 = off.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

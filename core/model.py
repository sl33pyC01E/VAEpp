#!/usr/bin/env python3
"""Mini VAE — causal temporal video autoencoder.

Architecture follows TAEHV (madebyollin/taehv) with custom channel widths.
Building blocks (conv, Clamp, MemBlock, TPool, TGrow) inlined below.

Input:  (N, T, C_in, H, W)
Latent: (N, T', C_lat, H/8, W/8)  — 8x spatial, up to 4x temporal
Output: (N, T'', C_out, H, W)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from tqdm.auto import tqdm


# -- TAEHV building blocks (from madebyollin/taehv) ---------------------------

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True),
                                  conv(n_out, n_out), nn.ReLU(inplace=True),
                                  conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    assert x.ndim == 5
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)
    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, MemBlock):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)

def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, MemBlock):
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N * b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(xt.view(NT // b.stride, b.stride * C, H, W).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return None

def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    assert x.ndim == 5
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar)
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)


# -- Model --------------------------------------------------------------------

OutputChannels = namedtuple("OutputChannels", ("rgb", "depth", "flow", "semantic"))


class MiniVAE(nn.Module):
    """Causal temporal VAE, 9ch.

    RGB(3) + depth(1) + flow(2) + semantic(3) = 9ch.
    TAEHV architecture. Continuous latent (FSQ added in Phase 2).
    """

    def __init__(
        self,
        latent_channels=16,
        image_channels=9,
        output_channels=9,
        encoder_channels=64,
        decoder_channels=(192, 96, 64),
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=(False, True, True),
        checkpoint_path=None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.image_channels = image_channels
        self.output_channels = output_channels

        ec = encoder_channels

        # Encoder: RGB -> latent
        self.encoder = nn.Sequential(
            conv(image_channels, ec), nn.ReLU(inplace=True),
            TPool(ec, 2 if encoder_time_downscale[0] else 1),
            conv(ec, ec, stride=2, bias=False),
            MemBlock(ec, ec), MemBlock(ec, ec), MemBlock(ec, ec),
            TPool(ec, 2 if encoder_time_downscale[1] else 1),
            conv(ec, ec, stride=2, bias=False),
            MemBlock(ec, ec), MemBlock(ec, ec), MemBlock(ec, ec),
            TPool(ec, 1 if not encoder_time_downscale[2] else 2),
            conv(ec, ec, stride=2, bias=False),
            MemBlock(ec, ec), MemBlock(ec, ec), MemBlock(ec, ec),
            conv(ec, latent_channels),
        )

        # Decoder: latent -> multi-modal output
        n_f = list(decoder_channels)  # [192, 96, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(latent_channels, n_f[0]), nn.ReLU(inplace=True),
            # Stage 1: bottleneck resolution (48x84)
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[0], n_f[1], bias=False),
            # Stage 2: mid resolution (96x168)
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[1], n_f[2], bias=False),
            # Stage 3: high resolution (192x336)
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1),
            conv(n_f[2], n_f[2], bias=False),
            # Output head (384x672)
            nn.ReLU(inplace=True),
            conv(n_f[2], output_channels),
        )

        # Computed properties
        self.t_downscale = 1
        for m in self.encoder:
            if isinstance(m, TPool) and m.stride > 1:
                self.t_downscale *= m.stride

        self.t_upscale = 1
        for m in self.decoder:
            if isinstance(m, TGrow) and m.stride > 1:
                self.t_upscale *= m.stride

        self.frames_to_trim = self.t_upscale - 1

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(
                checkpoint_path, map_location="cpu", weights_only=True))

    @classmethod
    def from_pretrained_taehv(cls, taehv_path="taehv/taehv.pth", **kwargs):
        """Create MiniVAE initialized from pretrained 3ch TAEHV weights.

        Loads TAEHV weights, copies matching internal layers (MemBlocks,
        stride convs), inflates first encoder conv (3→6ch, zero-init semantic)
        and last decoder conv (3→6ch, zero-init semantic). Latent projection
        layers are randomly initialized if latent_channels differs from TAEHV's 16.
        """
        # Build with TAEHV-matching defaults for max weight transfer
        model = cls(
            latent_channels=kwargs.get("latent_channels", 16),
            image_channels=9,
            output_channels=9,
            encoder_channels=64,
            decoder_channels=(256, 128, 64),
            **{k: v for k, v in kwargs.items()
               if k not in ("latent_channels", "encoder_channels", "decoder_channels")},
        )

        # Load TAEHV state dict
        taehv_sd = torch.load(taehv_path, map_location="cpu", weights_only=True)
        model_sd = model.state_dict()

        loaded, skipped = 0, 0
        for key, param in taehv_sd.items():
            if key not in model_sd:
                skipped += 1
                continue
            target = model_sd[key]
            if param.shape == target.shape:
                model_sd[key] = param
                loaded += 1
            elif key == "encoder.0.weight":
                # First conv: (64, 3, 3, 3) → (64, 9, 3, 3)
                model_sd[key][:, :3] = param
                loaded += 1
            elif key == "decoder.22.weight":
                # Last conv: (3, 64, 3, 3) → (9, 64, 3, 3)
                model_sd[key][:3] = param
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(model_sd)
        print(f"TAEHV init: {loaded} layers loaded, {skipped} skipped")
        return model

    def encode_video(self, x, parallel=True, show_progress_bar=False):
        """Encode RGB video to latents.

        Args:
            x: (N, T, C, H, W) RGB tensor in [0, 1], C=image_channels.
            parallel: process all frames at once (fast, more memory).

        Returns:
            (N, T', latent_channels, H/8, W/8) latent tensor.
        """
        assert x.ndim == 5, f"Expected NTCHW, got {x.ndim}D"
        # Pad T to multiple of t_downscale
        if x.shape[1] % self.t_downscale != 0:
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(
            self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        """Decode latents to multi-modal output.

        Args:
            x: (N, T, latent_channels, H, W) latent tensor.
            parallel: process all frames at once (fast, more memory).

        Returns:
            (N, T'', output_channels, H*8, W*8) output tensor.
            T'' = T * t_upscale - frames_to_trim.
        """
        assert x.ndim == 5, f"Expected NTCHW, got {x.ndim}D"
        x = apply_model_with_memblocks(
            self.decoder, x, parallel, show_progress_bar)
        return x[:, self.frames_to_trim:]

    def forward(self, x):
        """Full encode-decode pass.

        Args:
            x: (N, T, 9, H, W) tensor in [0, 1].
               RGB(3) + depth(1) + flow(2) + semantic(3).

        Returns:
            (recon, latent) where recon is (N, T'', 9, H, W) and
            latent is (N, T', latent_channels, H/8, W/8).
        """
        latent = self.encode_video(x)
        recon = self.decode_video(latent)
        return recon, latent

    def postprocess_output(self, x):
        """Split and postprocess decoder output channels.

        Args:
            x: (N, T, 9, H, W) raw decoder output.

        Returns:
            OutputChannels namedtuple with:
                rgb: (N, T, 3, H, W) clamped to [0, 1]
                depth: (N, T, 1, H, W) clamped to [0, 1]
                flow: (N, T, 2, H, W)
                semantic: (N, T, 3, H, W) clamped to [0, 1]
        """
        return OutputChannels(
            rgb=x[:, :, :3].clamp(0, 1),
            depth=x[:, :, 3:4].clamp(0, 1),
            flow=x[:, :, 4:6],
            semantic=x[:, :, 6:9].clamp(0, 1),
        )

    def param_count(self):
        """Return parameter counts for encoder, decoder, and total."""
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


class StreamingMiniVAE(nn.Module):
    """Streaming wrapper for real-time encode/decode.

    Encode-decode (video-to-video) usage:
        streaming = StreamingMiniVAE(model)
        for frame in video_frames:
            latent = streaming.encode(frame_tensor)
            decoded = streaming.decode(latent)
            if decoded is not None:
                display(decoded)
        for frame in streaming.flush():
            display(frame)

    Decode-only (world model) usage:
        streaming = StreamingMiniVAE(model)
        while running:
            latent = world_model.step()
            frame = streaming.decode(latent)
            while frame is not None:
                display(frame)
                frame = streaming.decode()
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset()

    def reset(self):
        """Reset all internal state for a new stream."""
        self.encoder_work_queue = []
        self.encoder_memory = [None] * len(self.model.encoder)
        self.decoder_work_queue = []
        self.decoder_memory = [None] * len(self.model.decoder)
        self.n_frames_encoded = 0
        self.n_frames_decoded = 0
        self._last_encoder_input_frame = None

    def encode(self, x=None):
        """Feed an input frame and try to produce an encoder output.

        Args:
            x: (N, 1, C, H, W) RGB frame, or None to process pending work.

        Returns:
            (N, 1, latent_channels, H/8, W/8) latent, or None.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.model.image_channels
            self._last_encoder_input_frame = x[:, -1:]
            self.encoder_work_queue.extend(
                TWorkItem(xt, 0) for xt in x.unbind(1))
            self.n_frames_encoded += x.shape[1]
        return apply_model_with_memblocks_sequential_single_step(
            self.model.encoder, self.encoder_memory, self.encoder_work_queue)

    def decode(self, x=None):
        """Feed a latent and try to produce a decoded frame.

        Args:
            x: (N, 1, latent_channels, H, W) latent, or None.

        Returns:
            (N, 1, output_channels, H*8, W*8) decoded frame, or None.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.model.latent_channels
            self.decoder_work_queue.extend(
                TWorkItem(xt, 0) for xt in x.unbind(1))
        while True:
            xt = apply_model_with_memblocks_sequential_single_step(
                self.model.decoder, self.decoder_memory,
                self.decoder_work_queue)
            if xt is None:
                return None
            self.n_frames_decoded += 1
            # Skip startup frames (causal alignment)
            if self.n_frames_decoded <= self.model.frames_to_trim:
                continue
            return xt

    def flush_encoder(self):
        """Pad and drain remaining latents from the encoder."""
        latents = []
        if (self._last_encoder_input_frame is not None
                and self.n_frames_encoded % self.model.t_downscale != 0):
            n_pad = (self.model.t_downscale
                     - self.n_frames_encoded % self.model.t_downscale)
            for _ in range(n_pad):
                lat = self.encode(self._last_encoder_input_frame)
                if lat is not None:
                    latents.append(lat)
        while (lat := self.encode()) is not None:
            latents.append(lat)
        return latents

    def flush_decoder(self):
        """Drain all remaining decoded frames."""
        frames = []
        while (frame := self.decode()) is not None:
            frames.append(frame)
        return frames

    def flush(self):
        """Flush encoder then decoder, returning all remaining frames."""
        frames = []
        for latent in self.flush_encoder():
            frame = self.decode(latent)
            if frame is not None:
                frames.append(frame)
        frames.extend(self.flush_decoder())
        return frames

"""Prefilled clip bank for ElasticTok training.

Generate clips once via the VAEpp0r generator, cache them on disk as
H.264 mp4 files (yuv420p, crf 18), sample during training. Mp4 gets
us ~30-100x compression vs raw uint8 tensors, so 8000 clips is low
single-digit GB instead of ~23 GB.

Storage choice recap:
  raw .pt uint8:  2.83 MB / 4-frame 368x640 clip -> 22.6 GB / 8k
  mp4 yuv420p 18: ~30-80 KB / same clip         -> ~0.3-0.6 GB / 8k

Load path uses ffmpeg to decode; ~10-30 ms per clip on a modern CPU,
small vs. training-step time.

Usage:
    bank = PrefillBank(root="prefill_bank", H=368, W=640, T=4)
    bank.fill_to(8000, gen)          # at startup
    clips = bank.sample_batch(B)     # each training step
    bank.maybe_refresh(step, every=5000, n_add=1000, max_size=50000,
                       gen=gen)
"""

from __future__ import annotations

import os
import random
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


_CLIP_GLOB = "clip_*.mp4"


class PrefillBank:
    def __init__(self, root: str | os.PathLike, H: int, W: int, T: int,
                 verbose: bool = True,
                 crf: int = 18,
                 preset: str = "veryfast"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.H = int(H)
        self.W = int(W)
        self.T = int(T)
        self.verbose = verbose
        self.crf = int(crf)
        self.preset = str(preset)
        self._seq = 0

    # ---- Bank state --------------------------------------------------
    def _file_list(self) -> List[Path]:
        return sorted(self.root.glob(_CLIP_GLOB),
                       key=lambda p: p.stat().st_mtime)

    def count(self) -> int:
        return len(list(self.root.glob(_CLIP_GLOB)))

    def _mkname(self) -> str:
        self._seq += 1
        return (f"clip_{int(time.time()*1e6):016d}_"
                f"{self._seq:08d}.mp4")

    # ---- Write (ffmpeg encode) --------------------------------------
    def add_clip(self, clip_uint8: torch.Tensor):
        """Append one clip. Expects (T, 3, H, W) uint8 on CPU.
        Encoded as H.264 yuv420p (crf=self.crf) mp4."""
        assert clip_uint8.dtype == torch.uint8, clip_uint8.dtype
        assert clip_uint8.shape == (self.T, 3, self.H, self.W), \
            f"expected {(self.T, 3, self.H, self.W)}, got " \
            f"{tuple(clip_uint8.shape)}"
        # (T, 3, H, W) -> (T, H, W, 3) raw bytes for rgb24 ffmpeg input
        arr = clip_uint8.permute(0, 2, 3, 1).contiguous().numpy()
        path = self.root / self._mkname()
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.W}x{self.H}",
            "-r", "30",
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            str(path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            proc.stdin.write(arr.tobytes())
            proc.stdin.close()
            _, err = proc.communicate(timeout=30)
            if proc.returncode != 0:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise RuntimeError(
                    f"ffmpeg encode failed ({proc.returncode}): "
                    f"{err.decode(errors='replace')[:200]}")
        finally:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

    def add_batch(self, clips_float: torch.Tensor):
        """Append a batch of clips. Expects (B, T, 3, H, W) float
        in [0, 1]. Per-clip encoding — ffmpeg subprocess per clip."""
        assert clips_float.dim() == 5, clips_float.shape
        assert clips_float.shape[1:] == (self.T, 3, self.H, self.W), \
            f"shape mismatch: {tuple(clips_float.shape)}"
        x = clips_float.clamp(0, 1).mul(255).to(
            torch.uint8).cpu().contiguous()
        for b in range(x.shape[0]):
            self.add_clip(x[b])

    # ---- Read (ffmpeg decode) ---------------------------------------
    def _decode(self, path: Path) -> Optional[torch.Tensor]:
        """Decode one mp4 into (T, 3, H, W) uint8 tensor. None on
        failure (the file may be half-written / corrupt)."""
        cmd = [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30)
            raw = result.stdout
            frame_bytes = self.H * self.W * 3
            n = len(raw) // frame_bytes
            if n < self.T:
                return None
            arr = np.frombuffer(
                raw, dtype=np.uint8,
                count=self.T * frame_bytes).reshape(
                    self.T, self.H, self.W, 3)
            # (T, H, W, 3) -> (T, 3, H, W)
            return torch.from_numpy(
                arr).permute(0, 3, 1, 2).contiguous()
        except Exception:
            return None

    def sample_batch(self, B: int) -> Optional[torch.Tensor]:
        """Random-sample B clips from the bank. Returns
        (B, T, 3, H, W) float in [0, 1] on CPU, or None if the bank
        is empty."""
        files = list(self.root.glob(_CLIP_GLOB))
        if not files:
            return None
        if B <= len(files):
            picks = random.sample(files, B)
        else:
            picks = [random.choice(files) for _ in range(B)]
        clips = []
        for p in picks:
            t = self._decode(p)
            if t is None:
                # Corrupt file — delete + retry once.
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
                return self.sample_batch(B)
            clips.append(t.float() / 255.0)
        return torch.stack(clips)

    # ---- Fill / refresh ---------------------------------------------
    def fill_to(self, target: int, gen, batch_size: int = 8,
                per_clip_log: bool = True,
                rolling_window: int = 100):
        """Generate clips via gen.generate_from_pool until count
        reaches `target`. No-op if already at or above.

        Per-clip log shows rolling-window rate (not cumulative) so the
        reported clips/s reflects RECENT throughput. A [batch] line
        prints once per generator batch with gen vs. disk (encode)
        time so you can see where the time goes.
        """
        from collections import deque

        cur = self.count()
        if cur >= target:
            if self.verbose:
                print(f"  [prefill] bank at {cur}/{target} — skip fill",
                      flush=True)
            return
        if self.verbose:
            print(f"  [prefill] filling bank {cur} -> {target} "
                  f"via generator (batch {batch_size}, mp4 crf="
                  f"{self.crf} preset={self.preset})...", flush=True)
        t0 = time.time()
        window = deque(maxlen=rolling_window)
        window.append(t0)
        while cur < target:
            n = min(batch_size, target - cur)
            t_gen0 = time.time()
            batch = gen.generate_from_pool(n).detach().cpu()
            t_gen1 = time.time()
            gen_dt = t_gen1 - t_gen0
            u8 = batch.clamp(0, 1).mul(255).to(torch.uint8).contiguous()
            t_enc0 = time.time()
            for b in range(u8.shape[0]):
                self.add_clip(u8[b])
                cur += 1
                now = time.time()
                window.append(now)
                if self.verbose and per_clip_log:
                    if len(window) >= 2:
                        span = window[-1] - window[0]
                        roll = (len(window) - 1) / max(span, 1e-6)
                    else:
                        roll = 0.0
                    remain = ((target - cur) / max(roll, 1e-6)
                              if roll > 0 else float("inf"))
                    print(f"  [prefill] {cur}/{target}  "
                          f"roll({len(window)-1})={roll:5.2f} clips/s  "
                          f"ETA {remain/60:6.1f}m", flush=True)
            t_enc1 = time.time()
            if self.verbose:
                print(f"  [prefill][batch] gen={gen_dt:5.2f}s  "
                      f"encode={t_enc1 - t_enc0:5.2f}s  "
                      f"({n} clips, "
                      f"{n / max(gen_dt + t_enc1 - t_enc0, 1e-6):5.2f} "
                      f"clips/s batch-local)", flush=True)
        if self.verbose:
            dt = time.time() - t0
            # Estimate bank size
            total_bytes = sum(
                f.stat().st_size for f in self.root.glob(_CLIP_GLOB))
            gb = total_bytes / (1024 ** 3)
            print(f"  [prefill] fill done: {target} clips in "
                  f"{dt/60:.1f}m  "
                  f"({target/dt:.1f} clips/s cumulative)  "
                  f"size={gb:.2f} GB",
                  flush=True)

    def _evict_to(self, max_size: int):
        files = self._file_list()
        while len(files) > max_size:
            old = files.pop(0)
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass

    def maybe_refresh(self, step: int, every: int, n_add: int,
                      max_size: int, gen,
                      batch_size: int = 8,
                      per_clip_log: bool = True,
                      rolling_window: int = 100) -> bool:
        """Every `every` steps, add `n_add` new clips and LRU-evict
        down to `max_size`. Returns True when a refresh fired."""
        from collections import deque

        if every <= 0 or n_add <= 0:
            return False
        if step == 0 or step % every != 0:
            return False
        t0 = time.time()
        cur = self.count()
        if self.verbose:
            print(f"  [prefill] refresh @ step {step}: "
                  f"adding {n_add}, cap {max_size} "
                  f"(current {cur})...", flush=True)
        window = deque(maxlen=rolling_window)
        window.append(t0)
        added = 0
        while added < n_add:
            n = min(batch_size, n_add - added)
            t_gen0 = time.time()
            batch = gen.generate_from_pool(n).detach().cpu()
            gen_dt = time.time() - t_gen0
            u8 = batch.clamp(0, 1).mul(255).to(torch.uint8).contiguous()
            t_enc0 = time.time()
            for b in range(u8.shape[0]):
                self.add_clip(u8[b])
                added += 1
                now = time.time()
                window.append(now)
                if self.verbose and per_clip_log:
                    if len(window) >= 2:
                        span = window[-1] - window[0]
                        roll = (len(window) - 1) / max(span, 1e-6)
                    else:
                        roll = 0.0
                    remain = ((n_add - added) / max(roll, 1e-6)
                              if roll > 0 else float("inf"))
                    print(f"  [prefill] +{added}/{n_add}  "
                          f"roll({len(window)-1})={roll:5.2f} clips/s  "
                          f"ETA {remain/60:6.1f}m", flush=True)
            enc_dt = time.time() - t_enc0
            if self.verbose:
                print(f"  [prefill][batch] gen={gen_dt:5.2f}s  "
                      f"encode={enc_dt:5.2f}s", flush=True)
        before_evict = self.count()
        self._evict_to(max_size)
        after = self.count()
        dt = time.time() - t0
        if self.verbose:
            print(f"  [prefill] refresh done: added {n_add} "
                  f"({before_evict} pre-evict -> {after})  "
                  f"in {dt:.1f}s", flush=True)
        return True

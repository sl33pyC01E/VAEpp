"""Auxiliary real-video dataset loader for ElasticTok training.

Walks a directory (optionally recursive) for supported video files,
ffprobes each once at startup for duration + fps, and yields random
T-frame windows as (T, 3, H, W) uint8 tensors stretched to exactly
(W, H). Designed to live behind a torch.utils.data.DataLoader with
`num_workers > 0` so ffmpeg decode runs in parallel with the training
GPU step.

Use case: swap out / mix with generator clips when you have a real
video dataset sitting on disk. See `--aux-data-*` flags in
train_elastictok.py.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


_MANIFEST_NAME = ".aux_manifest.json"


def aux_single_collate(batch):
    """DataLoader collate_fn for single-clip batches.

    Must be module-level (not a local lambda) so DataLoader worker
    processes on Windows can pickle it via `spawn`. With batch_size=1
    the `batch` arg is a length-1 list — we unwrap it so each
    `next(iter(loader))` call returns one `(T, 3, H, W)` tensor
    directly, not a list.
    """
    return batch[0]


_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")


def _probe_video(path: Path) -> Tuple[int, float] | None:
    """Return (n_frames, fps) for a video file, or None if unreadable.

    Uses `duration * fps` rather than `-count_frames` — the latter
    decodes every frame which is too slow on large datasets. Works for
    every container I've tested (mp4/mkv/webm/mov); videos with missing
    duration metadata are skipped.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate:format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        if len(lines) < 2:
            return None
        # ffprobe's default=nk=1 returns values in the order they were
        # requested: stream entries first, then format entries. So
        # lines[0] = r_frame_rate, lines[1] = duration.
        fps_str = lines[0]
        dur_str = lines[1]
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(fps_str)
        duration = float(dur_str)
        if fps <= 0 or duration <= 0:
            return None
        n_frames = int(duration * fps)
        return (n_frames, fps)
    except Exception:
        return None


def _decode_window(path: Path, start_time: float, T: int,
                   W: int, H: int) -> np.ndarray | None:
    """Decode T frames starting at `start_time` (seconds) from `path`,
    stretched to exactly (W, H). Returns (T, H, W, 3) uint8 or None on
    failure / short read."""
    try:
        # -ss BEFORE -i = input seek (fast, uses container index,
        # may land slightly off the keyframe, fine for training).
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-ss", f"{start_time:.3f}",
            "-i", str(path),
            "-frames:v", str(T),
            "-vf", f"scale={W}:{H}",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        raw = result.stdout
        frame_bytes = W * H * 3
        n = len(raw) // frame_bytes
        if n < T:
            return None
        arr = np.frombuffer(
            raw, dtype=np.uint8, count=T * frame_bytes).reshape(T, H, W, 3)
        return arr
    except Exception:
        return None


def _discover_videos(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        it = root.rglob("*")
    else:
        it = root.iterdir()
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS)


class AuxVideoDataset(IterableDataset):
    """Random T-frame windows from a directory of video files.

    Each iterator instance (including per-worker in a DataLoader) keeps
    a local RNG so workers don't sample identical files in lockstep.
    Yields single clips as `(T, 3, H, W)` uint8 tensors.
    """

    def __init__(self, root: str | os.PathLike, H: int, W: int, T: int,
                 recursive: bool = True,
                 skip_shorter_than: int | None = None,
                 max_probe: int | None = None,
                 verbose: bool = True,
                 probe_workers: int = 16,
                 use_manifest: bool = True):
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise RuntimeError(f"aux-data-dir not a directory: {root}")
        self.H = int(H)
        self.W = int(W)
        self.T = int(T)
        self.recursive = bool(recursive)
        self.skip_shorter = int(skip_shorter_than
                                 if skip_shorter_than is not None
                                 else self.T)

        files = _discover_videos(self.root, self.recursive)
        if verbose:
            print(f"  [aux-data] discovered {len(files)} video files under "
                  f"{self.root} (recursive={self.recursive})", flush=True)
        if max_probe is not None and max_probe > 0:
            files = files[:max_probe]

        # Manifest cache: probe once, persist result to
        # .aux_manifest.json at the dataset root. Next startup only
        # probes files whose mtime is newer than the cached entry.
        manifest_path = self.root / _MANIFEST_NAME
        manifest: dict = {}
        if use_manifest and manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                if verbose:
                    print(f"  [aux-data] manifest cache: loaded "
                          f"{len(manifest)} entries from "
                          f"{_MANIFEST_NAME}", flush=True)
            except Exception as e:
                if verbose:
                    print(f"  [aux-data] WARN: manifest unreadable "
                          f"({e}) — re-probing", flush=True)
                manifest = {}

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(self.root))
            except ValueError:
                return str(p)

        # Split into "cached" (manifest hit, mtime matches) and
        # "to_probe" (new / stale / never-seen).
        cached: List[Tuple[Path, int, float]] = []
        to_probe: List[Path] = []
        skipped_short = 0
        for f in files:
            try:
                mtime = f.stat().st_mtime
            except Exception:
                to_probe.append(f)
                continue
            entry = manifest.get(_rel(f))
            # 1-second tolerance for fs mtime granularity (FAT32 et al).
            if entry and entry.get("mtime", 0) >= mtime - 1.0:
                n_frames = int(entry.get("n_frames", 0))
                fps = float(entry.get("fps", 0.0))
                if n_frames < self.skip_shorter:
                    skipped_short += 1
                    continue
                if n_frames <= 0 or fps <= 0:
                    to_probe.append(f)
                    continue
                cached.append((f, n_frames, fps))
            else:
                to_probe.append(f)

        skipped_bad = 0
        probed_new: List[Tuple[Path, int, float]] = []
        if to_probe:
            if verbose:
                print(f"  [aux-data] cached={len(cached)}  "
                      f"probing {len(to_probe)} files "
                      f"with {probe_workers} workers...", flush=True)
            t0 = time.time()
            done = 0
            with ThreadPoolExecutor(
                    max_workers=max(1, probe_workers)) as ex:
                futures = {ex.submit(_probe_video, f): f
                           for f in to_probe}
                for fut in as_completed(futures):
                    f = futures[fut]
                    info = fut.result()
                    done += 1
                    if info is None:
                        skipped_bad += 1
                        continue
                    n_frames, fps = info
                    if n_frames < self.skip_shorter:
                        skipped_short += 1
                        # Still cache the result so we don't re-probe
                        # (a too-short file stays too-short).
                        if use_manifest:
                            try:
                                manifest[_rel(f)] = {
                                    "n_frames": int(n_frames),
                                    "fps": float(fps),
                                    "mtime": f.stat().st_mtime,
                                }
                            except Exception:
                                pass
                        continue
                    probed_new.append((f, n_frames, fps))
                    if use_manifest:
                        try:
                            manifest[_rel(f)] = {
                                "n_frames": int(n_frames),
                                "fps": float(fps),
                                "mtime": f.stat().st_mtime,
                            }
                        except Exception:
                            pass
                    if verbose and done % 500 == 0:
                        rate = done / max(time.time() - t0, 1e-6)
                        eta = (len(to_probe) - done) / max(rate, 1e-6)
                        print(f"  [aux-data] probed {done}/"
                              f"{len(to_probe)}  "
                              f"{rate:.0f}/s  ETA {eta:.0f}s",
                              flush=True)
            dt = time.time() - t0
            if verbose:
                print(f"  [aux-data] probe done: "
                      f"{len(to_probe)} files in {dt:.1f}s "
                      f"({len(to_probe)/max(dt,1e-6):.0f} probes/s)",
                      flush=True)

            # Persist the updated manifest.
            if use_manifest:
                try:
                    tmp = manifest_path.with_suffix(".json.tmp")
                    with open(tmp, "w") as f:
                        json.dump(manifest, f)
                    os.replace(tmp, manifest_path)
                    if verbose:
                        print(f"  [aux-data] manifest saved: "
                              f"{len(manifest)} entries", flush=True)
                except Exception as e:
                    if verbose:
                        print(f"  [aux-data] WARN: manifest save "
                              f"failed: {e}", flush=True)

        probed = cached + probed_new
        if not probed:
            raise RuntimeError(
                f"aux-data: no usable videos in {root} "
                f"(discovered={len(files)}, probe-failed={skipped_bad}, "
                f"too-short={skipped_short})")
        self._probed = probed
        if verbose:
            total_frames = sum(n for _, n, _ in probed)
            total_hours = sum(n / max(fps, 1e-6)
                              for _, n, fps in probed) / 3600.0
            print(f"  [aux-data] usable: {len(probed)} files  "
                  f"total {total_frames:,} frames  "
                  f"({total_hours:.2f} source-hours)  "
                  f"skipped: {skipped_bad} bad, {skipped_short} too-short",
                  flush=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed = None
            wid = 0
        else:
            seed = (worker_info.seed or 0) & 0xFFFFFFFF
            wid = worker_info.id
        rng = random.Random(seed if seed is not None else None)
        # Each worker gets a rotating offset into the file list so the
        # first few files don't dominate the batch when num_workers is
        # small.
        start_off = wid * 17

        max_retry = 6
        while True:
            retry = 0
            while retry < max_retry:
                idx = (start_off + rng.randrange(len(self._probed))) \
                    % len(self._probed)
                path, n_frames, fps = self._probed[idx]
                max_start = max(0, n_frames - self.T - 1)
                start_frame = rng.randint(0, max_start)
                start_time = start_frame / max(fps, 1e-6)
                arr = _decode_window(path, start_time,
                                      self.T, self.W, self.H)
                if arr is not None:
                    # (T, H, W, 3) uint8 -> (T, 3, H, W) uint8
                    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                    yield t
                    break
                retry += 1
            else:
                # All retries failed; yield a zero clip so training
                # doesn't deadlock. Shouldn't happen on a healthy
                # dataset.
                yield torch.zeros(self.T, 3, self.H, self.W,
                                   dtype=torch.uint8)

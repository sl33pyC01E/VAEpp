#!/usr/bin/env python3
"""Build / grow / trim a prefill clip bank.

Subprocessed from the GUI's Data -> Prefill Bank tab. Sets up the
generator exactly like train_elastictok does (same bank / pool /
disco / kwargs), renders clips via gen.generate_from_pool, and writes
them to the bank as uint8 .pt files.

CLI:
  --bank-dir PATH              where the bank lives
  --mode grow|fill|trim|clear  what to do:
    grow  - add --n-clips new clips (default)
    fill  - add clips until bank reaches --target-size
    trim  - LRU-evict to --target-size
    clear - delete every clip in the bank
  --n-clips N                  for grow
  --target-size N              for fill / trim
  --H / --W / --T              clip geometry (must match what the
                               training tab uses)
  --bank-size / --n-layers / --pool-size / --disco ... same as train
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.generator import VAEpp0rGenerator
from training.prefill_bank import PrefillBank


def _build_gen(args, device):
    """Same generator init train_elastictok uses, minus the
    elastictok-specific bits. Kept in sync so prefilled clips match
    what training would see."""
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
    return gen


def main():
    p = argparse.ArgumentParser(
        description="Build / grow / trim a prefill clip bank")
    p.add_argument("--bank-dir", required=True,
                   help="Directory for the clip bank.")
    p.add_argument("--mode", default="grow",
                   choices=["grow", "fill", "trim", "clear"])
    p.add_argument("--n-clips", type=int, default=1000,
                   help="For mode=grow: how many clips to add.")
    p.add_argument("--target-size", type=int, default=8000,
                   help="For mode=fill (fill to N) or mode=trim "
                        "(evict down to N).")
    # Clip geometry — must match training.
    p.add_argument("--H", type=int, default=368)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--T", type=int, default=4)
    # Generator init (same defaults as train_elastictok).
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--disco", action="store_true")
    # Generator gen batch size (internal, just affects throughput).
    p.add_argument("--gen-batch", type=int, default=8)
    args = p.parse_args()

    bank_dir = args.bank_dir
    if not os.path.isabs(bank_dir):
        bank_dir = os.path.join(PROJECT_ROOT, bank_dir)
    os.makedirs(bank_dir, exist_ok=True)

    print(f"[prefill-build] mode={args.mode} dir={bank_dir}", flush=True)

    if args.mode == "clear":
        # Delete both legacy .pt and current .mp4 bank files.
        pt_files = list(pathlib.Path(bank_dir).glob("clip_*.pt"))
        mp4_files = list(pathlib.Path(bank_dir).glob("clip_*.mp4"))
        files = pt_files + mp4_files
        for f in files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        print(f"[prefill-build] cleared {len(files)} clips from "
              f"{bank_dir} ({len(pt_files)} .pt + {len(mp4_files)} .mp4)",
              flush=True)
        return

    if args.mode == "trim":
        bank = PrefillBank(bank_dir, H=args.H, W=args.W, T=args.T,
                            verbose=False)
        before = bank.count()
        bank._evict_to(int(args.target_size))
        after = bank.count()
        print(f"[prefill-build] trim: {before} -> {after} "
              f"(target {args.target_size})", flush=True)
        return

    # grow / fill need the generator.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prefill-build] device={device}", flush=True)
    gen = _build_gen(args, device)
    bank = PrefillBank(bank_dir, H=args.H, W=args.W, T=args.T)

    cur = bank.count()
    print(f"[prefill-build] bank currently has {cur} clips", flush=True)

    if args.mode == "grow":
        target = cur + int(args.n_clips)
    else:  # fill
        target = int(args.target_size)
    bank.fill_to(target, gen, batch_size=max(1, args.gen_batch))

    final = bank.count()
    print(f"[prefill-build] done. bank now has {final} clips",
          flush=True)


if __name__ == "__main__":
    main()

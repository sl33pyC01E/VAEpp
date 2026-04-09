#!/usr/bin/env python3
"""Bank persistence and dynamic loading for VAEpp0r generator.

Save/load shape banks and base layers, dynamic bank streaming from disk.
"""

import os
import torch


class IOMixin:
    """Mixin providing I/O and bank management methods for VAEpp0rGenerator."""

    def refresh_base_layers(self):
        """Regenerate base layers (keeps shape bank)."""
        self.build_base_layers()

    def refresh_all(self):
        """Regenerate everything."""
        self.build_banks()

    def bank_stats(self):
        """Return stats about current banks."""
        stats = {}
        if self.shape_bank is not None:
            stats["shape_bank"] = {
                "count": self.shape_bank.shape[0],
                "res": self.shape_res,
                "mb": self.shape_bank.element_size() * self.shape_bank.nelement() / 1e6,
            }
        if self.base_layers is not None:
            stats["base_layers"] = {
                "count": self.base_layers.shape[0],
                "res": f"{self.H}x{self.W}",
                "mb": self.base_layers.element_size() * self.base_layers.nelement() / 1e6,
            }
        return stats

    def save_shape_bank(self, path):
        """Save shape bank to disk."""
        if self.shape_bank is None:
            return
        torch.save(self.shape_bank.cpu().half(), path)
        n = self.shape_bank.shape[0]
        mb = os.path.getsize(path) / 1e6
        print(f"Saved {n} shapes to {path} ({mb:.1f} MB)", flush=True)

    def load_shape_bank(self, path):
        """Load shape bank from disk, appending to existing bank."""
        loaded = torch.load(path, map_location=self.device, weights_only=True).float()
        if self.shape_bank is None:
            self.shape_bank = loaded
        else:
            self.shape_bank = torch.cat([self.shape_bank, loaded], dim=0)
        self.bank_size = self.shape_bank.shape[0]
        print(f"Loaded {loaded.shape[0]} shapes from {path} "
              f"(total: {self.bank_size})", flush=True)

    def save_base_layers(self, path):
        """Save base layers to disk."""
        if self.base_layers is None:
            return
        torch.save(self.base_layers.cpu().half(), path)
        n = self.base_layers.shape[0]
        mb = os.path.getsize(path) / 1e6
        print(f"Saved {n} layers to {path} ({mb:.1f} MB)", flush=True)

    def load_base_layers(self, path):
        """Load base layers from disk, appending to existing."""
        loaded = torch.load(path, map_location=self.device, weights_only=True).float()
        if self.base_layers is None:
            self.base_layers = loaded
        else:
            self.base_layers = torch.cat([self.base_layers, loaded], dim=0)
        self.n_base_layers = self.base_layers.shape[0]
        print(f"Loaded {loaded.shape[0]} layers from {path} "
              f"(total: {self.n_base_layers})", flush=True)

    def load_bank_dir(self, bank_dir):
        """Load all .pt files from a directory, accumulating into banks."""
        if not os.path.isdir(bank_dir):
            print(f"Bank dir not found: {bank_dir}", flush=True)
            return
        shape_files = sorted(f for f in os.listdir(bank_dir)
                             if f.startswith("shapes_") and f.endswith(".pt"))
        layer_files = sorted(f for f in os.listdir(bank_dir)
                             if f.startswith("layers_") and f.endswith(".pt"))
        for f in shape_files:
            self.load_shape_bank(os.path.join(bank_dir, f))
        for f in layer_files:
            self.load_base_layers(os.path.join(bank_dir, f))
        if not shape_files and not layer_files:
            print(f"No bank files found in {bank_dir}", flush=True)

    def setup_dynamic_bank(self, bank_dir, working_size=1000,
                            refresh_interval=50):
        """Configure dynamic bank loading from disk.

        Keeps `working_size` shapes in VRAM. Every `refresh_interval`
        generate() calls, swaps a random portion of the working set
        with shapes from disk.

        Args:
            bank_dir: directory containing shapes_*.pt files
            working_size: shapes to keep in VRAM at once
            refresh_interval: batches between partial swaps
        """
        self._dyn_bank_dir = bank_dir
        self._dyn_working_size = working_size
        self._dyn_refresh_interval = refresh_interval
        self._dyn_call_count = 0

        # Load all shapes to CPU (cheap — system RAM)
        shape_files = sorted(f for f in os.listdir(bank_dir)
                             if f.startswith("shapes_") and f.endswith(".pt"))
        if not shape_files:
            print(f"No shape files in {bank_dir}", flush=True)
            return

        all_shapes = []
        for f in shape_files:
            loaded = torch.load(os.path.join(bank_dir, f),
                                map_location="cpu", weights_only=True).float()
            all_shapes.append(loaded)
            print(f"  Loaded {loaded.shape[0]} shapes from {f} (CPU)",
                  flush=True)
        self._dyn_cpu_bank = torch.cat(all_shapes, dim=0)
        total = self._dyn_cpu_bank.shape[0]
        print(f"Dynamic bank: {total} shapes on CPU, "
              f"{working_size} working set on GPU", flush=True)

        # Initial working set: random sample
        self._dyn_refresh_working_set(full=True)

    def _dyn_refresh_working_set(self, full=False, swap_frac=0.25):
        """Swap part of the GPU working set with random CPU shapes."""
        total = self._dyn_cpu_bank.shape[0]
        ws = min(self._dyn_working_size, total)

        if full or self.shape_bank is None:
            # Full refresh
            idx = torch.randperm(total)[:ws]
            self.shape_bank = self._dyn_cpu_bank[idx].to(self.device)
            self.bank_size = ws
        else:
            # Partial swap
            n_swap = max(1, int(ws * swap_frac))
            # Pick random positions in working set to replace
            replace_idx = torch.randperm(ws)[:n_swap]
            # Pick random shapes from CPU bank
            source_idx = torch.randint(0, total, (n_swap,))
            self.shape_bank[replace_idx] = \
                self._dyn_cpu_bank[source_idx].to(self.device)

    def _maybe_refresh_dynamic(self):
        """Called each generate() — refreshes working set periodically."""
        if not hasattr(self, '_dyn_cpu_bank') or self._dyn_cpu_bank is None:
            return
        self._dyn_call_count += 1
        if self._dyn_call_count % self._dyn_refresh_interval == 0:
            self._dyn_refresh_working_set(swap_frac=0.25)
            print(f"  [bank refresh: swapped 25% of working set]", flush=True)

    def save_to_bank_dir(self, bank_dir):
        """Save current banks to directory with timestamped names."""
        import time
        os.makedirs(bank_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.shape_bank is not None:
            self.save_shape_bank(
                os.path.join(bank_dir, f"shapes_{ts}.pt"))
        if self.base_layers is not None:
            self.save_base_layers(
                os.path.join(bank_dir, f"layers_{ts}.pt"))

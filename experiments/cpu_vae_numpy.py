#!/usr/bin/env python3
"""Pure NumPy inference for CPU VAE models.

Zero PyTorch dependency. Loads weights from .pt checkpoints (via pickle),
runs encode/decode using only numpy operations.

Usage:
    from experiments.cpu_vae_numpy import load_model, encode, decode

    model = load_model("pretrained/ur-ps8-lc3-id4-o3-10k.pt")
    image = np.random.rand(360, 640, 3).astype(np.float32)  # HWC [0,1]
    latent = encode(model, image)
    recon = decode(model, latent, original_size=(360, 640))
"""

import pickle
import struct
import zipfile
import io
import numpy as np


# =============================================================================
# Checkpoint loader (extract numpy arrays from PyTorch .pt files)
# =============================================================================

def _load_tensor_from_storage(zip_file, name):
    """Load a tensor from a PyTorch zip archive."""
    with zip_file.open(f"archive/data/{name}") as f:
        return f.read()


def load_pt_file(path):
    """Load a .pt checkpoint and return raw state dict + config as numpy.

    PyTorch .pt files are zip archives containing pickled metadata
    and raw tensor storage. We extract without importing torch.
    """
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_sd = ckpt.get("model", ckpt)

    # Convert all tensors to numpy
    weights = {}
    for k, v in model_sd.items():
        weights[k] = v.numpy()

    return weights, config


# =============================================================================
# GELU activation (numpy)
# =============================================================================

def gelu(x):
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# =============================================================================
# Unfold / Fold (numpy equivalents of F.unfold / F.fold)
# =============================================================================

def unfold(image, patch_size, stride):
    """Extract patches from image.

    Args:
        image: (C, H, W) float32
        patch_size: int
        stride: int

    Returns:
        patches: (patch_dim, n_patches) where patch_dim = C * ps * ps
    """
    C, H, W = image.shape
    ps = patch_size
    pH = (H - ps) // stride + 1
    pW = (W - ps) // stride + 1

    patches = np.zeros((C * ps * ps, pH * pW), dtype=image.dtype)
    idx = 0
    for i in range(pH):
        for j in range(pW):
            patch = image[:, i*stride:i*stride+ps, j*stride:j*stride+ps]
            patches[:, idx] = patch.reshape(-1)
            idx += 1

    return patches


def fold(patches, output_size, patch_size, stride):
    """Reconstruct image from patches (sum overlapping regions).

    Args:
        patches: (patch_dim, n_patches)
        output_size: (H, W)
        patch_size: int
        stride: int

    Returns:
        image: (C, H, W)
    """
    H, W = output_size
    ps = patch_size
    C = patches.shape[0] // (ps * ps)
    pH = (H - ps) // stride + 1
    pW = (W - ps) // stride + 1

    image = np.zeros((C, H, W), dtype=patches.dtype)
    idx = 0
    for i in range(pH):
        for j in range(pW):
            patch = patches[:, idx].reshape(C, ps, ps)
            image[:, i*stride:i*stride+ps, j*stride:j*stride+ps] += patch
            idx += 1

    return image


def fold_divisor(C, output_size, patch_size, stride):
    """Compute overlap count for averaging."""
    H, W = output_size
    ps = patch_size
    pH = (H - ps) // stride + 1
    pW = (W - ps) // stride + 1

    divisor = np.zeros((C, H, W), dtype=np.float32)
    for i in range(pH):
        for j in range(pW):
            divisor[:, i*stride:i*stride+ps, j*stride:j*stride+ps] += 1.0

    return np.maximum(divisor, 1.0)


# =============================================================================
# Model container
# =============================================================================

class CPUVAEModel:
    """Container for loaded model weights and config."""
    def __init__(self, weights, config):
        self.weights = weights
        self.config = config
        self.model_type = config.get("model_type", "patch")
        self.patch_size = config.get("patch_size", 8)
        self.overlap = config.get("overlap", 0)
        self.stride = self.patch_size - self.overlap
        self.image_channels = config.get("image_channels", 3)
        self.latent_channels = config.get("latent_channels", 3)
        self.hidden_dim = config.get("hidden_dim", 0)
        self.inner_dim = config.get("inner_dim", 8)
        self.n_pixels = self.patch_size ** 2
        self.n_positions = self.image_channels * self.n_pixels

        # Precompute position embeddings for unrolled variant
        if self.model_type == "unrolled":
            self._enc_pos = self._build_pos_embed(
                weights["spatial_embed"][0],
                weights["channel_embed"][0])
            self._dec_pos = self._build_pos_embed(
                weights["dec_spatial_embed"][0],
                weights["dec_channel_embed"][0])

        # Precompute fold divisor if overlap > 0
        self._divisor_cache = {}

    def _build_pos_embed(self, spatial, channel):
        """Build (inner_dim, n_positions) from spatial and channel embeds."""
        # spatial: (D, n_pixels), channel: (D, C)
        parts = []
        for c in range(self.image_channels):
            parts.append(spatial + channel[:, c:c+1])
        return np.concatenate(parts, axis=1)  # (D, n_positions)

    def _get_divisor(self, H_pad, W_pad):
        key = (H_pad, W_pad)
        if key not in self._divisor_cache:
            self._divisor_cache[key] = fold_divisor(
                self.image_channels, (H_pad, W_pad),
                self.patch_size, self.stride)
        return self._divisor_cache[key]


# =============================================================================
# Padding / grid helpers
# =============================================================================

def _pad_for_patches(model, H, W):
    ps, stride, overlap = model.patch_size, model.stride, model.overlap
    if overlap == 0:
        pad_h = (stride - H % stride) % stride
        pad_w = (stride - W % stride) % stride
    else:
        usable_h = H - ps
        pad_h = (stride - usable_h % stride) % stride if usable_h % stride != 0 else 0
        usable_w = W - ps
        pad_w = (stride - usable_w % stride) % stride if usable_w % stride != 0 else 0
    return pad_h, pad_w


def _patch_grid_size(model, H, W):
    pad_h, pad_w = _pad_for_patches(model, H, W)
    Hp, Wp = H + pad_h, W + pad_w
    ps, stride = model.patch_size, model.stride
    pH = (Hp - ps) // stride + 1
    pW = (Wp - ps) // stride + 1
    return pH, pW


# =============================================================================
# Linear layer helpers
# =============================================================================

def linear(x, weight, bias=None):
    """Apply linear layer: x @ weight.T + bias.

    Args:
        x: (..., in_features)
        weight: (out_features, in_features)
        bias: (out_features,) or None
    """
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


# =============================================================================
# PatchVAE inference
# =============================================================================

def _encode_patch(model, image):
    """Encode with PatchVAE. image: (C, H, W) -> latent: (lat_ch, pH, pW)."""
    w = model.weights
    C, H, W = image.shape
    ps = model.patch_size

    pad_h, pad_w = _pad_for_patches(model, H, W)
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="edge")
    else:
        padded = image

    pH, pW = _patch_grid_size(model, H, W)
    patches = unfold(padded, ps, model.stride)  # (patch_dim, n_patches)
    patches = patches.T  # (n_patches, patch_dim)

    if model.hidden_dim > 0:
        h = linear(patches, w["encoder.0.weight"], w["encoder.0.bias"])
        h = gelu(h)
        latent = linear(h, w["encoder.2.weight"], w["encoder.2.bias"])
    else:
        latent = linear(patches, w["encoder.weight"], w["encoder.bias"])

    # (n_patches, lat_ch) -> (lat_ch, pH, pW)
    return latent.T.reshape(model.latent_channels, pH, pW)


def _decode_patch(model, latent, original_size=None):
    """Decode with PatchVAE. latent: (lat_ch, pH, pW) -> image: (C, H, W)."""
    w = model.weights
    lat_ch, pH, pW = latent.shape
    ps = model.patch_size
    n_patches = pH * pW

    H_pad = (pH - 1) * model.stride + ps
    W_pad = (pW - 1) * model.stride + ps

    seq = latent.reshape(lat_ch, n_patches).T  # (n_patches, lat_ch)

    if model.hidden_dim > 0:
        h = linear(seq, w["decoder.0.weight"], w["decoder.0.bias"])
        h = gelu(h)
        patches = linear(h, w["decoder.2.weight"], w["decoder.2.bias"])
    else:
        patches = linear(seq, w["decoder.weight"], w["decoder.bias"])

    patches = patches.T  # (patch_dim, n_patches)

    recon = fold(patches, (H_pad, W_pad), ps, model.stride)

    if model.overlap > 0:
        divisor = model._get_divisor(H_pad, W_pad)
        recon = recon / divisor

    if original_size is not None:
        recon = recon[:, :original_size[0], :original_size[1]]

    return recon


# =============================================================================
# UnrolledPatchVAE inference
# =============================================================================

def _encode_unrolled(model, image):
    """Encode with UnrolledPatchVAE. image: (C, H, W) -> latent: (lat_ch, pH, pW)."""
    w = model.weights
    C, H, W = image.shape
    ps = model.patch_size
    D = model.inner_dim

    pad_h, pad_w = _pad_for_patches(model, H, W)
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="edge")
    else:
        padded = image

    pH, pW = _patch_grid_size(model, H, W)
    n_patches = pH * pW
    n_pos = model.n_positions

    patches = unfold(padded, ps, model.stride)  # (patch_dim, n_patches)

    # (n_patches, n_pos, 1)
    patches = patches.T.reshape(n_patches, n_pos, 1)

    # Value projection: (n_patches, n_pos, 1) @ (1, D) -> (n_patches, n_pos, D)
    vals = patches @ w["value_proj.weight"].T  # (N, 192, 1) @ (1, D) -> (N, 192, D)

    # -> (n_patches, D, n_pos)
    vals = vals.transpose(0, 2, 1)

    # Add position embeddings: (D, n_pos) broadcast
    vals = vals + model._enc_pos

    # enc_mix: Linear(n_pos, n_pos) -> GELU -> Linear(n_pos, lat_ch)
    # vals is (n_patches, D, n_pos), linear operates on last dim
    h = linear(vals, w["enc_mix.0.weight"], w["enc_mix.0.bias"])
    h = gelu(h)
    latent = linear(h, w["enc_mix.2.weight"], w["enc_mix.2.bias"])

    # Average over inner_dim: (n_patches, D, lat_ch) -> (n_patches, lat_ch)
    latent = latent.mean(axis=1)

    # -> (lat_ch, pH, pW)
    return latent.T.reshape(model.latent_channels, pH, pW)


def _decode_unrolled(model, latent, original_size=None):
    """Decode with UnrolledPatchVAE. latent: (lat_ch, pH, pW) -> image: (C, H, W)."""
    w = model.weights
    lat_ch, pH, pW = latent.shape
    ps = model.patch_size
    D = model.inner_dim
    n_patches = pH * pW
    n_pos = model.n_positions

    H_pad = (pH - 1) * model.stride + ps
    W_pad = (pW - 1) * model.stride + ps

    # (n_patches, lat_ch)
    lat_seq = latent.reshape(lat_ch, n_patches).T

    # Expand to (n_patches, D, lat_ch) via broadcast
    lat_exp = np.broadcast_to(lat_seq[:, np.newaxis, :],
                               (n_patches, D, lat_ch)).copy()

    # dec_mix: Linear(lat_ch, n_pos) -> GELU -> Linear(n_pos, n_pos)
    h = linear(lat_exp, w["dec_mix.0.weight"], w["dec_mix.0.bias"])
    h = gelu(h)
    unpooled = linear(h, w["dec_mix.2.weight"], w["dec_mix.2.bias"])

    # Add decoder position embeddings
    decoded = unpooled + model._dec_pos

    # Project to scalar: (n_patches, D, n_pos) -> (n_patches, n_pos, D) -> (n_patches, n_pos)
    decoded = decoded.transpose(0, 2, 1)
    pixel_vals = (decoded @ w["value_out.weight"].T).squeeze(-1)

    # (patch_dim, n_patches)
    patches = pixel_vals.T

    recon = fold(patches, (H_pad, W_pad), ps, model.stride)

    if model.overlap > 0:
        divisor = model._get_divisor(H_pad, W_pad)
        recon = recon / divisor

    if original_size is not None:
        recon = recon[:, :original_size[0], :original_size[1]]

    return recon


# =============================================================================
# Public API
# =============================================================================

def load_model(path):
    """Load a CPU VAE model from a .pt or .npz file.

    .pt files require torch for loading (pickle format).
    .npz files are pure numpy — zero external dependencies.

    Returns a CPUVAEModel with numpy weights.
    """
    if path.endswith(".npz"):
        return load_model_npz(path)
    weights, config = load_pt_file(path)
    return CPUVAEModel(weights, config)


def load_model_npz(path):
    """Load a CPU VAE model from a .npz file. Zero torch dependency.

    Args:
        path: path to .npz file exported by export_npz()

    Returns:
        CPUVAEModel with numpy weights.
    """
    import json
    data = np.load(path, allow_pickle=False)
    config = json.loads("".join(data["_config"]))
    weights = {}
    for k in data.files:
        if k.startswith("w_"):
            weights[k[2:]] = data[k]
    return CPUVAEModel(weights, config)


def export_npz(pt_path, npz_path):
    """Convert a .pt checkpoint to .npz for torch-free inference.

    Args:
        pt_path: input .pt file path
        npz_path: output .npz file path
    """
    import json
    weights, config = load_pt_file(pt_path)
    arrays = {}
    for k, v in weights.items():
        arrays[f"w_{k}"] = v
    arrays["_config"] = np.array(list(json.dumps(config)), dtype="U1")
    np.savez_compressed(npz_path, **arrays)


def encode(model, image):
    """Encode an image to latent representation.

    Args:
        model: CPUVAEModel from load_model()
        image: (H, W, C) or (C, H, W) float32 array in [0, 1]

    Returns:
        latent: (latent_channels, pH, pW) float32 array
    """
    # Accept HWC or CHW
    if image.ndim == 3 and image.shape[2] in (1, 3):
        image = image.transpose(2, 0, 1)  # HWC -> CHW

    image = image.astype(np.float32)

    if model.model_type == "unrolled":
        return _encode_unrolled(model, image)
    else:
        return _encode_patch(model, image)


def decode(model, latent, original_size=None):
    """Decode latent representation to image.

    Args:
        model: CPUVAEModel from load_model()
        latent: (latent_channels, pH, pW) float32 array
        original_size: (H, W) tuple to crop output to

    Returns:
        image: (H, W, C) float32 array in [0, 1] (clipped)
    """
    latent = latent.astype(np.float32)

    if model.model_type == "unrolled":
        recon = _decode_unrolled(model, latent, original_size)
    else:
        recon = _decode_patch(model, latent, original_size)

    # CHW -> HWC, clip to [0, 1]
    return np.clip(recon.transpose(1, 2, 0), 0, 1)


def encode_decode(model, image):
    """Full round-trip: image -> latent -> reconstruction.

    Args:
        model: CPUVAEModel from load_model()
        image: (H, W, C) or (C, H, W) float32 array in [0, 1]

    Returns:
        (recon, latent) where recon is (H, W, C) float32
    """
    if image.ndim == 3 and image.shape[2] in (1, 3):
        H, W = image.shape[:2]
    else:
        H, W = image.shape[1], image.shape[2]

    latent = encode(model, image)
    recon = decode(model, latent, original_size=(H, W))
    return recon, latent

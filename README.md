# VAEpp0r
Variable Auto Encoder: procedural priors, zero real
<img width="1490" height="423" alt="training_data1" src="https://github.com/user-attachments/assets/4e4531aa-9ade-4f04-817f-de3c76127fae" />
<img width="1488" height="422" alt="training_data2" src="https://github.com/user-attachments/assets/a9ed4dfa-8c34-4e5e-a9e0-91acb4ce3768" />

*No dataset? No problem.*

Procedural synthetic data pretraining for causal temporal video autoencoders.

VAEpp0r trains a video VAE entirely on procedurally generated data -- zero real images required. The generator produces structured synthetic images and video clips with controllable shape composition, scene templates, mathematical patterns, and physics-driven motion. A VAE trained on this synthetic data achieves **pixel-perfect out-of-distribution reconstruction** on real video after temporal training, closing ~90% of the domain gap before any real data is introduced.

https://github.com/user-attachments/assets/c1cf2996-5148-4e41-bd4f-80eff7d2b278

> **Left:** Ground truth Samsung phone ad footage (never seen during training). **Right:** VAE reconstruction from 3-channel latent at 8x spatial + 4x temporal compression. Trained exclusively on procedural synthetic data.

https://github.com/user-attachments/assets/8b2cbde3-8ad3-4c39-abf1-2c5f3cfcdb01

> **Left:** Ground truth training data; procedurally generated prior soup. **Right:** VAE reconstruction from 3-channel latent at 8x spatial + 4x temporal compression.




## Architecture

- 8x spatial compression, configurable 1-4x temporal compression
- Configurable latent channels (3-32), encoder/decoder widths
- Optional FSQ (Finite Scalar Quantization) for discrete latent tokens
- Flatten/Deflatten bottleneck for 1D latent serialization (for downstream world models)

## Procedural Generator

Three-tier GPU-accelerated generation pipeline:

1. **Shape Bank** -- 10 SDF primitives (circle, rect, triangle, ellipse, blob, line, stroke, hatch, stipple, fractal) x 4 textures (flat, perlin, gradient, voronoi) x 3 edges (hard, soft, textured)
2. **Base Layers** -- Composited scenes from shape bank with 19 scene templates (horizon, perspective, block city, landscape, road, water, forest, etc.)
3. **Final Output** -- Fast layer compositing with transforms, stamps, micro-stamps, post-processing

### Pattern System (38 generators)

Clean mathematical/structural patterns with zero shape compositing:
- Gradients (linear, radial, angular, diamond, multi-stop)
- Tilings (checkerboard, stripes, hexagonal, brick, herringbone, basketweave, fish scale, chevron, argyle)
- Waves (sine, interference/moire, concentric rings, spirals, ripples)
- Mathematical surfaces (quadratic contours, Lissajous, rose curves, spirograph, Julia sets)
- Symmetry/Op Art (kaleidoscope, warped grids, Islamic star patterns)
- Procedural natural (reaction-diffusion, contour maps, wood grain, marble, cracked earth)
- Art exercises (zentangle, maze generation, contour lines, squiggle fill)
- Fine-grain (halftone, ordered dither, stipple density)

### Disco Quadrant Mode

Balanced training data diversity:
- 25% pure mathematical patterns
- 25% pattern collages with sparse shape overlay
- 25% dense random compositing (cranked micro-stamps)
- 25% structured scene templates

### Temporal

Physics-driven motion (gravity, velocity, bounce), viewport transforms (pan, zoom, rotation), fluid advection, parallax layers. Motion stored as compact JSON recipes (~1KB vs ~9MB rendered).

## Training Pipeline

| Stage | Data | Temporal | Description |
|-------|------|----------|-------------|
| 1 | Static synthetic | No | RGB image reconstruction |
| 2 | Temporal synthetic | 4x | Video reconstruction with temporal consistency |
| 3 | FSQ | Optional | Quantize continuous latent to discrete tokens |
| 4 | Flatten | Optional | 1D bottleneck for sequence input |

## GUI

Tkinter desktop app with nested tab layout:

- **Data** -- Static generator controls, video generator with motion pool
- **Models** -- Training, inference, checkpoint conversion (static + video)
- **Compress** -- FSQ quantization, flatten/deflatten experiments (static + video)

All tabs support disco quadrant mode, resume from checkpoint, and auto-save inference outputs.

## Model Presets

| Preset | Channels | Latent | Params | Size |
|--------|----------|--------|--------|------|
| Pico | 3 | 4 | 1.0M | 4 MB |
| Nano | 3 | 8 | 1.2M | 5 MB |
| Tiny | 3 | 16 | 3.3M | 13 MB |
| Small | 3 | 16 | 4.0M | 16 MB |
| Medium | 3 | 32 | 11.3M | 43 MB |

## Usage

## Setup

```bash
# Windows
setup.bat

# Manual
python -m venv venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Usage

```bash
# Launch GUI
gui.bat
# or: python -m gui.app

# Train static VAE (Stage 1)
python -m training.train_static --disco

# Train temporal VAE (Stage 2)
python -m training.train_video --disco

# FSQ quantization
python -m experiments.fsq --vae-ckpt synthyper_logs/latest.pt

# Flatten experiment
python -m experiments.flatten --vae-ckpt synthyper_logs/latest.pt
```

## Project Structure

```
setup.bat               # Create venv + install deps
gui.bat                 # Launch GUI
requirements.txt

core/                   # Core modules
  generator.py          # Procedural image/video generator
  patterns.py           # 38 mathematical pattern generators
  pattern_collage.py    # Pattern combination operations
  model.py              # MiniVAE architecture
  fsq.py                # Finite Scalar Quantization layer

gui/                    # Tkinter desktop GUI
  app.py                # Main window (Data | Models | Compress)
  common.py             # Shared theme, helpers, process runner
  data_tabs.py          # Static + Video generator tabs
  models_tabs.py        # Training + Inference + Convert tabs
  compress_tabs.py      # FSQ + Flatten experiment tabs

training/               # Training scripts
  train_static.py       # Stage 1: static image VAE
  train_video.py        # Stage 2: temporal video VAE

experiments/            # Compression experiments
  fsq.py                # FSQ quantization fine-tuning
  flatten.py            # Flatten/deflatten bottleneck (static)
  flatten_video.py      # Flatten/deflatten bottleneck (temporal)
  cpu_vae.py            # CPU VAE: convolution-free pipeline
  cpu_vae_gui.py        # CPU VAE standalone GUI
  cpu_vae_numpy.py      # Pure NumPy inference (zero PyTorch)

pretrained/             # Pretrained checkpoints
  3ch_S8x.pt            # 3ch RGB, 8x spatial, static
  3ch_S8x_T4x.pt       # 3ch RGB, 8x spatial, 4x temporal
  ur-ps8-lc9-id8-o1-30k.pt   # CPU VAE: unrolled, 9ch, 30K steps
  ur-ps8-lc3-id4-o3-10k.pt   # CPU VAE: unrolled, 3ch, 10K steps
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ffmpeg (for video preview/inference)
- PIL/Pillow, numpy

## CPU VAE Experiment

Convolution-free encoder/decoder that runs entirely on CPU. Replaces all Conv2d operations with `Unfold + Linear` projections and learned positional embeddings. Designed as a lightweight encode/decode block for downstream RSSM world models.

### Architecture: UnrolledPatchVAE

Each image patch is unrolled into a pixel line with explicit positional and channel identity embeddings, then compressed via linear projection. No spatial convolutions anywhere.

- **Encode:** `F.unfold` patches + learned (spatial, channel) embeddings + `nn.Linear` projection
- **Decode:** `nn.Linear` projection + `F.fold` with overlap averaging
- **Overlap blending:** configurable patch overlap eliminates boundary artifacts

### Pipeline Stages

Agnostic staged pipeline with unified checkpoint format. Every checkpoint is self-contained (fuses all upstream models). Stages chain arbitrarily via `--mode extend`.

| Stage | Type | Function |
|-------|------|----------|
| **S1** | UnrolledPatchVAE | Cascadable spatial compression (fresh/resume/extend) |
| **Refiner** | PatchAttentionRefiner | Transformer self-attention over patch tokens with 2D rotary positional embeddings. Global receptive field for cross-patch refinement |
| **Refiner** | LatentRefiner | Conv1d residual blocks along walk-ordered 1D sequence (alternative) |
| **S2** | FlattenDeflatten | Channel compression + 1D serialization via walk order (raster/hilbert/morton) |

### Cascaded Compression

S1 stages cascade with `--mode extend` — each stage halves the spatial grid:

```
360x640 -> 180x320 -> 90x160 -> 45x80    (3 stages, ps3 o1, 64:1)
```

Each cascade stage is identical (~4K params) and trains in ~11 minutes on synthetic data.

### Attention Refiner

Transformer self-attention over the latent grid with configurable patchification:
- Unfolds the latent into overlapping patches for richer per-token features
- 2D rotary positional embeddings encode spatial position
- Global receptive field: every token attends to every other
- Optional selective finetuning of upstream encoders/decoders

Example: `--attn-patch-size 3 --attn-patch-overlap 1` on a 45x80 grid = 858 tokens, 6ms, 21K params.

### CPU Inference Benchmarks

**Single-stage overlap sweep** (ps8, lc3, id4, 360p):

| Stride | Grid | Patches | Lat dims | Params | Encode | Decode | Round-trip |
|--------|------|---------|----------|--------|--------|--------|------------|
| 8 | 45x80 | 3,600 | 10,800 | 76K | 4.0ms | 4.2ms | 7.9ms |
| 7 | 52x92 | 4,784 | 14,352 | 76K | 7.0ms | 9.0ms | 13.9ms |
| **6** | **60x107** | **6,420** | **19,260** | **76K** | **9.8ms** | **13.7ms** | **23.0ms** |
| 5 | 72x128 | 9,216 | 27,648 | 76K | 16.3ms | 22.5ms | 38.1ms |
| 4 | 89x159 | 14,151 | 42,453 | 76K | 29.7ms | 37.9ms | 67.0ms |
| 3 | 119x212 | 25,228 | 75,684 | 76K | 57.3ms | 74.7ms | 132.8ms |
| 2 | 177x317 | 56,109 | 168,327 | 76K | 130.0ms | 184.4ms | 313.4ms |
| 1 | 353x633 | 223,449 | 670,347 | 76K | 594.1ms | 848.9ms | 1526.5ms |

**Cascaded compression** (ps3, o1, lc3, id4, hd32, 360p):

| Stages | Final grid | Latent dims | Total params | Encode | Decode | Round-trip | Compression |
|--------|-----------|-------------|-------------|--------|--------|------------|-------------|
| 1 | 180x320 | 172,800 | 4,230 | 16.1ms | 19.8ms | 35.9ms | 4:1 |
| 2 | 90x160 | 43,200 | 8,460 | 19.6ms | 24.8ms | 44.4ms | 16:1 |
| 3 | 45x80 | 10,800 | 12,690 | 24.3ms | 26.9ms | 51.3ms | 64:1 |
| **4** | **22x40** | **2,640** | **16,920** | **26.2ms** | **28.3ms** | **54.4ms** | **262:1** |
| 5 | 11x20 | 660 | 21,150 | 24.7ms | 27.3ms | 52.0ms | 1,047:1 |
| 6 | 5x10 | 150 | 25,380 | 25.4ms | 28.6ms | 54.0ms | 4,608:1 |
| 7 | 2x5 | 30 | 29,610 | 25.0ms | 28.4ms | 53.4ms | 23,040:1 |

**Pipeline configurations:**

| Config | Latent dims | Params | CPU decode | Compression |
|--------|-------------|--------|-----------|-------------|
| 3-stage cascade (ps3 o1 lc3 hd32) | 10,800 | 13K | 27ms | 64:1 |
| 3-stage + attention refiner | 10,800 | 34K | 33ms | 64:1 |
| 4-stage cascade | 2,640 | 17K | 28ms | 262:1 |
| 3-stage + S2 flatten (2ch) | 7,200 | 15K | 28ms | 96:1 |

Pure NumPy inference available (`cpu_vae_numpy.py`) — zero PyTorch dependency, loads from `.npz` files.

### GUI

Standalone Tkinter app (`cpu_vae_gui.bat`) with 4 tabs:
- **S1 Train** — Fresh/Resume/Extend with architecture controls
- **Refiner** — Attention or Conv1d with selective encoder/decoder finetuning
- **S2 Train** — Flatten bottleneck with walk order selection
- **Inference** — Load any pipeline checkpoint, per-stage latency logging

### Usage

```bash
# Launch CPU VAE GUI
cpu_vae_gui.bat

# Train S1 fresh
python -m experiments.cpu_vae s1 --mode fresh --patch-size 3 --overlap 1 --latent-ch 3 --inner-dim 4 --hidden-dim 32

# Extend with new cascade stage
python -m experiments.cpu_vae s1 --mode extend --input-ckpt cpu_vae_logs/latest.pt

# Add attention refiner
python -m experiments.cpu_vae refiner --input-ckpt cpu_vae_logs/latest.pt --refiner-type attention --attn-patch-size 3 --attn-patch-overlap 1

# Flatten for world model
python -m experiments.cpu_vae s2 --input-ckpt cpu_vae_logs/latest.pt --bottleneck-ch 2 --walk-order hilbert

# Unified inference with timing
python -m experiments.cpu_vae infer --ckpt cpu_vae_logs/latest.pt
```

### Pretrained CPU VAE Checkpoints

| Checkpoint | Config | Steps |
|-----------|--------|-------|
| `ur-ps8-lc9-id8-o1-30k.pt` | ps8, 9ch latent, inner_dim=8, overlap=1 | 30K |
| `ur-ps8-lc3-id4-o3-10k.pt` | ps8, 3ch latent, inner_dim=4, overlap=3 | 10K |

## Acknowledgments

- **[TAEHV](https://github.com/madebyollin/taehv)** by madebyollin — the causal temporal VAE architecture (MemBlock, TPool, TGrow) that this project builds on
- **[Revisiting Dead Leaves Model: Training with Synthetic Data](https://ieeexplore.ieee.org/document/9633158/)** (Madhusudana et al., 2021) — demonstrated that neural networks trained on procedural dead leaves images can approach the performance of networks trained on real data
- **[Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)** (Mentzer et al., ICLR 2024) — the FSQ quantization method used for discrete latent tokens

## License

MIT with Attribution — free to use, modify, and distribute, but you must credit the original author and link back to this repository. See [LICENSE](LICENSE) for details.

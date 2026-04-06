# SynthyperVAE

Procedural synthetic data pretraining for causal temporal video autoencoders.

SynthyperVAE trains a video VAE entirely on procedurally generated data -- zero real images required. The generator produces structured synthetic images and video clips with controllable shape composition, scene templates, mathematical patterns, and physics-driven motion. A VAE trained on this synthetic data achieves **pixel-perfect out-of-distribution reconstruction** on real video after temporal training, closing ~90% of the domain gap before any real data is introduced.

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

pretrained/             # Pretrained checkpoints
  3ch_S8x_T4x.pt       # 3ch RGB, 8x spatial, 4x temporal
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ffmpeg (for video preview/inference)
- PIL/Pillow, numpy

## Acknowledgments

- **[TAEHV](https://github.com/madebyollin/taehv)** by madebyollin — the causal temporal VAE architecture (MemBlock, TPool, TGrow) that this project builds on
- **[Revisiting Dead Leaves Model: Training with Synthetic Data](https://ieeexplore.ieee.org/document/9633158/)** (Madhusudana et al., 2021) — demonstrated that neural networks trained on procedural dead leaves images can approach the performance of networks trained on real data
- **[Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)** (Mentzer et al., ICLR 2024) — the FSQ quantization method used for discrete latent tokens

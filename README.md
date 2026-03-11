# KransFormer 👑

**KransFormer: Dual-Stream Gated KAN-Transformer for Medical Image Segmentation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025-red.svg)](#citation)

<p align="center">
  <img src="assets/architecture.png" width="800" alt="KransFormer Architecture"/>
</p>

> **KransFormer** (from *krans* — Dutch/Afrikaans for "crown") is a novel hybrid
> encoder–decoder architecture for medical image segmentation that replaces standard
> skip connections with a learnable **Dual-Stream Gated Feature Aggregation (DSGFA)**
> module, and introduces **KAN-Attention** and **Adaptive Spline Positional Encoding
> (ASPE)** for richer token interactions.

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Architecture Overview](#-architecture-overview)
- [Novel Contributions](#-novel-contributions)
  - [1. DSGFA — Dual-Stream Gated Feature Aggregation](#1-dsgfa--dual-stream-gated-feature-aggregation)
  - [2. KAN-Attention](#2-kan-attention)
  - [3. ASPE — Adaptive Spline Positional Encoding](#3-aspe--adaptive-spline-positional-encoding)
  - [4. KANFFN — KAN Feed-Forward Network](#4-kanffn--kan-feed-forward-network)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Programmatic API](#-programmatic-api)
- [Hyperparameters](#-hyperparameters)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## ✨ Highlights

- 🏗 **Hybrid CNN–KAN–Transformer** encoder: 3 convolutional stages for local
  texture + 2 KAN-Transformer stages for global context
- 🔀 **DSGFA** skip connections: gated dual-stream fusion guided by deep
  supervision maps — a direct replacement for standard additive skip connections
- 🧠 **KAN-Attention**: multi-head self-attention with independent B-spline
  KANLinear projections for Q, K, and V
- 📍 **ASPE**: lightweight sinusoidal PE refined by a single KANLinear layer for
  data-driven spatial adaptation
- 📉 **Progressive Deep Supervision** with 4 auxiliary heads and learnable loss
  weights
- 🎯 Designed for **binary medical image segmentation** (echocardiography,
  ultrasound, CT, MRI)

---

## 🏛 Architecture Overview
```
Input (B, 1, H, W)
      │
      ├─── ConvBlock + BN + MaxPool  →  x1  (B, 32,  H/2,  W/2)
      ├─── ConvBlock + BN + MaxPool  →  x2  (B, 64,  H/4,  W/4)
      ├─── ConvBlock + BN + MaxPool  →  x3  (B, 128, H/8,  W/8)
      │
      ├─── OverlapPatchEmbed → ASPE → KransFormerBlock × D  →  x4  (B, 160, H/16, W/16)
      └─── OverlapPatchEmbed → ASPE → KransFormerBlock × D  →  x5  (B, 256, H/32, W/32)
                                                                        │
                           ┌────────────────────────────────────────────┘
                           ▼
              DeconvBlock + Upsample
                    + DSGFA(x5, x4, DS_guide)   →  d1
              DeconvBlock + Upsample
                    + DSGFA(d1, x3, DS_guide)   →  d2
              DeconvBlock + Upsample
                    + DSGFA(d2, x2, DS_guide)   →  d3
              DeconvBlock + Upsample
                    + DSGFA(d3, x1, DS_guide)   →  d4
              DeconvBlock + Upsample
                    → Conv 1×1
                           │
                           ▼
              Segmentation Map  (B, num_classes, H, W)
```

Each **KransFormerBlock** consists of:
```
x  →  LN  →  KAN-Attention  →  DropPath  →  (+x)
   →  LN  →  KANFFN         →  DropPath  →  (+x)
```

---

## 🆕 Novel Contributions

### 1. DSGFA — Dual-Stream Gated Feature Aggregation

> **Replaces standard additive UNet skip connections.**

Standard skip connections simply add high-level and low-level features.
DSGFA introduces a two-stream gated mechanism with three key improvements:
```
xh (high-level, deeper)              xl (low-level, shallower)
        │                                       │
  proj_high (1×1 Conv)               KANLinear Refiner
        │                                       │
  bilinear upsample                         Stream B
  [optional: DS guide × soft gate]    (non-linear texture)
        │
  Dilated DW-Conv  d=1 ──┐
  Dilated DW-Conv  d=2 ──┤
  Dilated DW-Conv  d=4 ──┤── fuse (1×1) → Stream A
  Dilated DW-Conv  d=8 ──┘       (multi-scale semantics)
                    │
         sigmoid gate  ←  concat(Stream A, Stream B)
                    │
     fused = gate · A  +  (1 − gate) · B
                    │
         Squeeze-Excitation re-calibration
                    │
          out_conv (1×1) + residual(xl)
                    │
                output
```

**Why it's novel:**
- Standard attention-gated networks treat high-level and low-level features
  symmetrically. DSGFA processes each stream differently: Stream A focuses on
  multi-scale semantic capture, Stream B on non-linear texture preservation.
- The sigmoid gate is learned from *both* streams jointly, enabling
  location-specific control over semantic vs. detail emphasis.
- Deep supervision maps optionally bias Stream A toward likely foreground
  regions before gating, creating closed-loop feature refinement.

---

### 2. KAN-Attention

> **Replaces all three linear Q/K/V projections with independent KANLinear layers.**

Standard multi-head attention:
```
Q = x @ W_Q,    K = x @ W_K,    V = x @ W_V
```

KAN-Attention:
```
Q = KANLinear_Q(x),    K = KANLinear_K(x),    V = KANLinear_V(x)
```

Each KANLinear uses a **B-spline basis** of learnable control points:
```
y = SiLU(x) @ W_base   +   B-spline(x) @ W_spline
```

**Why it's novel:**
- Affine projections can only perform linear token mixing before the
  attention softmax. KANLinear projections apply per-element non-linear
  transformations, giving each attention head a unique spline activation
  space — richer token interactions with the same architectural footprint.
- The spline grid is adaptive: it can be updated at inference to better
  cover the distribution of test-time activations (`update_grid()`).

---

### 3. ASPE — Adaptive Spline Positional Encoding

> **Sinusoidal PE + KANLinear refiner for data-driven spatial adaptation.**

Standard sinusoidal PE:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

ASPE:
```
PE_base   = sinusoidal(pos)          # deterministic, no parameters
PE_adapt  = KANLinear(PE_base)       # learned spline correction
output    = x + PE_adapt
```

**Why it's novel:**
- Fully learnable absolute PE tables (e.g., ViT) require `max_len × dim`
  parameters and don't generalise to unseen sequence lengths.
- ASPE uses the sinusoidal table as a structured prior and applies a single
  KANLinear to learn *corrections* — zero positional parameters for the
  sinusoidal part, only `O(dim × grid_size)` spline parameters total.
- The KAN refiner can represent any continuous function of the PE, allowing
  the model to adapt to medical image spatial statistics (e.g., cardiac
  structures are not uniformly distributed across the field of view).

---

### 4. KANFFN — KAN Feed-Forward Network

> **KAN fc1 → DW-Conv → KAN fc2: combines non-linear mixing with local context.**

Standard Transformer FFN:
```
x → Linear(dim, 4·dim) → GELU → Linear(4·dim, dim)
```

KANFFN:
```
x → KANLinear(dim, hidden) → DW_BN_ReLU(H, W) → KANLinear(hidden, dim)
```

The interleaved depthwise convolution injects **local spatial context** into
the otherwise position-agnostic FFN, bridging the gap between convolutional
inductive biases and the global attention mechanism.

---

## 📁 Repository Structure
```
KransFormer/
│
├── models/
│   ├── __init__.py
│   └── kransformer.py          # Full architecture
│                               #   KANLinear, ASPE, DW_BN_ReLU
│                               #   KANAttention, KANFFN
│                               #   KransFormerBlock, OverlapPatchEmbed
│                               #   ConvBlock, DeconvBlock
│                               #   DSGFA, KransFormer
│
├── utils/
│   ├── __init__.py
│   ├── losses.py               # TverskyLoss, DiceLoss, BCEDiceLoss, CombinedLoss
│   ├── metrics.py              # Dice, IoU, Accuracy, Precision, Recall, F2
│   └── dataset.py              # SegmentationDataset, CAMUSDataset, build_loaders
│
├── scripts/
│   └── prepare_camus.py        # CAMUS .mhd → images/ + masks/ pre-processor
│
├── configs/
│   └── default.yaml            # All hyperparameters
│
├── train.py                    # Training entry-point
├── test.py                     # Evaluation + visualisation entry-point
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-username/KransFormer.git
cd KransFormer

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, CUDA ≥ 11.7 (recommended)

---

## 🗂 Dataset Preparation

### Option A — CAMUS (Cardiac Echocardiography)

1. Download CAMUS from the
   [official challenge page](https://www.creatis.insa-lyon.fr/Challenge/camus/).
2. Run the preparation script to convert `.mhd` files to PNG pairs:
```bash
python scripts/prepare_camus.py \
    --raw_dir  /data/CAMUS_public \
    --out_dir  /data/camus_processed \
    --img_size 128 \
    --views    2CH_ED 2CH_ES 4CH_ED 4CH_ES
```

### Option B — Custom Dataset

Organise your data in the following flat structure:
```
your_dataset/
├── images/
│   ├── case_001.png
│   ├── case_002.png
│   └── ...
└── masks/
    ├── case_001.png    # binary, 0 = background, 255 = foreground
    ├── case_002.png
    └── ...
```

The `build_loaders()` factory automatically splits into train / val / test
using the `--val_fraction` and `--test_fraction` arguments.

---

## 🚀 Training

### Quick Start
```bash
python train.py \
    --data_root  /data/camus_processed \
    --img_size   128 \
    --in_chans   1 \
    --epochs     150 \
    --batch_size 4 \
    --lr         0.001 \
    --deep_sup \
    --save_dir   ./outputs
```

### All CLI Flags

| Category | Flag | Default | Description |
|----------|------|---------|-------------|
| **Data** | `--data_root` | *(required)* | Dataset root with `images/` + `masks/` |
| | `--img_size` | `128` | Square input resolution |
| | `--in_chans` | `1` | `1` = greyscale, `3` = RGB |
| | `--num_workers` | `4` | DataLoader worker threads |
| **Model** | `--embed_dims` | `32 64 128 160 256` | Channel dims for 5 stages |
| | `--trans_depth` | `2` | KransFormerBlocks per transformer stage |
| | `--num_heads` | `8` | Attention heads |
| | `--deep_sup` | flag | Enable progressive deep supervision |
| | `--drop_rate` | `0.1` | Token dropout |
| | `--drop_path_rate` | `0.1` | Stochastic depth max rate |
| | `--mlp_ratio` | `1.0` | FFN hidden-dim multiplier |
| **Training** | `--epochs` | `150` | Maximum training epochs |
| | `--batch_size` | `4` | Samples per batch |
| | `--lr` | `0.001` | Initial learning rate |
| | `--weight_decay` | `1e-4` | AdamW weight decay |
| | `--patience` | `25` | Early stopping patience |
| | `--scheduler_patience` | `5` | LR reduction patience |
| | `--min_lr` | `1e-6` | Minimum learning rate |
| **Loss** | `--tversky_alpha` | `0.7` | FN weight in Tversky loss |
| | `--tversky_beta` | `0.3` | FP weight in Tversky loss |
| | `--lambda_aux` | `0.5` | Deep supervision loss weight |
| | `--lambda_reg` | `0.1` | KAN spline regularization weight |
| **Output** | `--save_dir` | `./outputs` | Directory for checkpoints and logs |
| | `--seed` | `42` | Global random seed |
| | `--device` | *(auto)* | `cuda`, `mps`, or `cpu` |

### Output Files

After training, `--save_dir` will contain:
```
outputs/
├── kransformer_best.pth      # Best checkpoint (by val_loss)
├── training_curves.png       # Loss / Dice / IoU / Accuracy plots
├── training_log.csv          # Per-epoch metrics
├── hyperparameters.csv       # Full experiment configuration
└── train.log                 # Training log
```

---

## 🧪 Evaluation
```bash
python test.py \
    --data_root  /data/camus_processed \
    --checkpoint outputs/kransformer_best.pth \
    --img_size   128 \
    --n_vis      8 \
    --save_dir   ./outputs
```

Generates:

| File | Contents |
|------|----------|
| `test_results.csv` | Dice, IoU, Accuracy, Precision, Recall, F2 |
| `test_vis.png` | Grid of (Image \| Ground Truth \| Prediction) |

---

## 📊 Results

Results on CAMUS (4-view echocardiographic segmentation, 128×128):

| Model | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | F2 ↑ | Params |
|-------|--------|-------|-------------|----------|------|--------|
| U-Net | 0.871 | 0.792 | 0.883 | 0.862 | 0.865 | 31.0M |
| UKAN | 0.889 | 0.812 | 0.897 | 0.881 | 0.883 | 27.3M |
| **KransFormer** | **0.912** | **0.841** | **0.918** | **0.907** | **0.908** | **~24M** |

> Results are indicative. Reproduce with the commands above on your split.

---

## 🔧 Programmatic API
```python
import torch
from models.kransformer import KransFormer

# ── Instantiate ──────────────────────────────────────────────────────────────
model = KransFormer(
    num_classes    = 1,
    in_chans       = 1,
    img_size       = 128,
    embed_dims     = (32, 64, 128, 160, 256),
    deep_sup       = True,
    trans_depth    = 2,
    num_heads      = 8,
    drop_rate      = 0.1,
    drop_path_rate = 0.1,
    mlp_ratio      = 1.0,
)

# ── Forward pass ─────────────────────────────────────────────────────────────
x = torch.randn(2, 1, 128, 128)
(aux_preds, out) = model(x)    # deep_sup=True → (tuple of 4 aux maps, main map)

print(out.shape)               # torch.Size([2, 1, 128, 128])
print([a.shape for a in aux_preds])
# [torch.Size([2,1,128,128]), ×4]

# ── Inference only (no deep supervision output) ───────────────────────────
model_infer = KransFormer(deep_sup=False)
out = model_infer(x)           # plain Tensor
seg = torch.sigmoid(out) > 0.5

# ── KAN regularization loss ───────────────────────────────────────────────
reg = model.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0)

# ── Count trainable parameters ────────────────────────────────────────────
print(f"Parameters: {model.count_parameters():,}")

# ── Load checkpoint ───────────────────────────────────────────────────────
ckpt = torch.load("outputs/kransformer_best.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
```

---

## 🔬 Hyperparameters

Full hyperparameter set used in experiments:

| Hyperparameter | Value |
|----------------|-------|
| Image size | 128 × 128 |
| Input channels | 1 (greyscale) |
| Embed dims | 32, 64, 128, 160, 256 |
| Transformer depth | 2 blocks per stage |
| Attention heads | 8 |
| MLP ratio | 1.0 |
| KAN grid size | 5 |
| KAN spline order | 3 |
| Drop rate | 0.1 |
| Drop-path rate | 0.1 |
| Batch size | 4 |
| Epochs | 150 (early stopping, patience 25) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Min LR | 1e-6 |
| Tversky α / β | 0.7 / 0.3 |
| λ_aux (DS loss weight) | 0.5 |
| λ_reg (KAN reg weight) | 0.1 |
| Grad clip norm | 1.0 |

---

## 🔁 Reproducibility
```bash
# Fix all seeds
python train.py --seed 42 ...

# The following are set internally:
#   random.seed(42)
#   numpy.random.seed(42)
#   torch.manual_seed(42)
#   torch.cuda.manual_seed_all(42)
```

---

## 🏗 DSGFA — Detailed Design Diagram
```
┌─────────────────────────────────────────────────────────────────────┐
│                    DSGFA Forward Pass                               │
│                                                                     │
│  xh (B, C_h, H', W')          xl (B, C_l, H, W)                   │
│         │                              │                            │
│   proj_high (1×1)              KANLinear refiner                   │
│         │                      (B·H·W, C_l) → (B·H·W, C_l)        │
│   bilinear upsample → (H,W)           │                            │
│         │                         Stream B                         │
│   [DS guide: soft weight]             │                            │
│         │                             │                            │
│   ┌─────┴──────┐                      │                            │
│   │ DW-Conv d=1│                      │                            │
│   │ DW-Conv d=2│                      │                            │
│   │ DW-Conv d=4│  → concat → 1×1 →   │                            │
│   │ DW-Conv d=8│   Stream A           │                            │
│   └────────────┘         │            │                            │
│                           └────┬───────┘                           │
│                          concat(A, B)                              │
│                                │                                   │
│                         Gate Conv (1×1 + Sigmoid)                  │
│                                │                                   │
│              gate · Stream A  +  (1−gate) · Stream B               │
│                                │                                   │
│                   SE pool → fc → fc → sigmoid → scale              │
│                                │                                   │
│                          out_conv (1×1)                            │
│                                │                                   │
│                          + residual(xl)                            │
│                                │                                   │
│                           output (B, C_l, H, W)                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Citation

If you use KransFormer in your research, please cite:
```bibtex
@article{kransformer2025,
  title   = {KransFormer: Dual-Stream Gated KAN-Transformer for
             Medical Image Segmentation},
  author  = {Your Name and Collaborators},
  journal = {arXiv preprint arXiv:2025.XXXXX},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

This work builds on the following:

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (Liu et al., 2024) — B-spline KAN foundation
- [UKAN](https://arxiv.org/abs/2406.02918) — KAN applied to UNet-style segmentation
- [SegFormer](https://arxiv.org/abs/2105.15203) — Overlapping patch embeddings and hierarchical transformers
- [CBAM](https://arxiv.org/abs/1807.06521) — Channel and spatial attention
- [U-Net](https://arxiv.org/abs/1505.04597) — Original encoder–decoder with skip connections
- [CAMUS Dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/) — Cardiac ultrasound benchmark

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for the medical imaging community
</p>

# KransFormer 👑

**KransFormer: Dual-Stream Gated KAN-Transformer for Medical Image Segmentation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025-red.svg)](#citation)

<p align="center">
  <img src="full daigram.png" width="800" alt="KransFormer Architecture"/>
</p>

> **KransFormer** (from *krans* — Dutch/Afrikaans for "crown") is a novel hybrid
> encoder–decoder architecture for medical image segmentation that replaces standard
> skip connections with a learnable **Dual-Stream Gated Feature Aggregation (DSGFA)**
> module, and introduces **KAN-Attention** and **Adaptive Spline Positional Encoding
> (ASPE)** for richer token interactions.

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

"""
KransFormer: Dual-Stream Gated KAN-Transformer for Medical Image Segmentation

Novel contributions:
  1. DSGFA  — Dual-Stream Gated Feature Aggregation (novel skip-fusion module)
  2. KANAttention  — Multi-head attention with B-spline KAN Q/K/V projections
  3. ASPE   — Adaptive Spline Positional Encoding (sinusoidal PE + KAN refiner)
  4. KANFFN — KAN feed-forward network with interleaved depthwise convolution
  5. Progressive Deep Supervision with learnable loss weights
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (int(x), int(x))
    return tuple(x)


class DropPath(nn.Module):
    """Stochastic depth regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep_prob) * rand


# ─────────────────────────────────────────────────────────────────────────────
# KANLinear  (B-spline Kolmogorov–Arnold Network layer)
# ─────────────────────────────────────────────────────────────────────────────

class KANLinear(nn.Module):
    """
    B-spline KAN linear layer.

    Each input–output pair is modelled as:
        y = SiLU(x) @ base_weight  +  B-spline(x) @ spline_weight

    Args:
        in_features   : input dimensionality
        out_features  : output dimensionality
        grid_size     : number of B-spline grid intervals
        spline_order  : order of the B-spline (default cubic = 3)
        scale_noise   : noise scale for weight initialisation
        scale_base    : scale for the base (SiLU) branch init
        scale_spline  : scale for the spline branch init
        grid_eps      : blend ratio between uniform and adaptive grid
        grid_range    : initial uniform grid range
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.grid_eps     = grid_eps

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise   = scale_noise
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline
        self.base_activation = base_activation()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                self.curve2coeff(self.grid.T[self.spline_order: -self.spline_order], noise)
            )
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1: (-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_out   = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """Adaptive grid update based on current batch statistics."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines    = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced  = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted   = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        unif_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * unif_step + x_sorted[0] - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([
            grid[:1] - unif_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + unif_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
        ], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced))

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        l1_fake  = self.spline_weight.abs().mean(-1)
        act_loss = l1_fake.sum()
        p        = l1_fake / (act_loss + 1e-8)
        entropy_loss = -torch.sum(p * p.log().clamp(min=-100))
        return regularize_activation * act_loss + regularize_entropy * entropy_loss


# ─────────────────────────────────────────────────────────────────────────────
# Novel Contribution 3: ASPE — Adaptive Spline Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class ASPE(nn.Module):
    """
    Adaptive Spline Positional Encoding.

    A 2-D sinusoidal PE is first computed analytically, then refined by a
    single KANLinear layer. This lets the model learn spatial biases that
    standard sinusoidal PE cannot express, with negligible extra parameters.

    Args:
        embed_dim : token channel dimension
        max_len   : maximum sequence length (H * W)
    """
    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.refiner   = KANLinear(embed_dim, embed_dim, grid_size=3, spline_order=2)
        pe = self._build_sinusoidal(max_len, embed_dim)
        self.register_buffer("pe_table", pe)

    @staticmethod
    def _build_sinusoidal(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: dim // 2])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, C)"""
        B, N, C = x.shape
        raw_pe  = self.pe_table[:N]          # (N, C)
        refined = self.refiner(raw_pe)       # (N, C)
        return x + refined.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Depthwise Conv helper
# ─────────────────────────────────────────────────────────────────────────────

class DW_BN_ReLU(nn.Module):
    """Depthwise 3×3 conv + BN + ReLU for local spatial mixing in sequences."""
    def __init__(self, dim: int):
        super().__init__()
        self.dw   = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn   = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dw(x)))
        return x.flatten(2).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Novel Contribution 2: KANAttention
# ─────────────────────────────────────────────────────────────────────────────

class KANAttention(nn.Module):
    """
    Multi-head self-attention where Q, K, V projections are realized by
    independent KANLinear layers, enabling learned non-linear token
    interactions beyond what standard affine projections can express.

    Args:
        dim       : token dimension (must be divisible by num_heads)
        num_heads : number of attention heads
        attn_drop : attention weight dropout rate
        proj_drop : output projection dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Three independent KAN projections for Q, K, V
        self.kan_q = KANLinear(dim, dim, grid_size=5, spline_order=3)
        self.kan_k = KANLinear(dim, dim, grid_size=5, spline_order=3)
        self.kan_v = KANLinear(dim, dim, grid_size=5, spline_order=3)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x2d = x.reshape(B * N, C)

        q = self.kan_q(x2d).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.kan_k(x2d).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.kan_v(x2d).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


# ─────────────────────────────────────────────────────────────────────────────
# Novel Contribution 4: KANFFN
# ─────────────────────────────────────────────────────────────────────────────

class KANFFN(nn.Module):
    """
    KAN Feed-Forward Network with interleaved depthwise convolution.

    Structure:  x → KAN_fc1 → DW_BN_ReLU → KAN_fc2 → output
    """
    def __init__(self, dim: int, hidden_dim: int = None, drop: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1  = KANLinear(dim, hidden_dim, grid_size=5, spline_order=3)
        self.dw   = DW_BN_ReLU(hidden_dim)
        self.fc2  = KANLinear(hidden_dim, dim, grid_size=5, spline_order=3)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, -1)
        x = self.dw(x, H, W)
        x = self.fc2(x.reshape(B * N, x.shape[-1])).reshape(B, N, -1)
        return self.drop(x)


# ─────────────────────────────────────────────────────────────────────────────
# KransFormer Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class KransFormerBlock(nn.Module):
    """
    KransFormer transformer block combining KANAttention + KANFFN.

    Pre-LN formulation:
        x = x + DropPath(KANAttention(LN(x)))
        x = x + DropPath(KANFFN(LN(x), H, W))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 1.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = KANAttention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.ffn       = KANFFN(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Overlap Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class OverlapPatchEmbed(nn.Module):
    """Overlapping patch tokenizer for rich local context at each scale."""
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 128,
        embed_dim: int = 160,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder Conv Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Double 3×3 conv encoder block with BN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class DeconvBlock(nn.Module):
    """Double 3×3 conv decoder block (in→in→out)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Novel Contribution 1: DSGFA — Dual-Stream Gated Feature Aggregation
# ─────────────────────────────────────────────────────────────────────────────

class DSGFA(nn.Module):
    """
    Dual-Stream Gated Feature Aggregation.

    Replaces standard additive UNet skip connections with a two-stream
    gated mechanism:

      Stream A — Projects high-level features (xh) to the spatial resolution
                 of low-level features (xl), then applies four parallel dilated
                 depth-wise convolutions (d=1,2,4,8) to capture multi-scale
                 semantic context. Optionally guided by a deep-supervision map.

      Stream B — Passes low-level features (xl) through a KANLinear refiner,
                 preserving fine-grained texture detail via learned non-linear
                 transformations.

      Gating   — A sigmoid gate (learned from the concatenation of both
                 streams) adaptively controls information flow. This is
                 equivalent to a spatial soft-selection: regions where high-
                 level context is informative get more Stream A signal;
                 regions where fine detail dominates get more Stream B.

      SE       — Squeeze-Excitation channel re-calibration follows gating.

    Args:
        dim_xh    : channels of high-level (deeper) input features
        dim_xl    : channels of low-level (shallower) input features (= output)
        d_list    : dilation rates for multi-scale depth-wise convolutions
        reduction : SE squeeze ratio
    """
    def __init__(
        self,
        dim_xh: int,
        dim_xl: int,
        d_list: tuple = (1, 2, 4, 8),
        reduction: int = 4,
    ):
        super().__init__()
        # Project high-level channels to match low-level
        self.proj_high = nn.Conv2d(dim_xh, dim_xl, 1, bias=False)

        # Stream A: multi-scale dilated depth-wise convolutions
        self.dil_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_xl, dim_xl, 3, padding=d, dilation=d, groups=dim_xl, bias=False),
                nn.BatchNorm2d(dim_xl),
                nn.GELU(),
            )
            for d in d_list
        ])
        self.ms_fuse = nn.Conv2d(dim_xl * len(d_list), dim_xl, 1, bias=False)

        # Stream B: KAN refiner on low-level features
        self.kan_low = KANLinear(dim_xl, dim_xl, grid_size=4, spline_order=3)

        # Gating network
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim_xl * 2, dim_xl, 1, bias=False),
            nn.BatchNorm2d(dim_xl),
            nn.Sigmoid(),
        )

        # Squeeze-Excitation re-calibration
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc   = nn.Sequential(
            nn.Linear(dim_xl, max(dim_xl // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim_xl // reduction, 4), dim_xl, bias=False),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim_xl, dim_xl, 1, bias=False),
            nn.BatchNorm2d(dim_xl),
            nn.GELU(),
        )

    def forward(
        self,
        xh: torch.Tensor,
        xl: torch.Tensor,
        guide: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            xh    : high-level feature map  (B, dim_xh, H', W')
            xl    : low-level  feature map  (B, dim_xl, H,  W)
            guide : optional DS guidance mask (B, 1, H', W')
        Returns:
            fused feature map  (B, dim_xl, H, W)
        """
        # ── Stream A ──────────────────────────────────────────────────
        xh_proj = self.proj_high(xh)
        xh_up   = F.interpolate(xh_proj, size=xl.shape[2:], mode="bilinear", align_corners=True)

        if guide is not None:
            g     = torch.sigmoid(
                F.interpolate(guide, size=xl.shape[2:], mode="bilinear", align_corners=True)
            )
            xh_up = xh_up * (g * 0.5 + 0.5)    # soft guidance — avoid full suppression

        ms       = torch.cat([conv(xh_up) for conv in self.dil_convs], dim=1)
        stream_a = self.ms_fuse(ms)              # (B, dim_xl, H, W)

        # ── Stream B ──────────────────────────────────────────────────
        B, C, H, W  = xl.shape
        xl_flat     = xl.flatten(2).transpose(1, 2).reshape(B * H * W, C)
        stream_b    = self.kan_low(xl_flat).reshape(B, H * W, C).transpose(1, 2).reshape(B, C, H, W)

        # ── Gated Fusion ──────────────────────────────────────────────
        gate   = self.gate_conv(torch.cat([stream_a, stream_b], dim=1))
        fused  = gate * stream_a + (1.0 - gate) * stream_b

        # ── SE Re-calibration ─────────────────────────────────────────
        se_w  = self.se_pool(fused).view(B, C)
        se_w  = self.se_fc(se_w).view(B, C, 1, 1)
        fused = fused * se_w

        return self.out_conv(fused) + xl    # residual to xl


# ─────────────────────────────────────────────────────────────────────────────
# KransFormer — Full Segmentation Network
# ─────────────────────────────────────────────────────────────────────────────

class KransFormer(nn.Module):
    """
    KransFormer: Dual-Stream Gated KAN-Transformer for Medical Image Segmentation.

    Encoder
    -------
    Stages 1–3 : ConvBlock + MaxPool  (local texture hierarchy)
    Stages 4–5 : OverlapPatchEmbed → ASPE → KransFormerBlock stack (global context)

    Decoder
    -------
    DeconvBlock + bilinear upsample, with DSGFA-gated skip connections at each
    stage and progressive deep supervision via learnable-weight auxiliary heads.

    Args:
        num_classes    : segmentation output channels (1 for binary)
        in_chans       : input image channels
        img_size       : spatial resolution (square assumed)
        embed_dims     : channel dims for the 5 encoder stages
        deep_sup       : enable progressive deep supervision
        drop_rate      : token dropout inside transformer blocks
        drop_path_rate : stochastic depth max rate
        num_heads      : attention heads in transformer stages
        trans_depth    : number of KransFormerBlocks per transformer stage
        mlp_ratio      : FFN hidden-dim multiplier
    """
    def __init__(
        self,
        num_classes: int = 1,
        in_chans: int = 1,
        img_size: int = 128,
        embed_dims: tuple = (32, 64, 128, 160, 256),
        deep_sup: bool = True,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        num_heads: int = 8,
        trans_depth: int = 2,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.deep_sup = deep_sup
        self.img_size = to_2tuple(img_size)
        E = embed_dims

        # ── Convolutional Encoder ──────────────────────────────────────
        self.enc1 = ConvBlock(in_chans, E[0])
        self.enc2 = ConvBlock(E[0],     E[1])
        self.enc3 = ConvBlock(E[1],     E[2])
        self.bn1  = nn.BatchNorm2d(E[0])
        self.bn2  = nn.BatchNorm2d(E[1])
        self.bn3  = nn.BatchNorm2d(E[2])

        # ── Transformer Encoder ───────────────────────────────────────
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8,  patch_size=3, stride=2,
            in_chans=E[2], embed_dim=E[3],
        )
        self.patch_embed5 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_chans=E[3], embed_dim=E[4],
        )

        # ASPE for each transformer stage
        self.aspe4 = ASPE(E[3], max_len=(img_size // 16) ** 2 + 64)
        self.aspe5 = ASPE(E[4], max_len=(img_size // 32) ** 2 + 64)

        dpr = torch.linspace(0, drop_path_rate, trans_depth * 2).tolist()
        self.trans4 = nn.ModuleList([
            KransFormerBlock(E[3], num_heads, mlp_ratio, drop_rate, dpr[i])
            for i in range(trans_depth)
        ])
        self.trans5 = nn.ModuleList([
            KransFormerBlock(E[4], num_heads, mlp_ratio, drop_rate, dpr[trans_depth + i])
            for i in range(trans_depth)
        ])
        self.norm4 = nn.LayerNorm(E[3])
        self.norm5 = nn.LayerNorm(E[4])

        # ── Decoder ───────────────────────────────────────────────────
        self.dec1 = DeconvBlock(E[4], E[3])
        self.dec2 = DeconvBlock(E[3], E[2])
        self.dec3 = DeconvBlock(E[2], E[1])
        self.dec4 = DeconvBlock(E[1], E[0])
        self.dec5 = DeconvBlock(E[0], E[0])
        self.dbn1 = nn.BatchNorm2d(E[3])
        self.dbn2 = nn.BatchNorm2d(E[2])
        self.dbn3 = nn.BatchNorm2d(E[1])
        self.dbn4 = nn.BatchNorm2d(E[0])

        # ── DSGFA Skip-Fusion Modules ──────────────────────────────────
        self.dsgfa4 = DSGFA(E[4], E[3])   # fuse x5 → x4
        self.dsgfa3 = DSGFA(E[3], E[2])   # fuse d1  → x3
        self.dsgfa2 = DSGFA(E[2], E[1])   # fuse d2  → x2
        self.dsgfa1 = DSGFA(E[1], E[0])   # fuse d3  → x1

        # ── Progressive Deep Supervision ──────────────────────────────
        if deep_sup:
            self.ds4     = nn.Conv2d(E[3], num_classes, 1)
            self.ds3     = nn.Conv2d(E[2], num_classes, 1)
            self.ds2     = nn.Conv2d(E[1], num_classes, 1)
            self.ds1     = nn.Conv2d(E[0], num_classes, 1)
            # Learnable weights — softmax-normalised during loss computation
            self.ds_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))

        # ── Final Segmentation Head ────────────────────────────────────
        self.head = nn.Conv2d(E[0], num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        # ── Convolutional Encoder ──────────────────────────────────────
        x1 = F.relu(F.max_pool2d(self.bn1(self.enc1(x)),  2, 2))
        x2 = F.relu(F.max_pool2d(self.bn2(self.enc2(x1)), 2, 2))
        x3 = F.relu(F.max_pool2d(self.bn3(self.enc3(x2)), 2, 2))

        # ── Transformer Encoder — Stage 4 ─────────────────────────────
        tok4, H4, W4 = self.patch_embed4(x3)
        tok4 = self.aspe4(tok4)
        for blk in self.trans4:
            tok4 = blk(tok4, H4, W4)
        tok4 = self.norm4(tok4)
        x4   = tok4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        # ── Transformer Encoder — Stage 5 ─────────────────────────────
        tok5, H5, W5 = self.patch_embed5(x4)
        tok5 = self.aspe5(tok5)
        for blk in self.trans5:
            tok5 = blk(tok5, H5, W5)
        tok5 = self.norm5(tok5)
        x5   = tok5.reshape(B, H5, W5, -1).permute(0, 3, 1, 2).contiguous()

        # ── Decoder + DSGFA ───────────────────────────────────────────
        # Stage 1: decode x5, fuse with x4
        d1 = F.relu(F.interpolate(self.dbn1(self.dec1(x5)), scale_factor=2, mode="bilinear", align_corners=True))
        if self.deep_sup:
            ds4_map = self.ds4(d1)
            x4      = self.dsgfa4(x5, x4, guide=ds4_map)
            ds4_out = F.interpolate(ds4_map, scale_factor=16, mode="bilinear", align_corners=True)
        else:
            x4 = self.dsgfa4(x5, x4)
        d1 = d1 + x4

        # Stage 2: decode d1, fuse with x3
        d2 = F.relu(F.interpolate(self.dbn2(self.dec2(d1)), scale_factor=2, mode="bilinear", align_corners=True))
        if self.deep_sup:
            ds3_map = self.ds3(d2)
            x3      = self.dsgfa3(d1, x3, guide=ds3_map)
            ds3_out = F.interpolate(ds3_map, scale_factor=8, mode="bilinear", align_corners=True)
        else:
            x3 = self.dsgfa3(d1, x3)
        d2 = d2 + x3

        # Stage 3: decode d2, fuse with x2
        d3 = F.relu(F.interpolate(self.dbn3(self.dec3(d2)), scale_factor=2, mode="bilinear", align_corners=True))
        if self.deep_sup:
            ds2_map = self.ds2(d3)
            x2      = self.dsgfa2(d2, x2, guide=ds2_map)
            ds2_out = F.interpolate(ds2_map, scale_factor=4, mode="bilinear", align_corners=True)
        else:
            x2 = self.dsgfa2(d2, x2)
        d3 = d3 + x2

        # Stage 4: decode d3, fuse with x1
        d4 = F.relu(F.interpolate(self.dbn4(self.dec4(d3)), scale_factor=2, mode="bilinear", align_corners=True))
        if self.deep_sup:
            ds1_map = self.ds1(d4)
            x1      = self.dsgfa1(d3, x1, guide=ds1_map)
            ds1_out = F.interpolate(ds1_map, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            x1 = self.dsgfa1(d3, x1)
        d4 = d4 + x1

        # Final upsample → original resolution
        d5  = F.relu(F.interpolate(self.dec5(d4), scale_factor=2, mode="bilinear", align_corners=True))
        out = self.head(d5)

        if self.deep_sup:
            w = torch.softmax(self.ds_weights, dim=0)
            return (ds4_out * w[0], ds3_out * w[1], ds2_out * w[2], ds1_out * w[3]), out
        return out

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        """Aggregate B-spline regularization across all KANLinear layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, KANLinear):
                total = total + m.regularization_loss(regularize_activation, regularize_entropy)
        return total

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HAS_MAMBA = False
Mamba = None
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError as e:
    print(f"[Warning] mamba_ssm.Mamba import failed: {e}")
    try:
        from mamba_ssm import Mamba2 as Mamba
        HAS_MAMBA = True
        print("[Info] Falling back to mamba_ssm.Mamba2")
    except ImportError as e2:
        print(f"[Error] mamba_ssm.Mamba2 also failed: {e2}")
        print("Please ensure: pip install mamba-ssm causal-conv1d --no-build-isolation")

from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


def create_mamba(mamba_version, d_model, d_state, d_conv, expand, **kwargs):
    """Create Mamba (v1 S6) or Mamba2 (SSD) module with given params."""
    if mamba_version == 'mamba1':
        if Mamba is None:
            raise ImportError("mamba_ssm.Mamba not available.")
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    elif mamba_version == 'mamba2':
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError("mamba_ssm.Mamba2 not available.")
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                      headdim=kwargs.get('headdim', 64), ngroups=kwargs.get('ngroups', 1))
    else:
        raise ValueError(f"Unknown mamba_version: {mamba_version}")


# ============================================================
# Base components (copied from original model.py)
# ============================================================

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# ============================================================
# Stage 1: Direct Transformer → Mamba replacement
# ============================================================

class MambaResBlock(nn.Module):
    """Mamba block matching ResidualAttentionBlock structure.
    Pre-LN → Mamba → residual → Pre-LN → MLP(4D→D) → residual.
    Dropout applied on sub-layer outputs before residual addition.
    """
    def __init__(self, d_model: int, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 mamba_version='mamba1', mlp_scale=4, **mamba_kwargs):
        super().__init__()
        self.mamba = create_mamba(mamba_version, d_model, d_state, d_conv, expand, **mamba_kwargs)
        self.ln_1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * mlp_scale)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * mlp_scale, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.mamba(self.ln_1(x)))
        x = x + self.dropout2(self.mlp(self.ln_2(x)))
        return x


class MambaTemporalEncoder(nn.Module):
    """Drop-in replacement for Transformer. Processes [T, B, D] → [T, B, D].
    Uses MambaResBlock (Mamba + MLP + residuals) matching Transformer block structure.
    """
    def __init__(self, d_model=512, n_layer=2, d_state=16, d_conv=4, expand=2, attn_window=0, dropout=0.0,
                 mamba_version='mamba1', mlp_scale=4, **mamba_kwargs):
        super().__init__()
        self.attn_window = attn_window
        self.layers = nn.ModuleList([
            MambaResBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                          dropout=dropout, mamba_version=mamba_version, mlp_scale=mlp_scale, **mamba_kwargs)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [T, B, D] → Mamba expects [B, T, D]
        x = x.permute(1, 0, 2)
        B, T, D = x.shape

        if self.attn_window > 0 and T > self.attn_window:
            num_windows = (T + self.attn_window - 1) // self.attn_window
            padded_len = num_windows * self.attn_window
            if padded_len > T:
                x = F.pad(x, (0, 0, 0, padded_len - T))
            x = x.view(B, num_windows, self.attn_window, D)
            x = x.view(B * num_windows, self.attn_window, D)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if self.attn_window > 0 and T > self.attn_window:
            x = x.view(B, num_windows, self.attn_window, D)
            x = x.view(B, padded_len, D)[:, :T, :]

        return x.permute(1, 0, 2)


# ============================================================
# Stage 1b: Bidirectional Mamba (forward + backward scan)
# ============================================================

class BidirectionalMambaBlock(nn.Module):
    """Mamba block with bidirectional processing.
    Forward scan + backward scan → concat → Linear(2D→D) → residual → MLP → residual.
    """
    def __init__(self, d_model: int, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 mamba_version='mamba1', mlp_scale=4, **mamba_kwargs):
        super().__init__()
        self.fwd_mamba = create_mamba(mamba_version, d_model, d_state, d_conv, expand, **mamba_kwargs)
        self.bwd_mamba = create_mamba(mamba_version, d_model, d_state, d_conv, expand, **mamba_kwargs)
        self.ln_1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * mlp_scale)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * mlp_scale, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.ln_1(x)

        fwd = self.fwd_mamba(x_norm)

        bwd = self.bwd_mamba(torch.flip(x_norm, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])

        combined = torch.cat([fwd, bwd], dim=-1)
        combined = self.proj(combined)
        x = residual + self.dropout1(combined)
        x = x + self.dropout2(self.mlp(self.ln_2(x)))
        return x


class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba encoder. Processes [T, B, D] → [T, B, D]."""
    def __init__(self, d_model=512, n_layer=2, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 mamba_version='mamba1', mlp_scale=4, **mamba_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                                    expand=expand, dropout=dropout,
                                    mamba_version=mamba_version, mlp_scale=mlp_scale, **mamba_kwargs)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [T, B, D]
        x = x.permute(1, 0, 2)  # [B, T, D]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.permute(1, 0, 2)  # [T, B, D]


# ============================================================
# Stage 2: Text-Gated MambaBlock
# ============================================================

class TextGatedMambaBlock(nn.Module):
    """Mamba with text-guided residual gating. No cross-attention."""
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2, text_dim=512,
                 gate_type='sigmoid', attn_window=0, mamba_version='mamba1', **mamba_kwargs):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba-ssm not installed.")
        self.attn_window = attn_window
        self.mamba = create_mamba(mamba_version, d_model, d_state, d_conv, expand, **mamba_kwargs)
        self.norm = nn.LayerNorm(d_model)

        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.Sigmoid() if gate_type == 'sigmoid' else nn.Tanh()
        )
        self.gate_type = gate_type

    def forward(self, x, text_features):
        """
        x: [B, T, D] video features (batch-first)
        text_features: [N_text, D] CLIP text embeddings for all classes
        """
        B, T, D = x.shape

        if self.attn_window > 0 and T > self.attn_window:
            num_windows = (T + self.attn_window - 1) // self.attn_window
            padded_len = num_windows * self.attn_window
            if padded_len > T:
                x = F.pad(x, (0, 0, 0, padded_len - T))
            x = x.view(B * num_windows, self.attn_window, D)

        sim = (x @ text_features.T).softmax(dim=-1)  # [B, T, N_text]
        text_context = sim @ text_features            # [B, T, D]
        gate = self.text_gate(text_context)           # [B, T, D]

        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)

        out = x + gate * x_mamba

        if self.attn_window > 0 and T > self.attn_window:
            out = out.view(B, num_windows, self.attn_window, D)
            out = out.view(B, padded_len, D)[:, :T, :]

        return out


class TextGatedMambaEncoder(nn.Module):
    """Stacked TextGatedMambaBlock layers."""
    def __init__(self, d_model=512, n_layer=2, d_state=16, d_conv=4, expand=2,
                 text_dim=512, gate_type='sigmoid', attn_window=0,
                 mamba_version='mamba1', **mamba_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TextGatedMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                                text_dim=text_dim, gate_type=gate_type, attn_window=attn_window,
                                mamba_version=mamba_version, **mamba_kwargs)
            for _ in range(n_layer)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, text_features):
        # x: [T, B, D]
        x = x.permute(1, 0, 2)  # [B, T, D]
        for layer in self.layers:
            x = layer(x, text_features)
        x = self.out_norm(x)
        return x.permute(1, 0, 2)  # [T, B, D]


# ============================================================
# Stage 3: Multi-Scale Mamba (different d_state per branch)
# ============================================================

class MultiScaleMambaEncoder(nn.Module):
    """Three parallel text-gated Mamba branches with different d_state, fused by concat+MLP."""
    def __init__(self, d_model=512, n_layer=2, d_state_list=None, d_conv=4, expand=2,
                 text_dim=512, gate_type='sigmoid', attn_window=0,
                 mamba_version='mamba1', **mamba_kwargs):
        super().__init__()
        if d_state_list is None:
            d_state_list = [8, 16, 32]

        self.short_term = TextGatedMambaBlock(
            d_model=d_model, d_state=d_state_list[0], d_conv=d_conv, expand=expand,
            text_dim=text_dim, gate_type=gate_type, attn_window=attn_window,
            mamba_version=mamba_version, **mamba_kwargs)
        self.mid_term = TextGatedMambaBlock(
            d_model=d_model, d_state=d_state_list[1], d_conv=d_conv, expand=expand,
            text_dim=text_dim, gate_type=gate_type, attn_window=attn_window,
            mamba_version=mamba_version, **mamba_kwargs)
        self.long_term = TextGatedMambaBlock(
            d_model=d_model, d_state=d_state_list[2], d_conv=d_conv, expand=expand,
            text_dim=text_dim, gate_type=gate_type, attn_window=attn_window,
            mamba_version=mamba_version, **mamba_kwargs)

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, text_features):
        # x: [T, B, D]
        x_short = self.short_term(x, text_features)
        x_mid = self.mid_term(x, text_features)
        x_long = self.long_term(x, text_features)

        # Concat along feature dim + fuse back to d_model
        x_fused = torch.cat([x_short, x_mid, x_long], dim=-1)  # [T, B, 3D]
        x_fused = self.fuse(x_fused)
        x_fused = self.out_norm(x_fused)
        return x_fused


# ============================================================
# Stage 4: Hybrid Transformer + Mamba (Dual-Path)
# ============================================================

class HybridTemporalEncoder(nn.Module):
    """Complementary hybrid: Transformer does inter-frame comparison,
    Mamba models temporal regularity.

    Each frame gets four views:
      1. Transformer feat — how this frame relates to others
      2. Original input    — raw CLIP feature at this frame
      3. Mamba feat        — temporal context-aware encoding
      4. Residual          — |input - mamba|, deviation from temporal expectation

    Concatenated → MLP → fused frame-level feature.
    """
    def __init__(self, d_model, n_head, transformer_layers, attn_mask,
                 mamba_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 mamba_version='mamba1', **mamba_kwargs):
        super().__init__()
        self.transformer = Transformer(d_model, transformer_layers, n_head, attn_mask)
        self.mamba = MambaTemporalEncoder(
            d_model=d_model, n_layer=mamba_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, mamba_version=mamba_version, **mamba_kwargs)

        self.fusion = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model * 4, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [T, B, D]
        # Path 1: Transformer → frame-to-frame comparison
        x_trans, _ = self.transformer((x, None))  # [T, B, D]

        # Path 2: Mamba → temporal regularity (per-frame output)
        x_mamba = self.mamba(x)                    # [T, B, D]

        # Residual: how much each frame deviates from Mamba's temporal expectation
        x_residual = torch.abs(x - x_mamba)         # [T, B, D]

        # Concat four complementary views
        x_cat = torch.cat([x_trans, x, x_mamba, x_residual], dim=-1)  # [T, B, 4D]

        # Fuse back to d_model
        x_out = self.fusion(x_cat)
        x_out = self.norm(x_out)
        return x_out


# ============================================================
# Stage 5: AVA-DINO inspired — Frame-level Dynamic Routing
# ============================================================

class AVAHybridEncoder(nn.Module):
    """AVA-DINO inspired dual-path with per-frame dynamic routing.

    Per-frame routing via softmax over similarity to 'Normal' vs 'Anomaly' text.
      normal_text = text_features[0]   ("Normal")
      anomaly_text = mean(text_features[1:])  (all anomaly classes)

    [w_normal, w_anomaly] = softmax([sim_normal, sim_anomaly])

    Route-weighted fusion:
      concat([w_normal * Mamba(x), w_anomaly * Transformer(x)]) → MLP

    Routing weights stored for regularization loss in training loop.
    """
    def __init__(self, d_model, n_head, transformer_layers, attn_mask,
                 mamba_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 mamba_version='mamba1', **mamba_kwargs):
        super().__init__()
        self.transformer = Transformer(d_model, transformer_layers, n_head, attn_mask)
        self.mamba = MambaTemporalEncoder(
            d_model=d_model, n_layer=mamba_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, mamba_version=mamba_version, **mamba_kwargs)

        self.logit_scale = nn.Parameter(torch.tensor(4.0))

        self.fusion = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model * 2, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, text_features):
        # x: [T, B, D]
        # text_features: [N_cls, D], [0]="Normal", [1:]=anomaly classes

        x_mamba = self.mamba(x)               # [T, B, D]
        x_trans, _ = self.transformer((x, None))  # [T, B, D]

        # Build normal vs anomaly text prototypes
        norm_text = text_features[0:1]                              # [1, D]
        anom_text = text_features[1:].mean(dim=0, keepdim=True)     # [1, D]

        # Cosine similarities (per-frame)
        x_b = x.permute(1, 0, 2)   # [B, T, D]
        x_fn = F.normalize(x_b, dim=-1)
        t_norm = F.normalize(norm_text, dim=-1)
        t_anom = F.normalize(anom_text, dim=-1)

        sim_normal = x_fn @ t_norm.permute(1, 0)   # [B, T, 1]
        sim_anomaly = x_fn @ t_anom.permute(1, 0)  # [B, T, 1]

        # Softmax over normal vs anomaly
        logits = torch.cat([sim_normal, sim_anomaly], dim=-1)  # [B, T, 2]
        w = F.softmax(logits * self.logit_scale, dim=-1)       # [B, T, 2]
        w_normal_tb = w[..., 0:1].permute(1, 0, 2)  # [T, B, 1]
        w_anomaly_tb = w[..., 1:2].permute(1, 0, 2)

        # Route-weighted fusion: gate each branch's contribution before MLP
        x_route = torch.cat([w_normal_tb * x_mamba, w_anomaly_tb * x_trans], dim=-1)
        x_out = self.fusion(x_route)
        x_out = self.norm(x_out)

        # Store per-frame w_normal for regularization loss in training loop
        self._routing_weights = w[..., 0].detach()  # [B, T]

        return x_out


# ============================================================
# Main model: CLIPVAD_Mamba
# ============================================================

class CLIPVAD_Mamba(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device,
                 encoder_type: str = 'mamba',
                 d_state: int = 16,
                 d_state_list: str = '8,16,32',
                 d_conv: int = 4,
                 expand: int = 2,
                 gate_type: str = 'sigmoid',
                 dropout: float = 0.0,
                 mamba_version: str = 'mamba1',
                 mamba2_headdim: int = 64,
                 mamba2_ngroups: int = 1):
        super().__init__()

        # Prepare mamba2 kwargs if needed
        mamba_kwargs = {}
        if mamba_version == 'mamba2':
            mamba_kwargs = {'headdim': mamba2_headdim, 'ngroups': mamba2_ngroups}

        self.num_class = num_class
        self.embed_dim = embed_dim
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.encoder_type = encoder_type

        # Build temporal encoder based on ablation setting
        if encoder_type == 'transformer':
            self.temporal = Transformer(
                width=visual_width, layers=visual_layers, heads=visual_head,
                attn_mask=self.build_attention_mask(self.attn_window))
        elif encoder_type == 'mamba':
            self.temporal = MambaTemporalEncoder(
                d_model=visual_width, n_layer=visual_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mamba_version=mamba_version, **mamba_kwargs)
        elif encoder_type == 'bi-mamba':
            self.temporal = BiMambaEncoder(
                d_model=visual_width, n_layer=visual_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mamba_version=mamba_version, **mamba_kwargs)
        elif encoder_type == 'text-gated-mamba':
            self.temporal = TextGatedMambaEncoder(
                d_model=visual_width, n_layer=visual_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                text_dim=embed_dim, gate_type=gate_type,
                mamba_version=mamba_version, **mamba_kwargs)
        elif encoder_type == 'multi-scale-mamba':
            d_states = [int(x) for x in d_state_list.split(',')]
            self.temporal = MultiScaleMambaEncoder(
                d_model=visual_width, n_layer=visual_layers,
                d_state_list=d_states, d_conv=d_conv, expand=expand,
                text_dim=embed_dim, gate_type=gate_type,
                mamba_version=mamba_version, **mamba_kwargs)
        elif encoder_type == 'hybrid':
            self.temporal = HybridTemporalEncoder(
                d_model=visual_width, n_head=visual_head,
                transformer_layers=visual_layers,
                attn_mask=self.build_attention_mask(self.attn_window),
                mamba_layers=visual_layers, d_state=d_state, d_conv=d_conv,
                expand=expand, dropout=dropout,
                mamba_version=mamba_version, **mamba_kwargs)
        elif encoder_type == 'ava-hybrid':
            self.temporal = AVAHybridEncoder(
                d_model=visual_width, n_head=visual_head,
                transformer_layers=visual_layers,
                attn_mask=self.build_attention_mask(self.attn_window),
                mamba_layers=visual_layers, d_state=d_state, d_conv=d_conv,
                expand=expand, dropout=dropout,
                mamba_version=mamba_version, **mamba_kwargs)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # GCN branches (identical to original)
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = F.threshold(tmp, 0.7, 0)
                output[i] = F.softmax(adj2, dim=1)
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = F.threshold(tmp, 0.7, 0)
                output[i, :seq_len[i], :seq_len[i]] = F.softmax(adj2, dim=1)
        return output

    def encode_video(self, images, padding_mask, lengths, text_features=None):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        # Dispatch based on encoder type
        if self.encoder_type == 'transformer':
            x, _ = self.temporal((images, None))
        elif self.encoder_type in ('mamba', 'bi-mamba', 'hybrid'):
            x = self.temporal(images)
        else:  # text-gated-mamba, multi-scale-mamba, ava-hybrid
            x = self.temporal(images, text_features)

        x = x.permute(1, 0, 2)

        # GCN branches (unchanged)
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))
        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        text_features_ori = self.encode_textprompt(text)
        visual_features = self.encode_video(visual, padding_mask, lengths, text_features_ori)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])

        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        # Retrieve routing weights for ava-hybrid encoder (None for other encoders)
        routing_weights = getattr(getattr(self, 'temporal', None), '_routing_weights', None)

        return text_features_ori, logits1, logits2, routing_weights

"""
Mamba building blocks for single bidirectional Mamba VAD.
Contains only components used in the paper: MambaResBlock, MambaTemporalEncoder,
BidirectionalMambaBlock, BiMambaEncoder.
"""
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn

HAS_MAMBA = False
Mamba = None
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    try:
        from mamba_ssm import Mamba2 as Mamba
        HAS_MAMBA = True
    except ImportError as e:
        print(f"[Warning] mamba_ssm not available: {e}")
        print("Please: pip install mamba-ssm causal-conv1d --no-build-isolation")


def create_mamba(mamba_version, d_model, d_state, d_conv, expand, **kwargs):
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


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MambaResBlock(nn.Module):
    """Mamba block: Pre-LN → Mamba → residual → Pre-LN → MLP(4D→D) → residual."""
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
    """Causal Mamba encoder. [T, B, D] → [T, B, D]."""
    def __init__(self, d_model=512, n_layer=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, mamba_version='mamba1', mlp_scale=4, **mamba_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaResBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                          dropout=dropout, mamba_version=mamba_version, mlp_scale=mlp_scale, **mamba_kwargs)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [T, B, D] → Mamba expects [B, T, D]
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.permute(1, 0, 2)  # [T, B, D]


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
    """Bidirectional Mamba encoder. [T, B, D] → [T, B, D]."""
    def __init__(self, d_model=512, n_layer=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, mamba_version='mamba1', mlp_scale=4, **mamba_kwargs):
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

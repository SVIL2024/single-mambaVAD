"""
Single bidirectional Mamba VAD model (paper version).

Architecture (single Mamba, no GCN, no dual):
  CLIP features → Position Embed → BiMambaEncoder → Multi-scale fusion → Classifier

Only bidirectional configurable.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mamba_model import MambaTemporalEncoder, BiMambaEncoder


class SingleMambaVAD(nn.Module):
    """Single bidirectional Mamba VAD.

    Training objective:
        L = L_cls + alpha * L_proto + beta * L_future
    """

    def __init__(self, visual_width=512, visual_length=256,
                 mamba_layers=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.0,
                 num_class=14,
                 device='cuda', mamba_version='mamba1',
                 multi_scale=False, multi_scale_factors='1,8,16',
                 bidirectional=False, mlp_scale=4,
                 mamba2_headdim=64, mamba2_ngroups=1):
        super().__init__()

        self.visual_length = visual_length
        self.visual_width = visual_width
        self.device = device
        self.num_class = num_class
        self.multi_scale = multi_scale

        # Mamba kwargs
        mamba_kwargs = {}
        if mamba_version == 'mamba2':
            mamba_kwargs = {'headdim': mamba2_headdim, 'ngroups': mamba2_ngroups}
            # causal_conv1d kernel: (d_model*expand + 2*d_state*ngroups) % 8 == 0
            orig = d_state
            while (visual_width * expand + 2 * d_state * mamba2_ngroups) % 8 != 0:
                d_state += 1
            if d_state != orig:
                print(f"[AutoFix] d_state bumped from {orig} → {d_state} "
                      f"to satisfy Mamba2 causal_conv1d %8 constraint")

        # ===== Position embedding =====
        max_pos = max(visual_length, 1024)
        self.frame_position_embeddings = nn.Embedding(max_pos, visual_width)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

        # ===== Single Mamba encoder =====
        if bidirectional:
            self.mamba = BiMambaEncoder(
                d_model=visual_width, n_layer=mamba_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mlp_scale=mlp_scale,
                mamba_version=mamba_version, **mamba_kwargs)
        else:
            self.mamba = MambaTemporalEncoder(
                d_model=visual_width, n_layer=mamba_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, attn_window=0, mlp_scale=mlp_scale,
                mamba_version=mamba_version, **mamba_kwargs)

        # ===== Future score prediction head =====
        self.future_score_head = nn.Linear(visual_width, 1)

        # ===== Multi-scale fusion =====
        if multi_scale:
            if isinstance(multi_scale_factors, str):
                parsed = tuple(int(x) for x in multi_scale_factors.split(','))
            else:
                parsed = tuple(multi_scale_factors)
            self.multi_scale_factors = parsed
            num_scales = len([f for f in parsed if f > 1]) + 1
        else:
            self.multi_scale_factors = ()
            num_scales = 1
        self.fusion_linear = nn.Linear(visual_width * num_scales, visual_width)

        # ===== MLP classifier (no GCN) =====
        self.classifier = nn.Sequential(
            nn.Linear(visual_width, visual_width // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(visual_width // 2, 1)
        )

    def trainable_state_dict(self):
        """Alias for state_dict (CLIP removed from model, no filtering needed)."""
        return self.state_dict()

    # ──────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────

    def forward(self, visual, lengths=None, normal_mask=None):
        B, T, D = visual.shape

        # Single Mamba encode
        x = self._encode_video(visual, self.mamba)       # [T, B, D]
        x_vis = x.permute(1, 0, 2)                       # [B, T, D]

        # Future score prediction from raw Mamba hidden state
        score_pred = self.future_score_head(x_vis[:, :-1, :])
        score_pred = F.pad(score_pred, (0, 0, 0, 1), mode='replicate')

        # Multi-scale fusion (shared Mamba encoder)
        if self.multi_scale:
            scales = [x_vis]
            for f in self.multi_scale_factors:
                if f == 1:
                    continue
                pooled = F.avg_pool1d(visual.float().permute(0, 2, 1),
                                      kernel_size=f, stride=f).permute(0, 2, 1).to(visual.dtype)
                xs = self._encode_video(pooled, self.mamba).permute(1, 0, 2)
                xs = F.interpolate(xs.permute(0, 2, 1), size=T, mode='linear', align_corners=False).permute(0, 2, 1)
                scales.append(xs)
            x_vis = torch.cat(scales, dim=-1)

        x_fused = self.fusion_linear(x_vis)

        # Classifier (MLP, no GCN)
        logits1 = self.classifier(x_fused)

        outputs = dict(
            hidden_state=x_vis,
            hidden_state_fused=x_fused,
            logits1=logits1,
            score_future=score_pred,
        )

        return outputs

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _encode_video(self, images, mamba_module):
        """[B, T, D] → position embed → mamba → [T, B, D]"""
        B, T, D = images.shape
        images = images.permute(1, 0, 2)
        pos_ids = torch.arange(T, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
        pos_emb = self.frame_position_embeddings(pos_ids)
        pos_emb = pos_emb.permute(1, 0, 2)
        images = images + pos_emb
        x = mamba_module(images)
        return x

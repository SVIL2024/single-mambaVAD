"""Parameter accounting for the reported "10.6M / 5.6M" figures.

Paper Table (direction ablation) and Sec. 4.3 report 10.6M for the
bidirectional model and 5.6M for the causal one.  This script prints the
component breakdown so the manuscript numbers can be replaced with measured
values and, if a narrower definition is intended, stated explicitly.
"""
import torch

import ucf_option
from mamba_vad_dual_model import MambaVAD_Dual


def build(bidirectional):
    o = ucf_option.parser.parse_args([])
    torch.manual_seed(234)
    return MambaVAD_Dual(
        visual_width=o.visual_width, visual_length=o.visual_length,
        mamba_layers=2, d_state=8, d_conv=4, expand=2, dropout=0.3, level=2,
        pred_stride=1, num_class=o.classes_num, embed_dim=o.embed_dim,
        prompt_prefix=o.prompt_prefix, prompt_postfix=o.prompt_postfix,
        device='cuda', mamba_version='mamba2', mamba_only_cls=True,
        multi_scale=True, multi_scale_factors='1,8,16', single_mamba=True,
        bidirectional=bidirectional, mlp_scale=2, mamba2_headdim=64,
        mamba2_ngroups=1,
    )


def groups(model):
    out = {}
    for name, p in model.named_parameters():
        if 'mamba' in name or 'encoder' in name:
            k = 'temporal_encoder'
        elif 'classifier' in name:
            k = 'classifier_head'
        elif 'prompt' in name or 'text' in name:
            k = 'text_prompt'
        elif 'position' in name:
            k = 'position_embedding'
        elif 'fusion' in name or 'proj' in name or 'scale' in name:
            k = 'multi_scale_fusion'
        else:
            k = 'other'
        out[k] = out.get(k, 0) + p.numel()
    return out


for flag in (True, False):
    m = build(flag)
    g = groups(m)
    total = sum(p.numel() for p in m.parameters())
    tag = 'bidirectional' if flag else 'causal'
    print(f'[{tag}] total={total:,}')
    for k, v in sorted(g.items(), key=lambda x: -x[1]):
        print(f'    {k:<22} {v:>12,}')

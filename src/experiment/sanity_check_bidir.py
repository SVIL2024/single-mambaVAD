"""Sanity check: does the ``bidirectional`` flag actually change the model?

The ablation records in ``results/`` report *identical* AUC/AP to four decimals
for ``bidirectional=True`` and ``bidirectional=False`` in all three scale
configurations, which is impossible for two genuinely different networks.
Either the flag is not wired through, or the numbers were copied.

This script builds both variants from the same seed and prints the evidence:
encoder class, parameter count, number of SSM submodules, and the maximum
absolute difference between their outputs on one random input.

Usage (from ``src/``):

    python -m experiment.sanity_check_bidir

Expected result if the flag works: the causal model has ~half the parameters
of the bidirectional one, and the output difference is clearly > 0.
"""
import sys

import torch

import ucf_option
from mamba_vad_dual_model import MambaVAD_Dual


def build(bidirectional, device='cpu'):
    o = ucf_option.parser.parse_args([])
    torch.manual_seed(234)
    return MambaVAD_Dual(
        visual_width=o.visual_width,
        visual_length=o.visual_length,
        mamba_layers=2,
        d_state=8,
        d_conv=4,
        expand=2,
        dropout=0.3,
        level=2,
        pred_stride=2,
        num_class=o.classes_num,
        embed_dim=o.embed_dim,
        prompt_prefix=o.prompt_prefix,
        prompt_postfix=o.prompt_postfix,
        device=device,
        mamba_version='mamba2',
        mamba_only_cls=True,
        multi_scale=True,
        multi_scale_factors='1,8,16',
        single_mamba=True,
        bidirectional=bidirectional,
        mlp_scale=2,
        mamba2_headdim=64,
        mamba2_ngroups=1,
    )


def describe(tag, model):
    enc = [type(m).__name__ for m in model.modules()
           if 'Encoder' in type(m).__name__]
    ssm = [type(m).__name__ for m in model.modules()
           if 'Mamba' in type(m).__name__]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{tag}] encoders={enc}')
    print(f'[{tag}] SSM modules={len(ssm)} -> {sorted(set(ssm))}')
    print(f'[{tag}] params: total={total:,} trainable={trainable:,}')
    return dict(enc=tuple(enc), n_ssm=len(ssm), params=total)


def main():
    torch.manual_seed(0)
    print('=' * 72)
    bi = build(True)
    ca = build(False)
    d_bi = describe('bidirectional', bi)
    d_ca = describe('causal', ca)
    print('=' * 72)

    same_enc = d_bi['enc'] == d_ca['enc']
    same_params = d_bi['params'] == d_ca['params']
    print(f'encoder class identical : {same_enc}')
    print(f'parameter count identical: {same_params}')
    print(f'param ratio (causal/bidir): {d_ca["params"] / d_bi["params"]:.3f}')

    x = torch.randn(1, 16, 512)
    out = {}
    for tag, m in (('bidirectional', bi), ('causal', ca)):
        m.eval()
        try:
            with torch.no_grad():
                y = m(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            out[tag] = y
            print(f'[{tag}] forward ok, output shape {tuple(y.shape)}')
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            print(f'[{tag}] forward failed: {type(exc).__name__}: {exc}')
    if len(out) == 2:
        a, b = out['bidirectional'], out['causal']
        if a.shape == b.shape:
            diff = (a - b).abs().max().item()
            print(f'max |output diff| = {diff:.6f}')
            print('VERDICT:',
                  'flag has NO effect' if diff < 1e-6 and same_params
                  else 'flag changes the model (as expected)')
        else:
            print(f'shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}')
    elif same_params and same_enc:
        print('VERDICT: flag has NO effect (identical module tree and params)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

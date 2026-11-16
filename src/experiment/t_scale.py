"""Does the forward cost scale with T?  Quick check at batch 1 and 8."""
import time

import torch

from mamba_vad_dual_model import MambaVAD_Dual

dev = 'cuda'
m = MambaVAD_Dual(
    visual_width=512, visual_length=256,
    mamba_layers=2, d_state=8, d_conv=4, expand=2, dropout=0.0,
    level=2, pred_stride=1, num_class=14, embed_dim=512,
    prompt_prefix=None, prompt_postfix=None, device=dev,
    mamba_version='mamba2', mamba_only_cls=True,
    multi_scale=True, multi_scale_factors='1,8,16', single_mamba=True,
    bidirectional=True, mlp_scale=2, mamba2_headdim=64, mamba2_ngroups=1,
).to(dev).eval()

with torch.no_grad():
    for B in (1, 8):
        for T in (32, 256, 1024):
            x = torch.randn(B, T, 512, device=dev)
            lengths = torch.full((B,), T, device=dev)
            for _ in range(3):
                out = m(x, lengths=lengths, text=None)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                out = m(x, lengths=lengths, text=None)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
            hid = out.get('hidden_state_fused', out.get('hidden_state'))
            print(f'B={B} T={T:>4}  {dt * 1000:7.1f} ms   hidden={tuple(hid.shape)}  '
                  f'logits={tuple(out["logits1"].shape)}')

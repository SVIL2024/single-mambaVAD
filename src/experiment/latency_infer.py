"""Inference-side latency / memory measurement (batch = 1) for MambaVAD.

Complements benchmark_scaling.json, which is a *training-step* measurement at
batch 32.  R1-5 asks for inference cost; this reports per-clip forward latency,
peak memory for a single sample, and throughput under a sliding-window
(chunked) streaming schedule.

Usage (from src/):
    python -m experiment.latency_infer --lengths 32 64 128 256 512 1024 --stream
"""
import argparse
import json
import os
import time

import torch

from mamba_vad_dual_model import MambaVAD_Dual


def build(device, bidirectional=True):
    model = MambaVAD_Dual(
        visual_width=512, visual_length=256,
        mamba_layers=2, d_state=8, d_conv=4, expand=2, dropout=0.0,
        level=2, pred_stride=1, num_class=14, embed_dim=512,
        prompt_prefix=None, prompt_postfix=None, device=device,
        mamba_version='mamba2', mamba_only_cls=True,
        multi_scale=True, multi_scale_factors='1,8,16', single_mamba=True,
        bidirectional=bidirectional, mlp_scale=2,
        mamba2_headdim=64, mamba2_ngroups=1,
    )
    return model.to(device).eval()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def time_forward(model, device, T, iters=20, warmup=5):
    x = torch.randn(1, T, 512, device=device)
    lengths = torch.tensor([T], device=device)
    for _ in range(warmup):
        model(x, lengths=lengths, text=None)
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x, lengths=lengths, text=None)
    _sync()
    dt = (time.perf_counter() - t0) / iters
    peak = (torch.cuda.max_memory_allocated() / 1024.0 ** 2) if torch.cuda.is_available() else 0.0
    return dt * 1000.0, peak


@torch.no_grad()
def stream(model, device, window=256, stride=128, total=1000):
    """Slide `window` over `total` frames; report wall time and per-chunk latency."""
    x = torch.randn(1, total, 512, device=device)
    starts = list(range(0, total - window + 1, stride))
    lengths = torch.tensor([window], device=device)
    for s in starts[:3]:
        model(x[:, s:s + window], lengths=lengths, text=None)
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for s in starts:
        model(x[:, s:s + window], lengths=lengths, text=None)
    _sync()
    wall = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024.0 ** 2) if torch.cuda.is_available() else 0.0
    return dict(window=window, stride=stride, total=total, chunks=len(starts),
                wall_s=round(wall, 3), fps=round(total / wall, 1),
                latency_per_chunk_ms=round(wall / len(starts) * 1000, 1),
                peak_mem_MB=round(peak, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengths', nargs='+', type=int, default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument('--stream', action='store_true')
    ap.add_argument('--causal', action='store_true')
    ap.add_argument('--out', default='experiment/results/inference_latency.json')
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build(device, bidirectional=not a.causal)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {'causal' if a.causal else 'bidirectional'}  params: {n_par:,}")

    rows = []
    for T in a.lengths:
        ms, mem = time_forward(model, device, T)
        rows.append(dict(T=T, latency_ms=round(ms, 1), per_frame_ms=round(ms / T, 3),
                         peak_mem_MB=round(mem, 1)))
        print(f'  T={T:>4}  {ms:7.1f} ms  ({ms / T:6.3f} ms/frame)  peak {mem:7.1f} MB')

    payload = dict(params=n_par, batch=1, directional=('causal' if a.causal else 'bidirectional'),
                   rows=rows)
    if a.stream:
        st = [stream(model, device, w, s) for w, s in ((128, 64), (256, 128), (256, 256))]
        payload['stream'] = st
        for r in st:
            print(f"  stream w={r['window']} s={r['stride']}: {r['fps']} fps, "
                  f"{r['latency_per_chunk_ms']} ms/chunk, peak {r['peak_mem_MB']} MB")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'saved -> {a.out}')


if __name__ == '__main__':
    main()

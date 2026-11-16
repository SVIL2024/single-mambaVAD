"""
benchmark_time.py — Efficiency benchmark: training step time vs sequence length.

Measures full forward+backward time per batch for each model variant
across a range of sequence lengths to demonstrate O(n) vs O(n²) scaling.

Usage:
    python experiment/benchmark_time.py              # run benchmark, print table
    python experiment/benchmark_time.py --plot       # + show plot
    python experiment/benchmark_time.py --plot-only  # re-plot from saved data

Saves results to: experiment/results/benchmark_scaling.json
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiment.config import results_dir
import torch

# ── Constants ──────────────────────────────────────────────────────
BATCH_SIZE = 16
FEATURE_DIM = 512
WARMUP_STEPS = 5
MEASURED_STEPS = 20
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024]

# UCF-Crime class names for CLIPVAD text branch
PROMPT_TEXT = [
    'normal', 'abuse', 'arrest', 'arson', 'assault',
    'burglary', 'explosion', 'fighting', 'roadAccidents',
    'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism',
]


# ── Model builders ─────────────────────────────────────────────────
def build_clipvad(T, device):
    from model import CLIPVAD
    model = CLIPVAD(
        num_class=14, embed_dim=512,
        visual_length=T, visual_width=FEATURE_DIM,
        visual_head=1, visual_layers=2,
        attn_window=min(8, T),
        prompt_prefix=10, prompt_postfix=10,
        device=device,
    ).to(device)
    return model


def build_mamba(T, device, bidirectional=False, multi_scale=False):
    from mamba_vad_dual_model import MambaVAD_Dual
    kwargs = dict(
        visual_width=FEATURE_DIM, visual_length=T,
        mamba_layers=2, d_state=8, d_conv=4, expand=2,
        dropout=0.0, level=2, num_class=14, embed_dim=512,
        prompt_prefix=10, prompt_postfix=10,
        device=device,
        single_mamba=True,
        bidirectional=bidirectional,
        multi_scale=multi_scale,
        multi_scale_factors='1,8,16',
        mamba_only_cls=True,
        mlp_scale=2,
    )
    model = MambaVAD_Dual(**kwargs).to(device)
    return model


# ── Timing helpers ─────────────────────────────────────────────────
def measure_step(model, visual, lengths, device, model_type):
    """One forward+backward pass, return elapsed seconds."""
    model.zero_grad(set_to_none=True)

    if model_type == 'clipvad':
        _, logits1, logits2 = model(visual, None, PROMPT_TEXT, lengths)
        loss = logits1.mean() + logits2.mean()
    else:
        outputs = model(visual, lengths=lengths, text=None)
        loss = outputs['logits1'].mean()

    loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter()


def benchmark_one(name, model_type, build_fn, lengths, device, batch_size):
    """Benchmark one model variant across all sequence lengths."""
    print(f"\n  {name}")
    results = {
        'name': name,
        'type': model_type,
        'lengths': {},
        'param_counts': {},
    }

    for T in lengths:
        # Fresh model per T so temporal encoder size matches sequence length
        model = build_fn(T, device)
        model.train()

        # Count parameters (first model only, should be identical for same T)
        if 'total' not in results['param_counts']:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results['param_counts'] = {'total': total, 'trainable': trainable}
            print(f"    Params: total={total:,}, trainable={trainable:,}")

        # Dummy input of length T
        visual = torch.randn(batch_size, T, FEATURE_DIM, device=device)
        lengths_tensor = torch.full((batch_size,), T, device=device, dtype=torch.long)

        # Warmup
        torch.cuda.synchronize()
        for _ in range(WARMUP_STEPS):
            measure_step(model, visual, lengths_tensor, device, model_type)

        # Measured runs
        torch.cuda.reset_peak_memory_stats()

        paired = []
        for _ in range(MEASURED_STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if model_type == 'clipvad':
                _, logits1, logits2 = model(visual, None, PROMPT_TEXT, lengths)
                loss = logits1.mean() + logits2.mean()
            else:
                outputs = model(visual, lengths=lengths_tensor, text=None)
                loss = outputs['logits1'].mean()

            loss.backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            paired.append((t1 - t0) * 1000)

        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

        mean_ms = float(np.mean(paired))
        std_ms = float(np.std(paired))

        print(f"    T={T:4d}  {mean_ms:8.1f} ± {std_ms:5.1f} ms/batch  "
              f"peak {peak_mem:6.0f} MB")

        results['lengths'][str(T)] = {
            'mean_ms': round(mean_ms, 2),
            'std_ms': round(std_ms, 2),
            'peak_memory_mb': round(peak_mem, 1),
        }

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Efficiency benchmark')
    parser.add_argument('--plot', action='store_true',
                        help='Show plot after benchmark')
    parser.add_argument('--plot-only', action='store_true',
                        help='Re-plot from saved data without re-running')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this benchmark.")
        sys.exit(1)

    device = args.device
    print(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
    print(f"Batch size: {args.batch_size},  "
          f"Warmup: {WARMUP_STEPS},  Measured: {MEASURED_STEPS}")
    print(f"Sequence lengths: {SEQUENCE_LENGTHS}")

    # ── Run benchmark ────────────────────────────────────────────
    if args.plot_only:
        with open(os.path.join(results_dir(), 'benchmark_scaling.json')) as f:
            payload = json.load(f)
        all_results = payload['results']
        batch_size_plot = payload.get('meta', {}).get('batch_size', BATCH_SIZE)
    else:
        batch_size_plot = args.batch_size
        models = [
            ('VadCLIP (Transformer+GCN)', 'clipvad',
             lambda T, d: build_clipvad(T, d)),
            ('MambaVAD (causal)', 'mamba',
             lambda T, d: build_mamba(T, d, bidirectional=False, multi_scale=False)),
            ('MambaVAD (bidirectional)', 'mamba',
             lambda T, d: build_mamba(T, d, bidirectional=True, multi_scale=False)),
            ('MambaVAD (multi-scale)', 'mamba',
             lambda T, d: build_mamba(T, d, bidirectional=True, multi_scale=True)),
        ]

        all_results = []
        for name, mtype, build_fn in models:
            r = benchmark_one(name, mtype, build_fn, SEQUENCE_LENGTHS,
                              device, args.batch_size)
            all_results.append(r)

        meta = {'batch_size': args.batch_size}
        rdir = results_dir()
        save_path = os.path.join(rdir, 'benchmark_scaling.json')
        with open(save_path, 'w') as f:
            json.dump({'meta': meta, 'results': all_results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    print_table(all_results)

    if args.plot or args.plot_only:
        plot_results(all_results, batch_size_plot)


def print_table(all_results):
    """Print results as a markdown-style table."""
    lengths = sorted(
        [int(k) for k in all_results[0]['lengths'].keys()]
    ) if all_results else []

    # Header
    print(f"\n{'=' * 80}")
    print(f"{'Model':<32}", end='')
    for T in lengths:
        print(f"{'T=' + str(T):>12}", end='')
    print()
    print(f"{'─' * 32}", end='')
    for _ in lengths:
        print(f"{'─' * 12}", end='')
    print()

    # Rows
    for r in all_results:
        print(f"{r['name']:<32}", end='')
        for T in lengths:
            d = r['lengths'].get(str(T), {})
            ms = d.get('mean_ms', None)
            if ms is not None:
                print(f"{ms:>8.1f} ms ", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

    # Params row
    print(f"{'─' * 32}", end='')
    for _ in lengths:
        print(f"{'─' * 12}", end='')
    print()
    for r in all_results:
        pc = r.get('param_counts', {})
        total = pc.get('total', 0)
        trainable = pc.get('trainable', 0)
        print(f"{r['name']:<32}{total:>8,} total  {trainable:>8,} trainable")

    print(f"{'=' * 80}")


def plot_results(all_results, batch_size=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, cannot plot.")
        return

    if batch_size is None:
        batch_size = BATCH_SIZE

    lengths = sorted(
        [int(k) for k in all_results[0]['lengths'].keys()]
    )

    plt.figure(figsize=(8, 5.5))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#e74c3c', '#27ae60', '#2980b9', '#f39c12', '#8e44ad']

    for i, r in enumerate(all_results):
        xs = lengths
        ys = [r['lengths'][str(T)]['mean_ms'] for T in lengths]
        errs = [r['lengths'][str(T)]['std_ms'] for T in lengths]
        plt.errorbar(
            xs, ys, yerr=errs,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2, markersize=7, capsize=3,
            label=r['name'],
        )

    T_ref = np.array(lengths)
    clip_ms = [all_results[0]['lengths'][str(T)]['mean_ms'] for T in lengths]
    coeff_quad = clip_ms[-1] / (lengths[-1] ** 2)
    plt.plot(T_ref, coeff_quad * T_ref ** 2, '--', color='#e74c3c',
             alpha=0.2, linewidth=1, label='_nolegend_')
    plt.text(lengths[-1] * 0.7, coeff_quad * lengths[-1] ** 2 * 0.9,
             'O(T²)', fontsize=9, color='#e74c3c', alpha=0.5)

    mamba_ms = [all_results[1]['lengths'][str(T)]['mean_ms'] for T in lengths]
    coeff_lin = mamba_ms[-1] / lengths[-1]
    plt.plot(T_ref, coeff_lin * T_ref, '--', color='#27ae60',
             alpha=0.2, linewidth=1, label='_nolegend_')
    plt.text(lengths[-1] * 0.7, coeff_lin * lengths[-1] * 0.9,
             'O(T)', fontsize=9, color='#27ae60', alpha=0.5)

    plt.xlabel('Sequence Length (T)', fontsize=13)
    plt.ylabel('Time per Batch (ms)', fontsize=13)
    plt.title('Training Step Time vs. Sequence Length\n'
              f'(batch={batch_size}, dim={FEATURE_DIM})',
              fontsize=13)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    save_path = os.path.join(results_dir(), 'benchmark_scaling.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    main()

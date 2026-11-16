"""E1 — how much of the "future score" is already inside h_t?

The paper trains a linear head ``s_hat_{t+1} = w^T h_t + b`` and calls the
objective future-score prediction.  In the bidirectional encoder ``h_t`` is a
concatenation of forward state (frames <= t) and backward state (frames >= t),
so ``h_t`` already sees frame t+1.  This script measures that directly with a
frozen probe: ridge regression from the frozen ``h_t`` to the *score* of frame
t+k.  A high R^2 for k>=1 means the target is contained in the input, i.e. the
head is a consistency term rather than a predictor.

Usage (from src/):
    python -m experiment.probe_future --ckpt ../model/xxx.pt --split test
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ucf_option  # noqa: E402
from mamba_vad_dual_model import MambaVAD_Dual  # noqa: E402
from utils.dataset import UCFDataset  # noqa: E402

LABEL_MAP = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest',
                  'Arson': 'arson', 'Assault': 'assault',
                  'Burglary': 'burglary', 'Explosion': 'explosion',
                  'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents',
                  'Robbery': 'robbery', 'Shooting': 'shooting',
                  'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
                  'Vandalism': 'vandalism'})


def build(device='cuda', bidirectional=True):
    o = ucf_option.parser.parse_args([])
    return MambaVAD_Dual(
        visual_width=o.visual_width, visual_length=o.visual_length,
        mamba_layers=2, d_state=8, d_conv=4, expand=2, dropout=0.0,
        level=2, pred_stride=1, num_class=o.classes_num,
        embed_dim=o.embed_dim, prompt_prefix=o.prompt_prefix,
        prompt_postfix=o.prompt_postfix, device=device, mamba_version='mamba2',
        mamba_only_cls=True, multi_scale=True, multi_scale_factors='1,8,16',
        single_mamba=True, bidirectional=bidirectional, mlp_scale=2,
        mamba2_headdim=64, mamba2_ngroups=1,
    )


def load_ckpt(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] missing={len(missing)} unexpected={len(unexpected)}')
    if unexpected[:5]:
        print('[ckpt] unexpected sample:', unexpected[:5])


@torch.no_grad()
def collect(model, csv, device, max_videos=200):
    ds = UCFDataset(256, csv, True, LABEL_MAP)
    rows = []
    for i in range(min(len(ds), max_videos)):
        feat, label, length = ds[i]
        # test videos are split into (num_splits, 256, D) crops; use the first one
        if feat.dim() == 3:
            feat = feat[0]
        n_valid = int(min(int(length), feat.shape[0]))
        feat = feat.unsqueeze(0).to(device).float()
        out = model(feat)
        h = out.get('hidden_state_fused', out.get('hidden_state'))
        s = out['logits1']
        h = h[0, :n_valid].cpu().numpy()
        s = s[0, :n_valid].cpu().numpy().reshape(-1)
        rows.append((h, s, str(label)))
        if (i + 1) % 20 == 0:
            print(f'  collected {i + 1}/{min(len(ds), max_videos)}')
    return rows


def fit_probe(rows, k, ridge=1e-2, seed=0):
    """Ridge regression h_t -> s_{t+k}; returns R^2 and MSE on held-out videos."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))
    n_tr = int(0.8 * len(rows))
    tr, te = [rows[j] for j in idx[:n_tr]], [rows[j] for j in idx[n_tr:]]

    def pairs(rs):
        xs, ys = [], []
        for h, s, _ in rs:
            if len(s) <= k:
                continue
            xs.append(h[:len(s) - k])
            ys.append(s[k:])
        return np.concatenate(xs), np.concatenate(ys)

    xtr, ytr = pairs(tr)
    xte, yte = pairs(te)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    A = xtr.T @ xtr + ridge * len(xtr) * np.eye(xtr.shape[1])
    w = np.linalg.solve(A, xtr.T @ ytr)
    pred = xte @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    return dict(k=k, r2=1.0 - ss_res / ss_tot, mse=float(((yte - pred) ** 2).mean()))


def control_probe(rows, k, ridge=1e-2, seed=0):
    """Same probe, but h_t of video i is paired with s of video (i+1).

    If the R^2 above were an artefact of score smoothness / high-dimensional
    ridge fit, this control would still be high.  It should collapse instead.
    """
    shifted = [(rows[i][0], rows[(i + 1) % len(rows)][1], rows[i][2])
               for i in range(len(rows))]
    return fit_probe(shifted, k, ridge=ridge, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--split', default='test', choices=['test', 'train'])
    p.add_argument('--max-videos', type=int, default=200)
    p.add_argument('--causal', action='store_true', help='ckpt was trained without a backward scan')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(device, bidirectional=not a.causal).to(device).eval()
    load_ckpt(model, a.ckpt)

    csv = ('../list/ucf_CLIP_rgb.csv' if a.split == 'train'
           else '../list/ucf_CLIP_rgbtest.csv')
    rows = collect(model, csv, device, a.max_videos)
    print(f'[probe] videos={len(rows)}')

    res = [fit_probe(rows, k) for k in (0, 1, 2, 4, 8)]
    for r in res:
        print(f"  k={r['k']:>2}  R2={r['r2']:.3f}  MSE={r['mse']:.4f}")
    ctl = control_probe(rows, 1)
    print(f"  control (h_i -> s_j, k=1): R2={ctl['r2']:.3f}  MSE={ctl['mse']:.4f}")
    payload = dict(ckpt=a.ckpt, split=a.split, n_videos=len(rows), probe=res,
                   control_k1=ctl, causal=bool(a.causal))
    out = a.out or os.path.join(os.path.dirname(__file__), 'results',
                                'probe_future_%s.json' % ('causal' if a.causal else 'bidi'))
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print('[probe] ->', out)


if __name__ == '__main__':
    main()

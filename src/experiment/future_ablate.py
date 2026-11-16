"""Does the score at frame t depend on frames after t+1?

R1-3a: the future-score head predicts s_{t+1} from h_t, but in the bidirectional
encoder h_t already contains a backward state that has read frames >= t, so the
target is partly inside the input.  This is a truncation test: keep the first
`cut+1` frames and replace everything after them by a repetition of frame `cut`
(the same thing the model does when a clip is shorter than the window).  If the
recomputed scores for t <= cut move, the network is using frames beyond the
target to produce them.

Bidirectional models should move; a causal model should not (control).

Usage (from src/):
    python -m experiment.future_ablate --ckpt ../model/x.pt --causal
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


def build(device, bidirectional):
    o = ucf_option.parser.parse_args([])
    m = MambaVAD_Dual(
        visual_width=o.visual_width, visual_length=o.visual_length,
        mamba_layers=2, d_state=8, d_conv=4, expand=2, dropout=0.0,
        level=2, pred_stride=1, num_class=o.classes_num,
        embed_dim=o.embed_dim, prompt_prefix=o.prompt_prefix,
        prompt_postfix=o.prompt_postfix, device=device, mamba_version='mamba2',
        mamba_only_cls=True, multi_scale=True, multi_scale_factors='1,8,16',
        single_mamba=True, bidirectional=bidirectional, mlp_scale=2,
        mamba2_headdim=64, mamba2_ngroups=1,
    )
    return m.eval()


def load_ckpt(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] missing={len(missing)} unexpected={len(unexpected)}')
    return missing, unexpected


@torch.no_grad()
def run(model, csv, device, n_videos=40, max_len=256):
    ds = UCFDataset(256, csv, True, LABEL_MAP)
    out_rows = []
    for i in range(min(len(ds), n_videos)):
        feat, label, length = ds[i]
        if feat.dim() == 3:
            feat = feat[0]
        T = int(min(int(length), feat.shape[0], max_len))
        if T < 64:
            continue
        x = feat[:T].unsqueeze(0).to(device).float()
        s_full = model(x)['logits1'][0, :T].cpu().numpy().reshape(-1)

        cuts = [T // 2, int(T * 0.75)]
        row = dict(video=str(label), T=T)
        for c in cuts:
            x2 = x.clone()
            x2[:, c + 1:] = x[:, c:c + 1]        # truncate: repeat frame c
            s_tr = model(x2)['logits1'][0, :T].cpu().numpy().reshape(-1)
            pre = slice(0, c)                     # frames strictly before the cut
            d = np.abs(s_full[pre] - s_tr[pre])
            # margin of 8 frames keeps the padded boundary out of the statistic
            pre_in = slice(0, max(c - 8, 1))
            d_in = np.abs(s_full[pre_in] - s_tr[pre_in])
            r = np.corrcoef(s_full[pre_in], s_tr[pre_in])[0, 1] if c > 10 else np.nan
            row[f'cut{c}'] = dict(mean_abs_delta=float(d.mean()),
                                  mean_abs_delta_outer=float(d_in.mean()),
                                  max_abs_delta=float(d_in.max()),
                                  corr=float(r))
        out_rows.append(row)
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{min(len(ds), n_videos)}')
    return out_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--causal', action='store_true')
    p.add_argument('--n-videos', type=int, default=40)
    p.add_argument('--out', default='experiment/results/future_ablation.json')
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(device, bidirectional=not a.causal).to(device)
    load_ckpt(model, a.ckpt)
    rows = run(model, '../list/ucf_CLIP_rgbtest.csv', device, a.n_videos)

    def agg(idx):
        v = [r[list(r.keys())[idx]]['mean_abs_delta_outer'] for r in rows]
        c = [r[list(r.keys())[idx]]['corr'] for r in rows]
        return dict(mean_abs_delta=float(np.mean(v)), sd=float(np.std(v)),
                    mean_corr=float(np.nanmean(c)))

    summary = dict(ckpt=a.ckpt, causal=bool(a.causal), n=len(rows),
                   cut_half=agg(2), cut_3q=agg(3), rows=rows)
    print(f"[future-ablate] causal={a.causal} n={len(rows)}")
    print(f"  cut=T/2 : |delta|={summary['cut_half']['mean_abs_delta']:.4f} "
          f"(sd {summary['cut_half']['sd']:.4f}) corr={summary['cut_half']['mean_corr']:.4f}")
    print(f"  cut=3T/4: |delta|={summary['cut_3q']['mean_abs_delta']:.4f} "
          f"(sd {summary['cut_3q']['sd']:.4f}) corr={summary['cut_3q']['mean_corr']:.4f}")
    tag = 'causal' if a.causal else 'bidi'
    base, ext = os.path.splitext(a.out)
    out = f'{base}_{tag}{ext}'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print('[future-ablate] ->', out)


if __name__ == '__main__':
    main()

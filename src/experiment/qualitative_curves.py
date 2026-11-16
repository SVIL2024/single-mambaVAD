"""
qualitative_curves.py — Per-frame anomaly score curves.

Compares VadCLIP vs MambaVAD on selected test videos.
Scores are cached after first run for fast re-plotting.

Usage:
    # UCF (default)
    python experiment/qualitative_curves.py --dataset ucf \\
        --vadclip ../model/model_ucf.pth \\
        --mamba ../model/final_full.pt

    # XD-Violence
    python experiment/qualitative_curves.py --dataset xd \\
        --vadclip ../model/model_xd.pth \\
        --mamba experiment/best_xd.pt

    # auto-select interesting videos, save scores only
    python experiment/qualitative_curves.py --mamba experiment/best_auc0.8629.pt --no-plot

    # re-plot from cached scores
    python experiment/qualitative_curves.py --from-cache --dataset xd
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiment.config import results_dir

# ── Constants ──────────────────────────────────────────────────────
VISUAL_LENGTH = 256
FEATURE_DIM = 512

DATASET_CFG = {
    'ucf': {
        'csv': '../list/ucf_CLIP_rgbtest.csv',
        'gt': '../list/gt_ucf.npy',
        'classes': 14,
        'vadclip_layers': 2,
        'vadclip_attn_window': 8,
        'mamba_layers': 2,
        'prompt_text': [
            'normal', 'abuse', 'arrest', 'arson', 'assault',
            'burglary', 'explosion', 'fighting', 'roadAccidents',
            'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism',
        ],
        'label_map': {
            'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest',
            'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary',
            'Explosion': 'explosion', 'Fighting': 'fighting',
            'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
            'Shooting': 'shooting', 'Shoplifting': 'shoplifting',
            'Stealing': 'stealing', 'Vandalism': 'vandalism',
        },
    },
    'xd': {
        'csv': '../list/xd_CLIP_rgbtest.csv',
        'gt': '../list/gt.npy',
        'classes': 7,
        'vadclip_layers': 1,
        'vadclip_attn_window': 64,
        'mamba_layers': 1,
        'prompt_text': [
            'normal', 'fighting', 'shooting', 'riot',
            'abuse', 'car accident', 'explosion',
        ],
        'label_map': {
            'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
            'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
            'G': 'explosion',
        },
    },
}


# ── Video info ─────────────────────────────────────────────────────
def load_test_csv(csv_path='../list/ucf_CLIP_rgbtest.csv'):
    df = pd.read_csv(csv_path)
    video_names = []
    frame_counts = []
    video_labels = []
    for _, row in df.iterrows():
        path = row['path']
        name = os.path.splitext(os.path.basename(path))[0]
        feat = np.load(path)
        frame_counts.append(feat.shape[0])
        video_names.append(name)
        video_labels.append(row['label'])
    return video_names, frame_counts, video_labels


def get_video_gt(gt_path, frame_counts):
    """Split flat gt.npy into per-video arrays at feature-level granularity.

    gt.npy is at 16x upsampled frame level.  We downsample back to
    feature level by max-pooling each 16-frame block.
    """
    gt_full = np.load(gt_path)
    per_video = []
    offset = 0
    for n_frames in frame_counts:
        n_gt = n_frames * 16
        block = gt_full[offset: offset + n_gt]
        block = block[:n_frames * 16]
        # Downsample 16x → 1 label per feature frame
        feat_gt = block.reshape(-1, 16).max(axis=1)
        per_video.append(feat_gt)
        offset += n_gt
    return per_video


def rank_candidates(frame_counts, per_video_gt, video_labels):
    """Score and rank all videos by how informative they'd be for visualization.

    Returns list of (score, idx, label, n_frames, anomaly_ratio) sorted descending.
    """
    scores = []
    for i, (n_f, gt, lbl) in enumerate(zip(frame_counts, per_video_gt, video_labels)):
        if n_f < 32:
            continue
        anomaly_ratio = gt.mean()
        balance_score = 1.0 - abs(anomaly_ratio - 0.3) / 0.7
        length_score = min(1.0, n_f / 512)
        has_anomaly = anomaly_ratio > 0.01
        if not has_anomaly:
            balance_score *= 0.3
        combined = balance_score * 0.7 + length_score * 0.3
        scores.append((combined, i, lbl, n_f, anomaly_ratio))
    scores.sort(reverse=True)
    return scores


# ── Model inference ────────────────────────────────────────────────
def build_vadclip(device, cfg):
    from model import CLIPVAD
    model = CLIPVAD(
        num_class=cfg['classes'], embed_dim=512,
        visual_length=VISUAL_LENGTH, visual_width=FEATURE_DIM,
        visual_head=1, visual_layers=cfg['vadclip_layers'],
        attn_window=cfg['vadclip_attn_window'],
        prompt_prefix=10, prompt_postfix=10,
        device=device,
    ).to(device)
    model.eval()
    return model


def build_mamba_from_ckpt(ckpt, device, cfg):
    """Infer model architecture from checkpoint keys and build matching model."""
    from mamba_vad_dual_model import MambaVAD_Dual

    all_keys = list(ckpt.keys())
    has_fwd = any('fwd_mamba' in k for k in all_keys)
    has_bwd = any('bwd_mamba' in k for k in all_keys)
    bidirectional = has_fwd and has_bwd

    # Infer mlp_scale from MLP weight shape
    mlp_key = [k for k in ckpt if 'mlp.c_fc.weight' in k]
    mlp_scale = 4
    if mlp_key:
        in_features = ckpt[mlp_key[0]].shape[1]
        out_features = ckpt[mlp_key[0]].shape[0]
        mlp_scale = out_features // in_features

    # Infer mamba_version from key structure
    mamba_version = 'mamba1'
    if any('norm.weight' in k for k in ckpt):
        mamba_version = 'mamba2'

    layer_ids = set()
    for k in all_keys:
        m = re.match(r'mamba\.layers\.(\d+)', k)
        if m:
            layer_ids.add(int(m.group(1)))
    mamba_layers = max(layer_ids) + 1 if layer_ids else cfg['mamba_layers']

    print(f"  Checkpoint: bidirectional={bidirectional}, "
          f"mamba_layers={mamba_layers}, mlp_scale={mlp_scale}, "
          f"mamba_version={mamba_version}")

    # Sanity check: verify MLP shape matches checkpoint
    mlp_keys_in_ckpt = [k for k in all_keys if 'mlp.c_fc.weight' in k]
    model = MambaVAD_Dual(
        visual_width=FEATURE_DIM, visual_length=VISUAL_LENGTH,
        mamba_layers=mamba_layers, d_state=8, d_conv=4, expand=2,
        dropout=0.0, level=2, num_class=cfg['classes'], embed_dim=512,
        prompt_prefix=10, prompt_postfix=10,
        device=device, single_mamba=True,
        bidirectional=bidirectional, multi_scale=True,
        multi_scale_factors='1,8,16',
        mamba_only_cls=True, mlp_scale=mlp_scale,
        mamba_version=mamba_version,
    ).to(device)
    model.eval()

    # Debug: print model MLP keys for comparison
    model_keys = list(model.state_dict().keys())
    model_mlp_keys = [k for k in model_keys if 'mlp.c_fc.weight' in k]
    print(f"  Model   MLP keys: {model_mlp_keys[:3]}")
    if model_mlp_keys:
        shape = model.state_dict()[model_mlp_keys[0]].shape
        print(f"  Model   MLP shape: {shape}")

    return model


def extract_scores_vadclip(model, video_feat, length, device, prompt_text):
    """Run VadCLIP on one video, return per-frame anomaly scores (numpy)."""
    with torch.no_grad():
        num_chunks = int(length / VISUAL_LENGTH) + 1
        chunk_lens = []
        remaining = length
        for _ in range(num_chunks):
            cl = min(VISUAL_LENGTH, remaining)
            chunk_lens.append(cl)
            remaining -= cl

        if video_feat.ndim == 2:
            video_feat = video_feat.unsqueeze(0)

        lengths = torch.tensor(chunk_lens, device=device)
        from utils.tools import get_batch_mask
        padding_mask = get_batch_mask(lengths, VISUAL_LENGTH).to(device)

        _, logits1, _ = model(video_feat.to(device), padding_mask,
                              prompt_text, lengths)
        logits1 = logits1.reshape(-1)
        prob = torch.sigmoid(logits1[:length]).cpu().numpy()
    return prob


def extract_scores_mamba(model, video_feat, length, device):
    with torch.no_grad():
        if video_feat.ndim == 2:
            video_feat = video_feat.unsqueeze(0)

        outputs = model(video_feat.to(device),
                        lengths=torch.tensor([length], device=device))
        logits1 = torch.sigmoid(outputs['logits1']).squeeze()
        prob = logits1[:length].cpu().numpy()
    return prob


def run_inference(model, model_type, video_feats, frame_counts, device, prompt_text=None):
    scores_list = []
    for feat, n_f in zip(video_feats, frame_counts):
        if model_type == 'vadclip':
            s = extract_scores_vadclip(model, feat, n_f, device, prompt_text)
        else:
            s = extract_scores_mamba(model, feat, n_f, device)
        scores_list.append(s)
    return scores_list


# ── Loading ─────────────────────────────────────────────────────────
def load_video_features(video_indices, video_names, csv_path):
    """Load .npy features for selected video indices."""
    df = pd.read_csv(csv_path)
    feats = []
    names = []
    counts = []
    for idx in video_indices:
        path = df.iloc[idx]['path']
        feat = torch.from_numpy(np.load(path)).float()
        feats.append(feat)
        names.append(video_names[idx])
        counts.append(feat.shape[0])
    return feats, names, counts


# ── Plotting ────────────────────────────────────────────────────────
def plot_curves(video_infos, save_path):
    """Plot per-frame anomaly scores for selected videos.

    video_infos: list of dicts with keys:
      name, label, gt, scores_vadclip, scores_mamba, n_frames
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n_videos = len(video_infos)
    fig, axes = plt.subplots(n_videos, 1, figsize=(10, 2.8 * n_videos),
                             sharex=False)
    if n_videos == 1:
        axes = [axes]

    colors = {'gt': '#f5b7b1', 'vadclip': '#e74c3c', 'mamba': '#2980b9'}

    for ax, info in zip(axes, video_infos):
        nf = info['n_frames']
        gt = info['gt'][:nf]
        x = np.arange(nf)

        ax.fill_between(x, 0, 1, where=(gt > 0.5),
                        color=colors['gt'], alpha=0.6,
                        label='Ground Truth (anomaly)')

        if info['scores_vadclip'] is not None:
            sc_v = info['scores_vadclip'][:nf]
            ax.plot(x, sc_v, color=colors['vadclip'], linewidth=1.2,
                    alpha=0.85, label=f"VadCLIP (AUC={info.get('auc_v', 0):.3f})")

        if info['scores_mamba'] is not None:
            sc_m = info['scores_mamba'][:nf]
            ax.plot(x, sc_m, color=colors['mamba'], linewidth=1.2,
                    alpha=0.85, label=f"MambaVAD (AUC={info.get('auc_m', 0):.3f})")

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Anomaly Score', fontsize=10)
        ax.set_title(f"{info['name']}  ({info['label']}, {nf} frames)",
                     fontsize=11)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Qualitative anomaly score curves')
    parser.add_argument('--vadclip', type=str, default=None,
                        help='Path to VadCLIP checkpoint')
    parser.add_argument('--mamba', type=str, default=None,
                        help='Path to MambaVAD checkpoint')
    parser.add_argument('--videos', type=int, nargs='*', default=None,
                        help='Video indices to plot (0-indexed). '
                             'Omit for auto-select.')
    parser.add_argument('--num-videos', type=int, default=2,
                        help='Number of videos to plot (default: 2)')
    parser.add_argument('--min-auc', type=float, default=None,
                        help='Only show videos with AUC >= this threshold '
                             '(after inference). Loops through more '
                             'candidates if needed.')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Output image path')
    parser.add_argument('--dataset', type=str, default='ucf',
                        choices=list(DATASET_CFG.keys()),
                        help='Dataset config: ucf or xd')
    parser.add_argument('--csv', type=str, default=None,
                        help='Override test CSV path')
    parser.add_argument('--gt', type=str, default=None,
                        help='Override ground truth path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-plot', action='store_true',
                        help='Only cache scores, skip plotting')
    parser.add_argument('--from-cache', action='store_true',
                        help='Re-plot from cached scores, skip inference')
    args = parser.parse_args()

    if not args.from_cache and args.vadclip is None and args.mamba is None:
        print("Provide at least one checkpoint (--vadclip or --mamba) "
              "or use --from-cache.")
        sys.exit(1)

    # Resolve model paths: try as-is, then relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ('vadclip', 'mamba'):
        p = getattr(args, attr)
        if p is not None and not os.path.isfile(p):
            alt = os.path.join(script_dir, p)
            if os.path.isfile(alt):
                setattr(args, attr, alt)
                print(f"  Resolved {attr} path to {alt}")

    device = args.device if torch.cuda.is_available() else 'cpu'
    cfg = DATASET_CFG[args.dataset]
    cache_path = os.path.join(results_dir(), f'qualitative_scores_{args.dataset}.json')
    rdir = results_dir()

    csv_path = args.csv or cfg['csv']
    gt_path = args.gt or cfg['gt']

    # ── Load video info ─────────────────────────────────────────
    print(f"Loading test CSV ({args.dataset})...")
    video_names, frame_counts, video_labels = load_test_csv(csv_path)
    per_video_gt = get_video_gt(gt_path, frame_counts)
    print(f"  {len(video_names)} videos, "
          f"{sum(frame_counts)} total feature frames / "
          f"{sum(frame_counts) * 16} gt frames")

    target_n = args.num_videos if args.videos is None else len(args.videos)

    # ── Select or validate videos ───────────────────────────────
    if args.videos is not None and len(args.videos) > 0:
        selected = list(args.videos)
        print(f"  Using specified videos: {selected}")
    else:
        ranked = rank_candidates(frame_counts, per_video_gt, video_labels)
        candidate_queue = [s[1] for s in ranked]
        random.shuffle(candidate_queue)
        if args.from_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            cached_indices = set(e['video_idx'] for e in cached)
            candidate_queue = [i for i in candidate_queue if i not in cached_indices]

        selected = []

    from sklearn.metrics import roc_auc_score

    all_selected = []
    all_feats = []
    all_scores_v = []
    all_scores_m = []
    all_auc_v = []
    all_auc_m = []
    all_gt = []
    all_counts = []

    def _auc_for_video(gt, scores):
        try:
            return float(roc_auc_score(gt, scores))
        except Exception:
            return 0.0

    def _passes_filter(v_auc, m_auc, threshold):
        auc_val = m_auc if m_auc is not None else v_auc
        if threshold is None:
            return True
        return auc_val is not None and auc_val >= threshold

    def _count_passing(v_aucs, m_aucs, threshold):
        return sum(1 for va, ma in zip(v_aucs, m_aucs)
                   if _passes_filter(va, ma, threshold))

    models_built = {'vadclip': None, 'mamba': None}
    batch_size = target_n

    def _pick_diverse(queue, selected_indices, n, all_labels):
        seen_labels = set(all_labels[i] for i in selected_indices)
        picked = []
        rest = []
        for idx in queue:
            label = all_labels[idx]
            if len(picked) < n and label not in seen_labels:
                picked.append(idx)
                seen_labels.add(label)
            else:
                rest.append(idx)
        for idx in rest:
            if len(picked) >= n:
                break
            if idx not in picked:
                picked.append(idx)
        return picked, [i for i in queue if i not in picked]

    while True:
        if args.videos is not None:
            pending = [i for i in args.videos if i not in all_selected]
        else:
            pending, candidate_queue = _pick_diverse(
                candidate_queue, all_selected, batch_size, video_labels)

        if not pending:
            break

        new_feats, new_names, new_counts = load_video_features(
            pending, video_names, csv_path)
        new_gt = [per_video_gt[i][:n] for i, n in zip(pending, new_counts)]

        new_scores_v = [None] * len(pending)
        new_scores_m = [None] * len(pending)

        if args.from_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            for entry in cached:
                idx = entry['video_idx']
                if idx in pending:
                    pos = pending.index(idx)
                    if entry.get('model') == 'vadclip':
                        new_scores_v[pos] = np.array(entry['scores'])
                    elif entry.get('model') == 'mamba':
                        new_scores_m[pos] = np.array(entry['scores'])
        else:
            if args.vadclip:
                if models_built['vadclip'] is None:
                    print("Building VadCLIP model ...")
                    model_v = build_vadclip(device, cfg)
                    ckpt_v = torch.load(args.vadclip, map_location=device,
                                        weights_only=False)
                    model_v.load_state_dict(
                        ckpt_v.get('model_state_dict', ckpt_v))
                    models_built['vadclip'] = model_v
                new_scores_v = run_inference(
                    models_built['vadclip'], 'vadclip', new_feats, new_counts,
                    device, prompt_text=cfg['prompt_text'])
                for idx, sc in zip(pending, new_scores_v):
                    _save_cache_entry(cache_path, idx, 'vadclip', sc.tolist())

            if args.mamba:
                if models_built['mamba'] is None:
                    print("Loading MambaVAD checkpoint ...")
                    ckpt_m = torch.load(args.mamba, map_location=device,
                                        weights_only=False)
                    model_m = build_mamba_from_ckpt(
                        ckpt_m['model_state_dict'], device, cfg)
                    model_m.load_state_dict(ckpt_m['model_state_dict'])
                    models_built['mamba'] = model_m
                new_scores_m = run_inference(
                    models_built['mamba'], 'mamba', new_feats, new_counts, device)
                for idx, sc in zip(pending, new_scores_m):
                    _save_cache_entry(cache_path, idx, 'mamba', sc.tolist())

        new_auc_v = []
        new_auc_m = []
        for i in range(len(pending)):
            new_auc_v.append(
                _auc_for_video(new_gt[i], new_scores_v[i])
                if new_scores_v[i] is not None else None)
            new_auc_m.append(
                _auc_for_video(new_gt[i], new_scores_m[i])
                if new_scores_m[i] is not None else None)

        offset = len(all_selected)
        all_selected.extend(pending)
        all_feats.extend(new_feats)
        all_scores_v.extend(new_scores_v)
        all_scores_m.extend(new_scores_m)
        all_auc_v.extend(new_auc_v)
        all_auc_m.extend(new_auc_m)
        all_gt.extend(new_gt)
        all_counts.extend(new_counts)

        # Print batch results
        for i, idx in enumerate(pending):
            pct = f"{new_gt[i].mean():.1%}"
            av = f"{new_auc_v[i]:.4f}" if new_auc_v[i] is not None else "—"
            am = f"{new_auc_m[i]:.4f}" if new_auc_m[i] is not None else "—"
            print(f"  [{idx}] {video_names[idx]:<30} {new_counts[i]:>5} {pct:>7}  "
                  f"V:{av}  M:{am}")

        passing = _count_passing(all_auc_v, all_auc_m, args.min_auc)
        if passing >= target_n:
            print(f"  Found {passing} videos meeting AUC >= {args.min_auc}, "
                  f"enough (target={target_n}).")
            break

        if not candidate_queue:
            print(f"  Only {passing} videos meet AUC >= {args.min_auc} "
                  f"(target={target_n}), no more candidates.")
            break

        print(f"  Only {passing}/{target_n} videos pass filter, "
              f"trying next batch ({len(candidate_queue)} remaining) ...")

    for m in models_built.values():
        if m is not None:
            del m
    torch.cuda.empty_cache()
    print(f"  Processed {len(all_selected)} videos total.")

    # ── Filter by min-auc ───────────────────────────────────────
    if args.min_auc is not None:
        keep = [i for i in range(len(all_selected))
                if _passes_filter(all_auc_v[i], all_auc_m[i], args.min_auc)]
        if keep:
            all_selected = [all_selected[i] for i in keep]
            all_gt = [all_gt[i] for i in keep]
            all_counts = [all_counts[i] for i in keep]
            all_scores_v = [all_scores_v[i] for i in keep]
            all_scores_m = [all_scores_m[i] for i in keep]
            all_auc_v = [all_auc_v[i] for i in keep]
            all_auc_m = [all_auc_m[i] for i in keep]
            print(f"  Kept {len(keep)} videos with AUC >= {args.min_auc}")

    # Take top N
    selected = all_selected[:target_n]
    sel_gt = all_gt[:target_n]
    sel_counts = all_counts[:target_n]
    scores_vadclip = all_scores_v[:target_n]
    scores_mamba = all_scores_m[:target_n]
    auc_v_list = all_auc_v[:target_n]
    auc_m_list = all_auc_m[:target_n]

    # ── Print table ─────────────────────────────────────────────
    print(f"\n{'Video':<30} {'Frames':>6} {'GT_anom%':>8} "
          f"{'AUC_V':>8} {'AUC_M':>8}")
    print("-" * 60)
    for i, idx in enumerate(selected):
        anom_pct = f"{sel_gt[i].mean():.1%}"
        av = f"{auc_v_list[i]:.4f}" if auc_v_list[i] is not None else "—"
        am = f"{auc_m_list[i]:.4f}" if auc_m_list[i] is not None else "—"
        print(f"{video_names[idx]:<30} {sel_counts[i]:>6} {anom_pct:>8} "
              f"{av:>8} {am:>8}")

    # ── Plot ────────────────────────────────────────────────────
    if not args.no_plot:
        output = args.output or os.path.join(
            rdir, f'qualitative_curves_{args.dataset}.png')
        video_infos = []
        for i, idx in enumerate(selected):
            video_infos.append(dict(
                name=video_names[idx],
                label=video_labels[idx],
                gt=sel_gt[i],
                scores_vadclip=scores_vadclip[i],
                scores_mamba=scores_mamba[i],
                n_frames=sel_counts[i],
                auc_v=auc_v_list[i],
                auc_m=auc_m_list[i],
            ))
        plot_curves(video_infos, output)
        print(f"\nDone.  {output}")


def _save_cache_entry(cache_path, video_idx, model_name, scores):
    """Append one entry to the JSON cache file."""
    entry = {'video_idx': video_idx, 'model': model_name, 'scores': scores}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
    else:
        data = []
    data = [d for d in data
            if not (d['video_idx'] == video_idx and d['model'] == model_name)]
    data.append(entry)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()

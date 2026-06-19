import argparse
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mamba_vad_dual_model import SingleMambaVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask
import xd_option


def test(model, testdataloader, maxlen, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()

    all_prob = []

    with torch.no_grad():
        for item in testdataloader:
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if visual.dim() == 2:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            num_chunks = int(length / maxlen) + 1
            chunk_lens = []
            remaining = length
            for _ in range(num_chunks):
                chunk_l = min(maxlen, remaining)
                chunk_lens.append(chunk_l)
                remaining -= chunk_l

            lengths_tensor = torch.tensor(chunk_lens, device=device)
            outputs = model(visual, lengths=lengths_tensor)

            # Classifier score (primary)
            logits1 = torch.sigmoid(outputs['logits1']).squeeze(-1)
            cls_chunks = [logits1[j, :chunk_lens[j]] for j in range(num_chunks) if chunk_lens[j] > 0]
            prob = torch.cat(cls_chunks)[:len_cur] if cls_chunks else torch.zeros(len_cur, device=device)

            all_prob.append(prob.cpu())

    # Video-level AUC/AP
    prob_all = torch.cat(all_prob).numpy()
    prob_all = np.repeat(prob_all, 16)[:len(gt)]
    gt = np.array(gt)[:len(prob_all)]
    ROC = roc_auc_score(gt, prob_all)
    AP = average_precision_score(gt, prob_all)
    print(f"  AUC: {ROC:.4f} | AP: {AP:.4f}")

    return ROC, AP, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[xd_option.parser], add_help=False)
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=1, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    parser.add_argument('--pred-stride', default=1, type=int)
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--multi-scale-factors', default='1,8,16')
    parser.add_argument('--bidirectional', action='store_true')
    args = parser.parse_args()
    args.lr = float(args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
                      'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
                      'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mamba_kwargs = {}
    if args.mamba_version == 'mamba2':
        mamba_kwargs = {'mamba2_headdim': args.mamba2_headdim, 'mamba2_ngroups': args.mamba2_ngroups}

    model = SingleMambaVAD(
        visual_width=args.visual_width, visual_length=args.visual_length,
        mamba_layers=args.mamba_layers, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand,
        dropout=args.dropout,
        num_class=args.classes_num,
        device=device, mamba_version=args.mamba_version,
        multi_scale=args.multi_scale,
        multi_scale_factors=args.multi_scale_factors,
        bidirectional=args.bidirectional, **mamba_kwargs)

    ckpt = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    t_start = time.time()
    test(model, test_loader, args.visual_length,
         gt, gtsegments, gtlabels, device)
    elapsed = time.time() - t_start
    print(f"  Test time: {elapsed:.0f}s")

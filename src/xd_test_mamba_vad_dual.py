import argparse
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mamba_vad_dual_model import MambaVAD_Dual
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
import xd_option


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    """XD evaluation. Returns (AUC_cls, AP_cls, 0).

    Scores from sigmoid(logits1) + optionally from prediction error.
    mAP disabled (binary classifier, no per-class outputs).
    """
    model.to(device)
    model.eval()

    all_prob = []
    all_pred = []

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
            outputs = model(visual, lengths=lengths_tensor, text=prompt_text)

            # Cls score
            logits1 = torch.sigmoid(outputs['logits1']).squeeze(-1)
            cls_chunks = [logits1[j, :chunk_lens[j]] for j in range(num_chunks) if chunk_lens[j] > 0]
            prob = torch.cat(cls_chunks)[:len_cur] if cls_chunks else torch.zeros(len_cur, device=device)

            # Pred error score
            stride = model.pred_stride
            if model.single_mamba:
                error = outputs['error']
            else:
                error = outputs['error_N']
            chunk_probs = []
            for j in range(num_chunks):
                if chunk_lens[j] > 0:
                    n_pred = max(0, chunk_lens[j] - stride)
                    if n_pred > 0:
                        chunk_prob = torch.cat([
                            torch.zeros(stride, device=device),
                            error[j, :n_pred]
                        ])
                    else:
                        chunk_prob = torch.zeros(chunk_lens[j], device=device)
                    chunk_probs.append(chunk_prob)
            pred_prob = torch.cat(chunk_probs)[:len_cur] if chunk_probs else torch.zeros(len_cur, device=device)

            all_prob.append(prob.cpu())
            all_pred.append(pred_prob.cpu())

    # Video-level AUC/AP from classifier
    prob_all = torch.cat(all_prob).numpy()
    prob_all = np.repeat(prob_all, 16)[:len(gt)]
    gt = np.array(gt)[:len(prob_all)]
    ROC = roc_auc_score(gt, prob_all)
    AP = average_precision_score(gt, prob_all)
    print(f"  Cls AUC: {ROC:.4f} | AP: {AP:.4f}")

    # Pred error score
    pred_all = torch.cat(all_pred).numpy()
    pred_all = np.repeat(pred_all, 16)[:len(gt)]
    pred_ROC = roc_auc_score(gt, pred_all)
    pred_AP = average_precision_score(gt, pred_all)
    print(f"  Pred AUC: {pred_ROC:.4f} | AP: {pred_AP:.4f}")

    return ROC, AP, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[xd_option.parser], add_help=False)
    parser.add_argument('--level', default=2, type=int, choices=[1, 2, 3])
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=1, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    parser.add_argument('--pred-stride', default=1, type=int)
    parser.add_argument('--mamba-only-cls', action='store_true')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--multi-scale-factors', default='1,8,16')
    parser.add_argument('--single-mamba', action='store_true')
    parser.add_argument('--model-path', required=True)
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
        mamba_kwargs = {'headdim': args.mamba2_headdim, 'ngroups': args.mamba2_ngroups}

    model = MambaVAD_Dual(
        visual_width=args.visual_width, visual_length=args.visual_length,
        mamba_layers=args.mamba_layers, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand,
        dropout=args.dropout, level=args.level,
        pred_stride=args.pred_stride,
        num_class=args.classes_num, embed_dim=args.embed_dim,
        prompt_prefix=args.prompt_prefix, prompt_postfix=args.prompt_postfix,
        device=device, mamba_version=args.mamba_version,
        mamba_only_cls=args.mamba_only_cls,
        multi_scale=args.multi_scale,
        multi_scale_factors=args.multi_scale_factors,
        single_mamba=args.single_mamba, **mamba_kwargs)

    ckpt = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    t_start = time.time()
    test(model, test_loader, args.visual_length, prompt_text,
         gt, gtsegments, gtlabels, device)
    elapsed = time.time() - t_start
    print(f"  Test time: {elapsed:.0f}s")

import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mamba_vad_dual_model import MambaVAD_Dual
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text
import ucf_option


def test(model, testdataloader, maxlen, device, gt_path, level=1, prompt_text=None,
         normal_proto=None):
    """Evaluate MambaVAD_Dual. Returns (AUC, AP).

    Level 1:   log1p(error_N) - log1p(error_A)  (dual prediction log-ratio)
    Level 2+   logits1 (classifier sigmoid) or prototype distance
      proto:   1 - cos(h_fused_pooled, normal_proto)
    """
    model.to(device)
    model.eval()
    use_proto = normal_proto is not None and level >= 2
    all_prob = []
    all_pred = []

    with torch.no_grad():
        for item in testdataloader:
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length
            visual = visual.to(device)

            num_chunks = int(length / maxlen) + 1
            chunk_lens = []
            remaining = length
            for _ in range(num_chunks):
                chunk_l = min(maxlen, remaining)
                chunk_lens.append(chunk_l)
                remaining -= chunk_l

            if visual.dim() == 2:
                visual = visual.unsqueeze(0)

            lengths_tensor = torch.tensor(chunk_lens, device=device)

            if level == 3 and prompt_text is not None:
                outputs = model(visual, lengths=lengths_tensor, text=prompt_text)
            else:
                outputs = model(visual, lengths=lengths_tensor)

            # Prediction error score
            stride = model.pred_stride
            if model.single_mamba:
                error = outputs['error']
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

                if level >= 2:
                    logits1 = torch.sigmoid(outputs['logits1']).squeeze(-1)
                    cls_chunks = [logits1[j, :chunk_lens[j]] for j in range(num_chunks) if chunk_lens[j] > 0]
                    prob = torch.cat(cls_chunks)[:len_cur] if cls_chunks else pred_prob
                else:
                    prob = pred_prob
            else:
                error_N = outputs['error_N']
                error_A = outputs['error_A']
                chunk_probs = []
                for j in range(num_chunks):
                    if chunk_lens[j] > 0:
                        n_pred = max(0, chunk_lens[j] - stride)
                        if n_pred > 0:
                            diff = torch.log1p(error_N[j, :n_pred]) - torch.log1p(error_A[j, :n_pred])
                            chunk_prob = torch.cat([torch.zeros(stride, device=device), diff])
                        else:
                            chunk_prob = torch.zeros(chunk_lens[j], device=device)
                        chunk_probs.append(chunk_prob)
                dual_prob = torch.cat(chunk_probs)[:len_cur] if chunk_probs else torch.zeros(len_cur, device=device)

                if level >= 2:
                    logits1 = torch.sigmoid(outputs['logits1']).squeeze(-1)
                    cls_chunks = [logits1[j, :chunk_lens[j]] for j in range(num_chunks) if chunk_lens[j] > 0]
                    prob = torch.cat(cls_chunks)[:len_cur] if cls_chunks else dual_prob
                else:
                    prob = dual_prob

            all_prob.append(prob.cpu())
            if level >= 2:
                all_pred.append(
                    (dual_prob.cpu() if not model.single_mamba else pred_prob.cpu()))

    prob_all = torch.cat(all_prob).numpy()
    prob_all = np.repeat(prob_all, 16)
    gt = np.load(gt_path)
    ROC = roc_auc_score(gt, prob_all)
    AP = average_precision_score(gt, prob_all)
    print(f"  Cls AUC: {ROC:.4f} | AP: {AP:.4f}")

    if level >= 2 and all_pred:
        pred_all = torch.cat(all_pred).numpy()
        pred_all = np.repeat(pred_all, 16)
        pred_ROC = roc_auc_score(gt, pred_all)
        pred_AP = average_precision_score(gt, pred_all)
        print(f"  Pred AUC: {pred_ROC:.4f} | AP: {pred_AP:.4f}")

    return ROC, AP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[ucf_option.parser], add_help=False)
    parser.add_argument('--level', default=1, type=int, choices=[1, 2, 3])
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    parser.add_argument('--pred-stride', default=1, type=int)
    parser.add_argument('--mamba-only-cls', action='store_true',
                        help='Skip GCN, use fused Mamba features + linear classifier (Option A)')
    parser.add_argument('--multi-scale', action='store_true',
                        help='Multi-scale temporal features fused via concat')
    parser.add_argument('--multi-scale-factors', default='1,8,16',
                        help='Comma-separated pooling factors (e.g. "1,8,16" or "1,4,12")')
    parser.add_argument('--single-mamba', action='store_true',
                        help='Single Mamba encoder instead of dual')
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    args.lr = float(args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest',
                      'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary',
                      'Explosion': 'explosion', 'Fighting': 'fighting',
                      'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
                      'Shooting': 'shooting', 'Shoplifting': 'shoplifting',
                      'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
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
    normal_proto_ema = ckpt.get('normal_proto_ema')

    prompt_text = get_prompt_text(label_map) if args.level >= 2 else None
    t_start = time.time()
    test(model, test_loader, args.visual_length, device, args.gt_path,
         level=args.level, prompt_text=prompt_text,
         normal_proto=normal_proto_ema)
    elapsed = time.time() - t_start
    print(f"  Test time: {elapsed:.0f}s")

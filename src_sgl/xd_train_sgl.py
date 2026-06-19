import argparse
import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random

from mamba_vad_dual_model import SingleMambaVAD
from xd_test_sgl import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option


def CLAS2(logits, labels, lengths, device, label_smooth=0.1):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    if label_smooth > 0:
        labels = labels * (1 - label_smooth) + label_smooth * 0.5
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        k = max(1, int(lengths[i] / 16 + 1))
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=k, largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def normal_prototype_loss(h_pooled, labels, margin=0.5, device='cuda'):
    h_norm = F.normalize(h_pooled, dim=-1)
    labels_binary = (labels[:, 0] == 1).float()
    h_normal = h_norm[labels_binary == 1]
    h_anomaly = h_norm[labels_binary == 0]
    if len(h_normal) < 1 or len(h_anomaly) < 1:
        return torch.tensor(0.0, device=device), h_norm.mean(dim=0)
    normal_proto = F.normalize(h_normal.mean(dim=0), dim=-1)
    sim_normal = h_normal @ normal_proto
    loss_normal = (1 - sim_normal).mean()
    sim_anomaly = h_anomaly @ normal_proto
    loss_anomaly = F.relu(sim_anomaly + margin).mean()
    return loss_normal + loss_anomaly, normal_proto.detach()


def train(model, train_loader, test_loader, args, label_map, device):
    os.makedirs('model', exist_ok=True)
    model.to(device)
    t_start = time.time()

    if args.mamba_lr_scale < 1.0:
        mamba_params = model.mamba.parameters()
        optimizer = torch.optim.AdamW([
            {'params': mamba_params, 'lr': args.lr * args.mamba_lr_scale},
            {'params': model.frame_position_embeddings.parameters(), 'lr': args.lr * args.mamba_lr_scale},
            {'params': [p for n, p in model.named_parameters()
                        if 'mamba' not in n and 'frame_position' not in n]},
        ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    best_model_path = None
    normal_proto_ema = None

    for e in range(args.max_epoch):
        model.train()
        t_epoch = time.time()
        loss_cls_total = 0
        loss_proto_total = 0
        loss_future_total = 0

        for i, item in enumerate(train_loader):
            visual_feat, text_labels, feat_lengths = item
            B = visual_feat.shape[0]
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)

            text_onehot = get_batch_label(text_labels, prompt_text, label_map).to(device)
            outputs = model(visual_feat, lengths=feat_lengths)

            # Classification loss (CLAS2)
            loss2 = CLAS2(outputs['logits1'], text_onehot, feat_lengths, device)
            loss_cls_total += loss2.item()
            loss = loss2

            # Normal prototype contrastive loss
            h_fused = outputs['hidden_state_fused']
            lengths = feat_lengths
            mask = torch.arange(h_fused.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            h_pooled = (h_fused * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
            loss_proto, normal_proto = normal_prototype_loss(h_pooled, text_onehot, device=device)
            loss_proto_total += loss_proto.item()
            loss = loss + loss_proto * 0.1

            # Future score prediction bridge
            logits_detach = outputs['logits1'].detach()
            loss_future = F.mse_loss(outputs['score_future'], logits_detach)
            loss_future_total += loss_future.item()
            loss = loss + loss_future * 0.1

            if normal_proto_ema is None:
                normal_proto_ema = normal_proto
            else:
                normal_proto_ema = F.normalize(0.999 * normal_proto_ema + 0.001 * normal_proto, dim=-1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        epoch_time = time.time() - t_epoch
        avg_cls = loss_cls_total / len(train_loader)
        avg_proto = loss_proto_total / len(train_loader)
        avg_future = loss_future_total / len(train_loader)
        print(f"[Epoch {e+1}/{args.max_epoch}] "
              f"Cls:{avg_cls:.4f} | Proto:{avg_proto:.4f} | "
              f"Future:{avg_future:.4f} | Time:{epoch_time:.0f}s")

        AUC, AP, _ = test(model, test_loader, args.visual_length,
                          gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP
            new_path = f"model/xd_best_ap{AP:.4f}_auc{AUC:.4f}_ep{e+1}.pt"
            torch.save({'model_state_dict': model.trainable_state_dict(),
                        'normal_proto_ema': normal_proto_ema}, new_path)
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = new_path

        print(f"[Epoch {e+1}/{args.max_epoch}] AUC: {AUC:.4f} | AP: {AP:.4f} | Best AP: {ap_best:.4f}")

    final = f"model/xd_final_ap{ap_best:.4f}.pt"
    torch.save({'model_state_dict': model.trainable_state_dict(),
                'normal_proto_ema': normal_proto_ema}, final)
    total_time = time.time() - t_start
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Final model saved to {final} (Best AP: {ap_best:.4f})")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[xd_option.parser], add_help=False)
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=1, type=int)
    parser.add_argument('--mlp-scale', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--multi-scale-factors', default='1,8,16')
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--mamba-lr-scale', default=1.0, type=float)
    parser.add_argument('--label-smooth', default=0.0, type=float)
    parser.add_argument('--bidirectional', action='store_true')
    args = parser.parse_args()
    args.lr = float(args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
                      'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
                      'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

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
        bidirectional=args.bidirectional,
        mlp_scale=args.mlp_scale, **mamba_kwargs)

    ms_str = f"scales={args.multi_scale_factors}" if args.multi_scale else "off"
    print(f"[Config] SingleMambaVAD XD, "
          f"mamba_layers={args.mamba_layers}, d_state={args.d_state}, "
          f"mlp_scale={args.mlp_scale}, mamba_version={args.mamba_version}, "
          f"multi_scale={ms_str}, bidirectional={args.bidirectional}")

    train(model, train_loader, test_loader, args, label_map, device)

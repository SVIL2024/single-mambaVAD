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
from ucf_test_sgl import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option


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


def normal_prototype_loss(h_pooled, binary_labels, margin=0.5, device='cuda'):
    h_norm = F.normalize(h_pooled, dim=-1)
    h_normal = h_norm[binary_labels == 0]
    h_anomaly = h_norm[binary_labels == 1]
    if len(h_normal) < 1 or len(h_anomaly) < 1:
        return torch.tensor(0.0, device=device), h_norm.mean(dim=0)
    normal_proto = F.normalize(h_normal.mean(dim=0), dim=-1)
    sim_normal = h_normal @ normal_proto
    loss_normal = (1 - sim_normal).mean()
    sim_anomaly = h_anomaly @ normal_proto
    loss_anomaly = F.relu(sim_anomaly + margin).mean()
    return loss_normal + loss_anomaly, normal_proto.detach()


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
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
    prompt_text = get_prompt_text(label_map)

    auc_best = 0
    ap_best = 0
    best_model_path = None
    normal_proto_ema = None

    for e in range(args.max_epoch):
        t_epoch = time.time()
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_batches = min(len(normal_loader), len(anomaly_loader))
        loss_cls_total = 0
        loss_proto_total = 0
        loss_future_total = 0

        for i in range(num_batches):
            normal_feat, normal_label, normal_len = next(normal_iter)
            anomaly_feat, anomaly_label, anomaly_len = next(anomaly_iter)

            B_normal = normal_feat.shape[0]
            B_anomaly = anomaly_feat.shape[0]

            visual = torch.cat([normal_feat, anomaly_feat], dim=0).to(device)
            feat_lengths = torch.cat([normal_len, anomaly_len], dim=0).to(device)

            normal_mask = torch.zeros(B_normal + B_anomaly, dtype=torch.bool, device=device)
            normal_mask[:B_normal] = True

            text_labels = list(normal_label) + list(anomaly_label)
            text_onehot = get_batch_label(text_labels, prompt_text, label_map).to(device)
            outputs = model(visual, lengths=feat_lengths)

            # Classification loss (CLAS2)
            loss2 = CLAS2(outputs['logits1'], text_onehot, feat_lengths, device, label_smooth=args.label_smooth)
            loss_cls_total += loss2.item()
            loss = loss2

            # Normal prototype contrastive loss
            h_fused = outputs['hidden_state_fused']
            lengths = feat_lengths
            mask = torch.arange(h_fused.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            h_pooled = (h_fused * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
            binary_labels = torch.cat([torch.zeros(B_normal), torch.ones(B_anomaly)]).to(device)
            loss_proto, normal_proto = normal_prototype_loss(h_pooled, binary_labels, device=device)
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
        avg_cls = loss_cls_total / num_batches
        avg_proto = loss_proto_total / num_batches
        avg_future = loss_future_total / num_batches
        epoch_time = time.time() - t_epoch
        print(f"[Epoch {e+1}/{args.max_epoch}] "
              f"Cls:{avg_cls:.4f} | Proto:{avg_proto:.4f} | "
              f"Future:{avg_future:.4f} | Time:{epoch_time:.0f}s")

        AUC, AP = test(model, testloader, args.visual_length, device, args.gt_path,
                       normal_proto=normal_proto_ema)

        if AUC > auc_best:
            auc_best = AUC
            ap_best = AP
            new_path = f"model/best_auc{auc_best:.4f}_ap{ap_best:.4f}_ep{e+1}.pt"
            torch.save({'model_state_dict': model.trainable_state_dict(),
                        'normal_proto_ema': normal_proto_ema}, new_path)
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = new_path

        print(f"[Epoch {e+1}/{args.max_epoch}] AUC: {AUC:.4f} | AP: {AP:.4f} | Best AUC: {auc_best:.4f}")

    final = f"model/final_auc{auc_best:.4f}.pt"
    if best_model_path and os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, weights_only=False)
        torch.save({'model_state_dict': ckpt['model_state_dict'],
                    'normal_proto_ema': ckpt.get('normal_proto_ema')}, final)
    else:
        torch.save({'model_state_dict': model.trainable_state_dict(),
                    'normal_proto_ema': normal_proto_ema}, final)
    total_time = time.time() - t_start
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Final model saved to {final} (AUC: {auc_best:.4f})")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[ucf_option.parser], add_help=False)
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=2, type=int)
    parser.add_argument('--mlp-scale', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--multi-scale-factors', default='1,8,16')
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--mamba-lr-sca
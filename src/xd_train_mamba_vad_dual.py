import argparse
import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import random

from mamba_vad_dual_model import MambaVAD_Dual
from xd_test_mamba_vad_dual import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
from utils.layers import GraphConvolution, DistanceAdj, GraphAttentionLayer
import xd_option


# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        k = max(1, int(lengths[i] / 16 + 1))
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=k, largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


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
    """labels: 1=normal, 0=anomaly (binary_labels from get_batch_label[:,0]==1)"""
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


# ──────────────────────────────────────────────
# Config tag for model filenames
# ──────────────────────────────────────────────

def build_config_tag(args, prefix='xd'):
    """Compact CLI-args tag for model filenames. Omits path/seed fields."""
    skip_keys = {
        'model_path', 'train_list', 'test_list', 'gt_path',
        'gt_segment_path', 'gt_label_path', 'checkpoint_path',
        'use_checkpoint', 'seed', 'scheduler_rate', 'scheduler_milestones',
        'visual_layers', 'attn_window', 'visual_head',
    }
    short = {
        'level': 'lv', 'd_state': 'ds', 'mamba_layers': 'ml',
        'mamba_version': 'mv', 'batch_size': 'bs', 'pred_stride': 'ps',
        'dropout': 'do', 'weight_decay': 'wd', 'mamba_lr_scale': 'mls',
        'freeze_after_epoch': 'fae', 'lr': 'lr',
        'mamba_only_cls': 'cls', 'single_mamba': 'sm',
        'multi_scale': 'ms', 'multi_scale_factors': 'mf',
        'max_epoch': 'ep', 'ghost_weight': 'gw',
        'd_conv': 'dc', 'expand': 'exp', 'mamba2_headdim': 'hd',
        'mamba2_ngroups': 'ng', 'label_smooth': 'ls',
        'classes_num': 'cn', 'visual_width': 'vw', 'visual_length': 'vl',
        'prompt_prefix': 'pp', 'prompt_postfix': 'po',
    }
    parts = [prefix]
    for k, v in sorted(vars(args).items()):
        if k in skip_keys or v is None:
            continue
        abbr = short.get(k, k[:4])
        if isinstance(v, bool):
            if v:
                parts.append(f"{abbr}")
        elif isinstance(v, float):
            s = f"{v:.0e}" if 1e-4 <= abs(v) < 0.01 else f"{v}"
            parts.append(f"{abbr}-{s}")
        elif isinstance(v, str):
            cleaned = v.replace(',', '_')
            parts.append(f"{abbr}-{cleaned}")
        elif isinstance(v, int):
            parts.append(f"{abbr}-{v}")
        elif isinstance(v, list):
            parts.append(f"{abbr}-{'_'.join(str(x) for x in v)}")
        else:
            parts.append(f"{abbr}-{v}")
    return '_'.join(parts)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(model, train_loader, test_loader, args, label_map, device):
    os.makedirs('model', exist_ok=True)
    model.to(device)
    t_start = time.time()

    if args.mamba_lr_scale < 1.0:
        if model.single_mamba:
            mamba_params = model.mamba.parameters()
        else:
            mamba_params = list(model.mamba_normal.parameters()) + list(model.mamba_anomaly.parameters())
        optimizer = torch.optim.AdamW([
            {'params': mamba_params, 'lr': args.lr * args.mamba_lr_scale},
            {'params': model.frame_position_embeddings.parameters(), 'lr': args.lr * args.mamba_lr_scale},
            {'params': [p for n, p in model.named_parameters()
                        if 'mamba' not in n and 'frame_position' not in n]},
        ], lr=args.lr, weight_decay=args.weight_decay)
        print(f"[Optim] Mamba lr={args.lr * args.mamba_lr_scale:.2e}, other lr={args.lr:.2e}")
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
        # Freeze Mamba after specified epoch
        if 0 < args.freeze_after_epoch <= e:
            if model.single_mamba:
                mamba_modules = [model.mamba]
            else:
                mamba_modules = [model.mamba_normal, model.mamba_anomaly]
            for m in mamba_modules:
                for p in m.parameters():
                    p.requires_grad = False
            for p in model.frame_position_embeddings.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, weight_decay=args.weight_decay)
            args.freeze_after_epoch = 0
            print(f"[Freeze] Mamba frozen at epoch {e+1}, re-created optimizer")

        model.train()
        t_epoch = time.time()
        loss_predict_total = 0
        loss_cls_total = 0
        loss_proto_total = 0
        loss_future_total = 0
        loss_align_total = 0

        for i, item in enumerate(train_loader):
            visual_feat, text_labels, feat_lengths = item
            B = visual_feat.shape[0]
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)

            # XD uses single mixed loader — build normal_mask from labels
            text_onehot = get_batch_label(text_labels, prompt_text, label_map).to(device)
            normal_mask = (text_onehot[:, 0] == 1)

            outputs = model(visual_feat, lengths=feat_lengths, text=prompt_text, normal_mask=normal_mask)

            # ===== Frame prediction loss =====
            error = outputs['error']
            k = max(1, int(error.shape[1] * 0.1))
            loss_pred = error.topk(k, dim=1)[0].mean()
            loss_predict_total += loss_pred.item()

            if args.level == 1:
                loss = loss_pred
            else:
                loss2 = CLAS2(outputs['logits1'], text_onehot, feat_lengths, device)
                loss_cls_total += loss2.item()
                loss = loss2

                # ===== Prototype loss =====
                h_fused = outputs['hidden_state_fused']
                lengths = feat_lengths
                mask = torch.arange(h_fused.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
                h_pooled = (h_fused * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)

                loss_proto, normal_proto = normal_prototype_loss(h_pooled, text_onehot, device=device)
                loss_proto_total += loss_proto.item()
                loss = loss + loss_proto * 1

                # ===== Future score prediction =====
                if model.single_mamba and 'score_future' in outputs:
                    logits_detach = outputs['logits1'].detach()
                    loss_future = F.mse_loss(outputs['score_future'], logits_detach)
                    loss_future_total += loss_future.item()
                    loss = loss + loss_future * 1

                if normal_proto_ema is None:
                    normal_proto_ema = normal_proto
                else:
                    normal_proto_ema = F.normalize(0.999 * normal_proto_ema + 0.001 * normal_proto, dim=-1)

                # ===== Level 3: Ghost Teacher =====
                if args.level >= 3:
                    text_features = outputs['text_features']
                    t_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
                    loss3 = torch.zeros(1).to(device)
                    for j in range(1, text_features.shape[0]):
                        t_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                        loss3 += torch.abs(t_normal @ t_abr)
                    loss3 = loss3 / (text_features.shape[0] - 1) * 1e-4

                    h_proj = outputs['h_proj']
                    logits1 = outputs['logits1']
                    scores = torch.sigmoid(logits1).squeeze(-1)
                    anomaly_mask = ~normal_mask
                    if anomaly_mask.any():
                        h_proj_a = h_proj[anomaly_mask]
                        scores_a = scores[anomaly_mask]
                        text_onehot_a = text_onehot[anomaly_mask]
                        lengths_a = feat_lengths[anomaly_mask]
                        Ta = h_proj_a.shape[1]
                        pad_mask = torch.arange(Ta, device=device).unsqueeze(0) < lengths_a.unsqueeze(1)
                        scores_a = scores_a * pad_mask.float() + (~pad_mask).float() * (-1e9)
                        k = max(1, int(Ta * 0.05))
                        _, topk_idx = torch.topk(scores_a, k, dim=1)
                        batch_idx = torch.arange(h_proj_a.shape[0], device=device).unsqueeze(1).expand(-1, k)
                        h_selected = h_proj_a[batch_idx, topk_idx]
                        h_norm = F.normalize(h_selected, dim=-1)
                        z_norm = F.normalize(text_features, dim=-1)
                        z_target = z_norm.unsqueeze(0) * text_onehot_a.unsqueeze(-1)
                        z_target = z_target.sum(dim=1)
                        z_target = F.normalize(z_target, dim=-1)
                        z_target = z_target.unsqueeze(1).expand(-1, k, -1)
                        loss_align = F.mse_loss(h_norm, z_target)
                        loss_align_total += loss_align.item()
                        loss = loss + loss_align * args.ghost_weight + loss3

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        epoch_time = time.time() - t_epoch
        avg_pred = loss_predict_total / len(train_loader)
        avg_cls = loss_cls_total / len(train_loader) if args.level >= 2 else 0
        avg_proto = loss_proto_total / len(train_loader) if args.level >= 2 else 0
        avg_future = loss_future_total / len(train_loader) if (args.level >= 2 and model.single_mamba) else 0
        avg_align = loss_align_total / len(train_loader) if args.level >= 3 else 0
        print(f"[Epoch {e+1}/{args.max_epoch}] "
              f"Pred:{avg_pred:.4f} | Cls:{avg_cls:.4f} | "
              f"Proto:{avg_proto:.4f} | Future:{avg_future:.4f} | "
              f"Align:{avg_align:.4f} | Time:{epoch_time:.0f}s")

        # Evaluate at end of each epoch
        AUC, AP, _ = test(model, test_loader, args.visual_length, prompt_text,
                          gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP
            cfg = build_config_tag(args, prefix='xd')
            new_path = f"model/xd_best_ap{AP:.4f}_auc{AUC:.4f}_{cfg}.pt"
            torch.save({'model_state_dict': model.state_dict(),
                        'normal_proto_ema': normal_proto_ema}, new_path)
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = new_path
            print(f"  >> Saved best: {new_path}")

        print(f"[Epoch {e+1}/{args.max_epoch}] AUC: {AUC:.4f} | AP: {AP:.4f} | Best AP: {ap_best:.4f}")

    cfg = build_config_tag(args, prefix='xd')
    final = f"model/xd_final_{cfg}.pt"
    torch.save({'model_state_dict': model.state_dict(),
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
    parser.add_argument('--level', default=2, type=int, choices=[1, 2, 3])
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=1, type=int)
    parser.add_argument('--mlp-scale', default=4, type=int,
                        help='MLP expansion factor in Mamba block (default=4, try 2 for faster)')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    parser.add_argument('--pred-stride', default=1, type=int)
    parser.add_argument('--mamba-only-cls', action='store_true')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--multi-scale-factors', default='1,8,16')
    parser.add_argument('--ghost-weight', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--single-mamba', action='store_true')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use Bidirectional Mamba (BiMambaEncoder) instead of causal Mamba')
    parser.add_argument('--label-smooth', default=0.1, type=float)
    parser.add_argument('--random-crop', action='store_true',
                        help='Random temporal crop instead of uniform sampling')
    parser.add_argument('--freeze-after-epoch', default=0, type=int,
                        help='Freeze Mamba encoders after this epoch (0 = never freeze)')
    parser.add_argument('--mamba-lr-scale', default=1.0, type=float,
                        help='Mamba encoder lr = args.lr * mamba_lr_scale (e.g. 0.1)')
    parser.set_defaults(model_path='model/xd_mamba.pth')
    args = parser.parse_args()
    args.lr = float(args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
                      'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
                      'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map,
                               is_random=args.random_crop)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)

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
        single_mamba=args.single_mamba,
        bidirectional=args.bidirectional,
        mlp_scale=args.mlp_scale, **mamba_kwargs)

    ms_str = f"scales={args.multi_scale_factors}" if args.multi_scale else "off"
    print(f"[Config] XD MambaVAD_Dual Level={args.level}, pred_stride={args.pred_stride}, "
          f"mamba_layers={args.mamba_layers}, d_state={args.d_state}, "
          f"mlp_scale={args.mlp_scale}, mamba_version={args.mamba_version}, "
          f"multi_scale={ms_str}, "
          f"single_mamba={args.single_mamba}, bidirectional={args.bidirectional}")

    train(model, train_loader, test_loader, args, label_map, device)

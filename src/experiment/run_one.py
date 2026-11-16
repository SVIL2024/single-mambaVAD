"""
Run a single ablation experiment by ID (1–6).

Usage:
    python experiment/run_one.py --id 1     # full baseline
    python experiment/run_one.py --id 6     # CLIP baseline
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

# Add parent dir (src/) to path so imports work as if run from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ucf_train_mamba_vad_dual import (
    setup_seed,
    CLAS2,
)
from ucf_test_mamba_vad_dual import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
from mamba_vad_dual_model import MambaVAD_Dual
import ucf_option

from experiment.config import get_exp_config, results_dir
from experiment import losses

MODEL_DIR = os.environ.get('VAD_MODEL_DIR', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
from experiment.models import CLIPBaseline


def build_config_tag(args):
    """Compact CLI-args tag for model filenames (same as ucf_train_mamba_vad_dual)."""
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
        # real-fix knobs: without explicit short names these fall back to
        # k[:4] ("prot"/"futu"), which both collides and blows past NAME_MAX
        'proto_pool': 'ppl', 'proto_topk_ratio': 'ptr', 'proto_tau': 'ptau',
        'future_horizon': 'fhk', 'future_align': 'fal',
        'future_weight': 'fw', 'proto_weight': 'pw',
    }
    parts = []
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
            parts.append(f"{abbr}-{v}")
        elif isinstance(v, int):
            parts.append(f"{abbr}-{v}")
        elif isinstance(v, list):
            parts.append(f"{abbr}-{'_'.join(str(x) for x in v)}")
        else:
            parts.append(f"{abbr}-{v}")
    tag = '_'.join(parts)
    # ext4 caps a single path component at 255 bytes; the full tag plus the
    # "best_aucX.XXXX_apX.XXXX_" prefix and the directory already spends ~50,
    # and the pre-fix tag was sitting right at the limit (ENAMETOOLONG on save).
    if len(tag) > 180:
        tag = tag[:180].rstrip('-_.')
    return tag


def train_and_eval(model, normal_loader, anomaly_loader, testloader,
                   args, label_map, device, diag=None):
    """Identical to ucf_train_mamba_vad_dual.train() but returns (best_auc, best_ap).

    ``diag`` (optional dict) collects per-step diagnostics of the two auxiliary
    terms: prototype similarity for each group, the fraction of anomalous frames
    that actually take part in the repulsion, and the number of supervised
    bridge pairs.  They are averaged over the run by the caller.
    """
    if diag is None:
        diag = {}
    os.makedirs('model', exist_ok=True)
    model.to(device)
    t_start = time.time()

    if args.mamba_lr_scale < 1.0:
        lr_mamba = args.lr * args.mamba_lr_scale
        if model.single_mamba:
            mamba_params = model.mamba.parameters()
        else:
            mamba_params = (list(model.mamba_normal.parameters())
                            + list(model.mamba_anomaly.parameters()))
        optimizer = torch.optim.AdamW([
            {'params': mamba_params, 'lr': lr_mamba},
            {'params': model.frame_position_embeddings.parameters(), 'lr': lr_mamba},
            {'params': [p for n, p in model.named_parameters()
                        if 'mamba' not in n and 'frame_position' not in n]},
        ], lr=args.lr, weight_decay=args.weight_decay)
        print(f"[Optim] Mamba lr={lr_mamba:.2e}, other lr={args.lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    prompt_text = get_prompt_text(label_map) if args.level >= 2 else None

    auc_best = 0
    ap_best = 0
    best_model_path = None
    normal_proto_ema = None

    for e in range(args.max_epoch):
        t_epoch = time.time()
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
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_batches = min(len(normal_loader), len(anomaly_loader))
        loss_predict_total = 0
        loss_cls_total = 0
        loss_proto_total = 0
        loss_future_total = 0
        loss_align_total = 0

        for i in range(num_batches):
            normal_feat, normal_label, normal_len = next(normal_iter)
            anomaly_feat, anomaly_label, anomaly_len = next(anomaly_iter)

            B_normal = normal_feat.shape[0]
            B_anomaly = anomaly_feat.shape[0]

            visual = torch.cat([normal_feat, anomaly_feat], dim=0).to(device)
            feat_lengths = torch.cat([normal_len, anomaly_len], dim=0).to(device)

            normal_mask = torch.zeros(B_normal + B_anomaly, dtype=torch.bool, device=device)
            normal_mask[:B_normal] = True

            if args.level >= 3:
                text_labels = list(normal_label) + list(anomaly_label)
                text_onehot = get_batch_label(text_labels, prompt_text, label_map).to(device)
                outputs = model(visual, lengths=feat_lengths, text=prompt_text,
                                normal_mask=normal_mask)
            else:
                text_onehot = None
                outputs = model(visual, lengths=feat_lengths, text=None,
                                normal_mask=normal_mask)

            if model.single_mamba:
                error = outputs['error']
                k = max(1, int(error.shape[1] * 0.1))
                loss_pred = error.topk(k, dim=1)[0].mean()
            else:
                error_N = outputs['error_N']
                error_A = outputs['error_A']
                k = max(1, int(error_N.shape[1] * 0.1))
                loss_n = error_N.topk(k, dim=1)[0].mean(dim=1)[normal_mask].mean()
                loss_a = 0
                count_a = 0
                for j in range(error_N.shape[0]):
                    if not normal_mask[j]:
                        _, idx = torch.topk(error_N[j], k)
                        loss_a = loss_a + error_A[j][idx].mean()
                        count_a = count_a + 1
                loss_a = loss_a / max(count_a, 1)
                loss_pred = loss_n + loss_a

            if args.level == 1:
                loss = loss_pred
            else:
                if text_onehot is None:
                    text_labels = list(normal_label) + list(anomaly_label)
                    text_onehot = get_batch_label(
                        text_labels, prompt_text, label_map).to(device)

                loss2 = CLAS2(outputs['logits1'], text_onehot, feat_lengths,
                              device, label_smooth=args.label_smooth)
                loss_cls_total += loss2.item()
                loss = loss2

                aux = losses.compute_aux_terms(
                    h_fused=outputs['hidden_state_fused'],
                    logits1=outputs['logits1'],
                    score_future=outputs.get('score_future'),
                    lengths=feat_lengths,
                    normal_mask=normal_mask,
                    margin=0.5,
                    pool=args.proto_pool,
                    topk_ratio=args.proto_topk_ratio,
                    tau=args.proto_tau,
                    horizon=args.future_horizon,
                    align=args.future_align,
                )
                loss_proto, normal_proto = aux['loss_proto'], aux['proto']
                loss_proto_total += loss_proto.item()
                loss = loss + loss_proto * args.proto_weight
                for _k in ('sim_normal', 'sim_anomaly', 'proto_coverage'):
                    diag[_k] = diag.get(_k, 0.0) + float(aux[_k])

                if model.single_mamba and aux['loss_future'] is not None:
                    loss_future = aux['loss_future']
                    loss_future_total += loss_future.item()
                    loss = loss + loss_future * args.future_weight
                    diag['future_pairs'] = (diag.get('future_pairs', 0.0)
                                            + float(aux['future_pairs']))
                    diag['future_coverage'] = (diag.get('future_coverage', 0.0)
                                               + float(aux['future_coverage']))

                if normal_proto_ema is None:
                    normal_proto_ema = normal_proto
                else:
                    normal_proto_ema = F.normalize(
                        0.999 * normal_proto_ema + 0.001 * normal_proto, dim=-1)

                if args.level >= 3:
                    text_features = outputs['text_features']
                    loss3 = torch.zeros(1).to(device)
                    t_normal = (text_features[0]
                                / text_features[0].norm(dim=-1, keepdim=True))
                    for j in range(1, text_features.shape[0]):
                        t_abr = (text_features[j]
                                 / text_features[j].norm(dim=-1, keepdim=True))
                        loss3 += torch.abs(t_normal @ t_abr)
                    loss3 = loss3 / (text_features.shape[0] - 1) * 1e-1

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
                        pad_mask = (torch.arange(Ta, device=device).unsqueeze(0)
                                    < lengths_a.unsqueeze(1))
                        scores_a = (scores_a * pad_mask.float()
                                    + (~pad_mask).float() * (-1e9))
                        k = max(1, int(Ta * 0.05))
                        _, topk_idx = torch.topk(scores_a, k, dim=1)
                        batch_idx = (torch.arange(h_proj_a.shape[0], device=device)
                                     .unsqueeze(1).expand(-1, k))
                        h_selected = h_proj_a[batch_idx, topk_idx]
                        h_norm = F.normalize(h_selected, dim=-1)
                        z_norm = F.normalize(text_features, dim=-1)
                        z_target = (z_norm.unsqueeze(0)
                                    * text_onehot_a.unsqueeze(-1))
                        z_target = z_target.sum(dim=1)
                        z_target = F.normalize(z_target, dim=-1)
                        z_target = (z_target.unsqueeze(1)
                                    .expand(-1, k, -1))
                        loss_align = F.mse_loss(h_norm, z_target)
                        loss_align_total += loss_align.item()
                        loss = (loss + loss_align * args.ghost_weight + loss3)
                    else:
                        loss = loss + loss3

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_predict_total += loss_pred.item()
            diag['steps'] = diag.get('steps', 0) + 1

        scheduler.step()
        avg_pred = loss_predict_total / num_batches
        avg_cls = (loss_cls_total / num_batches if args.level >= 2 else 0)
        avg_proto = (loss_proto_total / num_batches if args.level >= 2 else 0)
        avg_future = (loss_future_total / num_batches
                      if (args.level >= 2 and model.single_mamba) else 0)
        avg_align = (loss_align_total / num_batches if args.level >= 3 else 0)
        epoch_time = time.time() - t_epoch
        print(f"[Epoch {e+1}/{args.max_epoch}] "
              f"Pred:{avg_pred:.4f} | Cls:{avg_cls:.4f} | "
              f"Proto:{avg_proto:.4f} | Future:{avg_future:.4f} | "
              f"Align:{avg_align:.4f} | Time:{epoch_time:.0f}s")

        AUC, AP = test(model, testloader, args.visual_length, device,
                       args.gt_path, level=args.level, prompt_text=prompt_text,
                       normal_proto=normal_proto_ema)

        if AUC > auc_best:
            auc_best = AUC
            ap_best = AP
            cfg = build_config_tag(args)
            new_path = (f"{MODEL_DIR}/best_auc{auc_best:.4f}_ap{ap_best:.4f}_{cfg}.pt")
            torch.save({'model_state_dict': model.state_dict(),
                        'normal_proto_ema': normal_proto_ema}, new_path)
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = new_path
            print(f"  >> Saved best: {new_path}")

        print(f"[Epoch {e+1}/{args.max_epoch}] AUC: {AUC:.4f} | "
              f"AP: {AP:.4f} | Best AUC: {auc_best:.4f}")

    cfg = build_config_tag(args)
    final = f"{MODEL_DIR}/final_{cfg}.pt"
    if best_model_path and os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, weights_only=False)
        torch.save({'model_state_dict': ckpt['model_state_dict'],
                    'normal_proto_ema': ckpt.get('normal_proto_ema')}, final)
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'normal_proto_ema': normal_proto_ema}, final)
    total_time = time.time() - t_start
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Final model saved to {final} (AUC: {auc_best:.4f})")

    return auc_best, ap_best


def cfg_to_argv(cfg_dict):
    """Turn an experiment config dict into the CLI argv used by ``main``."""
    argv = []
    for k, v in cfg_dict.items():
        if k in ('exp_name', 'exp_id', 'model_type'):
            continue
        if isinstance(v, bool):
            if v:
                argv.append(f'--{k.replace("_", "-")}')
        else:
            argv.extend([f'--{k.replace("_", "-")}', str(v)])
    return argv


def build_opt_parser():
    """The ablation CLI surface, kept importable so it can be tested."""
    opt_parser = argparse.ArgumentParser(parents=[ucf_option.parser],
                                         add_help=False)
    opt_parser.add_argument('--level', default=1, type=int, choices=[1, 2, 3])
    opt_parser.add_argument('--d-state', default=16, type=int)
    opt_parser.add_argument('--d-conv', default=4, type=int)
    opt_parser.add_argument('--expand', default=2, type=int)
    opt_parser.add_argument('--mamba-layers', default=2, type=int)
    opt_parser.add_argument('--mlp-scale', default=4, type=int)
    opt_parser.add_argument('--dropout', default=0.0, type=float)
    opt_parser.add_argument('--mamba-version', default='mamba1',
                            choices=['mamba1', 'mamba2'])
    opt_parser.add_argument('--mamba2-headdim', default=64, type=int)
    opt_parser.add_argument('--mamba2-ngroups', default=1, type=int)
    opt_parser.add_argument('--pred-stride', default=1, type=int)
    opt_parser.add_argument('--mamba-only-cls', action='store_true')
    opt_parser.add_argument('--multi-scale', action='store_true')
    opt_parser.add_argument('--multi-scale-factors', default='1,8,16')
    opt_parser.add_argument('--ghost-weight', default=0.1, type=float)
    opt_parser.add_argument('--weight-decay', default=0.0, type=float)
    opt_parser.add_argument('--freeze-after-epoch', default=0, type=int)
    opt_parser.add_argument('--mamba-lr-scale', default=1.0, type=float)
    opt_parser.add_argument('--label-smooth', default=0.1, type=float)
    opt_parser.add_argument('--proto-weight', default=0.1, type=float)
    opt_parser.add_argument('--future-weight', default=0.1, type=float)
    opt_parser.add_argument('--proto-pool', default='mean',
                            choices=['mean', 'topk', 'soft'],
                            help='what the repulsive half of the prototype '
                                 'loss acts on (mean = submitted behaviour)')
    opt_parser.add_argument('--proto-topk-ratio', default=0.1, type=float,
                            help='fraction of frames kept by --proto-pool topk')
    opt_parser.add_argument('--proto-tau', default=0.1, type=float,
                            help='softmax temperature for --proto-pool soft')
    opt_parser.add_argument('--future-horizon', default=1, type=int,
                            help='h_t predicts s_{t+k}; k=1 is the one-step '
                                 'bridge')
    opt_parser.add_argument('--future-align', default='legacy',
                            choices=['shift', 'legacy'],
                            help='legacy = the exact configuration behind every '
                                 'number reported in the paper (delta=0, padded '
                                 'as in the original release); shift = padded-free '
                                 'delta=k together with --future-horizon k '
                                 'objective, kept for bit-exact reruns')
    opt_parser.add_argument('--bidirectional', action='store_true')
    opt_parser.add_argument('--single-mamba', action='store_true')
    return opt_parser


def main():
    parser = argparse.ArgumentParser(description='Run one ablation experiment')
    parser.add_argument('--id', type=int, required=True,
                        help='1=full,2=causal,3=no_ms,4=ms_fine,5=ms_coarse,'
                             '6=ms_fine_causal,7=ms_coarse_causal,'
                             '8=no_proto,9=no_future,10=cls_only')
    cli_args = parser.parse_args()

    cfg_dict = get_exp_config(cli_args.id)
    exp_name = cfg_dict['exp_name']
    is_clip_baseline = cfg_dict.get('model_type') == 'clip_baseline'
    print(f"\n{'='*60}")
    print(f"  Experiment {cli_args.id}: {exp_name}")
    print(f"{'='*60}")

    setup_seed(cfg_dict['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest',
                      'Arson': 'arson', 'Assault': 'assault',
                      'Burglary': 'burglary', 'Explosion': 'explosion',
                      'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents',
                      'Robbery': 'robbery', 'Shooting': 'shooting',
                      'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
                      'Vandalism': 'vandalism'})

    opt_parser = build_opt_parser()

    argv = cfg_to_argv(cfg_dict)
    args = opt_parser.parse_args(argv)
    args.lr = float(args.lr)

    # ── Datasets ──
    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, True)
    n_workers = int(os.environ.get('VAD_WORKERS', '4'))
    loader_kw = dict(num_workers=n_workers, pin_memory=True)
    if n_workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        **loader_kw)
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        **loader_kw)
    test_dataset = UCFDataset(
        args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, **loader_kw)

    # ── Build model ──
    mamba_kwargs = {}
    if args.mamba_version == 'mamba2':
        mamba_kwargs = {'headdim': args.mamba2_headdim,
                        'ngroups': args.mamba2_ngroups}

    if is_clip_baseline:
        model = CLIPBaseline(
            visual_width=args.visual_width,
            num_class=args.classes_num,
            dropout=args.dropout,
            pred_stride=args.pred_stride,
            device=device)
        print(f"[Config] CLIPBaseline (no Mamba)")
    else:
        model = MambaVAD_Dual(
            visual_width=args.visual_width,
            visual_length=args.visual_length,
            mamba_layers=args.mamba_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            dropout=args.dropout,
            level=args.level,
            pred_stride=args.pred_stride,
            num_class=args.classes_num,
            embed_dim=args.embed_dim,
            prompt_prefix=args.prompt_prefix,
            prompt_postfix=args.prompt_postfix,
            device=device,
            mamba_version=args.mamba_version,
            mamba_only_cls=args.mamba_only_cls,
            multi_scale=args.multi_scale,
            multi_scale_factors=args.multi_scale_factors,
            single_mamba=args.single_mamba,
            bidirectional=args.bidirectional,
            mlp_scale=args.mlp_scale,
            future_align=args.future_align,
            **mamba_kwargs)
        ms_str = (f"scales={args.multi_scale_factors}"
                  if args.multi_scale else "off")
        print(f"[Config] MambaVAD_Dual Level={args.level}, "
              f"pred_stride={args.pred_stride}, "
              f"mamba_layers={args.mamba_layers}, d_state={args.d_state}, "
              f"mlp_scale={args.mlp_scale}, "
              f"mamba_version={args.mamba_version}, multi_scale={ms_str}, "
              f"bidirectional={args.bidirectional}")

    # ── Train ──
    diag = {}
    best_auc, best_ap = train_and_eval(
        model, normal_loader, anomaly_loader, test_loader,
        args, label_map, device, diag=diag)
    n_steps = max(1, int(diag.pop('steps', 0)))
    diag = {k: round(v / n_steps, 4) for k, v in diag.items()}

    # ── Save result ──
    result = {
        'exp_id': cli_args.id,
        'exp_name': exp_name,
        'config': {k: str(v) for k, v in cfg_dict.items()
                   if k not in ('name',)},
        'best_auc': round(float(best_auc), 4),
        'best_ap': round(float(best_ap), 4),
        'aux_diagnostics': diag,
    }
    rdir = results_dir()
    save_path = os.path.join(rdir, f'{exp_name}.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[Result] {exp_name}: AUC={best_auc:.4f}, AP={best_ap:.4f}")
    print(f"         saved to {save_path}")


if __name__ == '__main__':
    # Need F for the training loop
    import torch.nn.functional as F
    main()

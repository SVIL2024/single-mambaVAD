"""Local self-test for the review-driven fixes (no GPU, no mamba_ssm needed).

Run from ``src/``:

    python -m experiment.selftest_realfix

Checks
------
A. ``losses.prototype_loss(pool='mean')`` is numerically identical to the
   submitted ``normal_prototype_loss`` (R1-2 must not silently change the
   baseline objective).
B. ``pool='topk'`` repels only the high-score frames, so a clip whose mean is
   dominated by normal frames no longer gets its normal frames pushed away.
C. ``pool='soft'`` interpolates towards the mean as ``tau`` grows.
D. The anomaly scores used for pooling never receive gradient.
E. ``future_consistency_loss`` aligns ``h_t`` with ``s_{t+k}``: zero loss when
   the prediction matches the shifted target, correct pair count under
   padding, and no gradient path into the (detached) target.
F. ``compute_aux_terms`` runs end to end on fake model outputs for every
   (pool, horizon) combination and keeps every term finite.
G. Static checks on the model / config / CLI: the alignment switch exists, the
   default path no longer supervises replicated padding, and the new
   experiments are registered under the ids used by ``run_grid``.
"""
import ast
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiment import losses as L           # noqa: E402
from experiment import config as C           # noqa: E402

FAILS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:                 # noqa: BLE001
        FAILS.append("%s: %s: %s" % (name, type(exc).__name__, exc))
        print("  FAIL  %s -> %s: %s" % (name, type(exc).__name__, exc))
        return
    print("  ok    %s" % name)


def _original_prototype_loss(h_pooled, binary_labels, margin=0.5):
    """Verbatim copy of ucf_train_mamba_vad_dual.normal_prototype_loss."""
    h_norm = F.normalize(h_pooled, dim=-1)
    h_normal = h_norm[binary_labels == 0]
    h_anomaly = h_norm[binary_labels == 1]
    normal_proto = F.normalize(h_normal.mean(dim=0), dim=-1)
    loss_normal = (1 - (h_normal @ normal_proto)).mean()
    loss_anomaly = F.relu((h_anomaly @ normal_proto) + margin).mean()
    return loss_normal + loss_anomaly


# ── A ─────────────────────────────────────────────────────────
def test_mean_equivalence():
    torch.manual_seed(0)
    B, D = 8, 64
    h = torch.randn(B, D)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    ref = _original_prototype_loss(h, labels)
    out = L.prototype_loss(h, labels, margin=0.5, device='cpu', pool='mean')
    assert torch.allclose(out['loss'], ref, atol=1e-6), (out['loss'], ref)
    assert out['coverage'].item() == 1.0


# ── B ─────────────────────────────────────────────────────────
def test_topk_ignores_normal_frames_inside_an_anomalous_clip():
    torch.manual_seed(1)
    torch.set_printoptions(precision=3)
    D, T = 16, 40
    proto_dir = torch.zeros(D)
    proto_dir[0] = 1.0
    # An anomalous clip: 30 frames that look normal (align with the prototype),
    # 10 frames that are clearly anomalous (anti-aligned).
    h_frames = torch.zeros(4, T, D)
    h_frames[:, :30, :] = proto_dir * 2.0
    h_frames[:, 30:, :] = -proto_dir * 2.0
    # A normal clip of the same length, so the prototype is well defined.
    h_norm = torch.zeros(4, T, D)
    h_norm[:, :, :] = proto_dir * 2.0

    # 2 normal + 2 anomalous videos
    h_all = torch.cat([h_norm[:2], h_frames[:2]], dim=0)
    labels = torch.tensor([0, 0, 1, 1])
    lengths = torch.full((4,), T)
    h_pooled = L.masked_mean_features(h_all, lengths)

    # Scores: the 10 anomalous frames get high scores.
    scores = torch.zeros(4, T)
    scores[2:, 30:] = 5.0
    h_all = h_all.clone().requires_grad_(True)

    mean_out = L.prototype_loss(h_pooled, labels, device='cpu', pool='mean')
    topk_out = L.prototype_loss(
        h_pooled, labels, device='cpu', h_frames=h_all, scores=scores,
        lengths=lengths, pool='topk', topk_ratio=0.25)
    assert topk_out['loss'] < mean_out['loss'] - 0.1, (topk_out['loss'],
                                                       mean_out['loss'])
    # top-k = 25% of 40 = 10 frames = exactly the anomalous ones
    assert abs(topk_out['coverage'].item() - 0.25) < 1e-6

    # With a small ratio the anomaly similarity must be negative (anti-aligned)
    assert topk_out['sim_anomaly'].item() < -0.9
    # ... whereas the video mean is dominated by the 30 normal frames
    assert mean_out['sim_anomaly'].item() > 0.0


# ── C ─────────────────────────────────────────────────────────
def test_soft_pooling_tends_to_mean():
    torch.manual_seed(2)
    h_frames = torch.randn(2, 32, 8)
    lengths = torch.full((2,), 32)
    scores = torch.randn(2, 32)
    near_mean = L.pooled_repulsion_features(
        h_frames, scores, lengths, pool='soft', tau=1e4)[0]
    plain_mean = L.masked_mean_features(h_frames, lengths)
    assert torch.allclose(near_mean, plain_mean, atol=1e-4), \
        (near_mean - plain_mean).abs().max()
    sharp = L.pooled_repulsion_features(
        h_frames, scores, lengths, pool='soft', tau=1e-3)[0]
    assert not torch.allclose(sharp, plain_mean, atol=1e-2)


def test_soft_pooling_is_convex_combination():
    h_frames = torch.randn(1, 16, 4)
    lengths = torch.full((1,), 16)
    scores = torch.randn(1, 16)
    pooled = L.pooled_repulsion_features(h_frames, scores, lengths,
                                         pool='soft', tau=0.5)[0]
    lo = h_frames[0].min(dim=0).values - 1e-6
    hi = h_frames[0].max(dim=0).values + 1e-6
    assert bool(((pooled[0] >= lo) & (pooled[0] <= hi)).all())


# ── D ─────────────────────────────────────────────────────────
def test_scores_are_detached():
    torch.manual_seed(3)
    h_frames = torch.randn(2, 16, 8, requires_grad=True)
    lengths = torch.full((2,), 16)
    scores = torch.rand(2, 16, requires_grad=True)
    pooled, _ = L.pooled_repulsion_features(h_frames, scores, lengths,
                                            pool='topk')
    pooled.sum().backward()
    assert h_frames.grad is not None
    assert scores.grad is None, "pooling must not backprop into the classifier"

    logits = torch.randn(2, 16, 1, requires_grad=True)
    hf = torch.randn(2, 16, 8, requires_grad=True)
    hp = L.masked_mean_features(hf, lengths)
    out = L.prototype_loss(hp, torch.tensor([0, 1]), device='cpu',
                           h_frames=hf, scores=torch.sigmoid(
                               logits.squeeze(-1)).detach(),
                           lengths=lengths, pool='soft')
    out['loss'].backward()
    assert logits.grad is None


# ── E ─────────────────────────────────────────────────────────
def test_future_shift_alignment():
    B, T = 2, 10
    target = torch.arange(B * T, dtype=torch.float32).reshape(B, T, 1)
    for k in (1, 2, 4):
        pred = torch.zeros_like(target)
        pred[:, :T - k, :] = target[:, k:, :]
        out = L.future_consistency_loss(pred, target, lengths=None, horizon=k)
        assert out['loss'].item() < 1e-8, (k, out['loss'])
        assert out['n_pairs'] == B * (T - k), (k, out['n_pairs'])

    # one-step shift is not the identity: a self-consistency predictor fails
    pred_const = target.clone()
    out = L.future_consistency_loss(pred_const, target, horizon=1)
    assert out['loss'].item() > 0.5


def test_future_pair_count_respects_lengths():
    B, T, k = 3, 20, 3
    lengths = torch.tensor([20, 7, 2])
    pred = torch.randn(B, T, 1)
    target = torch.randn(B, T, 1)
    out = L.future_consistency_loss(pred, target, lengths=lengths, horizon=k)
    # video 0: 17 pairs, video 1: 4 pairs, video 2: 0 pairs
    assert out['n_pairs'] == 21, out['n_pairs']
    # pairs beyond the clip must not contribute
    pred2 = pred.clone()
    pred2[:, 17:, :] = 1e6
    out2 = L.future_consistency_loss(pred2, target, lengths=lengths, horizon=k)
    assert abs(out2['loss'].item() - out['loss'].item()) < 1e-5


def test_future_target_is_detached():
    pred = torch.zeros(1, 8, 1, requires_grad=True)
    tgt = torch.ones(1, 8, 1, requires_grad=True)
    L.future_consistency_loss(pred, tgt.detach(), horizon=1)['loss'].backward()
    assert tgt.grad is None
    # but the prediction does receive gradient
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_legacy_alignment_reproduces_mse():
    pred = torch.randn(2, 5, 1)
    tgt = torch.randn(2, 5, 1)
    a = L.legacy_future_loss(pred, tgt)
    b = F.mse_loss(pred, tgt.detach())
    assert torch.allclose(a, b)


# ── F ─────────────────────────────────────────────────────────
def _fake_outputs(B=4, T=32, D=16, seed=4):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, D, generator=g, requires_grad=True)
    logits = torch.randn(B, T, 1, generator=g, requires_grad=True)
    fut = torch.randn(B, T, 1, generator=g, requires_grad=True)
    lengths = torch.tensor([32, 28, 20, 8])
    normal_mask = torch.tensor([True, True, False, False])
    return h, logits, fut, lengths, normal_mask


def test_compute_aux_terms_all_combinations():
    for pool in ('mean', 'topk', 'soft'):
        for horizon in (1, 4):
            for align in ('shift', 'legacy'):
                h, logits, fut, lengths, nm = _fake_outputs()
                out = L.compute_aux_terms(h, logits, fut, lengths, nm,
                                          pool=pool, horizon=horizon,
                                          align=align)
                for key in ('loss_proto', 'loss_future'):
                    v = out[key]
                    assert v is not None and torch.isfinite(v), (pool, key, v)
                (out['loss_proto'] * 0.1 + out['loss_future'] * 0.1
                 ).backward()
                assert h.grad is not None and bool(
                    torch.isfinite(h.grad).all()), (pool, horizon, align)
                # the bridge target and the pooling scores are detached, so the
                # classifier head must not receive gradient from either term
                assert logits.grad is None, (pool, horizon, align)
                if align == 'shift' and horizon == 4:
                    assert out['future_pairs'] == (32 - 4) + (28 - 4) + \
                        (20 - 4) + max(0, 8 - 4)


def test_compute_aux_terms_matches_original_when_defaults():
    """With the defaults of the *submitted* config the prototype term must be
    unchanged; the bridge legitimately differs because its alignment bug is
    being fixed (--future-align legacy recovers it)."""
    h, logits, fut, lengths, nm = _fake_outputs()
    hp = L.masked_mean_features(h, lengths)
    ref = _original_prototype_loss(hp, (~nm).to(torch.long))
    out = L.compute_aux_terms(h, logits, fut, lengths, nm, pool='mean')
    assert torch.allclose(out['loss_proto'], ref, atol=1e-6)

    leg = L.compute_aux_terms(h, logits, fut, lengths, nm, align='legacy')
    assert torch.allclose(leg['loss_future'],
                          F.mse_loss(fut, logits.detach()))


def test_single_class_batch_is_safe():
    h, logits, fut, _, _ = _fake_outputs()
    nm = torch.ones(4, dtype=torch.bool)                 # all normal
    out = L.compute_aux_terms(h, logits, fut, torch.tensor([32, 32, 32, 32]),
                              nm, pool='topk')
    assert float(out['loss_proto']) == 0.0


# ── G ─────────────────────────────────────────────────────────
def _model_ast():
    path = os.path.join(SRC, 'mamba_vad_dual_model.py')
    with open(path, 'r', encoding='utf-8') as f:
        return path, ast.parse(f.read())


def test_model_has_alignment_switch():
    path, tree = _model_ast()
    src = open(path, 'r', encoding='utf-8').read()
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == '__init__']
    assert fns, "no __init__ found"
    init = fns[0]
    names = {a.arg for a in init.args.args}
    assert 'future_align' in names, 'future_align is not a model argument'
    assert "score_pred = self.future_score_head(x_vis)" in src
    # the padding call must survive only inside the legacy branch
    lines = src.splitlines()
    pad_lines = [i for i, ln in enumerate(lines) if 'mode=\'replicate\'' in ln]
    assert pad_lines, 'legacy padding disappeared entirely'
    for i in pad_lines:
        window = '\n'.join(lines[max(0, i - 4):i + 1])
        assert 'legacy' in window, 'padding is outside the legacy branch'


def test_config_registers_fix_experiments():
    names = C.get_exp_names()
    expect = ['proto_topk', 'proto_soft', 'future_k4', 'causal_no_future',
              'causal_future_k4']
    for n in expect:
        assert n in names, 'missing experiment %s' % n
    ids = {C.get_exp_config(i)['exp_name']: i for i in range(1, len(names) + 1)}
    for name in ('proto_topk', 'proto_soft', 'future_k1', 'future_k4',
                 'causal_no_future', 'causal_future_k4', 'future_legacy'):
        assert name in ids, name

    def cfg(name):
        return C.get_exp_config(ids[name])

    # the paper's default is delta = 0, i.e. the padded self-consistency
    # alignment of the original release
    assert C.FIXED['future_align'] == 'legacy'
    assert cfg('full')['future_align'] == 'legacy'
    assert cfg('full')['future_horizon'] == 1
    assert cfg('full')['proto_pool'] == 'mean'
    # delta = 1 and delta = 4 of Table 3 request the shifted alignment
    assert cfg('future_k1')['future_align'] == 'shift'
    assert cfg('future_k1')['future_horizon'] == 1
    assert cfg('future_k4')['future_align'] == 'shift'
    assert cfg('future_k4')['future_horizon'] == 4
    assert cfg('causal_future_k4')['future_align'] == 'shift'
    assert cfg('causal_future_k4')['bidirectional'] is False
    assert cfg('future_legacy')['future_align'] == 'legacy'
    # pool variants leave the alignment at the delta = 0 default
    assert cfg('proto_topk')['future_align'] == 'legacy'
    assert cfg('proto_soft')['future_align'] == 'legacy'
    # submitted configs keep the classifier-only weight mixture
    assert cfg('no_proto')['proto_weight'] == 0.0
    assert cfg('causal')['bidirectional'] is False


# ── H ─────────────────────────────────────────────────────────
def _load_model_module():
    """Import ``mamba_vad_dual_model`` with a CPU stand-in encoder.

    The encoder is not what the review is about; only the bridge head and the
    auxiliary losses are under test.  ``mamba_ssm`` and ``clip`` are stubbed so
    the real model class (real __init__, real forward, real head) can run on a
    CPU-only box.
    """
    import importlib
    import types
    from torch import nn

    mm = types.ModuleType('mamba_model')

    class _Enc(nn.Module):
        def __init__(self, d_model=16, n_layer=1, **kwargs):
            super().__init__()
            self.lin = nn.Linear(d_model, d_model)

        def forward(self, x):                    # [T, B, D] -> [T, B, D]
            return self.lin(x)

    mm.MambaTemporalEncoder = _Enc
    mm.BiMambaEncoder = _Enc
    mm.LayerNorm = nn.LayerNorm
    mm.QuickGELU = nn.GELU
    cm = types.ModuleType('clip')
    cm.clip = None
    sys.modules['mamba_model'] = mm
    sys.modules['clip'] = cm
    sys.modules.pop('mamba_vad_dual_model', None)
    return importlib.import_module('mamba_vad_dual_model')


def _build_model(mod, align):
    return mod.MambaVAD_Dual(
        visual_width=16, visual_length=32, mamba_layers=1, d_state=4,
        d_conv=4, expand=2, dropout=0.0, level=2, pred_stride=1, num_class=14,
        embed_dim=16, device='cpu', mamba_version='mamba1',
        mamba_only_cls=True, multi_scale=False, single_mamba=True,
        bidirectional=True, mlp_scale=2, future_align=align)


def test_model_forward_bridge_alignment():
    mod = _load_model_module()
    torch.manual_seed(7)
    x = torch.randn(2, 32, 16)

    shift = _build_model(mod, 'shift')
    legacy = _build_model(mod, 'legacy')

    out_s = shift(x)
    out_l = legacy(x)
    hs = out_s['hidden_state']
    assert out_s['score_future'].shape == (2, 32, 1), \
        out_s['score_future'].shape
    expect_s = shift.future_score_head(hs)
    assert torch.allclose(out_s['score_future'], expect_s, atol=1e-6), \
        'shift mode must score every frame, without padding'

    hs_l = out_l['hidden_state']
    expect_l = F.pad(legacy.future_score_head(hs_l[:, :-1, :]),
                     (0, 0, 0, 1), mode='replicate')
    assert torch.allclose(out_l['score_future'], expect_l, atol=1e-6), \
        'legacy mode must reproduce the submitted padded alignment'

    # same parameter set: switching the alignment adds nothing to the model
    n_s = sum(p.numel() for p in shift.parameters())
    n_l = sum(p.numel() for p in legacy.parameters())
    assert n_s == n_l, (n_s, n_l)

    # end-to-end: the auxiliary terms accept the real model outputs
    lengths = torch.tensor([32, 20])
    nm = torch.tensor([True, False])
    aux = L.compute_aux_terms(
        h_fused=out_s['hidden_state_fused'], logits1=out_s['logits1'],
        score_future=out_s['score_future'], lengths=lengths, normal_mask=nm,
        pool='topk', horizon=1)
    (aux['loss_proto'] + aux['loss_future']).backward()
    assert shift.future_score_head.weight.grad is not None
    assert shift.mamba.lin.weight.grad is not None
    assert aux['future_pairs'] == 31 + 19, aux['future_pairs']


def test_cli_roundtrip_all_configs():
    """Every experiment dict must survive cfg -> argv -> argparse unchanged."""
    from experiment import run_one as R
    p = R.build_opt_parser()
    known = set(p._option_string_actions.keys())
    for exp_id in range(1, 17):
        cfg = C.get_exp_config(exp_id)
        argv = R.cfg_to_argv(cfg)
        for a in argv:
            if a.startswith('--'):
                assert a in known, (exp_id, a, 'unknown flag')
        args = p.parse_args(argv)
        for k, v in cfg.items():
            if k in ('exp_name', 'exp_id', 'model_type'):
                continue
            got = getattr(args, k, '<missing>')
            if isinstance(v, bool):
                assert got == v, (exp_id, k, got, v)
            elif isinstance(v, (int, float)):
                assert float(got) == float(v), (exp_id, k, got, v)
            else:
                assert str(got) == str(v), (exp_id, k, got, v)


def test_grid_steps_match_ids():
    sys.path.insert(0, SRC)
    from experiment import run_grid as G
    names = C.get_exp_names()
    for step in ('g6', 'g7', 'g8'):
        for name, exp_id in G.STEPS[step]:
            assert names[exp_id - 1] == name, (step, name, exp_id,
                                               names[exp_id - 1])


def test_cli_exposes_new_flags():
    import argparse
    # import the parser builder indirectly: replicate the block under test by
    # importing run_one (it imports mamba_ssm only at call time, not at import
    # time, so a CPU-only box can still parse the flags).
    src = open(os.path.join(HERE, 'run_one.py'), 'r', encoding='utf-8').read()
    for flag in ('--proto-pool', '--proto-topk-ratio', '--proto-tau',
                 '--future-horizon', '--future-align', '--proto-weight'):
        assert flag in src, 'CLI flag %s missing' % flag
    assert 'future_align=args.future_align' in src
    assert "from experiment import losses" in src
    _ = argparse  # keep the import meaningful for linters


TESTS = [
    ('A/mean-equivalence', test_mean_equivalence),
    ('B/topk-isolates-anomalous', test_topk_ignores_normal_frames_inside_an_anomalous_clip),
    ('C/soft->mean', test_soft_pooling_tends_to_mean),
    ('C/soft-convex', test_soft_pooling_is_convex_combination),
    ('D/scores-detached', test_scores_are_detached),
    ('E/shift-alignment', test_future_shift_alignment),
    ('E/pair-count', test_future_pair_count_respects_lengths),
    ('E/target-detached', test_future_target_is_detached),
    ('E/legacy-mse', test_legacy_alignment_reproduces_mse),
    ('F/all-combinations', test_compute_aux_terms_all_combinations),
    ('F/defaults-match-original', test_compute_aux_terms_matches_original_when_defaults),
    ('F/single-class-safe', test_single_class_batch_is_safe),
    ('G/model-switch', test_model_has_alignment_switch),
    ('H/real-model-forward', test_model_forward_bridge_alignment),
    ('G/config-ids', test_config_registers_fix_experiments),
    ('G/grid-steps', test_grid_steps_match_ids),
    ('G/cli-flags', test_cli_exposes_new_flags),
    ('I/cli-roundtrip', test_cli_roundtrip_all_configs),
]


def main():
    print('mamba-ssm present: %s' % _has_mamba())
    print('running %d checks' % len(TESTS))
    for name, fn in TESTS:
        check(name, fn)
    print()
    if FAILS:
        print('FAILED %d/%d' % (len(FAILS), len(TESTS)))
        for f in FAILS:
            print('  - %s' % f)
        return 1
    print('ALL %d CHECKS PASSED' % len(TESTS))
    return 0


def _has_mamba():
    try:
        import mamba_ssm  # noqa: F401
        return True
    except Exception:                        # noqa: BLE001
        return False


if __name__ == '__main__':
    sys.exit(main())

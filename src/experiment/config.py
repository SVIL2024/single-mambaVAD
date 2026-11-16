"""
Experiment configurations for single-Mamba study (UCF-Crime).

Two series:
  A — Multi-scale × bidirectional interaction (exp 1-7)
  B — Loss ablation (exp 8-10)
"""
import os

# ─── Fixed base parameters (applied to ALL experiments) ───
FIXED = dict(
    seed=234,
    max_epoch=5,
    level=2,
    single_mamba=True,
    mamba_only_cls=True,
    mamba_version='mamba2',
    mamba2_headdim=64,
    mamba2_ngroups=1,
    batch_size=64,
    dropout=0.3,
    weight_decay=1e-3,
    lr=2e-5,
    mamba_lr_scale=0.5,
    label_smooth=0.1,
    pred_stride=2,
    d_conv=4,
    expand=2,
    # Default Mamba arch
    d_state=8,
    mamba_layers=2,
    mlp_scale=2,
    bidirectional=True,
    multi_scale=True,
    multi_scale_factors='1,8,16',
    # Loss weights
    proto_weight=0.1,
    future_weight=0.1,
    # Review-driven auxiliary-term settings (see experiment/losses.py).
    # Defaults reproduce the submitted behaviour exactly: the consistency
    # target is the frame's own score (delta = 0) computed the way the
    # originally released pipeline computed it, i.e. --future-align legacy.
    proto_pool='mean',
    proto_topk_ratio=0.1,
    proto_tau=0.1,
    future_horizon=1,
    future_align='legacy',
)

EXPERIMENTS = {}

# ════════════════════════════════════════════════
# Series A — Multi-scale × bidirectional (7 experiments)
# ════════════════════════════════════════════════
SERIES_A = [
    ('full',              dict(bidirectional=True,  multi_scale=True,  multi_scale_factors='1,8,16')),
    ('causal',            dict(bidirectional=False, multi_scale=True,  multi_scale_factors='1,8,16')),
    ('no_ms',             dict(bidirectional=True,  multi_scale=False)),
    ('ms_fine',           dict(bidirectional=True,  multi_scale=True,  multi_scale_factors='1,4,8')),
    ('ms_coarse',         dict(bidirectional=True,  multi_scale=True,  multi_scale_factors='1,16,32')),
    ('ms_fine_causal',    dict(bidirectional=False, multi_scale=True,  multi_scale_factors='1,4,8')),
    ('ms_coarse_causal',  dict(bidirectional=False, multi_scale=True,  multi_scale_factors='1,16,32')),
]

for name, overrides in SERIES_A:
    EXPERIMENTS[name] = overrides

# ════════════════════════════════════════════════
# Series B — Loss ablation (3 experiments)
# ════════════════════════════════════════════════
SERIES_B = [
    ('no_proto',    dict(proto_weight=0.0,  future_weight=0.1)),
    ('no_future',   dict(proto_weight=0.1,  future_weight=0.0)),
    ('cls_only',    dict(proto_weight=0.0,  future_weight=0.0)),
]

for name, overrides in SERIES_B:
    EXPERIMENTS[name] = overrides

# ════════════════════════════════════════════════
# Series C — review-driven fixes (KBS R1-2, R1-3a)
#   proto_topk / proto_soft : repulsion pooled over high-score frames instead
#       of the video mean, so normal frames inside an anomalous clip are not
#       pushed away from the prototype.
#   future_k4 / causal_future_k4 : the bridge must predict s_{t+4} from h_t.
#       With a causal encoder this is a genuine forecast, which is what lets
#       the paper claim a predictive (rather than consistency) role.
#   causal_no_future : completes the 2x2 {bidi, causal} x {bridge, no bridge}
#       design used to test the leakage argument (R1-3a).
# ════════════════════════════════════════════════
SERIES_C = [
    ('proto_topk',        dict(proto_pool='topk')),
    ('proto_soft',        dict(proto_pool='soft')),
    # delta = 1 and delta = 4 must ask for the shifted alignment explicitly:
    # the default is delta = 0 (legacy), so an inherited horizon would be
    # silently ignored by the legacy branch.
    ('future_k1',         dict(future_align='shift', future_horizon=1)),
    ('future_k4',         dict(future_align='shift', future_horizon=4)),
    ('causal_no_future',  dict(bidirectional=False, future_weight=0.0)),
    ('causal_future_k4',  dict(bidirectional=False, future_align='shift',
                               future_horizon=4)),
    # control: the submitted objective (head(h_t) vs s_t, replicate-padded).
    # Only used to quantify what the alignment bug was worth at equal budget.
    ('future_legacy',     dict(future_align='legacy')),
]

for name, overrides in SERIES_C:
    EXPERIMENTS[name] = overrides


def get_exp_config(exp_id):
    """Return full config dict for experiment *exp_id* (int, 1-based)."""
    keys = list(EXPERIMENTS.keys())
    if exp_id < 1 or exp_id > len(keys):
        raise ValueError(f"exp_id must be 1–{len(keys)}, got {exp_id}")
    name = keys[exp_id - 1]
    overrides = EXPERIMENTS[name]
    cfg = dict(FIXED)
    cfg.update(overrides)
    cfg['exp_name'] = name
    cfg['exp_id'] = exp_id
    return cfg


def get_exp_names():
    return list(EXPERIMENTS.keys())


def results_dir():
    d = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(d, exist_ok=True)
    return d

"""Grid runner: run the ablation configs over multiple seeds / epoch budgets.

Why this exists
---------------
``experiment/run_one.py`` hard-codes the budget in ``experiment/config.py``
(5 epochs, batch 64, lr 2e-5, dropout 0.3, seed 234) and always writes
``results/<exp_name>.json``, so a 3-seed run at 10 epochs would silently
overwrite itself.  This wrapper patches the shared config in place, calls
``run_one.main()`` once per (config, seed), then renames each result file to
``<exp_name>_e<E>_s<S>.json`` and prints a mean+-std summary.

Usage (from ``src/``)
---------------------
    python -m experiment.run_grid --step g1                  # full,  seeds 1 2 3
    python -m experiment.run_grid --step g2                  # causal, seeds 1 2 3
    python -m experiment.run_grid --step g3 --seeds 1        # multi-scale sweep
    python -m experiment.run_grid --step g4 --seeds 1        # loss sweep
    python -m experiment.run_grid --step sanity              # 2 short runs, no 10ep
"""
import argparse
import csv
import json
import os
import shutil
import statistics
import sys

from experiment import config as C

# step -> [(exp_name, exp_id)]  (ids follow experiment/run_one.py --id)
STEPS = {
    'g1': [('full', 1)],
    'g2': [('causal', 2)],
    'g3': [('no_ms', 3), ('ms_fine', 4), ('ms_coarse', 5)],
    'g4': [('no_proto', 8), ('no_future', 9), ('cls_only', 10)],
    'g5': [('ms_fine_causal', 6), ('ms_coarse_causal', 7)],
    # review-driven fixes (ids follow EXPERIMENTS insertion order; Series C
    # starts at 11 because series A has 7 and series B has 3 entries)
    'g6': [('proto_topk', 11), ('proto_soft', 12)],
    'g7': [('future_k1', 13), ('future_k4', 14), ('causal_no_future', 15),
           ('causal_future_k4', 16)],
    # g8 -- alignment control: submitted (legacy) bridge at the same budget
    'g8': [('future_legacy', 17)],
}


def patch_budget(epochs=None, batch=None, lr=None, dropout=None, seed=None,
                 pred_stride=None, weight_decay=None):
    if epochs is not None:
        C.FIXED['max_epoch'] = int(epochs)
    if batch is not None:
        C.FIXED['batch_size'] = int(batch)
    if lr is not None:
        C.FIXED['lr'] = float(lr)
    if dropout is not None:
        C.FIXED['dropout'] = float(dropout)
    if seed is not None:
        C.FIXED['seed'] = int(seed)
    if pred_stride is not None:
        C.FIXED['pred_stride'] = int(pred_stride)
    if weight_decay is not None:
        C.FIXED['weight_decay'] = float(weight_decay)


def run_one(exp_id):
    """Call run_one.main() with a synthetic argv, then return the written path."""
    import experiment.run_one as R
    old_argv = sys.argv
    sys.argv = ['run_one', '--id', str(exp_id)]
    try:
        R.main()
    finally:
        sys.argv = old_argv


def summarize(rows, step):
    out = os.path.join(C.results_dir(), f'grid_{step}_summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['exp_name', 'n', 'auc_mean', 'auc_std', 'ap_mean', 'ap_std'])
        for name, aucs, aps in rows:
            w.writerow([
                name, len(aucs),
                f'{statistics.mean(aucs):.4f}',
                f'{statistics.pstdev(aucs):.4f}',
                f'{statistics.mean(aps):.4f}',
                f'{statistics.pstdev(aps):.4f}',
            ])
    print(f'\n[grid] summary -> {out}')
    for name, aucs, aps in rows:
        print(f'  {name:<14} n={len(aucs)}  AUC={statistics.mean(aucs):.4f}'
              f' +- {statistics.pstdev(aucs):.4f}   AP={statistics.mean(aps):.4f}'
              f' +- {statistics.pstdev(aps):.4f}')


def model_dir():
    """Where run_one() writes checkpoints; overridable so runs can be sharded."""
    rel = os.environ.get('VAD_MODEL_DIR', 'model')
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
    os.makedirs(d, exist_ok=True)
    return d


def snapshot_model_dir():
    d = model_dir()
    return d, {f for f in os.listdir(d)} if os.path.isdir(d) else set()


def archive_ckpt(before, name, epochs, seed):
    """Keep the checkpoint of this run so later probes can point at it.

    run_one() names checkpoints after the best score, so two seeds can collide;
    copying to ``grid_<name>_e<E>_s<S>.pt`` makes the provenance explicit.
    """
    d, after = snapshot_model_dir()
    new = [f for f in after - before if f.endswith('.pt') and 'final' not in f]
    if not new:
        return None
    best = max((os.path.join(d, f) for f in new), key=os.path.getsize)
    dst = os.path.join(d, f'grid_{name}_e{epochs}_s{seed}.pt')
    shutil.copyfile(best, dst)
    return dst


def main():
    p = argparse.ArgumentParser(description='Grid runner for the ablation configs')
    p.add_argument('--step', required=True, choices=sorted(STEPS) + ['sanity'])
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--pred-stride', type=int, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    a = p.parse_args()

    if a.step == 'sanity':
        # 1 epoch, single seed: verifies the seed flag is wired end to end
        patch_budget(epochs=1, seed=a.seeds[0])
        run_one(1)
        patch_budget(epochs=1, seed=a.seeds[-1])
        run_one(1)
        print('[grid] sanity done: compare the two result files by seed')
        return

    rdir = C.results_dir()
    rows = []
    for name, exp_id in STEPS[a.step]:
        aucs, aps = [], []
        for seed in a.seeds:
            patch_budget(epochs=a.epochs, batch=a.batch, lr=a.lr,
                         dropout=a.dropout, seed=seed,
                         pred_stride=a.pred_stride,
                         weight_decay=a.weight_decay)
            before = snapshot_model_dir()[1]
            run_one(exp_id)
            ck = archive_ckpt(before, name, a.epochs, seed)
            if ck:
                print(f'[grid] ckpt -> {os.path.basename(ck)}')
            src = os.path.join(rdir, f'{name}.json')
            dst = os.path.join(rdir, f'{name}_e{a.epochs}_s{seed}.json')
            with open(src) as f:
                rec = json.load(f)
            rec['seed'] = seed
            rec['epochs'] = a.epochs
            with open(dst, 'w') as f:
                json.dump(rec, f, indent=2)
            os.remove(src)
            aucs.append(rec['best_auc'])
            aps.append(rec['best_ap'])
            print(f'[grid] {name} seed={seed}: AUC={rec["best_auc"]:.4f} '
                  f'AP={rec["best_ap"]:.4f}')
        rows.append((name, aucs, aps))
    summarize(rows, a.step)


if __name__ == '__main__':
    main()

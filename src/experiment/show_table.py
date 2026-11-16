"""
Read saved experiment results and print tables grouped by series.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiment.config import get_exp_names, results_dir


def load_results():
    rdir = results_dir()
    results = {}
    for fname in os.listdir(rdir):
        if fname.endswith('.json'):
            with open(os.path.join(rdir, fname)) as f:
                data = json.load(f)
            results[data['exp_name']] = data
    return results


def fmt_bool(val):
    if val is None:
        return '—'
    if isinstance(val, bool):
        return '✓' if val else '✗'
    return '✓' if str(val) == 'True' else '✗'


def main():
    results = load_results()

    # ── Series A: Multi-scale × bidirectional ──
    series_a_names = ['full', 'causal', 'no_ms',
                      'ms_fine', 'ms_coarse',
                      'ms_fine_causal', 'ms_coarse_causal']

    sep_a = ("|" + "-"*18 + "|" + "-"*4 + "|" + "-"*4 + "|"
             + "-"*12 + "|" + "-"*8 + "|" + "-"*8 + "|")
    print("\nSeries A — Multi-scale × bidirectional interaction")
    print(sep_a)
    print("| Method           | bi | ms | ms_factors  | AUC(%)  | AP(%)   |")
    print(sep_a)
    for name in series_a_names:
        r = results.get(name)
        if r is None:
            print(f"| {name:<16} | — | — | —           | ——      | ——      |")
            continue
        cfg = r.get('config', {})
        bi = fmt_bool(cfg.get('bidirectional'))
        ms = fmt_bool(cfg.get('multi_scale'))
        mf = str(cfg.get('multi_scale_factors', '—'))
        auc = r.get('best_auc', '—')
        ap = r.get('best_ap', '—')
        auc_s = f"{auc*100:.2f}" if isinstance(auc, (int, float)) else str(auc)
        ap_s = f"{ap*100:.2f}" if isinstance(ap, (int, float)) else str(ap)
        print(f"| {name:<16} | {bi:>2} | {ms:>2} | {mf:<12} | {auc_s:>6} | {ap_s:>6} |")
    print(sep_a)

    # ── Series B: Loss ablation ──
    series_b_names = ['no_proto', 'no_future', 'cls_only']

    sep_b = ("|" + "-"*18 + "|" + "-"*14 + "|" + "-"*14 + "|"
             + "-"*8 + "|" + "-"*8 + "|")
    print("\nSeries B — Loss ablation (full config)")
    print(sep_b)
    print("| Method           | proto_weight  | future_weight | AUC(%)  | AP(%)   |")
    print(sep_b)
    for name in series_b_names:
        r = results.get(name)
        if r is None:
            print(f"| {name:<16} | —             | —             | ——      | ——      |")
            continue
        cfg = r.get('config', {})
        pw = str(cfg.get('proto_weight', '—'))
        fw = str(cfg.get('future_weight', '—'))
        auc = r.get('best_auc', '—')
        ap = r.get('best_ap', '—')
        auc_s = f"{auc*100:.2f}" if isinstance(auc, (int, float)) else str(auc)
        ap_s = f"{ap*100:.2f}" if isinstance(ap, (int, float)) else str(ap)
        print(f"| {name:<16} | {pw:<14} | {fw:<14} | {auc_s:>6} | {ap_s:>6} |")
    print(sep_b)

    print("\nLegend: ✓=enabled, ✗=disabled, —=not applicable")


if __name__ == '__main__':
    main()

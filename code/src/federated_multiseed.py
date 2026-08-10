"""
Gap 1: seed variance for the federated experiment.

federated.py reports a single run at seed=42. The subgroup gap changes it
measures are small (0.004 to 0.026), so a point estimate cannot distinguish
"the intervention does nothing" from "the intervention does something we
did not resolve".

This runs the whole simulation across several seeds and reports the paired
difference per seed: gap(intervention) - gap(none), computed within each seed
before averaging. Pairing matters because the baseline itself varies with the
client partition, which is also seed-dependent.

The claim this supports is a null: no intervention produces an effect large
or consistent enough to matter against the gap's own magnitude.
"""

import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from federated import run_condition, induce_imbalance, TASKS

COND = ['none', 'fedar', 'demo_bal']
COND_LABEL = {'none': 'no intervention',
              'fedar': 'FedAR resampling',
              'demo_bal': 'demographic balancing'}


def prepare_data(d, task, seed, skew, n_classes):
    """
    Standardise on train statistics, then optionally skew.

    Order matters and matches federated.py: standardisation is computed on the
    full train split and is therefore seed-independent, while the skew
    subsample is drawn per seed. That means the skew-selection variance is
    captured in the reported spread, which is what we want.
    """
    mu, split = d['mu'], d['split']
    tr, te = split == 'train', split == 'test'

    m, s = mu[tr].mean(0), mu[tr].std(0) + 1e-8
    data = {
        'train': ((mu[tr] - m) / s, d[task][tr], d['race'][tr], d['gender'][tr]),
        'test':  ((mu[te] - m) / s, d[task][te], d['race'][te], d['gender'][te]),
    }

    if skew:
        rng = np.random.default_rng(seed)
        fracs = [float(f) for f in skew.split(',')]
        assert len(fracs) == n_classes, \
            f"--skew needs {n_classes} fractions for task '{task}'"
        data['train'] = induce_imbalance(*data['train'], fracs, rng)

    return data


def paired_stats(deltas):
    """Mean, std, and sign consistency of a paired difference across seeds."""
    a = np.asarray(deltas, dtype=float)
    out = {
        'mean': float(a.mean()),
        'std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        'min': float(a.min()),
        'max': float(a.max()),
        'n_positive': int((a > 0).sum()),
        'n_negative': int((a < 0).sum()),
        'values': a.tolist(),
    }
    # A paired t-test against zero. With five seeds this is weak, but it is the
    # honest summary of "could this be zero?".
    try:
        from scipy import stats
        if len(a) > 1 and a.std(ddof=1) > 0:
            t, p = stats.ttest_1samp(a, 0.0)
            out['t'] = float(t)
            out['p'] = float(p)
    except ImportError:
        pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latents', default='../../latents/utkface_latents.npz')
    p.add_argument('--output', default='../../results/federated_multiseed.json')
    p.add_argument('--seeds', default='42,1,7,13,99')
    p.add_argument('--task', choices=['age_bucket', 'emotion'], default=None)
    p.add_argument('--n_clients', type=int, default=10)
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--local_epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--imr_threshold', type=float, default=1.8)
    p.add_argument('--eval_every', type=int, default=1000)   # silence per-round logs
    p.add_argument('--skew', default=None)
    args = p.parse_args()

    device = torch.device('cpu')
    seeds = [int(s) for s in args.seeds.split(',')]

    d = np.load(args.latents, allow_pickle=True)
    task = args.task or ('emotion' if 'emotion' in d.files else 'age_bucket')
    n_classes = TASKS[task]['n_classes']
    dataset = str(d['dataset']) if 'dataset' in d.files else 'unknown'

    print(f"Dataset: {dataset}  |  task: {task} ({n_classes} classes)")
    print(f"Seeds: {seeds}")
    if args.skew:
        print(f"Skew: {args.skew}")
    print()

    per_seed = {}
    for seed in seeds:
        print(f"{'='*70}\nSEED {seed}\n{'='*70}", flush=True)
        run_args = SimpleNamespace(**vars(args), seed=seed)
        data = prepare_data(d, task, seed, args.skew, n_classes)

        per_seed[seed] = {}
        for cond in COND:
            res = run_condition(cond, data, run_args, device, n_classes)
            f = res['final']
            worst_name = min(f['groups'], key=lambda k: f['groups'][k]['acc'])
            per_seed[seed][cond] = {
                'overall_acc': f['overall_acc'],
                'balanced_acc': f['balanced_acc'],
                'worst_group': f['worst_group'],
                'best_group': f['best_group'],
                'gap': f['gap'],
                'worst_group_name': worst_name,
            }
            print(f"  {cond:<12} bal {f['balanced_acc']:.4f}  "
                  f"gap {f['gap']:.4f}  worst {f['worst_group']:.4f} "
                  f"({worst_name})", flush=True)
        print()

    # ---- absolute values, mean over seeds ----
    print("=" * 78)
    print(f"ABSOLUTE  (mean +/- std over {len(seeds)} seeds)")
    print("-" * 78)
    print(f"{'condition':<22}{'balanced acc':>20}{'subgroup gap':>20}"
          f"{'worst group':>16}")
    for cond in COND:
        row = {k: np.array([per_seed[s][cond][k] for s in seeds])
               for k in ('balanced_acc', 'gap', 'worst_group')}
        print(f"{COND_LABEL[cond]:<22}"
              f"{row['balanced_acc'].mean():>12.4f} ±{row['balanced_acc'].std(ddof=1):<7.4f}"
              f"{row['gap'].mean():>12.4f} ±{row['gap'].std(ddof=1):<7.4f}"
              f"{row['worst_group'].mean():>10.4f}")

    # ---- paired deltas, the number the claim rests on ----
    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCE vs no intervention  (computed within each seed)")
    print("-" * 78)

    deltas = {}
    for cond in COND[1:]:
        d_gap = [per_seed[s][cond]['gap'] - per_seed[s]['none']['gap']
                 for s in seeds]
        d_bal = [per_seed[s][cond]['balanced_acc']
                 - per_seed[s]['none']['balanced_acc'] for s in seeds]
        deltas[cond] = {'gap': paired_stats(d_gap), 'balanced_acc': paired_stats(d_bal)}

        g, b = deltas[cond]['gap'], deltas[cond]['balanced_acc']
        print(f"\n  {COND_LABEL[cond]}")
        print(f"    gap        {g['mean']:+.4f} ± {g['std']:.4f}   "
              f"range [{g['min']:+.4f}, {g['max']:+.4f}]   "
              f"sign: {g['n_positive']}+ / {g['n_negative']}-"
              + (f"   p={g['p']:.3f}" if 'p' in g else ""))
        print(f"    balanced   {b['mean']:+.4f} ± {b['std']:.4f}   "
              f"range [{b['min']:+.4f}, {b['max']:+.4f}]"
              + (f"   p={b['p']:.3f}" if 'p' in b else ""))
        print(f"    per-seed gap deltas: "
              + "  ".join(f"{v:+.4f}" for v in g['values']))

    # ---- the comparison that decides the claim ----
    base_gap = np.array([per_seed[s]['none']['gap'] for s in seeds])
    print("\n" + "-" * 78)
    print(f"  Gap itself: {base_gap.mean():.4f} ± {base_gap.std(ddof=1):.4f}")
    print(f"  Largest intervention effect: "
          f"{max(abs(deltas[c]['gap']['mean']) for c in deltas):.4f}")
    ratio = max(abs(deltas[c]['gap']['mean']) for c in deltas) / base_gap.mean()
    print(f"  That is {100*ratio:.1f}% of the gap it would need to close.")
    print("=" * 78 + "\n")

    # ---- worst-group stability ----
    print("Worst group per condition, across seeds")
    for cond in COND:
        names = [per_seed[s][cond]['worst_group_name'] for s in seeds]
        from collections import Counter
        c = Counter(names)
        summary = ", ".join(f"{k} ({v}/{len(seeds)})" for k, v in c.most_common())
        print(f"  {COND_LABEL[cond]:<22} {summary}")
    print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump({'args': vars(args), 'task': task, 'dataset': dataset,
                   'seeds': seeds, 'per_seed': per_seed, 'deltas': deltas},
                  fh, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
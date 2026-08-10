"""
Experiment D figure: the adversarial removal trade-off.

Two panels:
  left  - removal (%) against lambda, one line per demographic attribute
  right - removal (%) against utility cost

On RAF-DB the utility axis is *task accuracy* (emotion probe), which is a far
more legible cost than reconstruction loss: it is the thing the model exists
to do. On UTKFace, where there is no task label, reconstruction loss is used.

The reference point is lambda=1, not the original checkpoint. Every debias run
adds 20 epochs of ordinary training on top of the pretrained VAE, which changes
reconstruction on its own. Anchoring to lambda=1 isolates the cost attributable
to adversarial pressure rather than to the extra epochs.
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CB = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
      'red': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999'}

ATTR_STYLE = {
    'gender':     (CB['blue'],   'o', 'Gender'),
    'race':       (CB['orange'], 's', 'Race'),
    'age_bucket': (CB['green'],  '^', 'Age bucket'),
}

DATASET_TITLE = {'utkface': 'UTKFace', 'rafdb': 'RAF-DB'}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_runs(paths_by_lambda, metric='LogReg'):
    """
    Read each debias result file and pull out removal, reconstruction, and
    (where present) the emotion probe used as a utility measure.

    -------------------------------------------------------------------------
    WHY metric='LogReg' (metric-consistency patch)
    -------------------------------------------------------------------------
    The stored 'pct_removed' field was computed inside debias.py from the MLP
    probe. MLPClassifier(early_stopping=True) draws an internal validation
    split from its random_state, so its balanced accuracy is not
    bit-reproducible across call sites (probes.py for fig1 vs debias.py for
    the baseline). The linear LogReg probe is deterministic and matches fig1
    to four decimals, so we recompute removal here from the LogReg before/after
    values that debias.py already saved. fig1, fig5 and fig6 are then locked to
    the same reproducible metric. Pass metric='MLP' to recover the old
    behaviour.
    -------------------------------------------------------------------------
    """
    runs = []
    for lam, path in sorted(paths_by_lambda.items()):
        if not os.path.exists(path):
            print(f"  missing: {path}, skipping lambda={lam}")
            continue
        with open(path) as f:
            d = json.load(f)

        def pct(attr):
            before = d['before'][attr][metric]
            after = d['after'][attr][metric]
            chance = d['before'][attr]['chance']
            return ((before - after) / (before - chance) * 100
                    if before > chance else 0.0)

        run = {
            'lambda': lam,
            'recon': d['history'][-1]['recon'],
            'kl': d['history'][-1]['kl'],
            'metric': metric,
            'dataset': d.get('dataset', 'unknown'),
            'removed': {a: pct(a) for a in ATTR_STYLE},
            'after_probe': {a: d['after'][a][metric] for a in ATTR_STYLE},
            'before_probe': {a: d['before'][a][metric] for a in ATTR_STYLE},
            'chance': {a: d['before'][a]['chance'] for a in ATTR_STYLE},
        }
        if 'emotion' in d['before']:
            run['task_before'] = d['before']['emotion'][metric]
            run['task_after'] = d['after']['emotion'][metric]
            run['task_chance'] = d['before']['emotion']['chance']
            run['task_removed'] = pct('emotion')
        runs.append(run)
    return runs


def make_figure(runs, out_path, ref_recon, ref_label, dataset):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    lams = [r['lambda'] for r in runs]
    has_task = 'task_after' in runs[0]

    # ---- left: removal vs lambda, full scale to show how far from complete ----
    all_vals = [v for r in runs for v in r['removed'].values()]
    band_lo = min(0, np.floor(min(all_vals) / 5) * 5)
    band_top = max(20, np.ceil(max(all_vals) / 5) * 5)

    ax1.axhspan(band_lo, band_top, color=CB['grey'], alpha=0.08, zorder=0)
    if band_lo < 0:
        # negative removal means the attribute became MORE recoverable
        ax1.axhline(0, color=CB['grey'], linewidth=1.2, zorder=1)
        ax1.text(lams[0], band_lo + 1.0, 'below 0: more recoverable than before',
                 fontsize=9, color=CB['red'], va='bottom')

    for attr, (color, marker, label) in ATTR_STYLE.items():
        vals = [r['removed'][attr] for r in runs]
        ax1.plot(lams, vals, marker=marker, color=color, label=label,
                 linewidth=2, markersize=8, zorder=3)

    ax1.set_xscale('log')
    ax1.set_xticks(lams)
    ax1.set_xticklabels([str(int(l)) for l in lams])
    ax1.set_xlabel('adversarial strength  $\\lambda$')
    ax1.set_ylabel('above-chance signal removed (%)')
    ax1.set_ylim(min(band_lo - 5, -5), 100)
    ax1.axhline(100, color=CB['grey'], linestyle=':', linewidth=1.5)
    ax1.text(lams[0], 97.5, 'complete removal', fontsize=10,
             color=CB['grey'], va='top')
    ax1.text(lams[-1], band_top + 1.5, 'observed range', fontsize=9.5,
             color=CB['grey'], ha='right', va='bottom')
    ax1.set_title('Removal never approaches complete')
    ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.88))
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)

    # ---- right: removal vs utility cost ----
    # Task accuracy is the better cost axis when we have it: reconstruction
    # loss is a proxy, task accuracy is the thing the model is for.
    if has_task:
        xs = [r['task_after'] for r in runs]
        xlabel = 'emotion probe balanced accuracy  (lower = worse task utility)'
        ref_x = runs[0]['task_after']
    else:
        xs = [r['recon'] for r in runs]
        xlabel = 'reconstruction loss  (higher = worse representation)'
        ref_x = ref_recon

    offsets = {'gender': (6, 5), 'race': (6, 8), 'age_bucket': (6, -12)}
    for attr, (color, marker, label) in ATTR_STYLE.items():
        y = [r['removed'][attr] for r in runs]
        ax2.plot(xs, y, marker=marker, color=color, label=label,
                 linewidth=2, markersize=8, alpha=0.9)
        for xi, yi, r in zip(xs, y, runs):
            ax2.annotate(f"$\\lambda$={int(r['lambda'])}", (xi, yi),
                         textcoords='offset points', xytext=offsets[attr],
                         fontsize=8.5, color=color)

    ax2.axvline(ref_x, color=CB['red'], linestyle='--', linewidth=1.8)
    ymax = ax2.get_ylim()[1]
    ax2.text(ref_x, ymax * 0.97, f' {ref_label}', fontsize=10,
             color=CB['red'], va='top')
    if has_task:
        ax2.axhline(0, color=CB['grey'], linewidth=1.0, zorder=1)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('above-chance signal removed (%)')
    ax2.set_title('No operating point offers removal without cost')
    ax2.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.42))
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)

    prefix = DATASET_TITLE.get(dataset, '')
    fig.suptitle((f'{prefix}: ' if prefix else '') +
                 'gradient-reversal debiasing, adversarial strength against utility',
                 y=1.02, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_residual_figure(runs, out_path, dataset):
    """Absolute probe performance at each lambda, against the chance floor."""
    has_task = 'task_after' in runs[0]
    style = dict(ATTR_STYLE)
    if has_task:
        style = {'emotion': (CB['purple'], 'D', 'Emotion (task)'), **ATTR_STYLE}

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    lams = [r['lambda'] for r in runs]
    n_series = len(style)
    width = 0.8 / n_series
    x = np.arange(len(lams))

    for i, (attr, (color, _, label)) in enumerate(style.items()):
        if attr == 'emotion':
            vals = [r['task_after'] for r in runs]
            chance = runs[0]['task_chance']
            before = runs[0]['task_before']
        else:
            vals = [r['after_probe'][attr] for r in runs]
            chance = runs[0]['chance'][attr]
            before = runs[0]['before_probe'][attr]

        off = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + off, vals, width, color=color, label=label)
        for b in bars:
            ax.hlines(chance, b.get_x(), b.get_x() + b.get_width(),
                      colors=color, linestyles=':', linewidth=1.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.3f}',
                    ha='center', fontsize=8)
        ax.hlines(before, x[0] + off - width / 2, x[-1] + off + width / 2,
                  colors=color, linestyles='--', linewidth=1.4, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'$\\lambda$={int(l)}' for l in lams])
    ax.set_ylabel('balanced accuracy of a fresh probe')
    ax.set_ylim(0, 1.0)
    prefix = DATASET_TITLE.get(dataset, '')
    ax.set_title((f'{prefix}: ' if prefix else '') +
                 'what a freshly initialised probe still recovers\n'
                 'dashed = before debiasing, dotted = chance')
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='../../results')
    p.add_argument('--out_dir', default='../../figures')
    p.add_argument('--dataset', choices=['utkface', 'rafdb'], default='utkface')
    p.add_argument('--suffix', default='',
                   help='Appended to output filenames, e.g. "_rafdb"')
    p.add_argument('--ref_lambda', type=float, default=1.0,
                   help='Which lambda run anchors the utility reference. '
                        'Defaults to the weakest adversarial setting, which '
                        'shares the same number of extra training epochs.')
    p.add_argument('--ref_recon', type=float, default=None)
    p.add_argument('--metric', choices=['LogReg', 'MLP'], default='LogReg',
                   help='Probe whose balanced accuracy defines removal. '
                        'LogReg is deterministic and matches fig1 exactly.')
    args = p.parse_args()

    if args.dataset == 'rafdb':
        paths = {lam: os.path.join(args.results_dir,
                                   f'debias_rafdb_lam{lam}.json')
                 for lam in (1, 5, 20, 50)}
    else:
        paths = {
            1:  os.path.join(args.results_dir, 'debias_results.json'),
            5:  os.path.join(args.results_dir, 'debias_lam5.json'),
            20: os.path.join(args.results_dir, 'debias_lam20.json'),
            50: os.path.join(args.results_dir, 'debias_lam50.json'),
        }

    runs = load_runs(paths, metric=args.metric)
    if len(runs) < 2:
        print("need at least two runs")
        return
    has_task = 'task_after' in runs[0]
    print(f"Dataset: {args.dataset}  |  removal metric: {args.metric} "
          f"({'deterministic, matches fig1' if args.metric == 'LogReg' else 'MLP'})")
    if has_task:
        print("Emotion present: used as the utility axis instead of reconstruction.")

    if args.ref_recon is not None:
        ref_recon, ref_label = args.ref_recon, 'reference'
    else:
        ref_run = min(runs, key=lambda r: abs(r['lambda'] - args.ref_lambda))
        ref_recon = ref_run['recon']
        ref_label = f"$\\lambda$={int(ref_run['lambda'])} (minimal pressure)"

    os.makedirs(args.out_dir, exist_ok=True)
    sfx = args.suffix
    print("Generating figures:")
    make_figure(runs, os.path.join(args.out_dir, f'fig5_tradeoff{sfx}.png'),
                ref_recon, ref_label, args.dataset)
    make_residual_figure(runs,
                         os.path.join(args.out_dir, f'fig6_residual{sfx}.png'),
                         args.dataset)

    # summary table for the poster
    cols = f"{'lambda':>7}{'recon':>9}{'gender':>11}{'race':>11}{'age':>11}"
    cols += f"{'emotion':>11}" if has_task else ""
    cols += f"{'recon cost':>13}"
    print("\n" + "=" * len(cols))
    print(cols)
    print("-" * len(cols))
    for r in runs:
        line = (f"{r['lambda']:>7.0f}{r['recon']:>9.1f}"
                f"{r['removed']['gender']:>10.1f}%"
                f"{r['removed']['race']:>10.1f}%"
                f"{r['removed']['age_bucket']:>10.1f}%")
        if has_task:
            line += f"{r['task_removed']:>10.1f}%"
        line += f"{r['recon'] - ref_recon:>+13.1f}"
        print(line)
    print("=" * len(cols))
    print(f"reconstruction cost measured against "
          f"{ref_label.replace(chr(92) + 'lambda', 'lambda').replace('$', '')}")
    if has_task:
        print("emotion column is the TASK: removal there is collateral damage, "
              "not success.")

    best = max(runs, key=lambda r: max(r['removed'].values()))
    ba = max(best['removed'], key=best['removed'].get)
    print(f"peak removal: {best['removed'][ba]:.1f}% ({ba}) at "
          f"lambda={best['lambda']:.0f}, reconstruction cost "
          f"{best['recon'] - ref_recon:+.1f}")

    neg = [(r['lambda'], a, v) for r in runs
           for a, v in r['removed'].items() if v < 0]
    if neg:
        print("\nNEGATIVE removal (attribute became MORE recoverable):")
        for lam, a, v in neg:
            print(f"  lambda={lam:.0f}  {a}: {v:+.1f}%")
    print()


if __name__ == "__main__":
    main()
"""
Experiment D figure: the adversarial removal trade-off.

Two panels:
  left  - removal (%) against lambda, one line per attribute
  right - removal (%) against reconstruction loss, the utility/invariance plane

The reference point is lambda=1, not the original checkpoint. Every debias run
adds 20 epochs of ordinary training on top of the pretrained VAE, which improves
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
      'red': '#D55E00', 'grey': '#999999'}

ATTR_STYLE = {
    'gender':     (CB['blue'],   'o', 'Gender'),
    'race':       (CB['orange'], 's', 'Race'),
    'age_bucket': (CB['green'],  '^', 'Age bucket'),
}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_runs(paths_by_lambda, metric='LogReg'):
    """
    Read each debias result file and pull out removal + final reconstruction.

    -------------------------------------------------------------------------
    WHY metric='LogReg' (metric-consistency patch)
    -------------------------------------------------------------------------
    The stored 'pct_removed' field was computed inside debias.py from the MLP
    probe. MLPClassifier(early_stopping=True) draws an internal validation
    split from its random_state, so its balanced accuracy is not
    bit-reproducible across the two call sites (probes.py for fig1 vs
    debias.py for the baseline); the MLP 'before' therefore reads ~0.007
    higher than fig1's MLP, and fig1's headline could not be reconciled with
    fig5/fig6.

    The linear LogReg probe is deterministic and matches fig1 to 4 decimals.
    So we IGNORE the stored MLP-derived pct_removed and recompute removal here
    from the LogReg before/after values that debias.py already saved for every
    attribute. Result: fig1, fig5 and fig6 are all locked to the same
    reproducible linear-probe metric. Pass metric='MLP' to recover the old
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

        removed = {}
        for a in ATTR_STYLE:
            before = d['before'][a][metric]
            after = d['after'][a][metric]
            chance = d['before'][a]['chance']
            removed[a] = ((before - after) / (before - chance) * 100
                          if before > chance else 0.0)

        runs.append({
            'lambda': lam,
            'recon': d['history'][-1]['recon'],
            'kl': d['history'][-1]['kl'],
            'metric': metric,
            'removed': removed,
            'after_probe': {a: d['after'][a][metric] for a in ATTR_STYLE},
            'before_probe': {a: d['before'][a][metric] for a in ATTR_STYLE},
            'chance': {a: d['before'][a]['chance'] for a in ATTR_STYLE},
        })
    return runs


def make_figure(runs, out_path, ref_recon, ref_label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    lams = [r['lambda'] for r in runs]

    # ---- left: removal vs lambda, full scale to show how far from complete ----
    all_vals = [v for r in runs for v in r['removed'].values()]
    band_top = max(20, np.ceil(max(all_vals) / 5) * 5)

    ax1.axhspan(0, band_top, color=CB['grey'], alpha=0.08, zorder=0)

    for attr, (color, marker, label) in ATTR_STYLE.items():
        vals = [r['removed'][attr] for r in runs]
        ax1.plot(lams, vals, marker=marker, color=color, label=label,
                 linewidth=2, markersize=8, zorder=3)

    ax1.set_xscale('log')
    ax1.set_xticks(lams)
    ax1.set_xticklabels([str(int(l)) for l in lams])
    ax1.set_xlabel('adversarial strength  $\\lambda$')
    ax1.set_ylabel('above-chance signal removed (%)')
    ax1.set_ylim(0, 100)
    ax1.axhline(100, color=CB['grey'], linestyle=':', linewidth=1.5)
    ax1.text(lams[0], 97.5, 'complete removal', fontsize=10,
             color=CB['grey'], va='top')
    ax1.text(lams[-1], band_top + 1.5, 'observed range', fontsize=9.5,
             color=CB['grey'], ha='right', va='bottom')
    ax1.set_title('Removal never approaches complete')
    ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.88))
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)

    # ---- right: removal vs reconstruction cost ----
    # per-attribute label offsets so the lambda annotations don't stack where
    # curves converge (esp. gender/age near lambda=50); race up, gender slight
    # up, age pushed below its marker
    offsets = {'gender': (6, 5), 'race': (6, 8), 'age_bucket': (6, -12)}
    for attr, (color, marker, label) in ATTR_STYLE.items():
        x = [r['recon'] for r in runs]
        y = [r['removed'][attr] for r in runs]
        ax2.plot(x, y, marker=marker, color=color, label=label,
                 linewidth=2, markersize=8, alpha=0.9)
        for xi, yi, r in zip(x, y, runs):
            ax2.annotate(f"$\\lambda$={int(r['lambda'])}", (xi, yi),
                         textcoords='offset points', xytext=offsets[attr],
                         fontsize=8.5, color=color)

    ax2.axvline(ref_recon, color=CB['red'], linestyle='--', linewidth=1.8)
    ymax = ax2.get_ylim()[1]
    ax2.text(ref_recon, ymax * 0.97, f' {ref_label}', fontsize=10,
             color=CB['red'], va='top')

    ax2.set_xlabel('reconstruction loss  (higher = worse representation)')
    ax2.set_ylabel('above-chance signal removed (%)')
    ax2.set_title('No operating point offers removal without cost')
    # origin cluster (lower-left) and the race peak (upper area) are both busy;
    # the empty band is mid-right, below the race descent
    ax2.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.42))
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.suptitle('Gradient-reversal debiasing: adversarial strength against utility',
                 y=1.02, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_residual_figure(runs, out_path):
    """Absolute probe performance at each lambda, against the chance floor."""
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    lams = [r['lambda'] for r in runs]
    width = 0.25
    x = np.arange(len(lams))

    for i, (attr, (color, _, label)) in enumerate(ATTR_STYLE.items()):
        vals = [r['after_probe'][attr] for r in runs]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      color=color, label=label)
        chance = runs[0]['chance'][attr]
        for b in bars:
            ax.hlines(chance, b.get_x(), b.get_x() + b.get_width(),
                      colors=color, linestyles=':', linewidth=1.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.3f}',
                    ha='center', fontsize=8.5)

    # baseline (before any debiasing) spans
    for i, (attr, (color, _, _)) in enumerate(ATTR_STYLE.items()):
        b0 = runs[0]['before_probe'][attr]
        ax.hlines(b0, x[0] + (i - 1) * width - width / 2,
                  x[-1] + (i - 1) * width + width / 2,
                  colors=color, linestyles='--', linewidth=1.4, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'$\\lambda$={int(l)}' for l in lams])
    ax.set_ylabel('balanced accuracy of a fresh probe')
    ax.set_ylim(0, 1.0)
    ax.set_title('What a freshly initialised probe still recovers\n'
                 'dashed = before debiasing, dotted = chance')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='../../results')
    p.add_argument('--out_dir', default='../../figures')
    p.add_argument('--ref_lambda', type=float, default=1.0,
                   help='Which lambda run to use as the utility reference. '
                        'Defaults to the weakest adversarial setting, which '
                        'shares the same number of extra training epochs.')
    p.add_argument('--ref_recon', type=float, default=None,
                   help='Override the reference reconstruction loss directly.')
    p.add_argument('--metric', choices=['LogReg', 'MLP'], default='LogReg',
                   help='Probe whose balanced accuracy defines removal. '
                        'LogReg is deterministic and matches fig1 exactly; '
                        'MLP reproduces the earlier (non-reproducible) figures.')
    args = p.parse_args()

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
    print(f"Removal metric: {args.metric} balanced accuracy "
          f"({'deterministic, matches fig1' if args.metric == 'LogReg' else 'MLP, not reproducible'})")

    # anchor utility cost to the weakest adversarial run unless overridden
    if args.ref_recon is not None:
        ref_recon = args.ref_recon
        ref_label = 'reference'
    else:
        ref_run = min(runs, key=lambda r: abs(r['lambda'] - args.ref_lambda))
        ref_recon = ref_run['recon']
        ref_label = f"$\\lambda$={int(ref_run['lambda'])} (minimal pressure)"

    os.makedirs(args.out_dir, exist_ok=True)
    print("Generating figures:")
    make_figure(runs, os.path.join(args.out_dir, 'fig5_tradeoff.png'),
                ref_recon, ref_label)
    make_residual_figure(runs, os.path.join(args.out_dir, 'fig6_residual.png'))

    # summary table for the poster
    print("\n" + "=" * 74)
    print(f"{'lambda':>7}{'recon':>9}{'gender':>11}{'race':>11}"
          f"{'age':>11}{'recon cost':>13}")
    print("-" * 74)
    for r in runs:
        cost = r['recon'] - ref_recon
        print(f"{r['lambda']:>7.0f}{r['recon']:>9.1f}"
              f"{r['removed']['gender']:>10.1f}%"
              f"{r['removed']['race']:>10.1f}%"
              f"{r['removed']['age_bucket']:>10.1f}%"
              f"{cost:>+13.1f}")
    print("=" * 74)
    print(f"reconstruction cost measured against {ref_label.replace('$\\lambda$', 'lambda')}")

    best = max(runs, key=lambda r: max(r['removed'].values()))
    ba = max(best['removed'], key=best['removed'].get)
    print(f"peak removal: {best['removed'][ba]:.1f}% ({ba}) at "
          f"lambda={best['lambda']:.0f}, reconstruction cost "
          f"{best['recon'] - ref_recon:+.1f}\n")


if __name__ == "__main__":
    main()
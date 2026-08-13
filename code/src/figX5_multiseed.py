#!/usr/bin/env python3
"""
figX5_removal_cross (multi-seed) — reads debias_multiseed_{dataset}.json.

Drop-in replacement for the two functions in cross_dataset.py that build figX5:
  - load_debias_sweep_multiseed()  (reads the aggregate block, mean+std+signs)
  - fig_removal_cross_multiseed()  (adds error bars, honest lambda=50)

Standalone here so the figure can be rendered without a GPU (it only reads the
result JSONs). On the CIP pool you can either run this file directly, or paste
these two functions into cross_dataset.py (see NOTE at bottom).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- match cross_dataset.py exactly ---
CB = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
      'grey': '#999999', 'red': '#D55E00'}
DS = {
    'utkface': {'label': 'UTKFace', 'colour': CB['blue']},
    'rafdb':   {'label': 'RAF-DB', 'colour': CB['orange']},
}
DEMO = ['gender', 'race', 'age_bucket']
DEMO_LABEL = {'gender': 'Gender', 'race': 'Race', 'age_bucket': 'Age'}
LAMS = [1, 5, 20, 50]

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'figure.dpi': 200,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans',
})


def load_multiseed(results_dir, dataset):
    """Return {lambda: {attr: (mean, std, n_pos, n_neg)}} from the aggregate block."""
    path = os.path.join(results_dir, f'debias_multiseed_{dataset}.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    agg = d['aggregate']
    out = {}
    for lam in LAMS:
        key = str(lam)
        if key not in agg:
            continue
        out[lam] = {}
        for a in DEMO:
            if a in agg[key]:
                s = agg[key][a]
                out[lam][a] = (s['mean'], s['std'], s['n_positive'], s['n_negative'])
    return out


def fig_removal_cross_multiseed(sweeps, out_path):
    """3-panel (gender/race/age) removal vs lambda, mean +/- std across seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)

    # find y-range
    allvals = []
    for ds in sweeps:
        if not sweeps[ds]:
            continue
        for lam in sweeps[ds]:
            for a in DEMO:
                if a in sweeps[ds][lam]:
                    m, sd, *_ = sweeps[ds][lam][a]
                    allvals += [m - sd, m + sd]
    lo = min(-5, min(allvals) - 2) if allvals else -5
    hi = max(30, max(allvals) + 3) if allvals else 30

    for ax, attr in zip(axes, DEMO):
        for ds, meta in DS.items():
            if ds not in sweeps or not sweeps[ds]:
                continue
            lams, means, stds = [], [], []
            for lam in LAMS:
                if lam in sweeps[ds] and attr in sweeps[ds][lam]:
                    m, sd, *_ = sweeps[ds][lam][attr]
                    lams.append(lam); means.append(m); stds.append(sd)
            ax.errorbar(lams, means, yerr=stds, marker='o', color=meta['colour'],
                        label=meta['label'], linewidth=2, markersize=7,
                        capsize=4, capthick=1.4, elinewidth=1.4)

        ax.axhline(0, color='#333333', linewidth=1.1)
        ax.set_xscale('log')
        ax.set_xticks(LAMS)
        ax.set_xticklabels([str(l) for l in LAMS])
        ax.set_xlabel('adversarial strength $\\lambda$')
        ax.set_ylim(lo, hi)
        ax.set_title(DEMO_LABEL[attr])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('above-chance signal removed (%)')
    axes[0].legend(frameon=False, loc='upper left')

    # honest annotation: peak-then-decline, no "goes negative" claim
    axes[1].annotate('peaks then declines\nwith more $\\lambda$',
                     xy=(20, 18), xytext=(5, 26), fontsize=9,
                     color=CB['grey'], ha='left', va='top')

    fig.suptitle('Adversarial removal peaks below ~20% and declines with $\\lambda$ '
                 '— on both corpora, across 5 seeds\n'
                 'mean ± std, freshly initialised probe after the adversary is discarded',
                 y=1.05, fontsize=13.5)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='.')
    p.add_argument('--out', default='figX5_removal_cross.png')
    args = p.parse_args()

    sweeps = {ds: load_multiseed(args.results_dir, ds) for ds in DS}
    present = [ds for ds in sweeps if sweeps[ds]]
    print("multi-seed sweeps found for:", present)
    if not present:
        print("No debias_multiseed_*.json found in", args.results_dir)
        return
    fig_removal_cross_multiseed(sweeps, args.out)

    # console summary
    print("\nPeak removal per dataset (race):")
    for ds in present:
        best = max(((lam, sweeps[ds][lam]['race'][0]) for lam in sweeps[ds]
                    if 'race' in sweeps[ds][lam]), key=lambda t: t[1])
        print(f"  {DS[ds]['label']}: {best[1]:.1f}% at lambda={best[0]}")


if __name__ == "__main__":
    main()

# NOTE for CIP pool: to fold into cross_dataset.py, replace the call
#   sweeps = {ds: load_debias_sweep(R, ds, metric=args.metric) for ds in DS}
# with
#   sweeps = {ds: load_multiseed(R, ds) for ds in DS}
# and call fig_removal_cross_multiseed(sweeps, ...) instead of fig_removal_cross.
# The other figures (figX1-figX4) are unchanged.
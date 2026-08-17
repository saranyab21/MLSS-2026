"""
Figures for the two control experiment.

figX6_pca_control  - is the encoder responsible for the leakage? (no)
figX7_multiseed    - do the interventions move the gap? (not meaningfully)

Both are "we ran the control that could have killed this" figures. On a poster
they do more work than another result panel, because they show the claim
survived a test designed to break it.
"""

from fedar_common.plotting import CB, load_json
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fedar_common.plotting import apply_poster_style
apply_poster_style()
from fedar_common.plotting import CB, POSTER, load_json


DS_LABEL = {'utkface': 'UTKFace', 'rafdb': 'RAF-DB'}
DEMO = ['gender', 'race', 'age_bucket']
NICE = {'gender': 'Gender', 'race': 'Race', 'age_bucket': 'Age',
        'emotion': 'Emotion\n(task)'}

COND = ['none', 'fedar', 'demo_bal']
COND_SHORT = {'none': 'no\nintervention', 'fedar': 'FedAR\nresampling',
              'demo_bal': 'demographic\nbalancing'}

plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 16, 'axes.titlesize': 16,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 16,
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load(path):
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------

def fig_pca_control(pix, out_path):
    """
    Lift over chance for the VAE latent against dimension-matched PCA.

    The message is the flatness. Every representation carries the same
    demographic signal, so the encoder is not the source.
    """
    datasets = [d for d in ('utkface', 'rafdb') if d in pix]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(7.6 * len(datasets), 5.6))
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, datasets):
        res = pix[ds]['results']
        latent_dim = pix[ds]['latent_dim']

        # order: VAE first, then the dimension-matched controls, then raw
        keys = [k for k in res if k.startswith('VAE')]
        keys += sorted(k for k in res if k.startswith('PCA'))
        keys += [k for k in res if k.startswith('raw')]

        short = {}
        for k in keys:
            if k.startswith('VAE'):
                short[k] = f'VAE latent ({latent_dim})'
            elif '16x16' in k and k.startswith('PCA'):
                short[k] = f'PCA{latent_dim} · 16×16 grey'
            elif '32x32' in k:
                short[k] = f'PCA{latent_dim} · 32×32 RGB'
            else:
                short[k] = 'raw 16×16 grey (256)'

        colours = {keys[0]: CB['blue']}
        rest = [CB['sky'], CB['orange'], CB['grey']]
        for k, c in zip(keys[1:], rest):
            colours[k] = c

        targets = [t for t in DEMO if t in res[keys[0]]['targets']]
        x = np.arange(len(targets))
        width = 0.8 / len(keys)

        for i, k in enumerate(keys):
            vals = [res[k]['targets'][t]['lift'] for t in targets]
            off = (i - (len(keys) - 1) / 2) * width
            bars = ax.bar(x + off, vals, width, color=colours[k],
                          label=short[k],
                          edgecolor='black' if i == 0 else 'none',
                          linewidth=1.4 if i == 0 else 0)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7.8,
                        rotation=90)

        ax.axhline(0, color=CB['red'], linestyle='--', linewidth=1.6)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[t] for t in targets])
        ax.set_ylabel('balanced accuracy above chance' if ax is axes[0] else '')
        ax.set_ylim(0, max(res[k]['targets'][t]['lift']
                           for k in keys for t in targets) * 1.28)

        adv = pix[ds]['mean_demographic_advantage']
        ax.set_title(f"{DS_LABEL[ds]}\nmean VAE advantage over PCA: "
                     f"{adv:+.3f}", pad=10)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(frameon=False, loc='upper left', fontsize=9.5)

    fig.suptitle('The encoder is not the source: a PCA of raw pixels retains '
                 'as much demographic signal\n'
                 'the claim is preservation, not learning',
                 y=1.03, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_multiseed(ms, out_path):
    """
    Per-seed paired differences, with the gap's own magnitude for scale.

    Individual seeds are plotted, not just mean and error bar, because sign
    consistency across seeds is the claim — not the p-value, which is fragile
    at n=5.
    """
    datasets = [d for d in ('utkface', 'rafdb') if d in ms]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(7.0 * len(datasets), 5.6), sharey=True)
    axes = np.atleast_1d(axes)

    # common y-range so the two panels are visually comparable
    allv = [v for d in datasets for c in ('fedar', 'demo_bal')
            for v in ms[d]['deltas'][c]['gap']['values']]
    lim = max(abs(min(allv)), abs(max(allv))) * 1.45

    for ax, ds in zip(axes, datasets):
        deltas = ms[ds]['deltas']
        seeds = ms[ds]['seeds']
        base = np.array([ms[ds]['per_seed'][str(s)]['none']['gap']
                         for s in seeds])

        conds = ['fedar', 'demo_bal']
        colours = [CB['orange'], CB['green']]

        for i, (c, col) in enumerate(zip(conds, colours)):
            g = deltas[c]['gap']
            vals = g['values']
            jitter = np.linspace(-0.13, 0.13, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=46,
                       color=col, alpha=0.75, zorder=3,
                       edgecolors='white', linewidths=0.8)
            # mean and standard deviation
            ax.hlines(g['mean'], i - 0.28, i + 0.28, colors=col,
                      linewidth=3, zorder=4)
            ax.add_patch(plt.Rectangle((i - 0.28, g['mean'] - g['std']),
                                       0.56, 2 * g['std'],
                                       color=col, alpha=0.16, zorder=1))
            sign = f"{g['n_positive']}+ / {g['n_negative']}−"
            ax.text(i, lim * 0.90, f"{g['mean']:+.4f}\n± {g['std']:.4f}\n{sign}",
                    ha='center', va='top', fontsize=9.5, color=col)

        # the band the gap itself occupies, for scale
        ax.axhspan(-base.mean(), base.mean(), color=CB['grey'], alpha=0.07,
                   zorder=0)
        ax.axhline(0, color='#333333', linewidth=1.3, zorder=2)

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([COND_SHORT[c] for c in conds])
        ax.set_ylabel('change in subgroup gap' if ax is axes[0] else '')
        ax.set_ylim(-lim, lim)

        largest = max(abs(deltas[c]['gap']['mean']) for c in conds)
        pct = 100 * largest / base.mean()
        ax.set_title(f"gap itself {base.mean():.3f} ± "
                     f"{base.std(ddof=1):.3f}   ·   largest effect = "
                     f"{pct:.1f}% of it", pad=8, fontsize=11)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

        # dataset name below the panel, clear of the two-line x-tick labels
        ax.text(0.5, -0.20, DS_LABEL[ds], transform=ax.transAxes,
                ha='center', va='top', fontsize=13, fontweight='bold')

    axes[0].text(-0.42, -lim * 0.93, 'each point is one seed',
                 fontsize=9.5, color='#555555')

    fig.suptitle('Neither intervention moves the subgroup gap by more than 8% '
                 'of its own magnitude',
                 y=1.06, fontsize=14, fontweight='bold')
    fig.text(0.5, 1.005,
             'five seeds · paired within-seed differences against no intervention',
             ha='center', va='bottom', fontsize=10.5, color='#555555')
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")

''' def fig_multiseed(ms, out_path):
    """
    Per-seed paired differences, with the gap's own magnitude for scale.

    Individual seeds are plotted, not just mean and error bar, because sign
    consistency across seeds is the claim — not the p-value, which is fragile
    at n=5.
    """
    datasets = [d for d in ('utkface', 'rafdb') if d in ms]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(7.0 * len(datasets), 5.6), sharey=True)
    axes = np.atleast_1d(axes)

    # common y-range so the two panels are visually comparable
    allv = [v for d in datasets for c in ('fedar', 'demo_bal')
            for v in ms[d]['deltas'][c]['gap']['values']]
    lim = max(abs(min(allv)), abs(max(allv))) * 1.45

    for ax, ds in zip(axes, datasets):
        deltas = ms[ds]['deltas']
        seeds = ms[ds]['seeds']
        base = np.array([ms[ds]['per_seed'][str(s)]['none']['gap']
                         for s in seeds])

        conds = ['fedar', 'demo_bal']
        colours = [CB['orange'], CB['green']]

        for i, (c, col) in enumerate(zip(conds, colours)):
            g = deltas[c]['gap']
            vals = g['values']
            jitter = np.linspace(-0.13, 0.13, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=46,
                       color=col, alpha=0.75, zorder=3,
                       edgecolors='white', linewidths=0.8)
            # mean and standard deviation
            ax.hlines(g['mean'], i - 0.28, i + 0.28, colors=col,
                      linewidth=3, zorder=4)
            ax.add_patch(plt.Rectangle((i - 0.28, g['mean'] - g['std']),
                                       0.56, 2 * g['std'],
                                       color=col, alpha=0.16, zorder=1))
            sign = f"{g['n_positive']}+ / {g['n_negative']}−"
            ax.text(i, lim * 0.90, f"{g['mean']:+.4f}\n± {g['std']:.4f}\n{sign}",
                    ha='center', va='top', fontsize=9.5, color=col)

        # the band the gap itself occupies, for scale
        ax.axhspan(-base.mean(), base.mean(), color=CB['grey'], alpha=0.07,
                   zorder=0)
        ax.axhline(0, color='#333333', linewidth=1.3, zorder=2)

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([COND_SHORT[c] for c in conds])
        ax.set_ylabel('change in subgroup gap' if ax is axes[0] else '')
        ax.set_ylim(-lim, lim)

        largest = max(abs(deltas[c]['gap']['mean']) for c in conds)
        pct = 100 * largest / base.mean()
        ax.set_title(f"{DS_LABEL[ds]}\ngap itself {base.mean():.3f} ± "
                     f"{base.std(ddof=1):.3f}   ·   largest effect = "
                     f"{pct:.1f}% of it", pad=10)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].text(-0.42, -lim * 0.93, 'each point is one seed',
                 fontsize=9.5, color='#555555')

    fig.suptitle('Neither intervention moves the subgroup gap by more than '
                 '8% of its own magnitude\n'
                 'five seeds, paired within-seed differences against no intervention',
                 y=1.03, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}") '''


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='../../results')
    p.add_argument('--out_dir', default='../../figures')
    args = p.parse_args()

    R = args.results_dir
    os.makedirs(args.out_dir, exist_ok=True)

    pix = {}
    for ds in ('utkface', 'rafdb'):
        d = load(os.path.join(R, f'pixel_baseline_{ds}.json'))
        if d:
            pix[ds] = d

    ms = {}
    for ds in ('utkface', 'rafdb'):
        d = load(os.path.join(R, f'federated_multiseed_{ds}.json'))
        if d:
            ms[ds] = d

    print("Generating control figures:")
    if pix:
        fig_pca_control(pix, os.path.join(args.out_dir,
                                          'figX6_pca_control.png'))
    if ms:
        fig_multiseed(ms, os.path.join(args.out_dir, 'figX7_multiseed.png'))

    # one-line summaries for the poster caption
    print("\n" + "=" * 70)
    for ds in pix:
        print(f"{DS_LABEL[ds]:<10} mean VAE advantage over PCA-128: "
              f"{pix[ds]['mean_demographic_advantage']:+.4f}")
    for ds in ms:
        base = np.mean([ms[ds]['per_seed'][str(s)]['none']['gap']
                        for s in ms[ds]['seeds']])
        largest = max(abs(ms[ds]['deltas'][c]['gap']['mean'])
                      for c in ('fedar', 'demo_bal'))
        print(f"{DS_LABEL[ds]:<10} largest gap effect: {largest:.4f} "
              f"= {100*largest/base:.1f}% of the gap ({base:.4f})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main() 
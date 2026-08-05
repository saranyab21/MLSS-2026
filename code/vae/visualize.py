"""
Poster figures from the probe results and cached latents.

Produces:
  fig1_probe_performance.png  - probe balanced accuracy vs chance, all targets
  fig2_umap_panels.png        - latent space coloured by each demographic attribute
  fig3_dataset_composition.png - UTKFace demographic breakdown
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Colourblind-safe palette (Okabe-Ito)
CB = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'sky':    '#56B4E9',
    'grey':   '#999999',
}

TARGET_LABELS = {
    'gender': 'Gender\n(2 classes)',
    'race': 'Race\n(5 classes)',
    'age_bucket': 'Age bucket\n(4 classes)',
}

CLASS_NAMES = {
    'gender': ['Male', 'Female'],
    'race': ['White', 'Black', 'Asian', 'Indian', 'Other'],
    'age_bucket': ['0-19', '20-34', '35-49', '50+'],
}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def fig_probe_performance(results, out_path):
    """Grouped bar chart: three probes per target, with chance line."""
    targets = list(TARGET_LABELS.keys())
    probes = ['LogReg', 'LinearSVM', 'MLP']
    colors = [CB['blue'], CB['sky'], CB['orange']]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.24
    x = np.arange(len(targets))

    for i, (probe, color) in enumerate(zip(probes, colors)):
        vals = [results[t]['probes'][probe]['balanced_accuracy'] for t in targets]
        errs = [results[t]['probes'][probe].get('cv_std', 0) for t in targets]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=probe,
                      color=color, yerr=errs, capsize=3,
                      error_kw={'linewidth': 1, 'ecolor': '#444444'})
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.022, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9.5)

    # chance level per target
    for i, t in enumerate(targets):
        chance = 1.0 / results[t]['n_classes']
        ax.hlines(chance, i - 0.42, i + 0.42, colors=CB['red'],
                  linestyles='--', linewidth=2,
                  label='Chance' if i == 0 else None)
        ax.text(i + 0.44, chance, f'{chance:.2f}', va='center',
                fontsize=9.5, color=CB['red'])

    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in targets])
    ax.set_ylabel('Balanced accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_title('Demographic attributes recovered from a frozen VAE latent space\n'
                 'trained with reconstruction and KL loss only',
                 pad=14)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_umap(latents_path, out_path, seed=42, n_sample=6000):
    """Three UMAP panels of the same latent space, coloured by attribute."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP figure")
        print("  install with: pip install umap-learn")
        return

    d = np.load(latents_path, allow_pickle=True)
    mu = d['mu']

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(mu), size=min(n_sample, len(mu)), replace=False)
    mu_s = mu[idx]

    print(f"  fitting UMAP on {len(mu_s)} points...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=seed)
    emb = reducer.fit_transform(mu_s)

    attrs = ['gender', 'race', 'age_bucket']
    palettes = {
        'gender': [CB['blue'], CB['orange']],
        'race': [CB['blue'], CB['orange'], CB['green'], CB['purple'], CB['grey']],
        'age_bucket': [CB['sky'], CB['green'], CB['orange'], CB['red']],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for ax, attr in zip(axes, attrs):
        y = d[attr][idx]
        names = CLASS_NAMES[attr]
        pal = palettes[attr]

        for c in range(len(names)):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=2.5, alpha=0.45,
                       color=pal[c], label=names[c], linewidths=0)

        ax.set_title(f'coloured by {attr.replace("_", " ")}')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color('#cccccc')

        handles = [Patch(facecolor=pal[c], label=names[c])
                   for c in range(len(names))]
        ax.legend(handles=handles, loc='upper right', frameon=False,
                  fontsize=9, markerscale=1)

    fig.suptitle('UMAP of the same frozen VAE latent space\n'
                 'Illustrative only: probe performance is the evidence',
                 y=1.04, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_composition(latents_path, out_path):
    """Dataset composition: the confound-check figure."""
    d = np.load(latents_path, allow_pickle=True)
    n = len(d['mu'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    attrs = ['gender', 'race', 'age_bucket']
    palettes = {
        'gender': [CB['blue'], CB['orange']],
        'race': [CB['blue'], CB['orange'], CB['green'], CB['purple'], CB['grey']],
        'age_bucket': [CB['sky'], CB['green'], CB['orange'], CB['red']],
    }

    for ax, attr in zip(axes, attrs):
        y = d[attr]
        names = CLASS_NAMES[attr]
        counts = np.bincount(y, minlength=len(names))
        pct = 100 * counts / n

        bars = ax.bar(range(len(names)), pct, color=palettes[attr], width=0.65)
        for b, p, c in zip(bars, pct, counts):
            ax.text(b.get_x() + b.get_width() / 2, p + 1.0,
                    f'{p:.1f}%\n({c})', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=0 if len(names) < 5 else 30,
                           ha='center' if len(names) < 5 else 'right')
        ax.set_ylabel('% of dataset' if attr == 'gender' else '')
        ax.set_ylim(0, max(pct) * 1.28)
        ax.set_title(attr.replace('_', ' '))
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(f'UTKFace composition (N = {n:,})', y=1.02, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")

def fig_lda_projection(latents_path, out_path):
    """
    Project latents onto the linear discriminant axes for each attribute.

    This is the honest visual counterpart to the linear-probe result:
    if demographic information is linearly decodable, there exist directions
    in latent space along which the classes separate. LDA finds exactly those
    directions. Unlike UMAP, this is a linear projection with no free
    hyperparameters and nothing to tune toward a desired outcome.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    d = np.load(latents_path, allow_pickle=True)
    mu = d['mu']
    split = d['split']
    tr, te = split == 'train', split == 'test'

    scaler = StandardScaler().fit(mu[tr])
    X_tr, X_te = scaler.transform(mu[tr]), scaler.transform(mu[te])

    attrs = ['gender', 'race', 'age_bucket']
    palettes = {
        'gender': [CB['blue'], CB['orange']],
        'race': [CB['blue'], CB['orange'], CB['green'], CB['purple'], CB['grey']],
        'age_bucket': [CB['sky'], CB['green'], CB['orange'], CB['red']],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for ax, attr in zip(axes, attrs):
        y_tr, y_te = d[attr][tr], d[attr][te]
        names = CLASS_NAMES[attr]
        pal = palettes[attr]
        n_classes = len(names)

        # LDA yields at most (n_classes - 1) discriminant directions.
        # Binary gender gives one axis, so plot it as a density instead.
        n_comp = min(2, n_classes - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        lda.fit(X_tr, y_tr)
        Z = lda.transform(X_te)   # fit on train, project test: no leakage

        if n_comp == 1:
            for c in range(n_classes):
                m = y_te == c
                ax.hist(Z[m, 0], bins=60, alpha=0.6, color=pal[c],
                        label=names[c], density=True)
            ax.set_xlabel('LD1')
            ax.set_ylabel('density')
            ax.set_yticks([])
        else:
            for c in range(n_classes):
                m = y_te == c
                ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.5,
                           color=pal[c], label=names[c], linewidths=0)
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            ax.set_xticks([]); ax.set_yticks([])

        ax.set_title(f'{attr.replace("_", " ")}')
        ax.legend(loc='upper right', frameon=False, fontsize=9, markerscale=2.5)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color('#cccccc')

    fig.suptitle('Linear discriminant projection of the frozen VAE latent space\n'
                 'Directions fitted on the probe-train split, points shown are held-out test',
                 y=1.05, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default='../../results/probe_results.json')
    parser.add_argument('--latents', type=str,
                        default='../../latents/utkface_latents.npz')
    parser.add_argument('--out_dir', type=str, default='../../figures')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    print("Generating figures:")
    fig_probe_performance(results,
                          os.path.join(args.out_dir, 'fig1_probe_performance.png'))
    fig_composition(args.latents,
                    os.path.join(args.out_dir, 'fig3_dataset_composition.png'))
    fig_lda_projection(args.latents,
                       os.path.join(args.out_dir, 'fig4_lda_projection.png'))
    fig_umap(args.latents,
             os.path.join(args.out_dir, 'fig2_umap_panels.png'), seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
"""
Poster figures from the probe results and cached latents.

Works on either corpus. When the latents carry an emotion label (RAF-DB),
emotion is plotted alongside the demographic targets: the point is that the
task signal and the identity signal occupy the same latent space.

Produces:
  fig1_probe_performance.png   - probe balanced accuracy vs chance, all targets
  fig2_umap_panels.png         - latent space coloured by each attribute
  fig3_dataset_composition.png - demographic (and emotion) breakdown
  fig4_lda_projection.png      - linear discriminant projection per attribute
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Colourblind-safe palette (Okabe-Ito) — single definition in fedar_common.
from fedar_common.plotting import CB

TARGET_LABELS = {
    'emotion': 'Emotion\n(7 classes)',
    'gender': 'Gender\n(2 classes)',
    'race': 'Race\n(5 classes)',
    'age_bucket': 'Age bucket\n(4 classes)',
}

CLASS_NAMES = {
    'emotion': ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral'],
    'gender': ['Male', 'Female'],
    'race': ['White', 'Black', 'Asian', 'Indian', 'Other'],
    'age_bucket': ['0-19', '20-34', '35-49', '50+'],
}

PALETTES = {
    'emotion': [CB['blue'], CB['orange'], CB['green'], CB['purple'],
                CB['sky'], CB['red'], CB['grey']],
    'gender': [CB['blue'], CB['orange']],
    'race': [CB['blue'], CB['orange'], CB['green'], CB['purple'], CB['grey']],
    'age_bucket': [CB['sky'], CB['green'], CB['orange'], CB['red']],
}

DATASET_TITLE = {'utkface': 'UTKFace', 'rafdb': 'RAF-DB', 'unknown': ''}

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


def present_targets(container):
    """Which targets exist here, in a stable order with emotion first."""
    keys = container.files if hasattr(container, 'files') else container.keys()
    order = ['emotion', 'gender', 'race', 'age_bucket']
    return [t for t in order if t in keys]


def label_provenance(dataset, target):
    """One-line note for the caption: which labels are human, which inferred."""
    if dataset != 'rafdb':
        return ''
    return 'human-annotated' if target == 'emotion' else 'FairFace-inferred'


def fig_probe_performance(results, out_path, dataset):
    """Grouped bar chart: three probes per target, with chance line."""
    targets = [t for t in ['emotion', 'gender', 'race', 'age_bucket']
               if t in results]
    probes = ['LogReg', 'LinearSVM', 'MLP']
    colors = [CB['blue'], CB['sky'], CB['orange']]

    fig, ax = plt.subplots(figsize=(2.6 * len(targets) + 2, 5.5))
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

    # mark the task, when there is one, so nobody reads emotion as a leak
    if 'emotion' in targets:
        ax.text(0, 0.955, 'the task', ha='center', fontsize=10,
                color=CB['grey'], style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in targets])
    ax.set_ylabel('Balanced accuracy')
    ax.set_ylim(0, 1.0)

    title = ('Attributes recovered from a frozen VAE latent space\n'
             'trained with reconstruction and KL loss only')
    if dataset in DATASET_TITLE and DATASET_TITLE[dataset]:
        title = f'{DATASET_TITLE[dataset]}: ' + title[0].lower() + title[1:]
    ax.set_title(title, pad=14)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_umap(latents_path, out_path, seed=42, n_sample=6000):
    """UMAP panels of the same latent space, coloured by each attribute."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP figure")
        return

    d = np.load(latents_path, allow_pickle=True)
    mu = d['mu']
    dataset = str(d['dataset']) if 'dataset' in d.files else 'unknown'
    attrs = present_targets(d)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(mu), size=min(n_sample, len(mu)), replace=False)
    mu_s = mu[idx]

    print(f"  fitting UMAP on {len(mu_s)} points...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=seed)
    emb = reducer.fit_transform(mu_s)

    fig, axes = plt.subplots(1, len(attrs), figsize=(5.3 * len(attrs), 5.2))
    axes = np.atleast_1d(axes)

    for ax, attr in zip(axes, attrs):
        y = d[attr][idx]
        names, pal = CLASS_NAMES[attr], PALETTES[attr]

        for c in range(len(names)):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=2.5, alpha=0.45,
                       color=pal[c], label=names[c], linewidths=0)

        prov = label_provenance(dataset, attr)
        ax.set_title(f'coloured by {attr.replace("_", " ")}'
                     + (f'\n({prov})' if prov else ''))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.6); s.set_color('#cccccc')

        handles = [Patch(facecolor=pal[c], label=names[c])
                   for c in range(len(names))]
        ax.legend(handles=handles, loc='upper right', frameon=False,
                  fontsize=8.5, markerscale=1)

    fig.suptitle(f'{DATASET_TITLE.get(dataset, "")}: UMAP of the same frozen '
                 f'VAE latent space\n'
                 f'Illustrative only: probe performance is the evidence',
                 y=1.04, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_composition(latents_path, out_path):
    """Dataset composition: the confound-check figure."""
    d = np.load(latents_path, allow_pickle=True)
    n = len(d['mu'])
    dataset = str(d['dataset']) if 'dataset' in d.files else 'unknown'
    attrs = present_targets(d)

    fig, axes = plt.subplots(1, len(attrs), figsize=(4.4 * len(attrs), 4.4))
    axes = np.atleast_1d(axes)

    for ax, attr in zip(axes, attrs):
        y = d[attr]
        names, pal = CLASS_NAMES[attr], PALETTES[attr]
        counts = np.bincount(y, minlength=len(names))
        pct = 100 * counts / n

        bars = ax.bar(range(len(names)), pct, color=pal, width=0.65)
        for b, p, c in zip(bars, pct, counts):
            ax.text(b.get_x() + b.get_width() / 2, p + 1.0,
                    f'{p:.1f}%\n({c})', ha='center', va='bottom', fontsize=8.5)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=0 if len(names) < 5 else 35,
                           ha='center' if len(names) < 5 else 'right')
        ax.set_ylabel('% of dataset' if ax is axes[0] else '')
        ax.set_ylim(0, max(pct) * 1.30)

        prov = label_provenance(dataset, attr)
        ax.set_title(attr.replace('_', ' ') + (f'\n({prov})' if prov else ''))
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    # imbalance ratio of the task, where there is one
    sub = ''
    if 'emotion' in attrs:
        c = np.bincount(d['emotion'], minlength=7)
        sub = f'   |   emotion imbalance ratio {c.max() / c[c > 0].min():.1f}'

    fig.suptitle(f'{DATASET_TITLE.get(dataset, "")} composition '
                 f'(N = {n:,}){sub}', y=1.02, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_lda_projection(latents_path, out_path):
    """
    Project latents onto the linear discriminant axes for each attribute.

    The honest visual counterpart to the linear-probe result: if information is
    linearly decodable, there exist directions along which the classes separate.
    LDA finds exactly those. Unlike UMAP this is a linear projection with no
    free hyperparameters and nothing to tune toward a desired outcome.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    d = np.load(latents_path, allow_pickle=True)
    mu = d['mu']
    split = d['split']
    tr, te = split == 'train', split == 'test'
    dataset = str(d['dataset']) if 'dataset' in d.files else 'unknown'
    attrs = present_targets(d)

    scaler = StandardScaler().fit(mu[tr])
    X_tr, X_te = scaler.transform(mu[tr]), scaler.transform(mu[te])

    fig, axes = plt.subplots(1, len(attrs), figsize=(5.3 * len(attrs), 5.2))
    axes = np.atleast_1d(axes)

    for ax, attr in zip(axes, attrs):
        y_tr, y_te = d[attr][tr], d[attr][te]
        names, pal = CLASS_NAMES[attr], PALETTES[attr]
        n_classes = len(names)

        # LDA yields at most (n_classes - 1) directions; binary gender gives
        # one axis, so plot it as a density instead of a scatter.
        n_comp = min(2, n_classes - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        lda.fit(X_tr, y_tr)
        Z = lda.transform(X_te)   # fit on train, project test: no leakage

        if n_comp == 1:
            for c in range(n_classes):
                m = y_te == c
                ax.hist(Z[m, 0], bins=60, alpha=0.6, color=pal[c],
                        label=names[c], density=True)
            ax.set_xlabel('LD1'); ax.set_ylabel('density'); ax.set_yticks([])
        else:
            for c in range(n_classes):
                m = y_te == c
                ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.5,
                           color=pal[c], label=names[c], linewidths=0)
            ax.set_xlabel('LD1'); ax.set_ylabel('LD2')
            ax.set_xticks([]); ax.set_yticks([])

        prov = label_provenance(dataset, attr)
        ax.set_title(attr.replace('_', ' ') + (f'\n({prov})' if prov else ''))
        ax.legend(loc='upper right', frameon=False, fontsize=8.5, markerscale=2.5)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.6); s.set_color('#cccccc')

    fig.suptitle(f'{DATASET_TITLE.get(dataset, "")}: linear discriminant '
                 f'projection of the frozen VAE latent space\n'
                 f'Directions fitted on probe-train, points shown are held-out test',
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
    parser.add_argument('--suffix', type=str, default='',
                        help='Appended to every filename, e.g. "_rafdb"')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_umap', action='store_true',
                        help='Skip the UMAP panel. It segfaults when '
                             'numba and TensorFlow coexist in the same env.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)
    dataset = results.get('_dataset', 'unknown')
    sfx = args.suffix

    print(f"Generating figures (dataset: {dataset}):")
    fig_probe_performance(results,
                          os.path.join(args.out_dir,
                                       f'fig1_probe_performance{sfx}.png'),
                          dataset)
    fig_composition(args.latents,
                    os.path.join(args.out_dir,
                                 f'fig3_dataset_composition{sfx}.png'))
    fig_lda_projection(args.latents,
                       os.path.join(args.out_dir,
                                    f'fig4_lda_projection{sfx}.png'))
    fig_umap(args.latents,
             os.path.join(args.out_dir, f'fig2_umap_panels{sfx}.png'),
             seed=args.seed)

    if not args.no_umap:
        fig_umap(args.latents,
                 os.path.join(args.out_dir, f'fig2_umap_panels{sfx}.png'),
                 seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
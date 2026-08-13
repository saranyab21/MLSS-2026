"""
Cross-dataset comparison: the figures that turn two experiments into one claim.

Each panel tests the same claim on both corpora, which differ in almost every
way that could confound it:

  UTKFace  23,705 faces   ground-truth demographics   age-bucket task (induced IMR 41.5)
  RAF-DB   15,339 faces   FairFace-inferred labels    emotion task (native IMR 17.0)

Different images, different label provenance, different downstream task. If a
finding survives both, it is not an artifact of either.

Produces:
  figX1_leakage_cross.png    - probe lift over chance, both datasets
  figX2_task_vs_demo.png     - RAF-DB: task AUROC against demographic AUROC
  figX3_federated_cross.png  - change in subgroup gap under three interventions
  figX4_worst_group.png      - who absorbs the disparity, and how it moves
  figX5_removal_cross.png    - adversarial removal ceiling, both datasets
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CB = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
      'red': '#D55E00', 'purple': '#CC79A7', 'sky': '#56B4E9',
      'yellow': '#F0E442', 'grey': '#999999'}

DS = {
    'utkface': {'label': 'UTKFace', 'colour': CB['blue'],
                'note': 'ground-truth labels, age task'},
    'rafdb':   {'label': 'RAF-DB', 'colour': CB['orange'],
                'note': 'FairFace-inferred labels, emotion task'},
}

DEMO = ['gender', 'race', 'age_bucket']
DEMO_LABEL = {'gender': 'Gender', 'race': 'Race', 'age_bucket': 'Age'}

COND = ['none', 'fedar', 'demo_bal']
COND_LABEL = {'none': 'No intervention',
              'fedar': 'FedAR resampling',
              'demo_bal': 'Demographic balancing'}

LAMS = [1, 5, 20, 50]

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def probe_lift(probes, metric='balanced_accuracy'):
    """Best probe's margin over chance, and its AUROC if available."""
    best = max(probes['probes'].values(), key=lambda r: r[metric])
    chance = 1.0 / probes['n_classes']
    auroc = max((r['auroc'] for r in probes['probes'].values()
                 if r.get('auroc') is not None), default=None)
    return {'value': best[metric], 'chance': chance,
            'lift': best[metric] - chance, 'auroc': auroc}


def load_debias_sweep(results_dir, dataset, lams=(1, 5, 20, 50), metric='LogReg'):
    """Removal per lambda, recomputed on the deterministic linear probe."""
    out = []
    for lam in lams:
        if dataset == 'rafdb':
            path = os.path.join(results_dir, f'debias_rafdb_lam{lam}.json')
        else:
            path = os.path.join(results_dir,
                                'debias_results.json' if lam == 1
                                else f'debias_lam{lam}.json')
        d = load_json(path)
        if d is None:
            continue
        rec = {'lambda': lam, 'recon': d['history'][-1]['recon'], 'removed': {}}
        for a in DEMO:
            b, af = d['before'][a][metric], d['after'][a][metric]
            ch = d['before'][a]['chance']
            rec['removed'][a] = (b - af) / (b - ch) * 100 if b > ch else 0.0
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_leakage_cross(probes, out_path):
    """
    Probe lift over chance for each demographic attribute, both datasets.

    Lift rather than raw accuracy, because the two corpora have different
    class counts and therefore different chance floors.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    width = 0.36
    x = np.arange(len(DEMO))

    for i, (ds, meta) in enumerate(DS.items()):
        if ds not in probes:
            continue
        lifts, texts = [], []
        for a in DEMO:
            r = probe_lift(probes[ds][a])
            lifts.append(r['lift'])
            texts.append(f"{r['value']:.3f}\n(ch {r['chance']:.2f})")
        off = (i - 0.5) * width
        bars = ax.bar(x + off, lifts, width, color=meta['colour'],
                      label=f"{meta['label']}  ·  {meta['note']}")
        for b, t in zip(bars, texts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    t, ha='center', va='bottom', fontsize=8.5)

    ax.axhline(0, color=CB['red'], linestyle='--', linewidth=1.8)
    ax.text(len(DEMO) - 0.45, 0.006, 'chance', color=CB['red'], fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([DEMO_LABEL[a] for a in DEMO])
    ax.set_ylabel('balanced accuracy above chance')
    ax.set_title('Demographic leakage replicates across corpora\n'
                 'best of three probes on a frozen reconstruction-only VAE latent',
                 pad=12)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_task_vs_demo(probes_rafdb, out_path):
    """
    RAF-DB only: the figure the poster abstract was asking for.

    AUROC is the right scale here because it is comparable across targets with
    different class counts, unlike accuracy.
    """
    targets = ['emotion'] + DEMO
    labels = ['Emotion\n(the task)', 'Gender', 'Race', 'Age']
    colours = [CB['purple'], CB['blue'], CB['orange'], CB['green']]

    aurocs, accs, chances = [], [], []
    for t in targets:
        r = probe_lift(probes_rafdb[t])
        aurocs.append(r['auroc'] if r['auroc'] is not None else np.nan)
        accs.append(r['value'])
        chances.append(r['chance'])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(targets))
    bars = ax.bar(x, aurocs, 0.6, color=colours)

    for b, a, acc, ch in zip(bars, aurocs, accs, chances):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.012, f'{a:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='medium')
        ax.text(b.get_x() + b.get_width() / 2, 0.06,
                f'bal acc {acc:.3f}\nchance {ch:.2f}',
                ha='center', fontsize=8.5, color='white')

    ax.axhline(0.5, color=CB['red'], linestyle='--', linewidth=1.8)
    ax.text(len(targets) - 0.42, 0.515, 'AUROC chance', color=CB['red'],
            fontsize=10)

    # the band the demographic bars occupy, to make the closeness legible
    lo, hi = min(aurocs[1:]), max(aurocs[1:])
    ax.axhspan(lo, hi, color=CB['grey'], alpha=0.10, zorder=0)
    ax.annotate('', xy=(3.42, lo), xytext=(3.42, hi),
                arrowprops=dict(arrowstyle='<->', color=CB['grey'], lw=1.2))
    ax.text(3.5, (lo + hi) / 2, f'demographic\nband\n{lo:.2f}–{hi:.2f}',
            fontsize=9, color='#555555', va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('AUROC (macro, one-vs-rest)')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, 4.1)
    ax.set_title('RAF-DB: the encoder separates race almost as well as it\n'
                 'separates the emotion it was built to represent',
                 pad=12)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_federated_cross(fed, out_path):
    """
    Change in subgroup gap under each intervention, both datasets.
    Positive means the gap widened.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    width = 0.36
    x = np.arange(len(COND))

    for i, (ds, meta) in enumerate(DS.items()):
        if ds not in fed:
            continue
        res = fed[ds]['results']
        base_gap = res['none']['final']['gap']
        base_bal = res['none']['final']['balanced_acc']
        d_gap = [res[c]['final']['gap'] - base_gap for c in COND]
        d_bal = [res[c]['final']['balanced_acc'] - base_bal for c in COND]
        off = (i - 0.5) * width

        for ax, vals in ((ax1, d_gap), (ax2, d_bal)):
            bars = ax.bar(x + off, vals, width, color=meta['colour'],
                          label=meta['label'])
            for b, v in zip(bars, vals):
                if abs(v) < 1e-9:
                    continue
                va = 'bottom' if v >= 0 else 'top'
                pad = 0.0012 if v >= 0 else -0.0012
                ax.text(b.get_x() + b.get_width() / 2, v + pad, f'{v:+.3f}',
                        ha='center', va=va, fontsize=8.5)

    for ax, ylab, title in (
        (ax1, 'change in subgroup gap', 'The gap does not close'),
        (ax2, 'change in balanced accuracy', 'and utility does not improve'),
    ):
        ax.axhline(0, color='#333333', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c].replace(' ', '\n') for c in COND])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    ax1.text(0.02, 0.96, 'above zero = worse', transform=ax1.transAxes,
             fontsize=9.5, color=CB['red'], va='top')
    ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.90))

    fig.suptitle('Label-distribution interventions do not reach '
                 'representation-level disparity\n'
                 'relative to no intervention, on both corpora',
                 y=1.03, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_worst_group(fed, out_path):
    """
    Worst-group accuracy and which group it is, per condition.

    The naming matters: the interventions relocate who absorbs the disparity
    rather than reducing it.
    """
    datasets = [ds for ds in DS if ds in fed]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(6.2 * len(datasets), 5.2))
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, datasets):
        res = fed[ds]['results']
        x = np.arange(len(COND))
        worst, best, names = [], [], []
        for c in COND:
            f = res[c]['final']
            worst.append(f['worst_group'])
            best.append(f['best_group'])
            names.append(min(f['groups'], key=lambda k: f['groups'][k]['acc']))

        ax.bar(x, best, 0.55, color=CB['grey'], alpha=0.35, label='best group')
        bars = ax.bar(x, worst, 0.55, color=DS[ds]['colour'], label='worst group')

        for b, w, bst, nm in zip(bars, worst, best, names):
            cx = b.get_x() + b.get_width() / 2
            ax.text(cx, w - 0.028, f'{w:.3f}', ha='center', va='top',
                    fontsize=9.5, color='white', fontweight='medium')
            ax.text(cx, w + 0.012, nm, ha='center', va='bottom',
                    fontsize=9, color='#333333')
            ax.annotate('', xy=(cx + 0.30, w), xytext=(cx + 0.30, bst),
                        arrowprops=dict(arrowstyle='<->', color='#666666', lw=1))
            ax.text(cx + 0.33, (w + bst) / 2, f'{bst - w:.3f}',
                    fontsize=8.5, color='#666666', va='center')

        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c].replace(' ', '\n') for c in COND])
        ax.set_ylabel('accuracy' if ax is axes[0] else '')
        ax.set_ylim(0, max(best) * 1.22)
        ax.set_title(f"{DS[ds]['label']}  ·  {DS[ds]['note']}")
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(frameon=False, loc='lower right')

    fig.suptitle('The interventions relocate the disparity rather than reduce it\n'
                 'labels name the subgroup with the lowest accuracy',
                 y=1.03, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")

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


def fig_removal_cross(sweeps, out_path):
    """
    Adversarial removal against lambda, both datasets, one panel per attribute.
    Full 0-100 scale so the distance from complete removal is unmissable.
    """
    fig, axes = plt.subplots(1, len(DEMO), figsize=(5.2 * len(DEMO), 5.0),
                             sharey=True)

    all_vals = [v for s in sweeps.values() for r in s
                for v in r['removed'].values()]
    lo = min(-5, np.floor(min(all_vals) / 5) * 5)

    for ax, attr in zip(axes, DEMO):
        for ds, meta in DS.items():
            if ds not in sweeps or not sweeps[ds]:
                continue
            runs = sweeps[ds]
            lams = [r['lambda'] for r in runs]
            vals = [r['removed'][attr] for r in runs]
            ax.plot(lams, vals, marker='o', color=meta['colour'],
                    label=meta['label'], linewidth=2, markersize=7)

        ax.axhline(0, color='#333333', linewidth=1.1)
        ax.axhline(100, color=CB['grey'], linestyle=':', linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xticks([1, 5, 20, 50])
        ax.set_xticklabels(['1', '5', '20', '50'])
        ax.set_xlabel('adversarial strength $\\lambda$')
        ax.set_ylim(lo, 105)
        ax.set_title(DEMO_LABEL[attr])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('above-chance signal removed (%)')
    axes[0].text(1, 97, 'complete removal', fontsize=9.5,
                 color=CB['grey'], va='top')
    axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.88))
    if lo < 0:
        axes[-1].text(50, lo + 2, 'below zero:\nmore recoverable\nthan before',
                      fontsize=9, color=CB['red'], ha='right', va='bottom')

    fig.suptitle('Gradient-reversal removal never approaches complete, '
                 'on either corpus\n'
                 'measured with a freshly initialised probe after the '
                 'adversary is discarded',
                 y=1.04, fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='../../results')
    p.add_argument('--out_dir', default='../../figures')
    p.add_argument('--metric', choices=['LogReg', 'MLP'], default='LogReg',
                   help='Probe defining removal in the sweep panel. LogReg is '
                        'deterministic and matches fig1.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    R = args.results_dir

    probes = {}
    for ds, fname in (('utkface', 'probe_results.json'),
                      ('rafdb', 'probe_results_rafdb.json')):
        d = load_json(os.path.join(R, fname))
        if d:
            probes[ds] = d

    fed = {}
    for ds, fname in (('utkface', 'federated_skewed.json'),
                      ('rafdb', 'federated_rafdb.json')):
        d = load_json(os.path.join(R, fname))
        if d:
            fed[ds] = d

#    sweeps = {ds: load_debias_sweep(R, ds, metric=args.metric)
#              for ds in DS}

    sweeps = {ds: load_multiseed(R, ds) for ds in DS}

    print("Generating cross-dataset figures:")
    if len(probes) == 2:
        fig_leakage_cross(probes, os.path.join(args.out_dir,
                                               'figX1_leakage_cross.png'))
    if 'rafdb' in probes and 'emotion' in probes['rafdb']:
        fig_task_vs_demo(probes['rafdb'],
                         os.path.join(args.out_dir, 'figX2_task_vs_demo.png'))
    if len(fed) >= 1:
        fig_federated_cross(fed, os.path.join(args.out_dir,
                                              'figX3_federated_cross.png'))
        fig_worst_group(fed, os.path.join(args.out_dir,
                                          'figX4_worst_group.png'))
    if any(sweeps.values()):
        fig_removal_cross_multiseed(sweeps, os.path.join(args.out_dir,
                                               'figX5_removal_cross.png'))

    # ---- console summary, ready to paste into the poster ----
    print("\n" + "=" * 78)
    print("CROSS-DATASET SUMMARY")
    print("=" * 78)

    print("\nLeakage (best probe, balanced accuracy above chance)")
    print(f"  {'':<12}" + "".join(f"{DS[d]['label']:>14}" for d in probes))
    for a in DEMO:
        row = f"  {DEMO_LABEL[a]:<12}"
        for ds in probes:
            r = probe_lift(probes[ds][a])
            row += f"{r['lift']:>+13.3f} "
        print(row)

    if 'rafdb' in probes and 'emotion' in probes['rafdb']:
        e = probe_lift(probes['rafdb']['emotion'])
        print(f"\n  RAF-DB emotion (the task): {e['value']:.3f} "
              f"(lift {e['lift']:+.3f}, AUROC {e['auroc']:.3f})")
        aur = [probe_lift(probes['rafdb'][a])['auroc'] for a in DEMO]
        print(f"  demographic AUROC band:    {min(aur):.3f}–{max(aur):.3f}")

    print("\nFederated: change in subgroup gap vs no intervention")
    print(f"  {'':<24}" + "".join(f"{DS[d]['label']:>14}" for d in fed))
    for c in COND[1:]:
        row = f"  {COND_LABEL[c]:<24}"
        for ds in fed:
            res = fed[ds]['results']
            row += f"{res[c]['final']['gap'] - res['none']['final']['gap']:>+13.4f} "
        print(row)

    print("\nWorst group per condition")
    for ds in fed:
        res = fed[ds]['results']
        print(f"  {DS[ds]['label']}")
        for c in COND:
            f = res[c]['final']
            nm = min(f['groups'], key=lambda k: f['groups'][k]['acc'])
            print(f"    {COND_LABEL[c]:<24} {f['worst_group']:.4f}  {nm}")

    print("\nPeak adversarial removal (fresh probe, "
          f"{args.metric})")
    for ds, runs in sweeps.items():
        # guard: skip empty or malformed sweeps (e.g. missing single-seed files)
        if not runs or not all(isinstance(r, dict) and 'removed' in r for r in runs):
            print(f"  {DS[ds]['label']:<10} (no valid debias sweep found — skipped)")
            continue
        best = max(runs, key=lambda r: max(r['removed'].values()))
        a = max(best['removed'], key=best['removed'].get)
        print(f"  {DS[ds]['label']:<10} {best['removed'][a]:5.1f}%  "
              f"({DEMO_LABEL[a]}, lambda={best['lambda']:.0f})")
        neg = [(r['lambda'], k, v) for r in runs
               for k, v in r['removed'].items() if v < 0]
        for lam, k, v in neg:
            print(f"             {v:+5.1f}%  ({DEMO_LABEL[k]}, lambda={lam:.0f})")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
"""
Gap 2: does the VAE encode demographics, or merely fail to destroy them?

Runs the identical probe suite on raw downsampled pixels and compares against
the VAE latent. The decisive condition is PCA-128: a generic linear compression
to exactly the latent's dimensionality. If the VAE latent does not beat PCA-128,
then the encoder is not doing anything demographic-specific and the claim must
be stated as preservation rather than learning.

Same split, same probes, same metric as probes.py. Sample order is verified
against the cached filenames before anything is computed.
"""

from fedar_common.data import build_dataset
import os
import json
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fedar_common.probing import make_probes, evaluate
from probes import TARGETS_COMMON, TARGET_EMOTION  # target definitions only

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)




def flatten_images(args, size, grayscale):
    """Load every image at a low resolution and flatten to a feature vector."""
    ops = [transforms.Resize((size, size))]
    if grayscale:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.append(transforms.ToTensor())
    tf = transforms.Compose(ops)

    ds, collate = build_dataset(args, tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    chunks = []
    for x, _ in loader:
        chunks.append(x.flatten(1).numpy())
    X = np.concatenate(chunks, axis=0)
    return X, ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--dataset', choices=['utkface', 'rafdb'], default='utkface')
    p.add_argument('--demographics', default=None)
    p.add_argument('--latents', default='../../latents/utkface_latents.npz')
    p.add_argument('--output', default='../../results/pixel_baseline.json')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    d = np.load(args.latents, allow_pickle=True)
    split = d['split']
    tr, te = split == 'train', split == 'test'
    latent_dim = d['mu'].shape[1]

    targets = dict(TARGETS_COMMON)
    if 'emotion' in d.files:
        targets = {**TARGET_EMOTION, **TARGETS_COMMON}

    print(f"Dataset: {args.dataset}  |  latent dim {latent_dim}")
    print(f"  probe-train {tr.sum()}  |  probe-test {te.sum()}\n")

    # ---- build the pixel representations ----
    print("Loading images...")
    X16g, ds = flatten_images(args, 16, grayscale=True)     # 256
    X32c, _ = flatten_images(args, 32, grayscale=False)     # 3072

    # order check: the pixel matrices must align with the cached latents
    names = ds.get_label_arrays()['filename']
    if not np.array_equal(names, d['filename']):
        raise RuntimeError(
            "Sample order does not match the cached latents. Aborting: every "
            "comparison below would be meaningless."
        )
    print(f"  order verified against cached filenames ({len(names)} samples)")

    # PCA fitted on train only, to exactly the latent dimensionality
    print(f"  fitting PCA to {latent_dim} components on the train split...")
    pca_g = PCA(n_components=latent_dim, random_state=args.seed).fit(X16g[tr])
    pca_c = PCA(n_components=latent_dim, random_state=args.seed).fit(X32c[tr])
    Xg_pca = pca_g.transform(X16g)
    Xc_pca = pca_c.transform(X32c)
    print(f"    16x16 grey  -> PCA{latent_dim}: "
          f"{100*pca_g.explained_variance_ratio_.sum():.1f}% variance retained")
    print(f"    32x32 RGB   -> PCA{latent_dim}: "
          f"{100*pca_c.explained_variance_ratio_.sum():.1f}% variance retained\n")

    reps = {
        f'VAE latent ({latent_dim})': d['mu'],
        f'PCA{latent_dim} of 16x16 grey': Xg_pca,
        f'PCA{latent_dim} of 32x32 RGB': Xc_pca,
        'raw 16x16 grey (256)': X16g,
    }

    # ---- identical probe suite on every representation ----
    results = {}
    for rep_name, X in reps.items():
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])

        print("=" * 72)
        print(f"REPRESENTATION: {rep_name}   (dim {X.shape[1]})")
        print("=" * 72)

        results[rep_name] = {'dim': int(X.shape[1]), 'targets': {}}
        for target, meta in targets.items():
            y = d[target]
            y_tr, y_te = y[tr], y[te]
            n_cls = meta['n_classes']

            best_name, best = None, None
            for name, clf in make_probes(args.seed).items():
                r = evaluate(clf, X_tr, y_tr, X_te, y_te, n_cls)
                if best is None or r['balanced_accuracy'] > best['balanced_accuracy']:
                    best_name, best = name, r

            chance = 1.0 / n_cls
            results[rep_name]['targets'][target] = {
                'best_probe': best_name,
                'balanced_accuracy': best['balanced_accuracy'],
                'auroc': best['auroc'],
                'chance': chance,
                'lift': best['balanced_accuracy'] - chance,
            }
            auroc_s = f"  auroc {best['auroc']:.4f}" if best['auroc'] else ""
            print(f"  {target:<12} {best['balanced_accuracy']:.4f} "
                  f"({best_name})  lift {best['balanced_accuracy']-chance:+.4f}"
                  f"{auroc_s}")
        print()

    # ---- the comparison table ----
    vae_key = f'VAE latent ({latent_dim})'
    pca_keys = [k for k in reps if k.startswith('PCA')]

    print("=" * 84)
    print("LIFT OVER CHANCE, by representation")
    print("-" * 84)
    hdr = f"{'target':<12}" + "".join(f"{k[:22]:>22}" for k in reps)
    print(hdr)
    for target in targets:
        row = f"{target:<12}"
        for k in reps:
            row += f"{results[k]['targets'][target]['lift']:>+21.4f} "
        print(row)

    print("\n" + "-" * 84)
    print("VAE ADVANTAGE over dimension-matched PCA (lift difference)")
    print("-" * 84)
    verdicts = {}
    for target in targets:
        v = results[vae_key]['targets'][target]['lift']
        best_pca = max(results[k]['targets'][target]['lift'] for k in pca_keys)
        adv = v - best_pca
        verdicts[target] = adv
        flag = ('VAE carries more' if adv > 0.02
                else 'comparable' if adv > -0.02
                else 'PCA carries more')
        print(f"  {target:<12} {adv:+.4f}   {flag}")

    demo = [t for t in targets if t != 'emotion']
    mean_adv = float(np.mean([verdicts[t] for t in demo]))
    print("\n" + "=" * 84)
    print(f"Mean VAE advantage on demographic targets: {mean_adv:+.4f}")
    if mean_adv > 0.02:
        print("  -> the encoder carries MORE demographic signal than generic")
        print("     compression of the same dimensionality. 'The VAE encodes")
        print("     demographics' is supported.")
    elif mean_adv > -0.02:
        print("  -> comparable to generic compression. State the claim as")
        print("     PRESERVATION, not learning: compression to 128 dimensions")
        print("     did not remove what was already linearly present in pixels.")
        print("     This still undercuts FedAR's implicit privacy assumption.")
    else:
        print("  -> PCA carries more. The VAE partially SUPPRESSES demographic")
        print("     signal relative to generic compression. Reframe accordingly:")
        print("     the residual leakage is the finding, not the encoder's role.")
    print("=" * 84 + "\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'dataset': args.dataset, 'latent_dim': int(latent_dim),
                   'results': results, 'vae_advantage': verdicts,
                   'mean_demographic_advantage': mean_adv}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

"""
Experiment C: adversarial removal of demographic information.

Fine-tunes the pretrained VAE encoder with a gradient reversal layer feeding
demographic adversaries, then freezes it and trains fresh probes from scratch.

The fresh-probe check is the point. Elazar & Goldberg (2018) showed that
adversarially removed attributes are frequently recoverable by a newly
initialised probe: the adversary was defeated, the information was not.

Loss:  L = L_recon + beta * L_KL - lambda * L_demographic
where the negative sign is implemented by the gradient reversal layer.

Works on either corpus. Only the demographic attributes are adversarially
targeted; on RAF-DB the emotion label is carried through untouched so the
downstream task can be re-evaluated on the debiased latents.
"""

import os
import json
import copy
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from model import VAE, vae_loss
from utkface_dataset import UTKFaceDataset, collate_labels
from rafdb_dataset import RAFDBDataset, collate_labels as collate_rafdb

# Attributes the adversaries try to strip. Emotion is deliberately excluded:
# it is the task we want to preserve, not remove.
ATTRS = {'gender': 2, 'race': 5, 'age_bucket': 4}


# --------------------------------------------------------------------------
# gradient reversal
# --------------------------------------------------------------------------

class GradientReversal(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward (Ganin & Lempitsky)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversal.apply(x, lambda_)


class Adversary(nn.Module):
    """Predicts a demographic attribute from the latent."""

    def __init__(self, latent_dim, n_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z):
        return self.net(z)


# --------------------------------------------------------------------------
# dataset construction
# --------------------------------------------------------------------------

def build_dataset(args, transform):
    """Return (dataset, collate_fn) for whichever corpus was requested."""
    if args.dataset == 'rafdb':
        if not args.demographics:
            raise ValueError("--demographics is required for --dataset rafdb")
        ds = RAFDBDataset(args.data_root, args.demographics,
                          transform=transform, split=None)
        return ds, collate_rafdb
    ds = UTKFaceDataset(args.data_root, transform=transform)
    return ds, collate_labels


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_debiased(model, adversaries, loader, device, args):
    """Fine-tune the VAE while adversaries fight the encoder through the GRL."""
    model.train()
    for a in adversaries.values():
        a.train()

    opt_vae = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_adv = torch.optim.Adam(
        [p for a in adversaries.values() for p in a.parameters()], lr=args.adv_lr
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        agg = {'loss': 0., 'recon': 0., 'kl': 0.,
               **{f'adv_{k}': 0. for k in adversaries}}
        n = 0

        # ramp lambda from 0 so the encoder learns to reconstruct before
        # the adversarial pressure kicks in; standard DANN practice
        for x, labels in loader:
            p = (epoch - 1) / max(1, args.epochs - 1)
            lam = args.lambda_max * (2. / (1. + np.exp(-10 * p)) - 1.)

            x = x.to(device, non_blocking=True)
            mu, logvar = model.encoder(x)
            z = model.reparameterize(mu, logvar)
            recon = model.decoder(z)

            vae_l, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=args.beta)

            # adversaries read the reversed latent
            z_rev = grad_reverse(mu, lam)
            adv_total = 0.
            adv_parts = {}
            for attr, adv in adversaries.items():
                y = labels[attr].to(device)
                l = F.cross_entropy(adv(z_rev), y)
                adv_total = adv_total + l
                adv_parts[attr] = l.item()

            total = vae_l + adv_total

            opt_vae.zero_grad()
            opt_adv.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_vae.step()
            opt_adv.step()

            agg['loss'] += total.item()
            agg['recon'] += recon_l
            agg['kl'] += kl_l
            for k, v in adv_parts.items():
                agg[f'adv_{k}'] += v
            n += 1

        for k in agg:
            agg[k] /= n
        agg['epoch'] = epoch
        agg['lambda'] = float(lam)
        agg['time'] = time.time() - t0
        history.append(agg)

        adv_str = "  ".join(f"{k[4:]} {agg[k]:.3f}"
                            for k in agg if k.startswith('adv_'))
        print(f"  epoch {epoch:2d} | lam {lam:.3f} | recon {agg['recon']:7.2f} | "
              f"kl {agg['kl']:6.2f} | {adv_str} | {agg['time']:.1f}s", flush=True)

    return history


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def extract(model, loader, device, keys):
    model.eval()
    mus, labs = [], {k: [] for k in keys}
    for x, labels in loader:
        mu, _ = model.encoder(x.to(device, non_blocking=True))
        mus.append(mu.cpu().numpy())
        for k in keys:
            labs[k].append(labels[k].numpy())
    return (np.concatenate(mus),
            {k: np.concatenate(v) for k, v in labs.items()})


def fresh_probe(X_tr, y_tr, X_te, y_te, seed):
    """
    Train a probe from scratch on the frozen representation.
    This is the honest test: the adversary is gone, only the encoder remains.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    sc = StandardScaler().fit(X_tr)
    A, B = sc.transform(X_tr), sc.transform(X_te)

    out = {}
    lr = LogisticRegression(max_iter=2000, random_state=seed).fit(A, y_tr)
    out['LogReg'] = balanced_accuracy_score(y_te, lr.predict(B))

    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                        early_stopping=True, random_state=seed).fit(A, y_tr)
    out['MLP'] = balanced_accuracy_score(y_te, mlp.predict(B))
    return out


def evaluate_from_saved(npz, seed, targets):
    """
    Score the BEFORE-debiasing encoder against the *saved* latents.

    ---------------------------------------------------------------------------
    WHY THIS EXISTS (baseline-consistency patch)
    ---------------------------------------------------------------------------
    fig1 (probes.py) reads the deterministic `mu` written once by
    extract_latents.py. Re-running model.encoder(x) at eval time is not
    bit-identical (nondeterministic cuDNN conv kernels, different DataLoader
    batching), so a live-extracted "before" baseline drifts from fig1 by a few
    thousandths of balanced accuracy. On an A0 poster where both panels are
    visible, that unexplained gap is an easy reviewer target.

    Reading the identical saved `mu`/`split`/labels makes the baseline equal
    fig1 by construction. The AFTER measurement still re-extracts live, because
    the encoder weights have changed and no saved `mu` exists for it.
    ---------------------------------------------------------------------------
    """
    mu = npz['mu']
    split = npz['split']
    tr, te = split == 'train', split == 'test'

    print("\n  baseline: fresh probes on SAVED frozen latents (matches fig1)")
    res = {}
    for attr, n_cls in targets.items():
        y = npz[attr]
        r = fresh_probe(mu[tr], y[tr], mu[te], y[te], seed)
        chance = 1. / n_cls
        res[attr] = {**r, 'chance': chance}
        print(f"    {attr:<11} LogReg {r['LogReg']:.4f}  MLP {r['MLP']:.4f}  "
              f"(chance {chance:.4f})")
    return res


def evaluate_encoder(model, tr_loader, te_loader, device, seed, tag, targets):
    """Extract latents and run fresh probes for every target."""
    keys = list(targets.keys())
    X_tr, y_tr = extract(model, tr_loader, device, keys)
    X_te, y_te = extract(model, te_loader, device, keys)

    print(f"\n  {tag}: fresh probes on frozen latents")
    res = {}
    for attr, n_cls in targets.items():
        r = fresh_probe(X_tr, y_tr[attr], X_te, y_te[attr], seed)
        chance = 1. / n_cls
        res[attr] = {**r, 'chance': chance}
        print(f"    {attr:<11} LogReg {r['LogReg']:.4f}  MLP {r['MLP']:.4f}  "
              f"(chance {chance:.4f})")
    return res, (X_tr, y_tr, X_te, y_te)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--dataset', choices=['utkface', 'rafdb'], default='utkface')
    p.add_argument('--demographics', type=str, default=None,
                   help='Path to rafdb_demographics.csv (rafdb only)')
    p.add_argument('--checkpoint', default='../../checkpoints/vae_best.pt')
    p.add_argument('--reference_latents', default=None,
                   help='Saved latents used for the BEFORE baseline. '
                        'Defaults to the matching dataset file.')
    p.add_argument('--output', default='../../results/debias_results.json')
    p.add_argument('--latent_out', default='../../latents/utkface_debiased.npz')
    p.add_argument('--ckpt_out', default='../../checkpoints/vae_debiased.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--adv_lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--lambda_max', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.reference_latents is None:
        args.reference_latents = (f'../../latents/{args.dataset}_latents.npz')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  dataset {args.dataset}  |  "
          f"lambda_max {args.lambda_max}\n")

    tf = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
    ds, collate = build_dataset(args, tf)

    # reuse the exact split AND saved latents from extract_latents so the
    # before-debiasing baseline is identical to fig1
    ref = np.load(args.reference_latents, allow_pickle=True)
    split = ref['split']
    if len(split) != len(ds):
        raise RuntimeError(
            f"Reference latents have {len(split)} rows but the dataset has "
            f"{len(ds)}. Check --reference_latents matches --dataset."
        )
    tr_idx = np.where(split == 'train')[0]
    te_idx = np.where(split == 'test')[0]

    tr_ds, te_ds = Subset(ds, tr_idx), Subset(ds, te_idx)

    def mk(dset, shuffle):
        return DataLoader(dset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate)

    train_loader = mk(tr_ds, True)
    tr_eval, te_eval = mk(tr_ds, False), mk(te_ds, False)
    print(f"train {len(tr_ds)} | test {len(te_ds)}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt['args']['latent_dim']
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded VAE from epoch {ckpt['epoch']}, latent_dim {latent_dim}")

    # Track the task label too, where present, so we can measure whether
    # adversarial pressure on demographics also damages the task signal.
    has_emotion = args.dataset == 'rafdb' and 'emotion' in ref.files
    eval_targets = dict(ATTRS)
    if has_emotion:
        eval_targets['emotion'] = 7
        print("Emotion present: tracked as a utility target, not adversarially removed.")

    results = {'args': vars(args), 'dataset': args.dataset}

    # --- before ---
    print("\n" + "=" * 70)
    print("BEFORE DEBIASING")
    print("=" * 70)
    results['before'] = evaluate_from_saved(ref, args.seed, eval_targets)

    # --- adversarial fine-tuning ---
    print("\n" + "=" * 70)
    print("ADVERSARIAL FINE-TUNING")
    print("=" * 70)
    advs = {a: Adversary(latent_dim, n).to(device) for a, n in ATTRS.items()}
    results['history'] = train_debiased(model, advs, train_loader, device, args)

    # --- after ---
    print("\n" + "=" * 70)
    print("AFTER DEBIASING (fresh probes, adversaries discarded)")
    print("=" * 70)
    results['after'], (Xtr, ytr, Xte, yte) = evaluate_encoder(
        model, tr_eval, te_eval, device, args.seed, 'debiased', eval_targets)

    # --- verdict ---
    print("\n" + "=" * 70)
    print(f"{'target':<13}{'chance':>9}{'before':>10}{'after':>10}"
          f"{'removed':>10}{'residual':>11}")
    print("-" * 70)
    for attr in eval_targets:
        ch = results['before'][attr]['chance']
        b = results['before'][attr]['MLP']
        a = results['after'][attr]['MLP']
        removed = (b - a) / (b - ch) * 100 if b > ch else 0.
        residual = a - ch
        results['after'][attr]['pct_removed'] = float(removed)
        results['after'][attr]['residual_above_chance'] = float(residual)
        tag = '  <- task, not targeted' if attr == 'emotion' else ''
        print(f"{attr:<13}{ch:>9.4f}{b:>10.4f}{a:>10.4f}"
              f"{removed:>9.1f}%{residual:>+11.4f}{tag}")
    print("=" * 70)
    print("'removed' = fraction of above-chance signal eliminated.")
    print("'residual' = what a fresh probe still recovers above chance.")
    if has_emotion:
        print("Emotion is reported for utility: removal there is collateral damage,")
        print("not success.")
    print()

    # save debiased latents so downstream experiments can reuse them
    os.makedirs(os.path.dirname(args.latent_out), exist_ok=True)
    mu_all = np.zeros((len(ds), latent_dim), dtype=np.float32)
    mu_all[tr_idx], mu_all[te_idx] = Xtr, Xte
    payload = {
        'mu': mu_all,
        'age_bucket': ref['age_bucket'], 'race': ref['race'],
        'gender': ref['gender'], 'age': ref['age'],
        'filename': ref['filename'], 'split': split,
        'dataset': args.dataset,
    }
    if has_emotion:
        payload['emotion'] = ref['emotion']
    np.savez_compressed(args.latent_out, **payload)
    print(f"Saved debiased latents to {args.latent_out}")

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'args': {'latent_dim': latent_dim, **vars(args)}},
               args.ckpt_out)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
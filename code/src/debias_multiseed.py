"""
Gap 2: seed variance for the removal experiment (Experiments C/D).

debias.py reports a single run at seed=42 for each lambda. The federated
experiment (federated_multiseed.py) taught us that at this data scale a
point estimate cannot distinguish "the intervention does X" from "seed noise
happens to look like X" -- the RAF-DB gap variance alone was 17% of the gap.

The removal sweep has the same exposure. In particular the single-seed run
showed, at lambda=50 on RAF-DB, race and age coming back *more* recoverable
than before debiasing (pct_removed of -1.7% and -6.6%). That is either a real
destabilisation effect or a one-seed artefact, and a point estimate cannot
tell us which. This script settles it the same way federated_multiseed.py
settled the gap-widening claim: run the identical, unmodified removal pipeline
across several seeds and report per-seed sign consistency, leading with signs
over p-values because n is small.

Design note. This is a *uniform* sweep: every lambda is run across every seed
with exactly the same code. It does not special-case lambda=50. That is
deliberate -- singling out the setting we already find interesting would bias
the design. The lambda=50 negative-removal result then either survives (removal
is negative in most seeds) or dissolves (it spans zero), and either way we can
state it honestly on the poster.

It reuses debias.py's own building blocks unchanged -- the GRL, the adversaries,
the fine-tuning loop, the saved-latent baseline, and the fresh-probe evaluation
-- so a single-seed slice of this script's output is bit-identical to running
debias.py directly. Nothing about the removal method is re-implemented here;
only the seed/lambda orchestration and the paired aggregation are new.

The headline metric is locked to the deterministic LogReg probe (matching the
figures, per the README note that MLPClassifier's internal validation split is
not bit-reproducible). The MLP number is also recorded for reference.

Usage mirrors federated_multiseed.py. UTKFace:

    python debias_multiseed.py \
        --data_root ../../data/utkface \
        --checkpoint ../../checkpoints/vae_best.pt \
        --reference_latents ../../latents/utkface_latents.npz \
        --output ../../results/debias_multiseed_utkface.json

RAF-DB:

    python debias_multiseed.py --dataset rafdb \
        --data_root ../../data/rafdb_raw \
        --demographics ../../data/rafdb_demographics.csv \
        --checkpoint ../../checkpoints/rafdb/vae_best.pt \
        --reference_latents ../../latents/rafdb_latents.npz \
        --output ../../results/debias_multiseed_rafdb.json

Runtime is the single-seed cost times (#seeds x #lambdas). With the defaults
(5 seeds x 4 lambdas = 20 fine-tuning runs of 20 epochs) budget a couple of
hours on the RTX 2080, the same order as the eight original debias runs plus
the federated multi-seed sweep. Use --seeds / --lambdas to trim while testing.
"""

import os
import json
import time
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from model import VAE
# Reuse debias.py's own components unchanged. If any of these move, this import
# breaks loudly rather than silently diverging from the single-seed pipeline.
from debias import (
    ATTRS,
    Adversary,
    build_dataset,
    train_debiased,
    evaluate_from_saved,
    evaluate_encoder,
)


# The metric the poster figures are locked to. LogReg is deterministic; the
# README explains why MLP is not used as the headline removal number.
HEADLINE_PROBE = 'LogReg'


def run_single(args, seed, lambda_max, ref, ds, collate, split,
               tr_idx, te_idx, device, eval_targets, latent_dim):
    """
    One (seed, lambda) removal run, reproducing debias.py.main() exactly for a
    single setting, and returning the per-target removal summary.

    Every step here is the same call debias.py makes; only the surrounding loop
    is new. torch.manual_seed(seed) is set before the adversaries and encoder
    fine-tuning so the run is reproducible per seed, matching debias.py.
    """
    torch.manual_seed(seed)

    run_args = SimpleNamespace(
        data_root=args.data_root, dataset=args.dataset,
        demographics=args.demographics, checkpoint=args.checkpoint,
        reference_latents=args.reference_latents,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, adv_lr=args.adv_lr, beta=args.beta,
        lambda_max=float(lambda_max), num_workers=args.num_workers,
        seed=seed,
    )

    # data loaders over the frozen split, identical to debias.py
    tr_ds, te_ds = Subset(ds, tr_idx), Subset(ds, te_idx)

    def mk(dset, shuffle):
        return DataLoader(dset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate)

    train_loader = mk(tr_ds, True)
    tr_eval, te_eval = mk(tr_ds, False), mk(te_ds, False)

    # fresh copy of the pretrained encoder for every run
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    # BEFORE: fresh probes on the SAVED latents (matches fig1 by construction)
    before = evaluate_from_saved(ref, seed, eval_targets)

    # adversarial fine-tuning (debias.py's own loop, GRL and all)
    advs = {a: Adversary(latent_dim, n).to(device) for a, n in ATTRS.items()}
    history = train_debiased(model, advs, train_loader, device, run_args)

    # AFTER: fresh probes on the debiased frozen encoder (adversaries discarded)
    after, _ = evaluate_encoder(
        model, tr_eval, te_eval, device, seed, 'debiased', eval_targets)

    # per-target removal, computed on BOTH probes; headline is LogReg
    summary = {}
    for attr in eval_targets:
        chance = before[attr]['chance']
        row = {'chance': chance}
        for probe in ('LogReg', 'MLP'):
            b = before[attr][probe]
            a = after[attr][probe]
            removed = (b - a) / (b - chance) * 100 if b > chance else 0.0
            row[probe] = {
                'before': float(b),
                'after': float(a),
                'pct_removed': float(removed),
                'residual_above_chance': float(a - chance),
            }
        summary[attr] = row

    return {
        'seed': seed, 'lambda_max': float(lambda_max),
        'summary': summary,
        'final_lambda': history[-1]['lambda'] if history else None,
        'final_recon': history[-1]['recon'] if history else None,
    }


def paired_stats(vals):
    """Mean, std, sign consistency of a per-seed quantity. Same shape as
    federated_multiseed.paired_stats so downstream reporting is consistent."""
    a = np.asarray(vals, dtype=float)
    out = {
        'mean': float(a.mean()),
        'std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        'min': float(a.min()),
        'max': float(a.max()),
        'n_positive': int((a > 0).sum()),
        'n_negative': int((a < 0).sum()),
        'values': a.tolist(),
    }
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
    p.add_argument('--data_root', required=True)
    p.add_argument('--dataset', choices=['utkface', 'rafdb'], default='utkface')
    p.add_argument('--demographics', type=str, default=None)
    p.add_argument('--checkpoint', default='../../checkpoints/vae_best.pt')
    p.add_argument('--reference_latents', default=None)
    p.add_argument('--output', default='../../results/debias_multiseed.json')
    p.add_argument('--seeds', default='42,1,7,13,99')
    p.add_argument('--lambdas', default='1,5,20,50')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--adv_lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    if args.reference_latents is None:
        args.reference_latents = f'../../latents/{args.dataset}_latents.npz'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')]
    lambdas = [float(x) for x in args.lambdas.split(',')]

    print(f"Device: {device}  |  dataset {args.dataset}")
    print(f"Seeds:   {seeds}")
    print(f"Lambdas: {lambdas}")
    print(f"Headline probe: {HEADLINE_PROBE} (deterministic; matches the figures)\n")

    # ---- build dataset and frozen split once; reused for every run ----
    tf = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
    ds, collate = build_dataset(args, tf)

    ref = np.load(args.reference_latents, allow_pickle=True)
    split = ref['split']
    if len(split) != len(ds):
        raise RuntimeError(
            f"Reference latents have {len(split)} rows but the dataset has "
            f"{len(ds)}. Check --reference_latents matches --dataset.")
    tr_idx = np.where(split == 'train')[0]
    te_idx = np.where(split == 'test')[0]

    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt['args']['latent_dim']

    has_emotion = args.dataset == 'rafdb' and 'emotion' in ref.files
    eval_targets = dict(ATTRS)
    if has_emotion:
        eval_targets['emotion'] = 7
        print("Emotion present: tracked as a utility target, not removed.\n")

    # ---- the sweep ----
    runs = []  # flat list of per-(seed,lambda) results
    t_start = time.time()
    for lam in lambdas:
        for seed in seeds:
            print(f"{'='*72}\nLAMBDA {lam:g}  |  SEED {seed}\n{'='*72}", flush=True)
            t0 = time.time()
            r = run_single(args, seed, lam, ref, ds, collate, split,
                           tr_idx, te_idx, device, eval_targets, latent_dim)
            runs.append(r)
            hp = HEADLINE_PROBE
            line = "  ".join(
                f"{attr[:4]} {r['summary'][attr][hp]['pct_removed']:+.1f}%"
                for attr in eval_targets)
            print(f"  removed ({hp}):  {line}   [{time.time()-t0:.0f}s]\n",
                  flush=True)

    # ---- aggregate per lambda x target, over seeds ----
    print("\n" + "=" * 78)
    print(f"PER-SEED REMOVAL, aggregated over {len(seeds)} seeds "
          f"(headline probe: {HEADLINE_PROBE})")
    print("Sign consistency leads; p-values are secondary at this n.")
    print("=" * 78)

    agg = {}
    for lam in lambdas:
        agg[f'{lam:g}'] = {}
        lam_runs = [r for r in runs if r['lambda_max'] == lam]
        print(f"\nlambda = {lam:g}")
        print(f"  {'target':<12}{'removed % (mean±std)':>26}"
              f"{'range':>20}{'signs':>12}")
        print("  " + "-" * 66)
        for attr in eval_targets:
            vals = [r['summary'][attr][HEADLINE_PROBE]['pct_removed']
                    for r in lam_runs]
            st = paired_stats(vals)
            agg[f'{lam:g}'][attr] = st
            tag = '  <- task' if attr == 'emotion' else ''
            pstr = f"  p={st['p']:.3f}" if 'p' in st else ""
            print(f"  {attr:<12}"
                  f"{st['mean']:>15.1f} ± {st['std']:<6.1f}"
                  f"[{st['min']:+.1f}, {st['max']:+.1f}]"
                  f"{st['n_positive']:>4}+ /{st['n_negative']:>2}-"
                  f"{pstr}{tag}")

    # ---- the lambda=50 verdict, stated explicitly ----
    print("\n" + "=" * 78)
    print("VERDICT on the lambda=50 negative-removal result")
    print("-" * 78)
    if 50.0 in lambdas:
        for attr in ('race', 'age_bucket'):
            if attr in eval_targets:
                st = agg['50'][attr]
                if st['n_negative'] > st['n_positive']:
                    holds = (f"negative in {st['n_negative']}/{len(seeds)} seeds "
                             f"-> the effect looks real, not a single-seed artefact")
                elif st['n_positive'] > st['n_negative'] and st['min'] >= 0:
                    holds = "positive in every seed -> the single-seed negative was noise"
                else:
                    holds = (f"signs split {st['n_positive']}+/{st['n_negative']}- "
                             f"and the range spans zero -> not resolvable; report as inconclusive")
                print(f"  {attr:<11} mean {st['mean']:+.1f}% "
                      f"[{st['min']:+.1f}, {st['max']:+.1f}]  ->  {holds}")
    else:
        print("  lambda=50 not in --lambdas; nothing to verdict.")
    print("=" * 78 + "\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump({
            'args': vars(args),
            'dataset': args.dataset,
            'seeds': seeds,
            'lambdas': lambdas,
            'headline_probe': HEADLINE_PROBE,
            'eval_targets': list(eval_targets.keys()),
            'runs': runs,
            'aggregate': agg,
        }, fh, indent=2)
    print(f"Saved to {args.output}")
    print(f"Total wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
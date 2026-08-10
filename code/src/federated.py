"""
Experiment B: does FedAR-style resampling fix subgroup disparity?

Simulates FedAvg on the cached VAE latents. The downstream task is emotion
classification on RAF-DB (7 classes, natively imbalanced at IMR ~16.8, close
to the AffectNet setting FedAR was designed for) or age-bucket classification
on UTKFace (4 classes, mildly imbalanced). Race and gender are held out as
sensitive attributes and never seen by any model.

Three conditions:
  none      - no resampling, the naive federated baseline
  fedar     - SMOTE on any client whose imbalance ratio exceeds a threshold
              (this is the FedAR intervention, applied to latent vectors)
  demo_bal  - resample so each client is balanced across race x gender

The question: does an intervention that balances the *task* labels do anything
for the *demographic* subgroup gap?
"""

import os
import json
import copy
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RACE_NAMES = ['White', 'Black', 'Asian', 'Indian', 'Other']
GENDER_NAMES = ['Male', 'Female']

TASKS = {
    'age_bucket': {'n_classes': 4,
                   'names': ['0-19', '20-34', '35-49', '50+']},
    'emotion':    {'n_classes': 7,
                   'names': ['Surprise', 'Fear', 'Disgust', 'Happy',
                             'Sad', 'Anger', 'Neutral']},
}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class Classifier(nn.Module):
    """Small MLP head over the frozen latent, matching FedAR's DMLP clients."""

    def __init__(self, in_dim=128, hidden=256, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# partitioning and resampling
# --------------------------------------------------------------------------

def partition_clients(n, n_clients, rng):
    """Random shuffle then contiguous split, as in FedAR."""
    idx = rng.permutation(n)
    return np.array_split(idx, n_clients)


def imbalance_ratio(y, n_classes):
    """Ratio of the largest class to the smallest non-empty class."""
    counts = np.bincount(y, minlength=n_classes)
    nz = counts[counts > 0]
    if len(nz) < 2:
        return np.inf
    return nz.max() / nz.min()


def smote_resample(X, y, seed):
    """Balance task classes with SMOTE. Falls back to random oversampling."""
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    counts = Counter(y)
    k = min(5, min(counts.values()) - 1)
    try:
        if k >= 1:
            return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
        return RandomOverSampler(random_state=seed).fit_resample(X, y)
    except ValueError:
        return RandomOverSampler(random_state=seed).fit_resample(X, y)


def demographic_balance(X, y, race, gender, rng):
    """Oversample so every race x gender cell is equally represented."""
    strata = race * 2 + gender
    counts = Counter(strata)
    target = max(counts.values())
    keep = []
    for s, c in counts.items():
        pool = np.where(strata == s)[0]
        keep.append(pool)
        if c < target:
            extra = rng.choice(pool, size=target - c, replace=True)
            keep.append(extra)
    sel = np.concatenate(keep)
    return X[sel], y[sel]


def induce_imbalance(X, y, race, gender, keep_fractions, rng):
    """
    Subsample task classes to create severe imbalance, mimicking the
    class distribution FedAR was designed for (AffectNet IMR ~18.7).

    keep_fractions: one fraction per task class, in class order.
    Unnecessary on RAF-DB emotion, which is already at IMR ~16.8.
    """
    keep = []
    for c, frac in enumerate(keep_fractions):
        pool = np.where(y == c)[0]
        n_keep = max(1, int(len(pool) * frac))
        keep.append(rng.choice(pool, size=n_keep, replace=False))
    sel = np.concatenate(keep)
    rng.shuffle(sel)
    return X[sel], y[sel], race[sel], gender[sel]


# --------------------------------------------------------------------------
# federated training
# --------------------------------------------------------------------------

def local_train(model, X, y, epochs, lr, batch_size, device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    n = len(X_t)

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(X_t[b]), y_t[b])
            loss.backward()
            opt.step()
    return model.state_dict(), n


def fedavg(states, sizes):
    """Weighted average of client state dicts, weighted by local sample count."""
    total = sum(sizes)
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = sum(s[k].float() * (n / total) for s, n in zip(states, sizes))
    return avg


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def subgroup_report(model, X_te, y_te, race_te, gender_te, device,
                    n_classes, min_group=25):
    """Overall accuracy plus a race x gender breakdown."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te, dtype=torch.float32, device=device))
        pred = logits.argmax(1).cpu().numpy()

    correct = (pred == y_te)
    overall = correct.mean()

    # balanced accuracy over the task classes
    per_class = [correct[y_te == c].mean()
                 for c in range(n_classes) if (y_te == c).sum() > 0]
    balanced = float(np.mean(per_class))

    groups = {}
    for r in range(len(RACE_NAMES)):
        for g in range(len(GENDER_NAMES)):
            m = (race_te == r) & (gender_te == g)
            if m.sum() < min_group:          # too small to be meaningful
                continue
            name = f"{RACE_NAMES[r]}/{GENDER_NAMES[g]}"
            groups[name] = {'acc': float(correct[m].mean()), 'n': int(m.sum())}

    accs = [v['acc'] for v in groups.values()]
    return {
        'overall_acc': float(overall),
        'balanced_acc': balanced,
        'groups': groups,
        'worst_group': float(min(accs)),
        'best_group': float(max(accs)),
        'gap': float(max(accs) - min(accs)),
        'std_across_groups': float(np.std(accs)),
        'n_groups_evaluated': len(groups),
    }


# --------------------------------------------------------------------------
# one full run
# --------------------------------------------------------------------------

def run_condition(condition, data, args, device, n_classes):
    rng = np.random.default_rng(args.seed)
    X_tr, y_tr, race_tr, gender_tr = data['train']
    X_te, y_te, race_te, gender_te = data['test']

    parts = partition_clients(len(X_tr), args.n_clients, rng)

    # build each client's local dataset under the given condition
    clients, meta = [], []
    for ci, idx in enumerate(parts):
        Xc, yc = X_tr[idx], y_tr[idx]
        rc, gc = race_tr[idx], gender_tr[idx]
        imr = imbalance_ratio(yc, n_classes)
        n_before = len(Xc)
        resampled = False

        if condition == 'fedar' and imr > args.imr_threshold:
            Xc, yc = smote_resample(Xc, yc, args.seed + ci)
            resampled = True
        elif condition == 'demo_bal':
            Xc, yc = demographic_balance(Xc, yc, rc, gc, rng)
            resampled = True

        clients.append((Xc, yc))
        meta.append({
            'client': ci,
            'n_before': n_before,
            'n_after': len(Xc),
            'imbalance_ratio': float(imr),
            'resampled': resampled,
            'race_dist': np.bincount(rc, minlength=len(RACE_NAMES)).tolist(),
            'gender_dist': np.bincount(gc, minlength=len(GENDER_NAMES)).tolist(),
            'task_dist': np.bincount(y_tr[idx], minlength=n_classes).tolist(),
        })

    # federated averaging
    torch.manual_seed(args.seed)
    global_model = Classifier(in_dim=X_tr.shape[1], n_classes=n_classes).to(device)
    history = []

    for rnd in range(1, args.rounds + 1):
        states, sizes = [], []
        for Xc, yc in clients:
            local = copy.deepcopy(global_model)
            st, n = local_train(local, Xc, yc, args.local_epochs,
                                args.lr, args.batch_size, device)
            states.append(st)
            sizes.append(n)

        global_model.load_state_dict(fedavg(states, sizes))

        if rnd % args.eval_every == 0 or rnd == args.rounds:
            rep = subgroup_report(global_model, X_te, y_te,
                                  race_te, gender_te, device, n_classes)
            history.append({'round': rnd, **{k: v for k, v in rep.items()
                                             if k != 'groups'}})
            print(f"  round {rnd:3d} | acc {rep['overall_acc']:.4f} | "
                  f"bal {rep['balanced_acc']:.4f} | worst {rep['worst_group']:.4f} | "
                  f"gap {rep['gap']:.4f}", flush=True)

    final = subgroup_report(global_model, X_te, y_te, race_te, gender_te,
                            device, n_classes)
    return {'final': final, 'history': history, 'clients': meta}


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latents', default='../../latents/utkface_latents.npz')
    p.add_argument('--output', default='../../results/federated_results.json')
    p.add_argument('--task', choices=['age_bucket', 'emotion'], default=None,
                   help='Downstream task. Defaults to emotion when the latents '
                        'carry it, otherwise age_bucket.')
    p.add_argument('--n_clients', type=int, default=10)
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--local_epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--imr_threshold', type=float, default=1.8)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skew', type=str, default=None,
                   help='Comma-separated keep-fractions, one per task class, '
                        'e.g. "0.05,1.0,0.5,0.15". Applied to the training '
                        'split only; the test split stays untouched. Not needed '
                        'for RAF-DB emotion, which is already severely imbalanced.')
    args = p.parse_args()

    device = torch.device('cpu')   # 128-dim vectors, CPU is fine and avoids contention

    d = np.load(args.latents, allow_pickle=True)
    mu, split = d['mu'], d['split']
    tr, te = split == 'train', split == 'test'

    # pick the task
    task = args.task
    if task is None:
        task = 'emotion' if 'emotion' in d.files else 'age_bucket'
    if task not in d.files:
        raise ValueError(f"Latents do not contain '{task}'. "
                         f"Available: {[f for f in d.files]}")
    n_classes = TASKS[task]['n_classes']
    dataset = str(d['dataset']) if 'dataset' in d.files else 'unknown'

    print(f"Dataset: {dataset}  |  task: {task} ({n_classes} classes)")
    if dataset == 'rafdb':
        print("Note: emotion labels are human-annotated; race/gender are "
              "model-inferred (FairFace).")

    # standardize on train statistics
    m, s = mu[tr].mean(0), mu[tr].std(0) + 1e-8
    data = {
        'train': ((mu[tr] - m) / s, d[task][tr], d['race'][tr], d['gender'][tr]),
        'test':  ((mu[te] - m) / s, d[task][te], d['race'][te], d['gender'][te]),
    }

    # report the native imbalance before any intervention
    native = np.bincount(data['train'][1], minlength=n_classes)
    native_imr = native.max() / native[native > 0].min()
    print(f"Native task distribution: {native.tolist()}  (IMR {native_imr:.1f})")

    # optionally induce severe task imbalance so the resampling intervention
    # has something meaningful to correct
    if args.skew:
        rng_skew = np.random.default_rng(args.seed)
        fracs = [float(f) for f in args.skew.split(',')]
        assert len(fracs) == n_classes, \
            f"--skew needs {n_classes} comma-separated fractions for task '{task}'"

        Xs, ys, rs, gs = induce_imbalance(*data['train'], fracs, rng_skew)
        data['train'] = (Xs, ys, rs, gs)

        after = np.bincount(ys, minlength=n_classes)
        imr_after = after.max() / after[after > 0].min()
        print(f"SKEW APPLIED: {args.skew}")
        print(f"  after:  {after.tolist()}  (IMR {imr_after:.1f}, n={after.sum()})")
        print(f"  test split unchanged: "
              f"{np.bincount(data['test'][1], minlength=n_classes).tolist()}")

    print(f"\ntrain {len(data['train'][0])} | test {len(data['test'][0])} | "
          f"{args.n_clients} clients | {args.rounds} rounds\n")

    results = {}
    for cond in ['none', 'fedar', 'demo_bal']:
        print(f"{'='*70}\nCONDITION: {cond}\n{'='*70}")
        results[cond] = run_condition(cond, data, args, device, n_classes)
        f = results[cond]['final']
        n_res = sum(c['resampled'] for c in results[cond]['clients'])
        imrs = [c['imbalance_ratio'] for c in results[cond]['clients']]
        print(f"\n  clients resampled: {n_res}/{args.n_clients}  "
              f"(client IMR range {min(imrs):.2f}-{max(imrs):.2f})")
        print(f"  overall {f['overall_acc']:.4f} | balanced {f['balanced_acc']:.4f}")
        print(f"  worst group {f['worst_group']:.4f} "
              f"({min(f['groups'], key=lambda k: f['groups'][k]['acc'])})")
        print(f"  best group  {f['best_group']:.4f}")
        print(f"  gap {f['gap']:.4f} | std {f['std_across_groups']:.4f}  "
              f"({f['n_groups_evaluated']} groups)\n")

    # side by side, with deltas against the no-intervention baseline
    base = results['none']['final']
    print("=" * 78)
    print(f"{'condition':<12}{'overall':>10}{'balanced':>10}{'worst':>10}"
          f"{'gap':>10}{'d_bal':>11}{'d_gap':>11}")
    print("-" * 78)
    for cond in ['none', 'fedar', 'demo_bal']:
        f = results[cond]['final']
        d_bal = f['balanced_acc'] - base['balanced_acc']
        d_gap = f['gap'] - base['gap']
        d_bal_s = '  --' if cond == 'none' else f'{d_bal:+.4f}'
        d_gap_s = '  --' if cond == 'none' else f'{d_gap:+.4f}'
        print(f"{cond:<12}{f['overall_acc']:>10.4f}{f['balanced_acc']:>10.4f}"
              f"{f['worst_group']:>10.4f}{f['gap']:>10.4f}"
              f"{d_bal_s:>11}{d_gap_s:>11}")
    print("=" * 78)
    print("d_bal / d_gap are changes relative to the 'none' baseline.")
    print("The finding to look for: resampling lifts d_bal but leaves d_gap flat.\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump({'args': vars(args), 'task': task, 'dataset': dataset,
                   'results': results}, fh, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
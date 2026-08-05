"""
Experiment A: demographic leakage probes.

Trains three probe families (logistic regression, linear SVM, MLP) to predict
gender, race, and age bucket from the frozen VAE latent space.

Three controls are reported alongside every result:
  - stratified dummy (respects class priors)
  - majority-class dummy
  - label-shuffled null (same probe, permuted labels; must score at chance)

The label-shuffled null is the important one. If it scores above chance,
there is a leak in the pipeline and every other number is meaningless.
"""

import os
import json
import argparse
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


TARGETS = {
    'gender': {'n_classes': 2, 'names': ['Male', 'Female']},
    'race':   {'n_classes': 5, 'names': ['White', 'Black', 'Asian', 'Indian', 'Other']},
    'age_bucket': {'n_classes': 4, 'names': ['0-19', '20-34', '35-49', '50+']},
}


def make_probes(seed):
    return {
        'LogReg': LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        'LinearSVM': LinearSVC(max_iter=5000, C=1.0, random_state=seed, dual='auto'),
        'MLP': MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                             early_stopping=True, random_state=seed),
    }


def make_baselines(seed):
    return {
        'Stratified': DummyClassifier(strategy='stratified', random_state=seed),
        'Majority': DummyClassifier(strategy='most_frequent'),
    }


def evaluate(clf, X_tr, y_tr, X_te, y_te, n_classes):
    """Fit and return balanced accuracy, macro-F1, and AUROC where available."""
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    bal_acc = balanced_accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)

    auroc = None
    if hasattr(clf, 'predict_proba'):
        try:
            proba = clf.predict_proba(X_te)
            if n_classes == 2:
                auroc = roc_auc_score(y_te, proba[:, 1])
            else:
                auroc = roc_auc_score(y_te, proba, multi_class='ovr',
                                      average='macro')
        except Exception:
            pass

    return {'balanced_accuracy': bal_acc, 'macro_f1': macro_f1, 'auroc': auroc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=str,
                        default='../../latents/utkface_latents.npz')
    parser.add_argument('--output', type=str,
                        default='../../results/probe_results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--skip_cv', action='store_true',
                        help='Skip cross-validation for a faster first pass')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    d = np.load(args.latents, allow_pickle=True)
    mu = d['mu']
    split = d['split']
    tr_mask = split == 'train'
    te_mask = split == 'test'

    print(f"Latents: {mu.shape}")
    print(f"  probe-train {tr_mask.sum()}  |  probe-test {te_mask.sum()}\n")

    # Standardize using train statistics only
    scaler = StandardScaler().fit(mu[tr_mask])
    X_tr = scaler.transform(mu[tr_mask])
    X_te = scaler.transform(mu[te_mask])

    results = {}

    for target, meta in TARGETS.items():
        y = d[target]
        y_tr, y_te = y[tr_mask], y[te_mask]
        n_classes = meta['n_classes']

        print("=" * 72)
        print(f"TARGET: {target}  ({n_classes} classes)")
        print("=" * 72)

        # class distribution in the test split, for context
        counts = np.bincount(y_te, minlength=n_classes)
        dist = "  ".join(f"{meta['names'][i]}:{counts[i]}"
                         for i in range(n_classes))
        print(f"test distribution: {dist}\n")

        results[target] = {'n_classes': n_classes, 'probes': {}, 'baselines': {},
                           'null_control': {}}

        # --- baselines ---
        for name, clf in make_baselines(args.seed).items():
            r = evaluate(clf, X_tr, y_tr, X_te, y_te, n_classes)
            results[target]['baselines'][name] = r
            print(f"  {name:<12} bal_acc {r['balanced_accuracy']:.4f}  "
                  f"macro_f1 {r['macro_f1']:.4f}")

        print()

        # --- probes ---
        for name, clf in make_probes(args.seed).items():
            r = evaluate(clf, X_tr, y_tr, X_te, y_te, n_classes)

            # cross-validated stability on the train split
            if not args.skip_cv:
                cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                                     random_state=args.seed)
                fresh = make_probes(args.seed)[name]
                scores = cross_val_score(fresh, X_tr, y_tr, cv=cv,
                                         scoring='balanced_accuracy', n_jobs=-1)
                r['cv_mean'] = float(scores.mean())
                r['cv_std'] = float(scores.std())

            results[target]['probes'][name] = r

            auroc_str = f"  auroc {r['auroc']:.4f}" if r['auroc'] is not None else ""
            cv_str = (f"  cv {r['cv_mean']:.4f}±{r['cv_std']:.4f}"
                      if 'cv_mean' in r else "")
            print(f"  {name:<12} bal_acc {r['balanced_accuracy']:.4f}  "
                  f"macro_f1 {r['macro_f1']:.4f}{auroc_str}{cv_str}")

        # --- label-shuffled null control ---
        # Shuffle BOTH splits so the probe has no recoverable signal at all.
        y_tr_shuf = rng.permutation(y_tr)
        y_te_shuf = rng.permutation(y_te)
        null_clf = LogisticRegression(max_iter=2000, random_state=args.seed)
        r_null = evaluate(null_clf, X_tr, y_tr_shuf, X_te, y_te_shuf, n_classes)
        results[target]['null_control'] = r_null
        print(f"\n  {'NULL (shuffled)':<12} bal_acc {r_null['balanced_accuracy']:.4f}"
              f"   <- must sit at chance ({1/n_classes:.4f})")

        # headline number
        best = max(results[target]['probes'].items(),
                   key=lambda kv: kv[1]['balanced_accuracy'])
        chance = 1.0 / n_classes
        lift = best[1]['balanced_accuracy'] - chance
        print(f"\n  best probe: {best[0]} at {best[1]['balanced_accuracy']:.4f} "
              f"({lift:+.4f} over chance {chance:.4f})\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
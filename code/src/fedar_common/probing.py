"""
The probe suite — one definition, used by Experiment A and the controls.

make_probes / make_baselines / evaluate were defined in probes.py and used
nowhere else by import (pixel_baseline.py re-implemented an equivalent inline).
They live here now so the leakage probe (probes.py) and the PCA control
(pixel_baseline.py) provably score with the *same* estimators and the *same*
metric — which matters, because the whole PCA control rests on "identical
probe suite, different input".

Note on the removal experiment (Experiments C/D): debias.py keeps its own
`fresh_probe` (LogReg + MLP, deterministic LogReg headline). That is the
Elazar-Goldberg fresh-probe check and is deliberately co-located with the
removal pipeline; debias_multiseed.py imports it from debias.py by design.
This module is for the leakage/control probes only.
"""
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

__all__ = ['make_probes', 'make_baselines', 'evaluate']


def make_probes(seed):
    """The three probe families. LogReg is the deterministic headline probe."""
    return {
        'LogReg': LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        'LinearSVM': LinearSVC(max_iter=5000, C=1.0, random_state=seed, dual='auto'),
        'MLP': MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                             early_stopping=True, random_state=seed),
    }


def make_baselines(seed):
    """Stratified (respects priors) and majority-class dummies."""
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

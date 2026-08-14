"""
paired_stats — per-seed aggregation for the multi-seed drivers.

Identical copies previously lived in federated_multiseed.py and
debias_multiseed.py. Lifted here unchanged so both import one definition.
Sign consistency (n_positive / n_negative) is the headline; the paired
t-test p-value is recorded but secondary at n=5, per the README.
"""
import numpy as np

__all__ = ['paired_stats']


def paired_stats(vals):
    """Mean, std, range and sign consistency of a paired per-seed quantity.

    A one-sample t-test against zero is included when scipy is available and
    there is nonzero spread. With five seeds the p-value is fragile, so the
    sign split is what downstream reporting leads with.
    """
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

"""
fedar_common — shared building blocks for the MLSS-2026 poster codebase.

Everything in this package existed before, copy-pasted across the flat scripts
in code/src/. It has been lifted here verbatim (behaviour-preserving) so there
is exactly one definition of each shared piece:

    data      one build_dataset(), age_to_bucket(), split helpers
    probing   the probe suite, evaluate(), fresh-probe recovery
    stats     paired_stats() (was duplicated in the two *_multiseed drivers)
    plotting  the Okabe-Ito palette + JSON/figure helpers (was in 8 files)

The scripts still run exactly as the README documents them
(`python probes.py`, `python federated_multiseed.py`, ...). They now import
from this package instead of carrying private copies.
"""

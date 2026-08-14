#!/usr/bin/env python3
"""
make_umap_panels.py  —  clean single-panel UMAP projections.

The repo's visualize.py can make a combined UMAP figure, but the default runs pass
--no_umap, and the team demoted UMAP because it showed structure for gender but not
race/age. This script makes isolated square UMAP panels ONLY if you want them as a
laptop backup or to demonstrate the "gender separates, race/age don't" point in
conversation. It is deliberately NOT the poster's main latent panel — the LDA panels
(make_lda_panels.py) are, because LDA is a supervised linear projection with no free
hyperparameters and is what the poster already uses.

Requires umap-learn (segfaults if TensorFlow is present in the same env — see repo README).

Run from code/src/ :

    python make_umap_panels.py \
        --latents ../../latents/utkface_latents.npz \
        --attrs gender race age_bucket \
        --out_dir ../../figures/umap_panels

Outputs: <out_dir>/umap_<attr>.png  (square, transparent bg)
"""
# ---------------------------------------------------------------------------
# TensorFlow-segfault guard. deepface/retina-face pull TensorFlow into this venv,
# and TF's native CUDA-stub init collides with numba/llvmlite (which umap uses),
# producing the "Segmentation fault" seen on the CIP pool. We never use TF here,
# so we stop it initialising before numba/umap load. These env vars must be set
# BEFORE numpy/numba/umap are imported, so this block stays at the very top.
from fedar_common.plotting import OI
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # hide GPU from TF entirely
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # silence TF logging
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "false")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # avoid TBB/omp clash
# Belt-and-suspenders: import umap (and thus numba) FIRST, before anything can
# transitively import tensorflow.
try:
    import umap  # noqa: F401  (imported early on purpose)
    _UMAP_OK = True
except Exception as _e:            # ImportError or the numba/TF segfault surrogate
    _UMAP_OK = False
    _UMAP_ERR = _e
# ---------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_NAMES = {
    "gender":     ["Male", "Female"],
    "race":       ["White", "Black", "Asian", "Indian", "Other"],
    "age_bucket": ["0-19", "20-34", "35-49", "50+"],
    "emotion":    ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--attrs", nargs="+", default=["gender", "race", "age_bucket"])
    ap.add_argument("--out_dir", default="../../figures/umap_panels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sample", type=int, default=6000)
    args = ap.parse_args()

    if not _UMAP_OK:
        print(f"UMAP unavailable ({_UMAP_ERR!r}).")
        print("If this is the TensorFlow segfault, run with TF hidden, e.g.:")
        print("  CUDA_VISIBLE_DEVICES=-1 python make_umap_panels.py ...")
        print("or, most reliably, use make_lda_panels.py instead — LDA needs no umap/TF")
        print("and is the projection the poster actually uses.")
        return

    d = np.load(args.latents, allow_pickle=True)
    mu = d["mu"]
    dataset = str(d["dataset"]) if "dataset" in d.files else "unknown"

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(mu), size=min(args.n_sample, len(mu)), replace=False)
    mu_s = mu[idx]

    print(f"  fitting UMAP on {len(mu_s)} points ({dataset})...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=args.seed)
    emb = reducer.fit_transform(mu_s)

    os.makedirs(args.out_dir, exist_ok=True)
    for attr in args.attrs:
        if attr not in d.files:
            print(f"  [skip] {attr} not present")
            continue
        y = d[attr][idx]
        n_classes = len(np.unique(y))
        names = CLASS_NAMES.get(attr, [str(i) for i in range(n_classes)])

        fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=180)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        for c in np.unique(y):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=3, color=OI[int(c) % len(OI)],
                       alpha=0.55, linewidths=0,
                       label=names[int(c)] if int(c) < len(names) else str(c))
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, framealpha=0.85, loc="best", markerscale=2.5)
        ax.set_title(f"{dataset} UMAP · {attr}", fontsize=11)
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"umap_{attr}.png")
        fig.savefig(out, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
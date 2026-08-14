#!/usr/bin/env python3
"""
make_lda_panels.py  —  clean single-panel LDA projections for the hero diagram.

The repo's visualize.py produces a combined 3-panel fig4_lda_projection*.png. For the
hero pipeline diagram we want each attribute as its own square panel with a transparent
background, so it drops cleanly into the SVG without cropping a multi-panel figure.

Reuses the exact LDA recipe from visualize.py (directions fitted on the probe-train
split, points shown are held-out test) so these panels are identical in method to the
figure already on the poster — just isolated and squared.

Run from code/src/ :

    # UTKFace (ground-truth labels) — race and age separate most cleanly
    python make_lda_panels.py \
        --latents ../../latents/utkface_latents.npz \
        --attrs race age_bucket gender \
        --out_dir ../../figures/lda_panels

    # RAF-DB (FairFace-inferred + human emotion)
    python make_lda_panels.py \
        --latents ../../latents/rafdb_latents.npz \
        --attrs race age_bucket gender emotion \
        --out_dir ../../figures/lda_panels_rafdb

Outputs: <out_dir>/lda_<attr>.png  (square, ~600x600, transparent bg)
"""
# Defensive: this venv has TensorFlow (via deepface). We don't use it, and TF's
# native init has been segfaulting other scripts on the CIP pool, so hide the GPU
# from TF in case anything imports it transitively. Harmless for this sklearn-only
# script. Must run before other imports.
from fedar_common.plotting import OI
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Okabe-Ito, matching the repo palette

CLASS_NAMES = {
    "gender":     ["Male", "Female"],
    "race":       ["White", "Black", "Asian", "Indian", "Other"],
    "age_bucket": ["0-19", "20-34", "35-49", "50+"],
    "emotion":    ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--attrs", nargs="+", default=["race", "age_bucket"])
    ap.add_argument("--out_dir", default="../../figures/lda_panels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--point_size", type=float, default=4.0)
    args = ap.parse_args()

    d = np.load(args.latents, allow_pickle=True)
    mu = d["mu"]
    dataset = str(d["dataset"]) if "dataset" in d.files else "unknown"
    split = d["split"].astype("<U5") if "split" in d.files else None

    os.makedirs(args.out_dir, exist_ok=True)

    for attr in args.attrs:
        if attr not in d.files:
            print(f"  [skip] {attr} not in latents ({dataset})")
            continue
        y = d[attr]
        # fit LDA directions on train, show test — same protocol as visualize.py
        if split is not None:
            tr = split == "train"
            te = split == "test"
        else:
            rng = np.random.default_rng(args.seed)
            te = rng.random(len(mu)) < 0.3
            tr = ~te

        n_classes = len(np.unique(y))
        n_comp = min(2, n_classes - 1)
        if n_comp < 1:
            print(f"  [skip] {attr}: only one class")
            continue

        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        lda.fit(mu[tr], y[tr])
        proj = lda.transform(mu[te])
        yt = y[te]

        fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=180)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        names = CLASS_NAMES.get(attr, [str(i) for i in range(n_classes)])

        if n_comp == 1:
            # 1D (binary): density histogram per class
            for c in np.unique(yt):
                ax.hist(proj[yt == c, 0], bins=40, alpha=0.55,
                        color=OI[int(c) % len(OI)],
                        label=names[int(c)] if int(c) < len(names) else str(c))
            ax.set_xlabel("LD1"); ax.set_ylabel("density")
        else:
            for c in np.unique(yt):
                m = yt == c
                ax.scatter(proj[m, 0], proj[m, 1], s=args.point_size,
                           color=OI[int(c) % len(OI)], alpha=0.55, linewidths=0,
                           label=names[int(c)] if int(c) < len(names) else str(c))
            ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
            ax.set_xticks([]); ax.set_yticks([])

        ax.legend(fontsize=7, framealpha=0.85, loc="best", markerscale=2)
        ax.set_title(f"{dataset} · {attr}", fontsize=11)
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"lda_{attr}.png")
        fig.savefig(out, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
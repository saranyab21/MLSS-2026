#!/usr/bin/env python3
"""
make_umap_panels_isolated.py  —  guaranteed-no-segfault UMAP via subprocess isolation.

Use this ONLY if make_umap_panels.py still segfaults on the CIP pool. It computes the
UMAP embedding in a clean child process whose import graph never touches TensorFlow
(the cause of the numba/llvmlite segfault), writes the 2-D embedding to a temp .npy,
then plots it in the parent. Same output as make_umap_panels.py.

Reminder: you almost certainly do NOT need UMAP for the poster — the LDA panels from
make_lda_panels.py are the projection the poster uses, need no umap/TF, and can't
segfault. This exists only so a UMAP request never blocks you.

Run from code/src/ :

    python make_umap_panels_isolated.py \
        --latents ../../latents/utkface_latents.npz \
        --attrs gender race age_bucket \
        --out_dir ../../figures/umap_panels
"""
import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#999999",
      "#56B4E9", "#D55E00", "#F0E442"]
CLASS_NAMES = {
    "gender":     ["Male", "Female"],
    "race":       ["White", "Black", "Asian", "Indian", "Other"],
    "age_bucket": ["0-19", "20-34", "35-49", "50+"],
    "emotion":    ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
}

# Child program: pure numpy + umap, TF hidden, no other project imports.
CHILD = r'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import numpy as np
import umap
latents, out_emb, seed, n_sample = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = np.load(latents, allow_pickle=True)
mu = d["mu"]
rng = np.random.default_rng(seed)
idx = rng.choice(len(mu), size=min(n_sample, len(mu)), replace=False)
reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=seed)
emb = reducer.fit_transform(mu[idx])
np.save(out_emb, np.column_stack([idx.astype(np.float64), emb]))
print("child: embedding done", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--attrs", nargs="+", default=["gender", "race", "age_bucket"])
    ap.add_argument("--out_dir", default="../../figures/umap_panels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sample", type=int, default=6000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        child_py = os.path.join(td, "child.py")
        out_emb = os.path.join(td, "emb.npy")
        with open(child_py, "w") as f:
            f.write(CHILD)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        print("running UMAP in isolated subprocess (TF hidden)...")
        r = subprocess.run(
            [sys.executable, child_py, args.latents, out_emb,
             str(args.seed), str(args.n_sample)],
            env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("child failed:\n", r.stdout, r.stderr)
            print("Use make_lda_panels.py instead — no umap/TF, cannot segfault.")
            return
        print(r.stdout.strip())
        arr = np.load(out_emb)

    idx = arr[:, 0].astype(int)
    emb = arr[:, 1:]

    d = np.load(args.latents, allow_pickle=True)
    dataset = str(d["dataset"]) if "dataset" in d.files else "unknown"

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
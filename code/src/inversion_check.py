#!/usr/bin/env python3
"""
inversion_check.py  —  OPTIONAL. Make the "invert z -> image?" block genuinely yours.

WHY THIS EXISTS
---------------
FedAR's privacy claim ("similarity 0.06, so raw images are safe") comes from inverting
the latent through an EXTERNAL decoder. Your repo does not reproduce that experiment, so
in the hero diagram the invert-z block is currently (correctly) labelled as *FedAR's*
check, not yours. If you would rather show a real inversion result of your own — so the
green block is an experiment in your codebase, not a citation — run this.

WHAT IT DOES
------------
Takes the frozen VAE, runs held-out TEST faces z -> decoder -> reconstruction, and reports
image-similarity between original and reconstruction with several metrics, so you can state
an honest number. It also saves a small original/reconstruction grid for the poster/laptop.

IMPORTANT HONESTY NOTE (read before using the number)
-----------------------------------------------------
This inverts through YOUR OWN decoder, which was trained jointly with the encoder. FedAR's
0.06 used an *external* decoder (a stronger attacker model), which is a different — and
arguably more meaningful — threat model. So:
  * A LOW similarity here does NOT reprove FedAR's 0.06 (different decoder).
  * A HIGH similarity here would just mean your own decoder reconstructs well (expected).
Frame whatever you report as "reconstruction fidelity of our own decoder", and keep the
poster's actual claim about *attribute recoverability*, which is the point. If you want to
match FedAR's threat model exactly, you'd train a separate decoder on a disjoint split and
invert with that — a bigger job; ask and I'll write it.

Run from code/src/ :

    python inversion_check.py \
        --dataset rafdb \
        --data_root ../../data/rafdb_raw \
        --demographics ../../data/rafdb_demographics.csv \
        --checkpoint ../../checkpoints/rafdb/vae_best.pt \
        --out_json ../../results/inversion_rafdb.json \
        --out_grid ../../figures/inversion_rafdb.png

    # UTKFace
    python inversion_check.py \
        --dataset utkface \
        --data_root ../../data/utkface \
        --checkpoint ../../checkpoints/vae_best.pt \
        --out_json ../../results/inversion_utkface.json \
        --out_grid ../../figures/inversion_utkface.png
"""
import argparse
import json
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from model import VAE
from utkface_dataset import UTKFaceDataset, collate_labels
from rafdb_dataset import RAFDBDataset, collate_labels as collate_rafdb


def build_test_loader(args, transform):
    """
    Return a loader over the TEST split so the inversion number is comparable to the
    probe/leakage results (which are all on held-out test).

    RAF-DB has a native split, so RAFDBDataset(split='test') is exact.
    UTKFace's probe split lives in the latents .npz (drawn in extract_latents.py,
    stratified on race). If --latents is given we restrict UTKFace to those test
    indices; otherwise we fall back to the whole set and print a warning.
    """
    if args.dataset == "rafdb":
        ds = RAFDBDataset(args.data_root, demographics_csv=args.demographics,
                          transform=transform, split="test")
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_rafdb), ds

    # UTKFace: samples are dicts with a 'filename' key (see utkface_dataset.py)
    full = UTKFaceDataset(args.data_root, transform=transform)
    if args.latents and os.path.exists(args.latents):
        d = np.load(args.latents, allow_pickle=True)
        if "split" in d.files and "filename" in d.files:
            fn = np.asarray(d["filename"])
            test_names = set(fn[d["split"].astype("<U5") == "test"].tolist())
            keep = [i for i, s in enumerate(full.samples)
                    if s["filename"] in test_names]
            if keep:
                from torch.utils.data import Subset
                ds = Subset(full, keep)
                print(f"UTKFace: restricted to {len(keep)} test-split images from latents.")
                return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_labels), ds
    print("WARNING: no --latents test split for UTKFace; using the FULL set. "
          "The number will not be strictly comparable to the probe results.")
    return DataLoader(full, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, collate_fn=collate_labels), full


def ssim_batch(a, b):
    """Lightweight SSIM (mean over batch, channels). a,b in [0,1], shape (N,C,H,W)."""
    # constants for [0,1] range
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = a.mean(dim=(2, 3), keepdim=True)
    mu_b = b.mean(dim=(2, 3), keepdim=True)
    va = ((a - mu_a) ** 2).mean(dim=(2, 3), keepdim=True)
    vb = ((b - mu_b) ** 2).mean(dim=(2, 3), keepdim=True)
    cov = ((a - mu_a) * (b - mu_b)).mean(dim=(2, 3), keepdim=True)
    ssim = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / \
           ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return ssim.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["utkface", "rafdb"], required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--demographics", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents", default=None,
                    help="UTKFace only: latents .npz, used to restrict to the test "
                         "split so the number matches the probe results.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_grid", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = all test batches")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((96, 96)),
        torchvision.transforms.ToTensor(),
    ])

    model = VAE(latent_dim=128).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # repo saves checkpoints as {'model_state_dict': ..., 'optimizer_state_dict': ...,
    # 'epoch', 'loss', 'args'} — see train.py / extract_latents.py. Support that plus
    # a couple of common fallbacks.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()

    loader, ds = build_test_loader(args, transform)

    mses, l1s, ssims, cosines = [], [], [], []
    grid_orig, grid_recon = None, None

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            mu, logvar = model.encoder(x)
            recon = model.decoder(mu)              # deterministic: invert the mean latent
            mses.append(torch.mean((recon - x) ** 2).item())
            l1s.append(torch.mean(torch.abs(recon - x)).item())
            ssims.append(ssim_batch(recon, x))
            a = x.flatten(1); b = recon.flatten(1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
            cosines.append(cos)
            if grid_orig is None:
                k = min(8, x.size(0))
                grid_orig = x[:k].cpu()
                grid_recon = recon[:k].cpu()
            if args.max_batches and (i + 1) >= args.max_batches:
                break

    result = {
        "dataset": args.dataset,
        "n_test_batches": len(mses),
        "decoder": "own (trained jointly with encoder) — NOT FedAR's external decoder",
        "pixel_mse": float(np.mean(mses)),
        "pixel_l1": float(np.mean(l1s)),
        "ssim": float(np.mean(ssims)),
        "cosine_pixel": float(np.mean(cosines)),
        "note": ("Reconstruction fidelity of our own decoder. Not comparable to FedAR's "
                 "0.06, which used an external decoder (a different threat model)."),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    if args.out_grid and grid_orig is not None:
        grid = torch.cat([grid_orig, grid_recon], dim=0)
        os.makedirs(os.path.dirname(args.out_grid), exist_ok=True)
        torchvision.utils.save_image(grid, args.out_grid, nrow=grid_orig.size(0))
        print(f"wrote grid {args.out_grid} (top row original, bottom row inverted)")


if __name__ == "__main__":
    main()
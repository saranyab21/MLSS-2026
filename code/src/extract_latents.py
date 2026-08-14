"""
Extract latent vectors from the frozen VAE encoder for the full dataset.

Saves mu (deterministic, not sampled), logvar, and all labels to a single
.npz. Everything downstream reads from this file, so after this script runs
you never need the GPU again.

UTKFace: probe split is drawn here, stratified on race.
RAF-DB:  the dataset's own train/test split is reused, and the emotion label
         is saved alongside the demographic attributes.
"""

from fedar_common.data import build_dataset
import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from model import VAE




def make_split(args, ds, labels):
    """
    RAF-DB: reuse the official train/test partition.
    UTKFace: draw a stratified split on race, the most imbalanced attribute,
             so every group appears in both probe-train and probe-test.
    """
    n = len(ds)
    if args.dataset == 'rafdb':
        split = labels['orig_split'].astype('<U5')
        print(f"Using RAF-DB's native split: "
              f"{(split == 'train').sum()} train / {(split == 'test').sum()} test")
        return split

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed,
        stratify=labels['race'],
    )
    split = np.empty(n, dtype='<U5')
    split[train_idx] = 'train'
    split[test_idx] = 'test'
    print(f"Drew a stratified split: "
          f"{(split == 'train').sum()} train / {(split == 'test').sum()} test")
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', choices=['utkface', 'rafdb'], default='utkface')
    parser.add_argument('--demographics', type=str, default=None,
                        help='Path to rafdb_demographics.csv (rafdb only)')
    parser.add_argument('--checkpoint', type=str,
                        default='../../checkpoints/vae_best.pt')
    parser.add_argument('--output', type=str,
                        default='../../latents/utkface_latents.npz')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='UTKFace only; RAF-DB uses its native split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt['args']['latent_dim']
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, "
          f"latent_dim {latent_dim}, loss {ckpt['loss']:.2f}")

    # Warn loudly if the encoder was trained on the other corpus.
    ckpt_ds = ckpt['args'].get('dataset', 'utkface')
    if ckpt_ds != args.dataset:
        print(f"\n  WARNING: checkpoint was trained on '{ckpt_ds}' but you are "
              f"extracting '{args.dataset}'.\n  This is a cross-dataset transfer "
              f"setting. Intentional? If not, check --checkpoint.\n")

    # No augmentation at extraction time: deterministic representations only
    eval_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    ds, collate = build_dataset(args, eval_tf)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate,
    )

    all_mu, all_logvar = [], []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            mu, logvar = model.encoder(x)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"  {(i+1) * args.batch_size} / {len(ds)}", flush=True)

    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    labels = ds.get_label_arrays()
    split = make_split(args, ds, labels)

    payload = {
        'mu': mu,
        'logvar': logvar,
        'age': labels['age'],
        'age_bucket': labels['age_bucket'],
        'gender': labels['gender'],
        'race': labels['race'],
        'filename': labels['filename'],
        'split': split,
        'checkpoint_epoch': ckpt['epoch'],
        'latent_dim': latent_dim,
        'dataset': args.dataset,
    }
    if args.dataset == 'rafdb':
        payload['emotion'] = labels['emotion']

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **payload)

    print(f"\nSaved {mu.shape[0]} latents of dim {mu.shape[1]} to {args.output}")
    print(f"  probe-train: {(split == 'train').sum()}")
    print(f"  probe-test:  {(split == 'test').sum()}")
    if args.dataset == 'rafdb':
        print(f"  emotion classes: {len(np.unique(labels['emotion']))}")
    print(f"  mu range: [{mu.min():.3f}, {mu.max():.3f}], "
          f"mean {mu.mean():.3f}, std {mu.std():.3f}")


if __name__ == "__main__":
    main()

"""
Extract latent vectors from the frozen VAE encoder for the full UTKFace set.

Saves mu (deterministic, not sampled), logvar, and all demographic labels
to a single .npz. Everything downstream reads from this file, so after this
script runs you never need the GPU again.
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from model import VAE
from utkface_dataset import UTKFaceDataset, collate_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str,
                        default='../../checkpoints/vae_best.pt')
    parser.add_argument('--output', type=str,
                        default='../../latents/utkface_latents.npz')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_size', type=float, default=0.2)
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

    # No augmentation at extraction time: deterministic representations only
    eval_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    ds = UTKFaceDataset(args.data_root, transform=eval_tf)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_labels,
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

    # Stratify on race, the most imbalanced attribute, so every group is
    # represented in both probe-train and probe-test.
    idx = np.arange(len(ds))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed,
        stratify=labels['race'],
    )
    split = np.empty(len(ds), dtype='<U5')
    split[train_idx] = 'train'
    split[test_idx] = 'test'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        mu=mu,
        logvar=logvar,
        age=labels['age'],
        age_bucket=labels['age_bucket'],
        gender=labels['gender'],
        race=labels['race'],
        filename=labels['filename'],
        split=split,
        checkpoint_epoch=ckpt['epoch'],
        latent_dim=latent_dim,
    )

    print(f"\nSaved {mu.shape[0]} latents of dim {mu.shape[1]} to {args.output}")
    print(f"  probe-train: {(split == 'train').sum()}")
    print(f"  probe-test:  {(split == 'test').sum()}")
    print(f"  mu range: [{mu.min():.3f}, {mu.max():.3f}], "
          f"mean {mu.mean():.3f}, std {mu.std():.3f}")


if __name__ == "__main__":
    main()
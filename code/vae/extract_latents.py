"""
Extract 128-dim latent vectors from the frozen VAE encoder for the RAF-DB test set.
Saves latents, emotion labels, and image identifiers to a single .npz file.
This is the input to every probe and fairness analysis downstream.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import VAE
from train import RAFDBDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to vae_best.pt')
    parser.add_argument('--output', type=str, default='../../latents/rafdb_test_latents.npz')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt['args']['latent_dim']
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, latent_dim={latent_dim}")

    # Same preprocessing as training but no augmentation
    eval_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    ds = RAFDBDataset(args.data_root, split=args.split, transform=eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"{args.split} samples: {len(ds)}")

    all_mu = []
    all_logvar = []
    all_labels = []
    all_names = [name for name, _ in ds.samples]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            mu, logvar = model.encoder(x)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_labels.append(y.numpy())

    mu_arr = np.concatenate(all_mu, axis=0)
    logvar_arr = np.concatenate(all_logvar, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        mu=mu_arr,
        logvar=logvar_arr,
        emotion=labels_arr,
        names=np.array(all_names),
    )
    print(f"Saved {mu_arr.shape[0]} latents of dim {mu_arr.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
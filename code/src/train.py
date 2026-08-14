"""
Train a ResNet-18 style VAE on UTKFace or RAF-DB.

Deliberately naive: reconstruction + KL loss only.
No demographic supervision, no fairness constraint, no disentanglement term.
The point is to observe what the representation learns when we never tell it
what not to learn.

UTKFace trains on all images; the probe split is drawn later in
extract_latents.py. RAF-DB trains only on its native train split, so the
official test images are never seen by the encoder.
"""

from fedar_common.data import build_dataset
import os
import argparse
import time

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms

from model import VAE, vae_loss


def save_recon_grid(model, loader, device, epoch, out_dir, n=8):
    """
    Save a grid: top row original images, bottom row their reconstructions.
    This is the fastest way to tell whether training is actually working.
    """
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x[:n].to(device)
        recon, _, _ = model(x)
        grid = torch.cat([x.cpu(), recon.cpu()], dim=0)
        os.makedirs(out_dir, exist_ok=True)
        vutils.save_image(
            grid,
            os.path.join(out_dir, f'recon_epoch{epoch:03d}.png'),
            nrow=n,
            normalize=False,
        )
    model.train()




def train_one_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    n_batches = 0

    for x, _ in loader:                      # labels ignored: the VAE never sees them
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_l
        total_kl += kl_l
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='UTKFace flat image folder, or RAF-DB root')
    parser.add_argument('--dataset', choices=['utkface', 'rafdb'], default='utkface')
    parser.add_argument('--demographics', type=str, default=None,
                        help='Path to rafdb_demographics.csv (rafdb only)')
    parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoints')
    parser.add_argument('--figure_dir', type=str, default='../../figures/recon')
    parser.add_argument('--log_file', type=str, default='../../logs/train.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")

    # Guard against silently overwriting the other dataset's checkpoint.
    existing = os.path.join(args.checkpoint_dir, 'vae_best.pt')
    if os.path.exists(existing):
        try:
            prev = torch.load(existing, map_location='cpu')['args'].get('dataset',
                                                                       'utkface')
            if prev != args.dataset:
                print(f"\n  WARNING: {existing} was trained on '{prev}' but you are "
                      f"training '{args.dataset}'.\n  It will be overwritten. "
                      f"Pass a different --checkpoint_dir if that is not intended.\n")
        except Exception:
            pass

    # Horizontal flip only. No colour jitter, which would interfere with
    # exactly the appearance attributes we are studying.
    train_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # rafdb_split='train' holds out the official RAF-DB test split during VAE
    # training (no-op for UTKFace). This was the one behavioural difference
    # between train.py's old private build_dataset and the downstream copies.
    train_ds, collate = build_dataset(args, train_tf, rafdb_split='train')
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    print(f"Train samples: {len(train_ds)}  |  batches/epoch: {len(train_loader)}")

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: latent_dim={args.latent_dim}, beta={args.beta}, "
          f"lr={args.lr}, batch_size={args.batch_size}\n")

    log_f = open(args.log_file, 'w')
    log_f.write("epoch,loss,recon,kl,time_sec\n")

    snapshot_epochs = {1, 5, 10, 25, args.epochs}
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, recon, kl = train_one_epoch(model, train_loader, optimizer, device, args.beta)
        dt = time.time() - t0

        print(f"epoch {epoch:3d}/{args.epochs} | loss {loss:9.2f} | "
              f"recon {recon:9.2f} | kl {kl:7.2f} | {dt:5.1f}s", flush=True)
        log_f.write(f"{epoch},{loss:.4f},{recon:.4f},{kl:.4f},{dt:.2f}\n")
        log_f.flush()

        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': vars(args),
            }, os.path.join(args.checkpoint_dir, 'vae_best.pt'))

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'args': vars(args),
            }, os.path.join(args.checkpoint_dir, f'vae_epoch{epoch:03d}.pt'))

        if epoch in snapshot_epochs:
            save_recon_grid(model, train_loader, device, epoch, args.figure_dir)

    log_f.close()
    print(f"\nDone. Best loss: {best_loss:.2f}")
    print(f"Checkpoint: {os.path.join(args.checkpoint_dir, 'vae_best.pt')}")
    print(f"Reconstruction grids: {args.figure_dir}")


if __name__ == "__main__":
    main()

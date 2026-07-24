"""
Train the ResNet-18 VAE on RAF-DB training set.
Standard reconstruction + KL loss only, no fairness or disentanglement objectives.
The point is to demonstrate what a *vanilla* VAE encodes.
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import VAE, vae_loss


class RAFDBDataset(Dataset):
    """
    RAF-DB loader for VAE training.
    Reads the aligned face images from basic/Image/aligned/.
    Uses list_patition_label.txt to filter train (prefix 'train_') vs test (prefix 'test_').
    """
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.transform = transform
        label_file = self.root / 'basic' / 'EmoLabel' / 'list_patition_label.txt'
        with open(label_file, 'r') as f:
            lines = [l.strip().split() for l in f if l.strip()]
        prefix = 'train' if split == 'train' else 'test'
        self.samples = [(name, int(label)) for name, label in lines if name.startswith(prefix)]
        self.img_dir = self.root / 'basic' / 'Image' / 'aligned'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        # RAF-DB aligned images are named like train_00001_aligned.jpg
        img_name = name.replace('.jpg', '_aligned.jpg')
        img_path = self.img_dir / img_name
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label - 1  # emotion labels are 1-indexed in the file


def train_one_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    n_batches = 0
    for x, _ in loader:
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
                        help='Path to RAF-DB root (containing basic/)')
    parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoints')
    parser.add_argument('--log_file', type=str, default='../../logs/train.log')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Preprocessing: resize aligned faces to 96x96, normalize to [0, 1]
    train_tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    train_ds = RAFDBDataset(args.data_root, split='train', transform=train_tf)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    print(f"Train samples: {len(train_ds)}")

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    log_f = open(args.log_file, 'w')
    log_f.write("epoch,loss,recon,kl,time_sec\n")

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, recon, kl = train_one_epoch(model, train_loader, optimizer, device, args.beta)
        dt = time.time() - t0
        msg = f"epoch {epoch:3d} | loss {loss:8.2f} | recon {recon:8.2f} | kl {kl:6.2f} | {dt:.1f}s"
        print(msg)
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
            }, os.path.join(args.checkpoint_dir, f'vae_epoch{epoch}.pt'))

    log_f.close()
    print(f"Done. Best loss: {best_loss:.2f}")


if __name__ == "__main__":
    main()
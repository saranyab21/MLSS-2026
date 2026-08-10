"""
UTKFace dataset loader with demographic label parsing.

Filename format: [age]_[gender]_[race]_[timestamp].jpg.chip.jpg
  age:    int, 0-116
  gender: 0 = male, 1 = female
  race:   0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Other

A small number of files are malformed (missing the race field) and are skipped.
"""

import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


GENDER_NAMES = {0: 'Male', 1: 'Female'}
RACE_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}

# Age buckets chosen to keep each class populated given UTKFace's skew toward 20-40
AGE_BUCKETS = [(0, 19), (20, 34), (35, 49), (50, 116)]
AGE_BUCKET_NAMES = {0: '0-19', 1: '20-34', 2: '35-49', 3: '50+'}


def age_to_bucket(age):
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        if lo <= age <= hi:
            return i
    return None


def parse_filename(fname):
    """
    Parse UTKFace filename into (age, gender, race).
    Returns None if the filename is malformed.
    """
    stem = fname.split('.')[0]           # strip .jpg.chip.jpg
    parts = stem.split('_')
    if len(parts) != 4:
        return None
    try:
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if gender not in GENDER_NAMES or race not in RACE_NAMES:
        return None
    if not (0 <= age <= 116):
        return None
    return age, gender, race


class UTKFaceDataset(Dataset):
    """
    Returns (image_tensor, label_dict) where label_dict contains
    age, age_bucket, gender, race, and the filename.
    """

    def __init__(self, root, transform=None, indices=None, verbose=True):
        self.root = Path(root)
        self.transform = transform

        all_files = sorted(self.root.glob('*.jpg'))
        self.samples = []
        n_skipped = 0

        for f in all_files:
            parsed = parse_filename(f.name)
            if parsed is None:
                n_skipped += 1
                continue
            age, gender, race = parsed
            bucket = age_to_bucket(age)
            if bucket is None:
                n_skipped += 1
                continue
            self.samples.append({
                'path': f,
                'age': age,
                'age_bucket': bucket,
                'gender': gender,
                'race': race,
                'filename': f.name,
            })

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        if verbose:
            print(f"UTKFaceDataset: {len(self.samples)} valid, {n_skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        labels = {
            'age': s['age'],
            'age_bucket': s['age_bucket'],
            'gender': s['gender'],
            'race': s['race'],
        }
        return img, labels

    def get_label_arrays(self):
        """Return all labels as numpy arrays, for stratification and probes."""
        return {
            'age': np.array([s['age'] for s in self.samples]),
            'age_bucket': np.array([s['age_bucket'] for s in self.samples]),
            'gender': np.array([s['gender'] for s in self.samples]),
            'race': np.array([s['race'] for s in self.samples]),
            'filename': np.array([s['filename'] for s in self.samples]),
        }

    def composition_report(self):
        """Print the dataset composition table for the poster."""
        n = len(self.samples)
        labels = self.get_label_arrays()

        print(f"\n{'='*55}")
        print(f"UTKFace composition (N = {n})")
        print(f"{'='*55}")

        print("\nGender")
        for k, cnt in sorted(Counter(labels['gender']).items()):
            print(f"  {GENDER_NAMES[k]:<10} {cnt:>6}  ({100*cnt/n:5.1f}%)")

        print("\nRace")
        for k, cnt in sorted(Counter(labels['race']).items()):
            print(f"  {RACE_NAMES[k]:<10} {cnt:>6}  ({100*cnt/n:5.1f}%)")

        print("\nAge bucket")
        for k, cnt in sorted(Counter(labels['age_bucket']).items()):
            print(f"  {AGE_BUCKET_NAMES[k]:<10} {cnt:>6}  ({100*cnt/n:5.1f}%)")

        print("\nGender x Race")
        header = "  " + " " * 10 + "".join(f"{RACE_NAMES[r]:>9}" for r in range(5))
        print(header)
        for g in range(2):
            row = f"  {GENDER_NAMES[g]:<10}"
            for r in range(5):
                cnt = int(((labels['gender'] == g) & (labels['race'] == r)).sum())
                row += f"{cnt:>9}"
            print(row)

        print(f"\nAge: min {labels['age'].min()}, max {labels['age'].max()}, "
              f"median {int(np.median(labels['age']))}")
        print(f"{'='*55}\n")


def collate_labels(batch):
    """Custom collate so label dicts stack into tensors."""
    imgs = torch.stack([b[0] for b in batch])
    labels = {
        k: torch.tensor([b[1][k] for b in batch])
        for k in batch[0][1].keys()
    }
    return imgs, labels


if __name__ == "__main__":
    import argparse
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=str(Path.home() / 'mlss-poster/data/utkface'))
    args = parser.parse_args()

    tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    ds = UTKFaceDataset(args.data_root, transform=tf)
    ds.composition_report()

    # Sanity check one sample
    img, labels = ds[0]
    print(f"Sample 0: image {tuple(img.shape)}, "
          f"range [{img.min():.3f}, {img.max():.3f}]")
    print(f"  age={labels['age']}, bucket={AGE_BUCKET_NAMES[labels['age_bucket']]}, "
          f"gender={GENDER_NAMES[labels['gender']]}, race={RACE_NAMES[labels['race']]}")
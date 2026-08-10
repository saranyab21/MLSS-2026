"""
Annotate RAF-DB with demographic attributes using the FairFace classifier.

RAF-DB's official release carries human race/age/gender annotations, but the
public Kaggle mirrors redistribute only the emotion labels. We therefore infer
demographics with FairFace (Karkkainen & Joo, WACV 2021), the model used by
the published fairness-in-FER studies we compare against.

These labels are MODEL-INFERRED, not human ground truth. Every downstream
claim must be stated as such.

Output schema matches the DeepFace version, so rafdb_dataset.py is unchanged.
"""

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# FairFace head layout: 18 outputs = 7 race + 2 gender + 9 age
FF_RACE = ['White', 'Black', 'Latino_Hispanic', 'East Asian',
           'Southeast Asian', 'Indian', 'Middle Eastern']
FF_GENDER = ['Male', 'Female']
FF_AGE = ['0-2', '3-9', '10-19', '20-29', '30-39',
          '40-49', '50-59', '60-69', '70+']

# Collapse FairFace's 7 races onto UTKFace's 5-class scheme so the two datasets
# share a race axis on the poster. The raw 7-class label is kept in race_raw.
FF_TO_UTK = {
    'White': 0, 'Black': 1,
    'East Asian': 2, 'Southeast Asian': 2,
    'Indian': 3,
    'Latino_Hispanic': 4, 'Middle Eastern': 4,   # -> Other
}

# Representative age per FairFace bin, fed through the shared bucket function.
FF_AGE_MIDPOINT = [1, 6, 15, 25, 35, 45, 55, 65, 75]

AGE_BUCKETS = [(0, 19), (20, 34), (35, 49), (50, 116)]


def age_to_bucket(age):
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        if lo <= age <= hi:
            return i
    return 3


def build_model(weights_path, device):
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


class ImageFolderFlat(torch.utils.data.Dataset):
    """Every image under root, whatever the folder layout."""

    def __init__(self, root, transform, limit=None, seed=42):
        self.root = Path(root)
        files = []
        for e in ('*.jpg', '*.jpeg', '*.png'):
            files.extend(self.root.rglob(e))
        files = sorted(files)
        if limit and limit < len(files):
            # random subset, not the first N, so a smoke test is representative
            rng = np.random.default_rng(seed)
            files = [files[i] for i in
                     sorted(rng.choice(len(files), limit, replace=False))]
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        return self.transform(Image.open(p).convert('RGB')), i


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image_root', required=True)
    p.add_argument('--weights', required=True,
                   help='Path to res34_fair_align_multi_7_20190809.pt')
    p.add_argument('--output', default='../../data/rafdb_demographics.csv')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = ImageFolderFlat(args.image_root, tf, args.limit, args.seed)
    print(f"Found {len(ds)} images under {args.image_root}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.weights, device)

    rows = []
    with torch.no_grad():
        for x, idx in tqdm(loader, desc='annotating'):
            out = model(x.to(device, non_blocking=True)).cpu()

            race_p = torch.softmax(out[:, :7], dim=1)
            gend_p = torch.softmax(out[:, 7:9], dim=1)
            age_p = torch.softmax(out[:, 9:18], dim=1)

            race_i = race_p.argmax(1).numpy()
            race_c = race_p.max(1).values.numpy()
            gend_i = gend_p.argmax(1).numpy()
            age_i = age_p.argmax(1).numpy()

            for j, k in enumerate(idx.numpy()):
                path = ds.files[k]
                raw = FF_RACE[race_i[j]]
                age = FF_AGE_MIDPOINT[age_i[j]]
                rows.append({
                    'filename': path.name,
                    'relpath': str(path.relative_to(ds.root)),
                    'age': age,
                    'age_bucket': age_to_bucket(age),
                    'gender': int(gend_i[j]),          # 0 Male, 1 Female
                    'race': FF_TO_UTK[raw],
                    'race_raw': raw,
                    'race_confidence': round(100 * float(race_c[j]), 1),
                    'age_bin_raw': FF_AGE[age_i[j]],
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")

    # composition summary
    import collections
    n = len(rows)
    print(f"\nInferred composition (N = {n})")
    for key, names in [('gender', FF_GENDER),
                       ('race', ['White', 'Black', 'Asian', 'Indian', 'Other']),
                       ('age_bucket', ['0-19', '20-34', '35-49', '50+'])]:
        c = collections.Counter(r[key] for r in rows)
        print(f"  {key}")
        for k in sorted(c):
            print(f"    {names[k]:<10} {c[k]:>6}  ({100*c[k]/n:5.1f}%)")

    print("\n  raw FairFace race (7-class)")
    c = collections.Counter(r['race_raw'] for r in rows)
    for k, v in c.most_common():
        print(f"    {k:<18} {v:>6}  ({100*v/n:5.1f}%)")

    conf = np.array([r['race_confidence'] for r in rows])
    print(f"\n  mean race confidence {conf.mean():.1f}%  "
          f"(below ~60% suggests unreliable labels)")


if __name__ == "__main__":
    main()
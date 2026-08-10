"""
RAF-DB loader with emotion labels and model-inferred demographic attributes.

Emotion labels come from the dataset itself (human-annotated, 7 basic classes).
Demographic labels come from annotate_demographics.py and are MODEL-INFERRED.

Interface mirrors utkface_dataset.py exactly, so extract_latents.py, probes.py
and federated.py work with either dataset.
"""

import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


GENDER_NAMES = {0: 'Male', 1: 'Female'}
RACE_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
AGE_BUCKET_NAMES = {0: '0-19', 1: '20-34', 2: '35-49', 3: '50+'}

# RAF-DB basic emotion labels are 1-indexed in the official label file.
EMOTION_NAMES = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happy',
                 4: 'Sad', 5: 'Anger', 6: 'Neutral'}


def _load_emotion_labels(root):
    """
    Read emotion labels, tolerating the two layouts the public mirrors use.

    Layout A (original):  EmoLabel/list_patition_label.txt   "train_00001.jpg 5"
    Layout B (Kaggle):    train_labels.csv / test_labels.csv  with image,label
    Layout C (folders):   DATASET/train/<class>/<file>.jpg
    Returns {filename_stem: (emotion_0indexed, split_hint)}.
    """
    root = Path(root)
    labels = {}

    # A
    for cand in root.rglob('list_patition_label.txt'):
        with open(cand) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    stem = Path(parts[0]).stem
                    labels[stem] = (int(parts[1]) - 1,
                                    'train' if parts[0].startswith('train') else 'test')
        if labels:
            return labels

    # B
    for cand in root.rglob('*_labels.csv'):
        split = 'train' if 'train' in cand.name else 'test'
        with open(cand) as f:
            for row in csv.DictReader(f):
                keys = {k.lower(): v for k, v in row.items()}
                name = keys.get('image') or keys.get('filename') or keys.get('name')
                lab = keys.get('label') or keys.get('emotion') or keys.get('class')
                if name is None or lab is None:
                    continue
                lab = int(lab)
                labels[Path(name).stem] = (lab - 1 if lab >= 1 else lab, split)
    if labels:
        return labels

    # C
    for img in root.rglob('*.jpg'):
        parts = img.parts
        for i, part in enumerate(parts):
            if part.lower() in ('train', 'test') and i + 1 < len(parts) - 1:
                cls = parts[i + 1]
                if cls.isdigit():
                    labels[img.stem] = (int(cls) - 1, part.lower())
                break
    return labels


class RAFDBDataset(Dataset):
    """
    Returns (image_tensor, label_dict) with keys:
      emotion, age, age_bucket, gender, race
    matching UTKFaceDataset plus the emotion target.
    """

    def __init__(self, root, demographics_csv, transform=None,
                 split=None, indices=None, verbose=True):
        self.root = Path(root)
        self.transform = transform

        emo = _load_emotion_labels(self.root)
        if not emo:
            raise RuntimeError(
                f"No emotion labels found under {root}. Inspect the folder "
                f"layout and extend _load_emotion_labels()."
            )

        demo = {}
        with open(demographics_csv) as f:
            for row in csv.DictReader(f):
                demo[Path(row['filename']).stem] = row

        by_stem = {}
        for p in self.root.rglob('*.jpg'):
            by_stem.setdefault(p.stem, p)

        self.samples = []
        n_no_demo = n_no_img = 0

        for stem, (emotion, split_hint) in sorted(emo.items()):
            if split is not None and split_hint != split:
                continue
            path = by_stem.get(stem)
            if path is None:
                n_no_img += 1
                continue
            d = demo.get(stem)
            if d is None:
                n_no_demo += 1
                continue
            self.samples.append({
                'path': path,
                'filename': path.name,
                'emotion': emotion,
                'age': int(d['age']),
                'age_bucket': int(d['age_bucket']),
                'gender': int(d['gender']),
                'race': int(d['race']),
                'orig_split': split_hint,
            })

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        if verbose:
            print(f"RAFDBDataset: {len(self.samples)} valid"
                  + (f", {n_no_demo} missing demographics" if n_no_demo else "")
                  + (f", {n_no_img} missing images" if n_no_img else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        labels = {
            'emotion': s['emotion'],
            'age': s['age'],
            'age_bucket': s['age_bucket'],
            'gender': s['gender'],
            'race': s['race'],
        }
        return img, labels

    def get_label_arrays(self):
        return {
            'emotion': np.array([s['emotion'] for s in self.samples]),
            'age': np.array([s['age'] for s in self.samples]),
            'age_bucket': np.array([s['age_bucket'] for s in self.samples]),
            'gender': np.array([s['gender'] for s in self.samples]),
            'race': np.array([s['race'] for s in self.samples]),
            'filename': np.array([s['filename'] for s in self.samples]),
            'orig_split': np.array([s['orig_split'] for s in self.samples]),
        }

    def composition_report(self):
        n = len(self.samples)
        L = self.get_label_arrays()

        print(f"\n{'='*58}")
        print(f"RAF-DB composition (N = {n})")
        print(f"demographics are MODEL-INFERRED (FairFace), not ground truth")
        print(f"{'='*58}")

        print("\nEmotion (human-annotated)")
        for k, c in sorted(Counter(L['emotion']).items()):
            print(f"  {EMOTION_NAMES[k]:<10} {c:>6}  ({100*c/n:5.1f}%)")

        for key, names in [('gender', GENDER_NAMES),
                           ('race', RACE_NAMES),
                           ('age_bucket', AGE_BUCKET_NAMES)]:
            print(f"\n{key.replace('_', ' ').title()} (inferred)")
            for k, c in sorted(Counter(L[key]).items()):
                print(f"  {names[k]:<10} {c:>6}  ({100*c/n:5.1f}%)")

        print("\nGender x Race")
        print("  " + " " * 10 + "".join(f"{RACE_NAMES[r]:>9}" for r in range(5)))
        for g in range(2):
            row = f"  {GENDER_NAMES[g]:<10}"
            for r in range(5):
                row += f"{int(((L['gender'] == g) & (L['race'] == r)).sum()):>9}"
            print(row)

        print(f"\nOriginal split: "
              f"{(L['orig_split'] == 'train').sum()} train / "
              f"{(L['orig_split'] == 'test').sum()} test")
        print(f"{'='*58}\n")


def collate_labels(batch):
    """Identical signature to utkface_dataset.collate_labels."""
    imgs = torch.stack([b[0] for b in batch])
    labels = {k: torch.tensor([b[1][k] for b in batch])
              for k in batch[0][1].keys()}
    return imgs, labels


if __name__ == "__main__":
    import argparse
    from torchvision import transforms

    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=str(Path.home() / 'mlss-poster/data/rafdb_raw'))
    p.add_argument('--demographics', default=str(Path.home() / 'mlss-poster/data/rafdb_demographics.csv'))
    args = p.parse_args()

    tf = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
    ds = RAFDBDataset(args.data_root, args.demographics, transform=tf)
    ds.composition_report()

    img, labels = ds[0]
    print(f"Sample 0: image {tuple(img.shape)}, range [{img.min():.3f}, {img.max():.3f}]")
    print(f"  emotion={EMOTION_NAMES[labels['emotion']]}, "
          f"gender={GENDER_NAMES[labels['gender']]}, "
          f"race={RACE_NAMES[labels['race']]}, "
          f"age={labels['age']}")
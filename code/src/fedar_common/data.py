"""
Dataset construction — the single source of truth.

Before this module, build_dataset() was copy-pasted into train.py,
extract_latents.py, pixel_baseline.py and debias.py. Three of those copies
were identical; train.py's differed in ONE important way — it passes
split='train' to RAFDBDataset so the official test split is held out during
VAE training, whereas the downstream scripts pass split=None to extract over
train AND test. That difference is real and load-bearing, so it is preserved
here as an explicit argument rather than silently collapsed.

    build_dataset(args, transform, rafdb_split=None)

    train.py            -> rafdb_split='train'   (hold out test during training)
    extract/pixel/debias-> rafdb_split=None      (all rows; split recorded per-sample)
"""
from utkface_dataset import UTKFaceDataset, collate_labels, age_to_bucket
from rafdb_dataset import RAFDBDataset, collate_labels as collate_rafdb

__all__ = ['build_dataset', 'age_to_bucket', 'collate_labels', 'collate_rafdb']


def build_dataset(args, transform, rafdb_split=None):
    """Return (dataset, collate_fn) for whichever corpus args.dataset names.

    rafdb_split is passed straight through to RAFDBDataset:
      'train' holds out the official test split (used by train.py);
      None extracts over the whole corpus (used by every downstream script,
      which then reads the per-sample split array from the cached latents).
    UTKFace ignores it — the split is drawn later from the cached latents.
    """
    if getattr(args, 'dataset', 'utkface') == 'rafdb':
        if not getattr(args, 'demographics', None):
            raise ValueError(
                "--demographics is required for --dataset rafdb. "
                "Run annotate_demographics.py first."
            )
        ds = RAFDBDataset(args.data_root, args.demographics,
                          transform=transform, split=rafdb_split)
        return ds, collate_rafdb

    ds = UTKFaceDataset(args.data_root, transform=transform)
    return ds, collate_labels

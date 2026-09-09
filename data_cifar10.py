"""CIFAR-10 train/test loaded from HuggingFace instead of torchvision.

torchvision pulls from cs.toronto.edu, which measured ~300 B/s from this
machine (a 170 MB download would take ~6 days). The same data on HF measured
~3.3 MB/s. `uoft-cs/cifar10` is also the source the study plan names, so this
is the intended provenance either way.

Arrays are cached once as .npz; after that nothing touches the network.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HF_REPO = "uoft-cs/cifar10"


def _load_split(split, cache_dir, mmap=True):
    """Return (images, labels) for one split.

    Cached as uncompressed .npy and opened with mmap_mode='r' so that every
    dataloader worker shares one set of OS page-cache pages instead of holding
    its own 150 MB copy. On Windows each worker is a fresh process that
    re-imports torch, so worker memory is the binding constraint on this
    machine; a compressed .npz cannot be mapped and forces a per-worker copy.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    img_path = cache / f"cifar10_{split}_images.npy"
    lbl_path = cache / f"cifar10_{split}_labels.npy"
    npz_path = cache / f"cifar10_{split}.npz"

    if not img_path.exists():
        if npz_path.exists():  # migrate the earlier compressed cache in place
            with np.load(npz_path) as d:
                images, labels = d["images"], d["labels"]
        else:
            from datasets import load_dataset

            ds = load_dataset(HF_REPO, split=split)
            img_key = "img" if "img" in ds.column_names else "image"
            images = np.stack([np.asarray(r.convert("RGB"), dtype=np.uint8)
                               for r in ds[img_key]])
            labels = np.asarray(ds["label"], dtype=np.int64)
        np.save(img_path, images)
        np.save(lbl_path, labels)

    mode = "r" if mmap else None
    return np.load(img_path, mmap_mode=mode), np.load(lbl_path, mmap_mode=mode)


class CIFAR10Numpy(Dataset):
    """Drop-in for torchvision.datasets.CIFAR10 over HF-sourced arrays.

    Transforms receive a PIL image, exactly as torchvision's version does, so
    the augmentation pipeline in train.py is unchanged.
    """

    def __init__(self, root="./data", train=True, transform=None, download=True):
        self.images, self.labels = _load_split("train" if train else "test", root)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


if __name__ == "__main__":
    for split in ("train", "test"):
        imgs, lbls = _load_split(split, "./data")
        print(f"{split}: images {imgs.shape} {imgs.dtype}, labels {lbls.shape}, "
              f"classes {sorted(set(lbls.tolist()))}")

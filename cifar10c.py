"""CIFAR-10-C loader for WNJXYK/TTA-CIFAR-10-C (HF Parquet, CC-BY-4.0).

The plan's evaluate.py assumed the Zenodo .npy layout. The confirmed-ungated
source ships Parquet instead, one config per corruption type with a split per
severity. This module hides that difference behind a numpy interface and caches
each (corruption, severity) pair as an .npz so the download happens once.

    images, labels = load_corruption("brightness", 3)   # (10000,32,32,3) uint8, (10000,)
"""

import numpy as np

CORRUPTIONS_4 = ["brightness", "contrast", "defocus_blur", "elastic_transform"]
CORRUPTIONS_15 = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
HF_REPO = "WNJXYK/TTA-CIFAR-10-C"
N_SEVERITIES = 5


def load_corruption(corruption, severity, cache_dir="./data/CIFAR-10-C"):
    """Return (images uint8 (N,32,32,3), labels int64 (N,)) for one severity.

    Cached as uncompressed .npy and memory-mapped: the evaluation sweep reads
    every split once per model, so a compressed .npz would decompress 30 MB
    twelve times per split. Mapped pages are shared and effectively free after
    the first read. An existing .npz cache is migrated in place.
    """
    from pathlib import Path

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    img_path = cache / f"{corruption}_s{severity}_images.npy"
    lbl_path = cache / f"{corruption}_s{severity}_labels.npy"
    npz_path = cache / f"{corruption}_s{severity}.npz"

    if not img_path.exists():
        if npz_path.exists():
            with np.load(npz_path) as d:
                images, labels = d["images"], d["labels"]
        else:
            from datasets import load_dataset

            ds = load_dataset(HF_REPO, corruption, split=f"severity_{severity}")
            images = np.stack([np.asarray(r.convert("RGB"), dtype=np.uint8)
                               for r in ds["image"]])
            labels = np.asarray(ds["label"], dtype=np.int64)
        np.save(img_path, images)
        np.save(lbl_path, labels)

    return np.load(img_path, mmap_mode="r"), np.load(lbl_path, mmap_mode="r")


def prefetch(corruptions, cache_dir="./data/CIFAR-10-C"):
    """Download and cache every (corruption, severity) pair up front."""
    for c in corruptions:
        for s in range(1, N_SEVERITIES + 1):
            imgs, _ = load_corruption(c, s, cache_dir)
            print(f"  cached {c} severity {s}: {imgs.shape}", flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pre-download CIFAR-10-C splits.")
    p.add_argument("--corruptions", nargs="+", default=CORRUPTIONS_4)
    p.add_argument("--cache-dir", default="./data/CIFAR-10-C")
    a = p.parse_args()
    prefetch(a.corruptions, a.cache_dir)
    print("done.")

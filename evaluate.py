"""Evaluate one trained checkpoint on clean CIFAR-10 and on CIFAR-10-C.

Expects CIFAR-10-C (Hendrycks & Dietterich, 2019 -- https://zenodo.org/record/2535967)
extracted under --corruption-dir as one .npy file per corruption type
(e.g. brightness.npy) plus a shared labels.npy. Each corruption file is
(50000, 32, 32, 3) uint8: 5 severity levels of 10000 images each, in
increasing-severity order; labels.npy (50000,) is the same for every
corruption type (the original test labels, tiled 5x in that same order).

Usage:
    python evaluate.py --model SE --seed 1
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from cifar10c import CORRUPTIONS_4, N_SEVERITIES, load_corruption
from data_cifar10 import CIFAR10Numpy
from model import resnet_cifar

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
VARIANTS = ["none", "SE", "BAM", "CBAM"]
MODEL_CHOICES = {"none": None, "SE": "SE", "BAM": "BAM", "CBAM": "CBAM"}


def to_normalized_tensor(images_uint8, device):
    # np.ascontiguousarray materialises the slice: the corruption cache is
    # memory-mapped and read-only, which torch.from_numpy will not accept.
    x = torch.from_numpy(np.ascontiguousarray(images_uint8)).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device)


@torch.no_grad()
def eval_numpy_batches(model, images, labels, device, batch_size):
    model.eval()
    correct, n = 0, images.shape[0]
    for i in range(0, n, batch_size):
        x = to_normalized_tensor(images[i:i + batch_size], device)
        y = torch.from_numpy(np.ascontiguousarray(labels[i:i + batch_size])).long().to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
    return 100.0 * correct / n


@torch.no_grad()
def eval_clean(model, data_dir, device, batch_size):
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_set = CIFAR10Numpy(root=data_dir, train=False, download=True, transform=test_tf)
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = resnet_cifar(attention_type=MODEL_CHOICES[ckpt["attention_type"]], num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on clean CIFAR-10 + CIFAR-10-C.")
    parser.add_argument("--model", required=True, choices=VARIANTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt-dir", default="./runs")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--corruption-dir", default="./data/CIFAR-10-C")
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS_4)
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_name = f"resnet18_{args.model.lower()}_seed{args.seed}"
    ckpt_path = Path(args.ckpt_dir) / f"{run_name}_best.pth"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_path, device)

    clean_acc = eval_clean(model, args.data_dir, device, args.batch_size)
    print(f"[{run_name}] clean top-1 accuracy: {clean_acc:.2f}%")

    rows = [{"model": args.model, "seed": args.seed, "corruption": "clean", "severity": 0, "accuracy": clean_acc}]
    per_corruption_mean = {}

    for corruption in args.corruptions:
        sev_accs = []
        for severity in range(1, N_SEVERITIES + 1):
            images, labels = load_corruption(corruption, severity, args.corruption_dir)
            acc = eval_numpy_batches(model, images, labels, device, args.batch_size)
            sev_accs.append(acc)
            rows.append({"model": args.model, "seed": args.seed, "corruption": corruption, "severity": severity, "accuracy": acc})
            print(f"  {corruption} severity {severity}: {acc:.2f}%")
        per_corruption_mean[corruption] = float(np.mean(sev_accs))

    all_sev_accs = [r["accuracy"] for r in rows if r["corruption"] != "clean"]
    mca = float(np.mean(all_sev_accs)) if all_sev_accs else float("nan")
    rel_drop = (clean_acc - mca) / clean_acc if all_sev_accs and clean_acc > 0 else float("nan")

    csv_path = out_dir / f"{run_name}_corruption_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "corruption", "severity", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_name": run_name, "model": args.model, "seed": args.seed,
        "clean_top1": clean_acc, "mCA": mca, "relative_robustness_drop": rel_drop,
        "per_corruption_mean_accuracy": per_corruption_mean,
    }
    summary_path = out_dir / f"{run_name}_corruption_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{run_name}] mCA={mca:.2f}%  relative_robustness_drop={rel_drop:.4f}")
    print(f"Saved: {csv_path}, {summary_path}")


if __name__ == "__main__":
    main()

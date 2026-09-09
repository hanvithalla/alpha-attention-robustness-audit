"""Train one (attention_type, seed) arm of the matched-budget audit.

Designed to run in short installments: full training state (model,
optimizer, scheduler, epoch counter, RNG state) is checkpointed to
"<run_name>_state.pth" after every epoch, and is auto-resumed the next
time you invoke this script for the same --model/--seed. Use
--time-limit-min to cap how long a single invocation runs before it
checkpoints and exits cleanly -- run it again (same command) to continue.

Usage:
    python train.py --model SE --seed 1 --epochs 100
    python train.py --model SE --seed 1 --epochs 100 --time-limit-min 30   # one installment
    python train.py --model none --seed 1 --epochs 1 --limit-batches 5    # fast smoke test

Every arm uses identical data augmentation, optimizer, LR schedule, batch
size, and total epoch count -- only --model (the attention slot) and
--seed vary, so any accuracy difference is attributable to the attention
mechanism.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from data_cifar10 import CIFAR10Numpy
from model import resnet_cifar

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

MODEL_CHOICES = {
    "none": None,
    "SE": "SE",
    "BAM": "BAM",
    "CBAM": "CBAM",
}

# Hyperparameters fixed at the first installment and reused on every resume,
# so a run's total training config can't drift between installments.
FROZEN_CONFIG_KEYS = ["model", "seed", "epochs", "batch_size", "lr", "momentum", "weight_decay", "scheduler", "amp"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    # set_rng_state requires CPU ByteTensors. Checkpoints written before the
    # map_location fix below were loaded onto the GPU, so coerce defensively
    # rather than failing the resume of an already-trained run.
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])


def get_dataloaders(data_dir, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = CIFAR10Numpy(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = CIFAR10Numpy(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train, limit_batches=None,
              scaler=None, use_amp=False):
    """One pass over `loader`.

    Mixed precision is applied to *training only*. Validation stays in fp32 so
    that the val_acc driving best-checkpoint selection is computed exactly the
    way evaluate.py computes the final reported accuracy -- otherwise the
    checkpoint chosen here and the number in the paper come from two slightly
    different numeric paths.
    """
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    amp_on = bool(use_amp and train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, (inputs, targets) in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if train:
                if amp_on:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    return total_loss / total, 100.0 * correct / total


def build_run_objects(args, device):
    model = resnet_cifar(attention_type=MODEL_CHOICES[args.model], num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return model, criterion, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train one arm of the attention-robustness audit.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CHOICES), help="Attention variant.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100, help="Total target epochs across all installments.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None,
                         help="Mixed-precision training. Defaults to on for CUDA, off for CPU.")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                         help="Force full fp32 training.")
    parser.add_argument("--time-limit-min", type=float, default=None,
                         help="Stop and checkpoint after this many minutes (this installment only), even if --epochs isn't reached yet.")
    parser.add_argument("--limit-batches", type=int, default=None,
                         help="Cap batches per epoch, for a fast smoke test. Not persisted across resumes.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing state file and start fresh.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.amp is None:
        args.amp = device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("[warn] --amp requested on a non-CUDA device; falling back to fp32.")
        args.amp = False

    run_name = f"resnet18_{args.model.lower()}_seed{args.seed}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / f"{run_name}_best.pth"
    state_path = out_dir / f"{run_name}_state.pth"
    log_path = out_dir / f"{run_name}.csv"

    resuming = state_path.exists() and not args.no_resume

    if resuming:
        # map_location="cpu", not the device: loading onto CUDA turns the saved
        # RNG state into a CUDA tensor, which torch.set_rng_state rejects.
        # load_state_dict copies CPU tensors into the already-on-device model,
        # and the optimizer moves its own state to the parameter device.
        state = torch.load(state_path, map_location="cpu", weights_only=False)  # trusted: written by this script
        frozen = state["config"]
        for key in FROZEN_CONFIG_KEYS:
            # .get keeps state files written before a key was frozen loadable
            setattr(args, key, frozen.get(key, getattr(args, key)))  # keep the original run's hyperparameters
        set_rng_state(state["rng_state"])
        model, criterion, optimizer, scheduler = build_run_objects(args, device)
        scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict") is not None:
            # the loss scale is training state: dropping it restarts scale
            # search mid-run and perturbs the first resumed steps
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_acc = state["best_acc"]
        print(f"[{run_name}] resuming from epoch {start_epoch}/{args.epochs} "
              f"(best_acc so far={best_acc:.2f}%, amp={args.amp})")
    else:
        set_seed(args.seed)
        model, criterion, optimizer, scheduler = build_run_objects(args, device)
        scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
        start_epoch = 1
        best_acc = 0.0
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"]).writeheader()
        print(f"[{run_name}] starting fresh, target {args.epochs} epochs (amp={args.amp})")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    installment_start = time.time()
    last_epoch_completed = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                          limit_batches=args.limit_batches, scaler=scaler, use_amp=args.amp)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False,
                                      limit_batches=args.limit_batches)
        scheduler.step()
        epoch_time = time.time() - t0
        last_epoch_completed = epoch

        row = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": epoch_time,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(row)).writerow(row)

        print(
            f"[{run_name}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% ({epoch_time:.1f}s)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "attention_type": args.model, "seed": args.seed,
                 "epoch": epoch, "val_acc": val_acc, "args": vars(args)},
                best_ckpt_path,
            )

        torch.save(
            {
                "config": {k: getattr(args, k) for k in FROZEN_CONFIG_KEYS},
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if args.amp else None,
                "epoch": epoch,
                "best_acc": best_acc,
                "rng_state": get_rng_state(),
            },
            state_path,
        )

        # Tiny sidecar so the driver can read progress without torch.load-ing a
        # ~135 MB checkpoint per cell just to get an integer.
        with open(out_dir / f"{run_name}_progress.json", "w") as f:
            json.dump({"epoch": epoch, "best_acc": best_acc, "epochs": args.epochs}, f)

        if args.time_limit_min is not None and (time.time() - installment_start) / 60.0 >= args.time_limit_min:
            remaining = args.epochs - epoch
            print(f"[{run_name}] time limit reached after epoch {epoch}. {remaining} epoch(s) remaining -- rerun the same command to continue.")
            return

    if last_epoch_completed >= args.epochs:
        summary_path = out_dir / f"{run_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"run_name": run_name, "model": args.model, "seed": args.seed, "best_val_acc": best_acc, "epochs": args.epochs}, f, indent=2)
        print(f"[{run_name}] done. best_val_acc={best_acc:.2f}%  checkpoint={best_ckpt_path}  log={log_path}")


if __name__ == "__main__":
    main()

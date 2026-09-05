"""Continue whichever variant has the fewest completed epochs, for a bounded
amount of time. Meant to be re-run once per session (each session is a
fresh terminal): it always figures out what to do next on its own.

Usage (run the same command every session):
    python run_next.py --seed 1 --epochs 15 --time-limit-min 90
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

VARIANTS = ["none", "SE", "BAM", "CBAM"]


def completed_epochs(out_dir, variant, seed):
    run_name = f"resnet18_{variant.lower()}_seed{seed}"
    state_path = Path(out_dir) / f"{run_name}_state.pth"
    if not state_path.exists():
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return state["epoch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--time-limit-min", type=float, required=True, help="Stop after this many minutes this session.")
    parser.add_argument("--out-dir", default="./runs")
    parser.add_argument("--num-workers", type=int, default=0)
    args, extra = parser.parse_known_args()

    progress = {v: completed_epochs(args.out_dir, v, args.seed) for v in VARIANTS}
    print(f"Progress so far (seed={args.seed}, target={args.epochs} epochs each): {progress}", flush=True)

    remaining = {v: e for v, e in progress.items() if e < args.epochs}
    if not remaining:
        print("All variants have reached the epoch target. Nothing left to run.", flush=True)
        return

    next_variant = min(remaining, key=lambda v: (progress[v], VARIANTS.index(v)))
    print(f"-> continuing '{next_variant}' ({progress[next_variant]}/{args.epochs} epochs done)", flush=True)

    cmd = [
        sys.executable, "train.py",
        "--model", next_variant,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--time-limit-min", str(args.time_limit_min),
        "--out-dir", args.out_dir,
        "--num-workers", str(args.num_workers),
    ] + extra
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

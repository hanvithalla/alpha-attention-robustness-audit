"""Continue whichever (variant, seed) run has the fewest completed epochs, for
a bounded amount of time. Meant to be re-run once per session (each session is
a fresh terminal): it always figures out what to do next on its own.

The full grid is 4 variants x 3 seeds = 12 runs. Advancing the least-finished
cell each session keeps the grid roughly level, so a partial grid is still a
usable (if lower-powered) result rather than three finished arms and one that
never started.

Usage (run the same command every session):
    python run_next.py --epochs 100 --time-limit-min 90
    python run_next.py --epochs 15 --seeds 1 --time-limit-min 30   # pilot
    python run_next.py --epochs 100 --time-limit-min 90 --status   # look, don't run
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


def print_grid(progress, variants, seeds, target):
    width = max(len(v) for v in variants) + 2
    header = "variant".ljust(width) + "".join(f"seed{s}".rjust(9) for s in seeds) + "   done"
    print(header)
    print("-" * len(header))
    for v in variants:
        cells = "".join(f"{progress[(v, s)]:>4}/{target:<4}".rjust(9) for s in seeds)
        done = sum(progress[(v, s)] for s in seeds)
        print(v.ljust(width) + cells + f"   {done}/{target * len(seeds)}")
    total = sum(progress.values())
    grand = target * len(variants) * len(seeds)
    pct = 100.0 * total / grand if grand else 0.0
    print("-" * len(header))
    print(f"{'TOTAL'.ljust(width)}{'':>{9 * len(seeds)}}   {total}/{grand} epochs ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Seed set for the grid. The design calls for 3.")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Target epochs per run. Frozen per-run at first installment.")
    parser.add_argument("--time-limit-min", type=float, required=True,
                        help="Stop after this many minutes this session.")
    parser.add_argument("--out-dir", default="./runs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--status", action="store_true",
                        help="Print the grid and exit without training.")
    args, extra = parser.parse_known_args()

    progress = {
        (v, s): completed_epochs(args.out_dir, v, s)
        for v in args.variants for s in args.seeds
    }

    print(f"Grid progress (target {args.epochs} epochs per run):", flush=True)
    print_grid(progress, args.variants, args.seeds, args.epochs)
    print(flush=True)

    if args.status:
        return

    remaining = {cell: e for cell, e in progress.items() if e < args.epochs}
    if not remaining:
        print("All runs have reached the epoch target. Nothing left to train.", flush=True)
        print("Next: python evaluate.py --model <variant> --seed <seed>", flush=True)
        return

    # Least-advanced cell wins; ties break by variant order, then seed order,
    # so the choice is deterministic across sessions.
    next_cell = min(
        remaining,
        key=lambda c: (progress[c], args.variants.index(c[0]), args.seeds.index(c[1])),
    )
    variant, seed = next_cell
    print(f"-> continuing '{variant}' seed {seed} "
          f"({progress[next_cell]}/{args.epochs} epochs done)", flush=True)

    cmd = [
        sys.executable, "train.py",
        "--model", variant,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--time-limit-min", str(args.time_limit_min),
        "--out-dir", args.out_dir,
        "--num-workers", str(args.num_workers),
    ] + extra
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

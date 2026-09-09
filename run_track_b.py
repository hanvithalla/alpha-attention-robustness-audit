"""Track B driver: turn finished checkpoints into paper-ready artefacts.

Runs the four steps that must happen in order once the training grid is
complete, and stops at the first failure rather than producing a half-updated
workspace:

  1. evaluate.py for every (variant, seed) -> results/*_corruption_results.csv
  2. aggregate.py                          -> ranking table, findings.json
  3. build_experimental_log.py             -> the log the paper is graded on
  4. make_figures.py                       -> experiment renders for Step 2

    python run_track_b.py                 # all of it
    python run_track_b.py --skip-eval     # re-aggregate without re-evaluating
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ["none", "SE", "BAM", "CBAM"]


def run(cmd, label):
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {r.returncode})")
    print(f"--- {label} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Evaluate whatever checkpoints exist instead of requiring a full grid.")
    args = ap.parse_args()

    cells = [(v, s) for v in args.variants for s in args.seeds]

    # Refuse to silently report a partial grid as if it were the full one.
    incomplete = []
    for v, s in cells:
        p = Path("runs") / f"resnet18_{v.lower()}_seed{s}_progress.json"
        done = json.loads(p.read_text())["epoch"] if p.exists() else 0
        if done < args.epochs:
            incomplete.append(f"{v}/seed{s} ({done}/{args.epochs})")
    if incomplete:
        msg = "Grid is incomplete: " + ", ".join(incomplete)
        if not args.allow_partial:
            raise SystemExit(msg + "\nPass --allow-partial to proceed anyway.")
        print("WARNING: " + msg, flush=True)

    if not args.skip_eval:
        for v, s in cells:
            ckpt = Path("runs") / f"resnet18_{v.lower()}_seed{s}_best.pth"
            if not ckpt.exists():
                print(f"  [skip] {v} seed{s}: no checkpoint", flush=True)
                continue
            run(["evaluate.py", "--model", v, "--seed", str(s)], f"evaluate {v} seed{s}")

    run(["aggregate.py"], "aggregate (milestones 8-9)")
    run(["build_experimental_log.py"], "rebuild experimental_log.md with real numbers")
    run(["make_figures.py"], "render figures")

    f = json.loads(Path("results/findings.json").read_text())
    print("\n" + "=" * 62)
    print(f"clean ranking      : {' > '.join(f['clean_ranking'])}")
    for s in sorted(f["ranking_by_severity"], key=int):
        print(f"severity {s} ranking : {' > '.join(f['ranking_by_severity'][s])}")
    print(f"mCA ranking        : {' > '.join(f['mca_ranking'])}")
    print(f"ranking holds      : {f['ranking_holds_at_every_severity']}")
    print(f"flips at severities: {f['severities_where_ranking_flips'] or 'none'}")
    print(f"max seed std       : {f['max_seed_std_pp']:.2f} pp")
    print(f"clean top-2 gap    : {f['clean_top_two_gap_pp']:.2f} pp vs sd sum "
          f"{f['clean_top_two_sd_sum_pp']:.2f} -> resolved={f['clean_gap_resolved']}")
    print(f"mCA   top-2 gap    : {f['mca_top_two_gap_pp']:.2f} pp vs sd sum "
          f"{f['mca_top_two_sd_sum_pp']:.2f} -> resolved={f['mca_gap_resolved']}")
    print("=" * 62)


if __name__ == "__main__":
    main()

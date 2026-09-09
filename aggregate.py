"""Milestones 8 and 9: aggregate per-seed results and test the ranking claim.

Reads every results/*_corruption_results.csv, averages over seeds, and answers
the study's actual question: does the clean-accuracy ranking survive at every
corruption severity, or does it reorder?

    python aggregate.py

Writes results/aggregate_by_severity.csv, results/ranking_table.md, and
results/findings.json.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANT_ORDER = ["none", "SE", "BAM", "CBAM"]


def load_rows(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("*_corruption_results.csv")):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r["seed"] = int(r["seed"])
                r["severity"] = int(r["severity"])
                r["accuracy"] = float(r["accuracy"])
                rows.append(r)
    return rows


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def rank(dct):
    """Variant names ordered best-first by value."""
    return [k for k, _ in sorted(dct.items(), key=lambda kv: -kv[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No *_corruption_results.csv found in {args.results_dir}")

    variants = [v for v in VARIANT_ORDER if any(r["model"] == v for r in rows)]
    seeds = sorted({r["seed"] for r in rows})
    corruptions = sorted({r["corruption"] for r in rows if r["corruption"] != "clean"})
    severities = sorted({r["severity"] for r in rows if r["severity"] > 0})

    # ---- per (variant, severity), averaged over corruption types then seeds
    by_seed = defaultdict(list)   # (variant, severity) -> [per-seed mean acc]
    for v in variants:
        for s in [0] + severities:
            for seed in seeds:
                vals = [r["accuracy"] for r in rows
                        if r["model"] == v and r["seed"] == seed and r["severity"] == s]
                if vals:
                    by_seed[(v, s)].append(float(np.mean(vals)))

    agg = {}
    for key, vals in by_seed.items():
        agg[key] = mean_std(vals)

    # ---- per (variant, corruption type), averaged over severities and seeds
    by_type = {}
    for v in variants:
        for c in corruptions:
            per_seed = []
            for seed in seeds:
                vals = [r["accuracy"] for r in rows
                        if r["model"] == v and r["seed"] == seed and r["corruption"] == c]
                if vals:
                    per_seed.append(float(np.mean(vals)))
            if per_seed:
                by_type[(v, c)] = mean_std(per_seed)

    # ---- mCA and relative drop
    mca, rel_drop = {}, {}
    for v in variants:
        per_seed_mca, per_seed_drop = [], []
        for seed in seeds:
            corrupted = [r["accuracy"] for r in rows
                         if r["model"] == v and r["seed"] == seed and r["severity"] > 0]
            clean = [r["accuracy"] for r in rows
                     if r["model"] == v and r["seed"] == seed and r["severity"] == 0]
            if corrupted and clean:
                m, c0 = float(np.mean(corrupted)), float(np.mean(clean))
                per_seed_mca.append(m)
                per_seed_drop.append((c0 - m) / c0)
        if per_seed_mca:
            mca[v] = mean_std(per_seed_mca)
            rel_drop[v] = mean_std(per_seed_drop)

    # ---- the ranking question
    clean_rank = rank({v: agg[(v, 0)][0] for v in variants if (v, 0) in agg})
    sev_ranks = {s: rank({v: agg[(v, s)][0] for v in variants if (v, s) in agg})
                 for s in severities}
    mca_rank = rank({v: mca[v][0] for v in variants if v in mca})
    drop_rank = [v for v, _ in sorted(rel_drop.items(), key=lambda kv: kv[1][0])]  # smaller = better

    flips = {s: r for s, r in sev_ranks.items() if r != clean_rank}
    ranking_holds = not flips and mca_rank == clean_rank

    # ---- outputs
    out = Path(args.results_dir)
    with open(out / "aggregate_by_severity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "severity", "mean_accuracy", "std_over_seeds", "n_seeds"])
        for v in variants:
            for s in [0] + severities:
                if (v, s) in agg:
                    m, sd = agg[(v, s)]
                    w.writerow([v, s, f"{m:.4f}", f"{sd:.4f}", len(by_seed[(v, s)])])

    lines = []
    lines.append(f"# Results ({len(seeds)} seeds: {seeds}; corruptions: {', '.join(corruptions)})\n")

    lines.append("\n## Accuracy by severity (mean +/- std over seeds)\n")
    hdr = "| Model | Clean | " + " | ".join(f"Sev {s}" for s in severities) + " | mCA | Rel. drop |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(severities) + 4))
    for v in variants:
        cells = []
        for s in [0] + severities:
            m, sd = agg.get((v, s), (float("nan"), 0.0))
            cells.append(f"{m:.2f} ± {sd:.2f}")
        m_mca, sd_mca = mca.get(v, (float("nan"), 0.0))
        m_rd, sd_rd = rel_drop.get(v, (float("nan"), 0.0))
        lines.append(f"| {v} | " + " | ".join(cells) +
                     f" | {m_mca:.2f} ± {sd_mca:.2f} | {m_rd*100:.2f}% ± {sd_rd*100:.2f} |")

    lines.append("\n## Ranking by severity\n")
    lines.append("| Condition | Ranking (best first) | Same as clean? |")
    lines.append("|---|---|---|")
    lines.append(f"| Clean (sev 0) | {' > '.join(clean_rank)} | -- |")
    for s in severities:
        same = "yes" if sev_ranks[s] == clean_rank else "**NO**"
        lines.append(f"| Severity {s} | {' > '.join(sev_ranks[s])} | {same} |")
    lines.append(f"| mCA (all sev) | {' > '.join(mca_rank)} | "
                 f"{'yes' if mca_rank == clean_rank else '**NO**'} |")
    lines.append(f"| Relative drop | {' > '.join(drop_rank)} | "
                 f"{'yes' if drop_rank == clean_rank else '**NO**'} |")

    lines.append("\n## Per-corruption-type mean accuracy (averaged over severities and seeds)\n")
    lines.append("| Model | " + " | ".join(corruptions) + " |")
    lines.append("|" + "---|" * (len(corruptions) + 1))
    for v in variants:
        cells = [f"{by_type[(v, c)][0]:.2f}" if (v, c) in by_type else "--" for c in corruptions]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")

    lines.append("\n## Verdict\n")
    if ranking_holds:
        lines.append("The clean-accuracy ranking is **preserved at every severity** and on mCA. "
                     "For this method set, clean top-1 was not a misleading basis for choosing "
                     "between these attention blocks.")
    else:
        flipped = ", ".join(f"severity {s}" for s in sorted(flips))
        lines.append(f"The clean-accuracy ranking **does not survive**: it reorders at {flipped}"
                     f"{' and on mCA' if mca_rank != clean_rank else ''}. "
                     "Choosing between these blocks on clean top-1 alone would pick a different "
                     "winner than corruption robustness does.")

    # Is the top-two gap bigger than seed noise? The comparison must use the
    # spread of the two arms being compared -- not the largest spread anywhere
    # in the grid, which here belongs to the noisy baseline/BAM arms and would
    # wrongly declare a resolvable difference unresolved.
    def resolve(pairs):
        """pairs: [(variant, mean, std)] -> (top, second, gap, sd_sum, resolved)."""
        ordered = sorted(pairs, key=lambda t: -t[1])
        if len(ordered) < 2:
            return None
        (v1, m1, s1), (v2, m2, s2) = ordered[0], ordered[1]
        gap, sd_sum = m1 - m2, s1 + s2
        return v1, v2, gap, sd_sum, gap > sd_sum

    clean_res = resolve([(v, agg[(v, 0)][0], agg[(v, 0)][1]) for v in variants if (v, 0) in agg])
    mca_res = resolve([(v, mca[v][0], mca[v][1]) for v in variants if v in mca])
    max_std = max((sd for (_, sd) in agg.values()), default=0.0)

    lines.append("\n## Is the top-two gap resolvable at this budget?\n")
    lines.append("| Metric | Top two | Gap (pp) | Sum of their seed sds | Resolved? |")
    lines.append("|---|---|---|---|---|")
    for name, r in (("Clean top-1", clean_res), ("mCA", mca_res)):
        if r:
            v1, v2, gap, sd_sum, ok = r
            lines.append(f"| {name} | {v1} vs {v2} | {gap:.2f} | {sd_sum:.2f} | "
                         f"{'**yes**' if ok else 'no'} |")
    lines.append(f"\nLargest per-cell seed std anywhere in the grid: {max_std:.2f} pp "
                 f"(from the higher-variance {'/'.join(v for v in variants if any(agg[(v, s)][1] > 1.5 for s in [0] + severities if (v, s) in agg))} "
                 f"arms). That figure is reported for completeness but is not the right "
                 f"denominator for a two-arm comparison.")
    if clean_res and mca_res and not clean_res[4] and mca_res[4]:
        lines.append(f"\n**The two metrics disagree about what is resolvable.** Clean accuracy "
                     f"cannot separate {clean_res[0]} from {clean_res[1]} ({clean_res[2]:.2f} pp, "
                     f"inside their combined seed spread of {clean_res[3]:.2f} pp), but mean "
                     f"corruption accuracy can ({mca_res[2]:.2f} pp against {mca_res[3]:.2f} pp). "
                     f"On this method set the corruption axis is the more discriminative "
                     f"measurement, not the noisier one.")

    (out / "ranking_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    findings = {
        "seeds": seeds, "variants": variants, "corruptions": corruptions,
        "clean_ranking": clean_rank,
        "ranking_by_severity": {str(s): r for s, r in sev_ranks.items()},
        "mca_ranking": mca_rank,
        "relative_drop_ranking_best_first": drop_rank,
        "ranking_holds_at_every_severity": ranking_holds,
        "severities_where_ranking_flips": sorted(flips),
        "accuracy": {f"{v}|sev{s}": {"mean": agg[(v, s)][0], "std": agg[(v, s)][1]}
                     for (v, s) in sorted(agg, key=lambda k: (VARIANT_ORDER.index(k[0]), k[1]))},
        "mCA": {v: {"mean": mca[v][0], "std": mca[v][1]} for v in mca},
        "relative_robustness_drop": {v: {"mean": rel_drop[v][0], "std": rel_drop[v][1]} for v in rel_drop},
        "max_seed_std_pp": max_std,
        "clean_top_two_gap_pp": clean_res[2] if clean_res else None,
        "clean_top_two_sd_sum_pp": clean_res[3] if clean_res else None,
        "clean_gap_resolved": bool(clean_res[4]) if clean_res else None,
        "mca_top_two_gap_pp": mca_res[2] if mca_res else None,
        "mca_top_two_sd_sum_pp": mca_res[3] if mca_res else None,
        "mca_gap_resolved": bool(mca_res[4]) if mca_res else None,
    }
    (out / "findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    # per-corruption-type means, keyed variant -> corruption; consumed by
    # make_figures.py for the breakdown figure that exposes a skewed average
    per_corruption = {v: {c: by_type[(v, c)][0] for c in corruptions if (v, c) in by_type}
                      for v in variants}
    (out / "per_corruption.json").write_text(json.dumps(per_corruption, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {out/'aggregate_by_severity.csv'}, {out/'ranking_table.md'}, {out/'findings.json'}")


if __name__ == "__main__":
    main()

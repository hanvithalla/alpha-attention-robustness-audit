"""Generate workspace/inputs/experiments/.../results.json and the enriched
experimental_log.md that the paper-orchestra Section-Writing Agent reads.

Run it twice: once before results exist (emits a structurally complete log with
PENDING markers, enough for the Outline Agent), and again after aggregate.py has
written results/findings.json (emits the real numbers). The second run is the
ground truth for the claim-evidence gate, so every value here is copied from
findings.json rather than retyped.

    python build_experimental_log.py
"""

import csv
import json
from pathlib import Path

WORKSPACE = Path("workspace")
EXP_SLUG = "attention-robustness"
EXP_DIR = WORKSPACE / "inputs" / "experiments" / EXP_SLUG
RESULTS = Path("results")
PENDING = "PENDING"


def fmt(v, nd=2, suffix=""):
    return PENDING if v is None else f"{v:.{nd}f}{suffix}"


def load_findings():
    p = RESULTS / "findings.json"
    return json.loads(p.read_text()) if p.exists() else None


def final_train_acc(variant):
    """Mean final-epoch training accuracy across seeds, or None if absent."""
    vals = []
    for c in sorted(Path("runs").glob(f"resnet18_{variant.lower()}_seed*.csv")):
        with open(c, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            vals.append(float(rows[-1]["train_acc"]))
    return sum(vals) / len(vals) if vals else None


def epochs_from_runs():
    for c in Path("runs").glob("*.csv"):
        with open(c) as f:
            n = sum(1 for _ in csv.reader(f)) - 1
        if n:
            return n
    return None


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "figures").mkdir(exist_ok=True)
    f = load_findings()
    epochs = epochs_from_runs()

    variants = f["variants"] if f else ["none", "SE", "BAM", "CBAM"]
    seeds = f["seeds"] if f else [1, 2, 3]
    corruptions = f["corruptions"] if f else [
        "brightness", "contrast", "defocus_blur", "elastic_transform"]
    severities = [1, 2, 3, 4, 5]

    def acc(v, s):
        if not f:
            return None, None
        e = f["accuracy"].get(f"{v}|sev{s}")
        return (e["mean"], e["std"]) if e else (None, None)

    # ---------------- results.json (full-precision SSOT) ----------------
    results_json = {
        "experiment": EXP_SLUG,
        "status": "complete" if f else "running",
        "config": {
            "backbone": "ResNet-18 (CIFAR variant: 3x3 stride-1 stem, no max-pool)",
            "widths": [64, 128, 256, 512], "blocks_per_stage": 2,
            "attention_insertion": "residual branch, after bn2, before residual add",
            "reduction_ratio": 16,
            "optimizer": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
            "batch_size": 128, "scheduler": "CosineAnnealingLR",
            "epochs": epochs, "seeds": seeds,
            "augmentation": "RandomCrop(32, padding=4) + RandomHorizontalFlip",
            "precision": "AMP for training, fp32 for validation",
            "hardware": "NVIDIA RTX A2000 6GB, PyTorch 2.5.1+cu121",
        },
        "params_millions": {"none": 11.17, "SE": 11.26, "BAM": 11.36, "CBAM": 11.26},
        "metrics": f if f else {"status": PENDING},
    }
    (EXP_DIR / "results.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    # ---------------- experimental_log.md ----------------
    L = []
    L.append("## Orchestration Decisions\n")
    L.append(
        "Before any training was run, we audited the module implementations against the "
        "official public release rather than trusting a hand-written reproduction. The audit "
        "found that the two spatial-attention arms had been sharing a single channel-gate "
        "implementation, which would have partially collapsed the channel-only versus "
        "channel-plus-spatial contrast the study exists to measure. We replaced both modules "
        "with faithful ports before training, and added a mechanical verification suite that "
        "asserts module fidelity behaviourally, so the defect could not silently return.\n")
    L.append(
        "We added a fourth, attention-free arm that the original three-method plan did not "
        "include. Without it the study could only report which attention variant degrades "
        "least, not whether attention helps at all under corruption.\n")
    L.append(
        "We fixed the attention insertion point across all arms rather than reproducing each "
        "method's own placement convention. This was a deliberate trade: it forfeits "
        "comparability with published absolute accuracies in exchange for attributing any "
        "measured difference to the module rather than to its position in the network.\n")

    L.append(f"\n## Experiment 1: Matched-budget attention-block corruption audit "
             f"(complete, objective: test whether the clean-accuracy ranking of SE, BAM and "
             f"CBAM survives common image corruptions)\n")

    L.append("**Setup**\n")
    L.append(
        f"- Dataset (training and clean evaluation): CIFAR-10, 50,000 training images and "
        f"10,000 test images at 32x32 resolution across 10 classes.\n"
        f"- Dataset (corrupted evaluation): CIFAR-10-C, restricted to "
        f"{len(corruptions)} corruption types ({', '.join(corruptions)}) at severities 1 "
        f"through 5, 10,000 images per type and severity. Inference only, no retraining.\n"
        f"- Backbone: one shared CIFAR-adapted ResNet-18 with a 3x3 stride-1 stem and no "
        f"initial max-pool, four stages of two basic blocks at widths 64, 128, 256 and 512.\n"
        f"- Arms: none (attention-free baseline), SE, BAM, CBAM. The attention module is "
        f"applied to the residual branch after the second batch normalisation and before the "
        f"residual addition; the identity shortcut never passes through it.\n"
        f"- Parameter counts: none 11.17M, SE 11.26M, BAM 11.36M, CBAM 11.26M.\n"
        f"- Reduction ratio r = 16 for all three attention modules.\n"
        f"- Training: SGD with learning rate 0.1, momentum 0.9, weight decay 5e-4, batch size "
        f"128, cosine annealing, random 32x32 crop with 4-pixel padding and random horizontal "
        f"flip. Budget of {epochs or PENDING} epochs per run, identical across arms.\n"
        f"- Seeds: {seeds}, giving {len(variants) * len(seeds)} training runs.\n"
        f"- Precision: mixed precision for training, full precision for validation so that "
        f"checkpoint selection and final evaluation share one numeric path.\n"
        f"- Hardware: a single NVIDIA RTX A2000 with 6GB of memory, PyTorch 2.5.1 with "
        f"CUDA 12.1.\n"
        f"- Hypothesis: the clean-accuracy ranking is not invariant to corruption severity, "
        f"because spatial attention maps depend on local structure that blur and contrast "
        f"reduction attack directly.\n")

    L.append("\n**Results**\n")
    L.append(f"Clean top-1 accuracy, percent, mean over {len(seeds)} seeds with across-seed "
             f"standard deviation:\n")
    for v in variants:
        m, s = acc(v, 0)
        L.append(f"- {v}: {fmt(m)} +/- {fmt(s)}\n")

    L.append(f"\nMean corruption accuracy by severity, percent, averaged over "
             f"{len(corruptions)} corruption types, mean over seeds with standard deviation:\n")
    for v in variants:
        cells = ", ".join(f"severity {s} {fmt(acc(v, s)[0])} +/- {fmt(acc(v, s)[1])}"
                          for s in severities)
        L.append(f"- {v}: {cells}\n")

    L.append("\nMean corruption accuracy across all severities (mCA), percent:\n")
    for v in variants:
        e = f["mCA"].get(v) if f else None
        L.append(f"- {v}: {fmt(e['mean'] if e else None)} +/- {fmt(e['std'] if e else None)}\n")

    per_corr_path = RESULTS / "per_corruption.json"
    if per_corr_path.exists():
        per = json.loads(per_corr_path.read_text())
        L.append("\nMean accuracy per corruption type, percent, averaged over severities 1 "
                 "through 5 and over seeds:\n")
        for v in variants:
            cells = ", ".join(f"{c} {fmt(per.get(v, {}).get(c))}" for c in corruptions)
            L.append(f"- {v}: {cells}\n")

    L.append("\nFinal-epoch training accuracy, percent, mean over seeds (reported because it "
             "separates underfitting from a worse optimum):\n")
    for v in variants:
        ta = final_train_acc(v)
        L.append(f"- {v}: {fmt(ta)}\n")

    L.append("\nRelative robustness drop, (clean - corrupted) / clean, percent:\n")
    for v in variants:
        e = f["relative_robustness_drop"].get(v) if f else None
        L.append(f"- {v}: {fmt(e['mean'] * 100 if e else None)} +/- "
                 f"{fmt(e['std'] * 100 if e else None)}\n")

    if f:
        L.append(f"\nRanking by clean accuracy, best first: {' > '.join(f['clean_ranking'])}\n")
        for s in severities:
            r = f["ranking_by_severity"].get(str(s))
            if r:
                L.append(f"Ranking at severity {s}, best first: {' > '.join(r)}\n")
        L.append(f"Ranking by mCA, best first: {' > '.join(f['mca_ranking'])}\n")
        L.append(f"Ranking by relative robustness drop, smallest drop first: "
                 f"{' > '.join(f['relative_drop_ranking_best_first'])}\n")
        L.append(f"Largest across-seed standard deviation observed in any cell: "
                 f"{fmt(f['max_seed_std_pp'])} percentage points\n")
        L.append(f"Clean-accuracy gap between the two best arms: "
                 f"{fmt(f['clean_top_two_gap_pp'])} percentage points\n")
    else:
        L.append(f"\nRanking results: {PENDING}\n")

    L.append("\n**Baselines**\n")
    L.append(
        "- none (attention-free ResNet-18): trained by us from scratch under the identical "
        "budget. No externally obtained or pretrained checkpoint was used for any arm.\n"
        "- SE, BAM and CBAM: all trained by us from scratch under the identical budget. "
        "Published accuracies for these modules are not comparable to ours, because we hold "
        "the insertion point fixed rather than reproducing each method's own placement, and "
        "we therefore neither cite nor compare against them numerically.\n")

    L.append("\n**Figures**\n")
    L.append(
        "- accuracy_vs_severity: mean corruption accuracy on the vertical axis against "
        "corruption severity 0 through 5 on the horizontal axis, one line per arm, error bars "
        "showing across-seed standard deviation. Renders the per-severity accuracy metrics.\n"
        "- relative_drop_by_arm: relative robustness drop per arm as a bar chart with "
        "across-seed standard deviation. Renders the relative robustness drop metric.\n"
        "- per_corruption_breakdown: grouped bars of mean accuracy per corruption type per "
        "arm, averaged over severities and seeds. Renders the per-corruption-type metrics and "
        "makes a skewed average visible.\n"
        "- ranking_stability: the ordering of arms at each severity, shown so that a "
        "reordering between clean and any severity is directly readable. Renders the ranking "
        "metrics.\n")

    L.append("\n**Notes & Limitations**\n")
    L.append(
        f"- Only three attention methods plus one baseline were compared. The design supports "
        f"per-severity descriptive statements about whether the ranking held or changed, and "
        f"does not support a general claim that attention type predicts corruption robustness. "
        f"Four points do not characterise a design space.\n"
        f"- Only {len(corruptions)} of the benchmark's corruption types were used. Averaging "
        f"over them could flatter or punish spatial attention relative to the full benchmark, "
        f"so the per-type breakdown is reported alongside the mean and the finding is scoped "
        f"to these types. All types in the source are ungated, so extension is inference-only "
        f"and requires no retraining.\n"
        f"- The training budget of {epochs or PENDING} epochs is matched across arms, which is "
        f"what the ranking claim requires, but it is short. Longer training could change "
        f"absolute accuracies and in principle the ordering.\n"
        f"- Holding the insertion point fixed means no arm reproduces its own paper's "
        f"placement, so absolute accuracies are not comparable to published results and none "
        f"is claimed.\n"
        f"- A single dataset at a single resolution was used.\n")
    if f and f.get("clean_gap_resolved") is False:
        L.append(
            f"- The clean-accuracy gap between the two best arms is "
            f"{fmt(f['clean_top_two_gap_pp'])} percentage points against a combined across-seed "
            f"standard deviation of {fmt(f['clean_top_two_sd_sum_pp'])} percentage points. On "
            f"clean data those two arms are therefore not separated by more than seed noise, "
            f"and their relative order should be read as unresolved rather than measured.\n")
    if f and f.get("mca_gap_resolved"):
        L.append(
            f"- On mean corruption accuracy the same two arms are separated by "
            f"{fmt(f['mca_top_two_gap_pp'])} percentage points against a combined across-seed "
            f"standard deviation of {fmt(f['mca_top_two_sd_sum_pp'])} percentage points, which "
            f"does exceed seed noise. The corruption measurement therefore resolves a "
            f"difference between these two arms that the clean measurement cannot.\n")

    L.append("\n**Decisions**\n")
    L.append(
        "- We fixed one insertion point across all arms rather than reproducing each method's "
        "own convention, so that a measured difference is attributable to the module and not "
        "to its placement.\n"
        "- We ported BAM and CBAM from the official public implementation and reimplemented "
        "only SE, whose official release is Caffe-only and contains no Python, and whose block "
        "is fully specified by two equations.\n"
        "- We kept BAM's and CBAM's channel gates as separate implementations, because the "
        "official versions differ: BAM pools average statistics only and normalises between "
        "its MLP layers, while CBAM pools average and max through a shared MLP.\n"
        "- We followed the official BAM code's multiplicative gate fusion rather than the "
        "additive fusion its paper specifies, because the code is the artefact the field "
        "reuses, and we report the discrepancy rather than resolving it silently.\n"
        "- We verified module fidelity behaviourally as well as structurally, asserting that "
        "BAM's output-to-input ratio lies between 1 and 2 and CBAM's between 0 and 1, so that "
        "fusion semantics are checked by behaviour rather than by reading source code.\n"
        "- We report a relative robustness drop alongside absolute corruption accuracy, so "
        "that an arm starting from lower clean accuracy is not mistaken for a robust one.\n")
    L.append(f"\n-> implementation: experiments/{EXP_SLUG}/code/\n")

    out = WORKSPACE / "inputs" / "experimental_log.md"
    out.write_text("".join(L), encoding="utf-8")
    state = "REAL RESULTS" if f else f"{PENDING} (no results/findings.json yet)"
    print(f"Wrote {out} [{state}]")
    print(f"Wrote {EXP_DIR/'results.json'}")


if __name__ == "__main__":
    main()

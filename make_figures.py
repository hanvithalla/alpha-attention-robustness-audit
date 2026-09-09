"""Render the paper's figures from the aggregated results.

These are EXPERIMENT renders, not paper-writing artefacts: they are written to
workspace/inputs/experiments/<slug>/figures/ so that the plotting agent (Step 2)
can copy them byte-identically and figure_provenance_gate.py can verify every
displayed data-plot has a validated render behind it.

Palette: #0072B2 / #D55E00 / #009E73 / #7C3AED, assigned to none/SE/BAM/CBAM in
fixed order and never cycled. Validated with the dataviz six-check validator
(light surface): lightness band, chroma floor, CVD separation, normal-vision
floor and contrast all PASS. Marker shape and dash pattern carry the same
identity as a secondary encoding, so the figures survive greyscale printing.

    python make_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("workspace/inputs/experiments/attention-robustness/figures")
RESULTS = Path("results")
DPI = 300

ORDER = ["none", "SE", "BAM", "CBAM"]
COLOR = {"none": "#0072B2", "SE": "#D55E00", "BAM": "#009E73", "CBAM": "#7C3AED"}
MARKER = {"none": "o", "SE": "s", "BAM": "^", "CBAM": "D"}
DASH = {"none": (0, ()), "SE": (0, (5, 2)), "BAM": (0, (1, 1.5)), "CBAM": (0, (4, 1.5, 1, 1.5))}
LABEL = {"none": "No attention", "SE": "SE", "BAM": "BAM", "CBAM": "CBAM"}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def style(ax):
    """Recessive grid and axes: the data is the figure, the frame is not."""
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)


def size(ratio, width=5.5):
    w, h = (float(x) for x in ratio.split(":"))
    return (width, width * h / w)


def load():
    f = json.loads((RESULTS / "findings.json").read_text())
    rows = []
    with open(RESULTS / "aggregate_by_severity.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            r["severity"] = int(r["severity"])
            r["mean_accuracy"] = float(r["mean_accuracy"])
            r["std_over_seeds"] = float(r["std_over_seeds"])
            rows.append(r)
    return f, rows


def variants_present(f):
    return [v for v in ORDER if v in f["variants"]]


# --------------------------------------------------------------- figure 1
def fig_backbone_overview():
    """Schematic (plot_type: diagram) of the shared block and the single slot."""
    fig, ax = plt.subplots(figsize=size("16:9", 7.0))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=MUTED, lw=1.0, fs=8.5, weight="normal"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3, weight=weight)

    def arrow(x1, y1, x2, y2, style_="-|>", color=MUTED, lw=1.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style_, color=color, lw=lw,
                                    shrinkA=0, shrinkB=0), zorder=1)

    y = 5.0
    box(0.4, y, 1.5, 1.1, "input\n$x$")
    arrow(1.9, y + 0.55, 2.6, y + 0.55)
    box(2.6, y, 2.3, 1.1, "conv1 3$\\times$3\nBN1, ReLU")
    arrow(4.9, y + 0.55, 5.6, y + 0.55)
    box(5.6, y, 2.3, 1.1, "conv2 3$\\times$3\nBN2")
    arrow(7.9, y + 0.55, 8.7, y + 0.55)

    # the attention slot
    box(8.7, y - 0.15, 2.4, 1.4, "attention\n$A(\\cdot)$", ec="#0072B2", lw=2.0,
        fs=9.5, weight="bold")
    arrow(11.1, y + 0.55, 12.1, y + 0.55)

    ax.add_patch(plt.Circle((12.5, y + 0.55), 0.36, facecolor="white",
                            edgecolor=MUTED, linewidth=1.0, zorder=2))
    ax.text(12.5, y + 0.55, "+", ha="center", va="center", fontsize=13,
            color=INK, zorder=3)
    arrow(12.86, y + 0.55, 13.7, y + 0.55)
    box(13.7, y, 1.9, 1.1, "ReLU\noutput")

    # identity shortcut bypasses the attention slot
    arrow(1.15, y, 1.15, 3.1, style_="-")
    arrow(1.15, 3.1, 12.5, 3.1, style_="-")
    arrow(12.5, 3.1, 12.5, y + 0.15)
    ax.text(6.8, 2.75, "identity shortcut — never passes through $A$",
            ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")

    # the four interchangeable settings, centred over the attention slot
    bw, gap, slot_cx = 2.35, 0.22, 9.9
    total = 4 * bw + 3 * gap
    x0 = slot_cx - total / 2
    for i, v in enumerate(ORDER):
        x = x0 + i * (bw + gap)
        ax.add_patch(plt.Rectangle((x, 7.55), bw, 0.74, facecolor="white",
                                   edgecolor=COLOR[v], linewidth=1.8, zorder=2))
        ax.text(x + bw / 2, 7.92, LABEL[v], ha="center", va="center", fontsize=8.2,
                color=INK, zorder=3)
    # label sits beside the connector, not on top of it
    ax.plot([slot_cx, slot_cx], [7.45, 6.62], color=MUTED, lw=0.8, ls=":")
    ax.text(slot_cx - 0.22, 7.03, "$A \\in$", ha="right", va="center",
            fontsize=9, color=INK)

    ax.text(0.4, 1.6, "One shared CIFAR-adapted ResNet-18 basic block. Only $A$ varies "
                      "across arms;\nthe backbone, its placement, and every training "
                      "hyperparameter are held fixed.",
            ha="left", va="center", fontsize=8.5, color=MUTED)

    fig.savefig(OUT / "fig_shared_backbone_overview.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------- figure 2
def fig_accuracy_vs_severity(f, rows):
    fig, ax = plt.subplots(figsize=size("4:3", 5.5))
    for v in variants_present(f):
        pts = sorted([r for r in rows if r["model"] == v], key=lambda r: r["severity"])
        xs = [r["severity"] for r in pts]
        ys = [r["mean_accuracy"] for r in pts]
        es = [r["std_over_seeds"] for r in pts]
        ax.errorbar(xs, ys, yerr=es, label=LABEL[v], color=COLOR[v],
                    marker=MARKER[v], markersize=5, linewidth=2.0,
                    linestyle=DASH[v], capsize=2.5, elinewidth=1.0,
                    markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("Corruption severity (0 = clean)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    style(ax)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.savefig(OUT / "fig_accuracy_vs_severity.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------- figure 3
def fig_relative_drop(f):
    vs = variants_present(f)
    fig, ax = plt.subplots(figsize=size("4:3", 5.0))
    means = [f["relative_robustness_drop"][v]["mean"] * 100 for v in vs]
    stds = [f["relative_robustness_drop"][v]["std"] * 100 for v in vs]
    x = np.arange(len(vs))
    bars = ax.bar(x, means, yerr=stds, width=0.62,
                  color=[COLOR[v] for v in vs], capsize=3,
                  error_kw=dict(elinewidth=1.0, ecolor=MUTED),
                  edgecolor="white", linewidth=2.0)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(means) * 0.02,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[v] for v in vs])
    ax.set_ylabel("Relative robustness drop (%)")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.18)
    style(ax)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig_relative_robustness_drop.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------- figure 4
def fig_per_corruption(f):
    vs = variants_present(f)
    corrs = f["corruptions"]
    per = json.loads((RESULTS / "per_corruption.json").read_text())
    fig, ax = plt.subplots(figsize=size("16:9", 7.0))
    x = np.arange(len(corrs))
    w = 0.8 / len(vs)
    for i, v in enumerate(vs):
        vals = [per[v][c] for c in corrs]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w * 0.9, label=LABEL[v],
               color=COLOR[v], edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in corrs])
    ax.set_ylabel("Mean accuracy over severities 1-5 (%)")
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=len(vs), loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    fig.savefig(OUT / "fig_per_corruption_breakdown.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------- figure 5
def fig_ranking_stability(f):
    vs = variants_present(f)
    conditions = ["clean"] + [f"sev {s}" for s in range(1, 6)]
    rankings = [f["clean_ranking"]] + [f["ranking_by_severity"][str(s)] for s in range(1, 6)]
    fig, ax = plt.subplots(figsize=size("4:3", 5.5))
    for v in vs:
        ys = [r.index(v) + 1 for r in rankings]
        ax.plot(range(len(conditions)), ys, color=COLOR[v], marker=MARKER[v],
                markersize=7, linewidth=2.0, linestyle=DASH[v], label=LABEL[v],
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(-0.12, ys[0], LABEL[v], ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticks(range(1, len(vs) + 1))
    ax.set_yticklabels([f"{i}" for i in range(1, len(vs) + 1)])
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis()
    ax.set_xlim(-1.0, len(conditions) - 0.6)
    style(ax)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig_ranking_stability.png", dpi=DPI)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_backbone_overview()
    print(f"  rendered fig_shared_backbone_overview.png (diagram)")

    if not (RESULTS / "findings.json").exists():
        print("  results/findings.json not found -- data plots skipped. "
              "Run evaluate.py + aggregate.py first.")
        return
    f, rows = load()
    fig_accuracy_vs_severity(f, rows)
    print("  rendered fig_accuracy_vs_severity.png")
    fig_relative_drop(f)
    print("  rendered fig_relative_robustness_drop.png")
    if (RESULTS / "per_corruption.json").exists():
        fig_per_corruption(f)
        print("  rendered fig_per_corruption_breakdown.png")
    fig_ranking_stability(f)
    print("  rendered fig_ranking_stability.png")
    print(f"\nFigures -> {OUT}")


if __name__ == "__main__":
    main()

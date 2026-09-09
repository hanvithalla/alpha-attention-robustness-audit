# Generic ML/AI Conference Submission Guidelines

These are generic guidelines covering NeurIPS, ICML, ICLR, and similar
venues. When the idea specifies a target venue, the paper-orchestra agent
should adapt style accordingly.

---

## Page Limits

- **NeurIPS / ICML / ICLR**: 9 pages of content + unlimited references.
- **AAAI**: 7 pages + 1 page references only.
- **EMNLP / ACL / NAACL**: 8 pages + unlimited references.
- Camera-ready versions typically receive one additional page.

## Formatting

- **Font**: 10pt serif (Latin Modern or Times New Roman).
- **Margins**: 1-inch top/bottom, 0.75-inch left/right.
- **Columns**: single-column (NeurIPS/ICLR) or double-column (ICML/AAAI).
- **Sections**: `\section`, `\subsection`, `\subsubsection` — no deeper.
- **Line spacing**: single-spaced.

## Required Sections

1. **Abstract** (150–250 words): problem, method, results, significance.
2. **Introduction**: motivation, research gap, contributions (bulleted), roadmap.
3. **Related Work**: at least 3 related threads; cite all compared baselines.
4. **Method**: formal problem statement; pseudocode for core algorithm.
5. **Experiments**: setup, main results table, ablations, analysis.
6. **Discussion / Limitations**: broader impact, failure modes, scope.
7. **Conclusion**: 1-paragraph summary + future work.
8. **References**: use `plainnat` or `abbrvnat` BibTeX style.

## Tables and Figures

- Every table needs a `\caption` above it; every figure needs a `\caption` below.
- Use `\toprule / \midrule / \bottomrule` (booktabs) — never `\hline`.
- Bold the best result in each column.
- All figures must be referenced in the main text before they appear.
- Figure files: use PDF (vector) for plots; PNG at ≥300 dpi for diagrams.

## Reproducibility Checklist (NeurIPS / ICML standard)

- [ ] State all hyperparameters.
- [ ] Report mean ± standard deviation over at least 3 seeds.
- [ ] Describe compute resources (GPU type, memory, wall-clock hours).
- [ ] List all datasets with download / preprocessing instructions.
- [ ] Code availability: anonymous link or promise of release.

## Ethics and Broader Impact

NeurIPS requires a Broader Impact section (can be in supplementary).
Address: potential misuse, environmental cost, fairness considerations.

## Anonymisation

Submissions are double-blind. Remove all author names, affiliations,
acknowledgements, and self-identifying repository links from the main text.
Use `[Anonymous, 20XX]` for forward citations to your own work.

## LaTeX Tips

- Use `\citep{key}` for parenthetical citations and `\citet{key}` for
  narrative ones (natbib).
- Prefer `equation` (single) and `align` (multi-line) environments over `$$`.
- Use `\mathbb{R}`, `\mathbf{x}`, `\boldsymbol{\theta}` for standard notation.
- Place floats with `[t]` (top of page) for compatibility with double-column
  layouts.

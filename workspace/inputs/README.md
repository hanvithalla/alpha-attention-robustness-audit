# Inputs

The paper-orchestra pipeline expects the four files below in this
directory before you start. Optional experiment figures go in each
experiment's `experiments/<exp_slug>/figures/` directory.

## Required

- `idea.md`                  — Idea Summary (Sparse or Dense; see io-contract.md)
- `experimental_log.md`      — Setup, raw numeric data, qualitative observations
- `template.tex`             — LaTeX template for the target conference
- `conference_guidelines.md` — Page limit, mandatory sections, formatting rules

## Optional

- `experiments/<exp_slug>/figures/` — Pre-existing figures (PNG/PDF) for that
                               experiment, beside its `results.json` and
                               `code/`. If none, the plotting agent generates
                               everything from scratch.

See `skills/paper-orchestra/references/io-contract.md` in the repo for the
exact schemas of each file.

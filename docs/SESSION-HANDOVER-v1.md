# Session Handover — v1

**Project:** Do Published Attention-Block Gains Survive Common Image Corruptions?
A Matched-Budget CIFAR-10 / CIFAR-10-C Stress Test of SE, BAM, and CBAM
**Repository:** `hanvithalla/alpha-attention-robustness-audit`
**Session date:** 2026-09-09, 11:30 – 16:01 IST (UTC+05:30)
**Document generated:** 2026-09-09 16:01 IST
**Version:** v1
**HEAD at handover:** `75f8272`
**Session outcome:** Milestones 1–9 complete; first full paper draft compiled and pushed.

> Read this document first in a new session. It is written to be self-contained:
> what the project is, what exists, what was decided and why, what broke, and
> what to do next.

---

## 1. What the project is

Every paper in the SE / BAM / CBAM sub-literature ranks attention blocks on a
single number: **clean top-1 accuracy**. CBAM positions itself as an improvement
over SE's channel-only design on exactly that basis. Nobody in this family
reports what happens to that ranking under distribution shift.

This project trains all three blocks — plus an attention-free baseline — into
**one shared backbone at one insertion point under one matched budget**, then
re-ranks them on CIFAR-10-C by degradation across corruption severities instead
of by clean accuracy.

**Why it is answerable cheaply:** CIFAR-10-C shares CIFAR-10's exact label space
and resolution, so corrupted evaluation is inference-only. No retraining, no
relabelling, no architecture change.

**The two confounds the design removes:**

1. Each original paper validates its block in its own backbone at its own scale
   under its own recipe, so cross-paper differences conflate module with host.
2. BAM and CBAM come from one shared public repo and inherit an insertion
   convention that an independently reimplemented SE does not — so even a
   same-codebase comparison confounds module identity with module placement.

Both are removed by fixing the backbone and the insertion point and varying only
the module.

**Pre-registered position:** a null result (ranking preserved at every severity)
was committed to as equally reportable, *before* results existed. This matters —
see §6, where the actual hypothesis was refuted.

---

## 2. Status at handover

| # | Milestone | Status |
|---|---|---|
| 1 | Read & digest / premise test | ✅ `docs/milestone-01-premise-test.md` |
| 2 | Shared backbone + pluggable slot | ✅ verified mechanically |
| 3 | SE reimplemented from Eq. 2–3 | ✅ verified mechanically |
| 4 | BAM/CBAM ported from official repo | ✅ verified mechanically |
| 5 | Train 4 variants × 3 seeds | ✅ 120/120 epochs |
| 6 | Clean CIFAR-10 evaluation | ✅ 12 checkpoints |
| 7 | CIFAR-10-C evaluation | ✅ 12 × 20 conditions |
| 8 | Aggregate per seed + spread | ✅ `results/findings.json` |
| 9 | Ranking-flip analysis | ✅ `results/ranking_table.md` |
| — | Paper first draft | ✅ `workspace/final/paper.pdf`, 9 pages |

**Tracker note:** the milestone system asked for a *new result filed with an
artifact attached*. The artifact to attach is
`results/findings.json`. See §11 for the caveat about how to word the filing.

---

## 3. Repository map

### Experiment code (repo root)

| File | Purpose |
|---|---|
| `attention.py` | SE (reimplemented), BAM + CBAM (ported from `Jongchan/attention-module`). Separate channel-gate classes per module — this separation is load-bearing, see §5.1. |
| `model.py` | Shared CIFAR ResNet-18. 3×3 stride-1 stem, no max-pool, widths 64/128/256/512. `attention_type` is the only varying argument. |
| `train.py` | Trains one (variant, seed) cell. AMP, per-epoch checkpointing, resumable installments, frozen hyperparameters across resumes, progress sidecar. |
| `run_next.py` | Grid driver. Advances the least-finished cell of the 4×3 grid. `--status` prints the grid without training. |
| `evaluate.py` | One checkpoint → clean top-1 + CIFAR-10-C across types × severities. |
| `aggregate.py` | Milestones 8–9. Seed averaging, mCA, relative drop, ranking-flip table, resolvability test. |
| `run_track_b.py` | Chains evaluate → aggregate → rebuild log → figures. Refuses to run on an incomplete grid without `--allow-partial`. |
| `data_cifar10.py` | CIFAR-10 from HuggingFace, memory-mapped `.npy` cache. |
| `cifar10c.py` | CIFAR-10-C from HuggingFace Parquet, memory-mapped `.npy` cache. |
| `make_figures.py` | Renders paper figures **as experiment artefacts** (see §5.3). |
| `build_experimental_log.py` | Generates `results.json` + `experimental_log.md` from `results/findings.json`. |
| `build_refs.py` | Verifies each citation against OpenAlex, emits `refs.bib`. |
| `verify_milestones.py` | **54 checks** covering milestones 2–4. Run before every training installment. |

### Documentation

| File | Purpose |
|---|---|
| `docs/milestone-01-premise-test.md` | The M1 submission. Premise claims verified against live sources. |
| `docs/PROVENANCE.md` | Module sources, licences, adaptations, and the three intentional protocol deviations. |
| `docs/SESSION-HANDOVER-v1.md` | This document. |

### Paper workspace (`workspace/`)

```
workspace/
├── inputs/
│   ├── idea.md                     # I — dense variant, 1117 words
│   ├── experimental_log.md         # E — GENERATED, do not hand-edit
│   ├── template.tex                # T
│   ├── conference_guidelines.md    # G
│   └── experiments/attention-robustness/
│       ├── results.json            # full-precision SSOT
│       ├── code/                   # 8 modules — Method ground truth for the gates
│       └── figures/                # experiment renders (provenance source)
├── outline.json                    # Step 1 — validates: 5 figures, 3 clusters, 7 sections
├── refs.bib                        # Step 3 — 15 OpenAlex-verified references
├── citation_pool.json              # verification record incl. canonical overrides
├── drafts/
│   ├── intro_relwork.tex           # Step 3 prose
│   └── paper.tex                   # Step 4 output
├── figures/                        # Step 2 — byte-copies + captions.json
├── final/                          # ← THE DELIVERABLE
│   ├── paper.pdf                   #   9 pages
│   ├── paper.tex
│   ├── refs.bib
│   └── figures/
├── tex_profile.json                # CORRECTED by hand — see §5.4
├── research_brief.md               # §1 outline decisions, §2 lit-review mode
└── provenance.json                 # input/output hashes + gate results
```

### Not in git (gitignored)

`data/` (~750 MB caches), `runs/` (checkpoints + logs), `workspace/cache/`,
`workspace/refinement/`. **`results/` IS committed** — it is the evidence.

---

## 4. Results

### Headline

**The clean-accuracy ranking survives.** `CBAM > SE > none > BAM` holds at all
five severities, on mCA, and on relative robustness drop. Zero flips.

| Arm | Clean | Sev 1 | Sev 3 | Sev 5 | mCA | Rel. drop |
|---|---|---|---|---|---|---|
| CBAM | **89.32** ±0.08 | **87.68** | **80.94** | **58.87** | **77.24** ±0.45 | **13.52%** |
| SE | 89.21 ±0.18 | 87.29 | 80.09 | 57.16 | 76.31 ±0.04 | 14.45% |
| none | 86.09 ±2.10 | 83.96 | 75.53 | 55.01 | 72.52 ±2.31 | 15.77% |
| BAM | 81.56 ±2.47 | 79.28 | 70.08 | 52.91 | 68.12 ±2.54 | 16.49% |

### Per corruption type (mean over severities 1–5, 3 seeds)

| Arm | brightness | contrast | defocus_blur | elastic_transform |
|---|---|---|---|---|
| none | 83.76 | 59.66 | 72.28 | 74.38 |
| SE | 86.83 | 67.06 | 74.63 | 76.75 |
| BAM | 78.66 | 53.27 | 69.40 | 71.16 |
| CBAM | **87.33** | **67.53** | **76.05** | **78.05** |

Ordering identical within every type — the aggregate hides no reversal.
Contrast is the most damaging corruption and where attention helps most
(CBAM leads baseline by 7.9 pp on contrast vs 3.6 pp on brightness).

### Final-epoch training accuracy (supports the BAM reading)

| none | SE | BAM | CBAM |
|---|---|---|---|
| 87.90% | 91.88% | **82.79%** | 92.48% |

### Three findings, in order of importance

1. **Null confirmed.** For this method set, clean top-1 was *not* a misleading
   basis for choosing between these blocks.
2. **The hypothesis was refuted.** We predicted spatial gates (BAM/CBAM) would
   be more fragile under blur/contrast than SE's global channel gate. CBAM has a
   spatial gate and degraded **most slowly** of all four arms. The paper reports
   this as refuted rather than restating it.
3. **The corruption axis is more discriminative than clean accuracy here.**
   Clean cannot separate CBAM from SE (0.11 pp gap vs 0.26 pp combined seed sd);
   mCA can (0.92 pp vs 0.49 pp). This is a *post-hoc* observation, not a
   pre-registered prediction — keep that distinction (§11).

Plus: **BAM finished 4.5 pp below the no-attention baseline.** The paper does
*not* claim BAM is a worse module. Its final train accuracy is also the lowest,
which is the signature of underfitting at a 10-epoch budget. Wording used:
"BAM did not pay for itself at this budget."

---

## 5. What was implemented today, and why

### 5.1 Milestone 4 — the fidelity bug that mattered most

The incoming `attention.py` hand-wrote BAM and CBAM instead of porting them, and
**both arms shared one `ChannelGate` class**. Diffing against
`Jongchan/attention-module` (fetched live) showed:

| | Official BAM | Official CBAM | What the repo had |
|---|---|---|---|
| Channel pooling | avg **only** | avg **and** max | avg+max for both |
| Channel MLP | `Linear → BatchNorm1d → ReLU → Linear` | `Linear → ReLU → Linear`, shared | no BN, shared |
| Fusion | `1 + σ(Mc · Ms)` **multiplicative** | sequential gates | `σ(Mc + Ms)` additive |
| CBAM spatial conv | — | `BasicConv` **with BatchNorm2d** | bare Conv2d |

**Why this was blocking:** the study's central contrast is channel-only (SE) vs
channel+spatial (BAM/CBAM). Two arms sharing a gate neither original uses
partially collapses the very thing being measured.

Ported faithfully. Also found and documented: **BAM's paper specifies additive
fusion, its official code multiplies.** Followed the code (it is the artefact the
field reuses) and reported the discrepancy — now a contribution in the paper.

### 5.2 Mechanical verification instead of inspection

`verify_milestones.py`, 54 checks. The important ones are **behavioural, not
structural**:

- BAM's output/input ratio must lie in `(1, 2)` — as `1 + σ(·)` requires.
  Measured: `[1.495, 1.505]`.
- CBAM's must lie in `(0, 1)` — as a pure gate requires. Measured: `[0.082, 0.437]`.
- `BAMChannelGate is not CBAMChannelGate` — so the §5.1 defect cannot return.

Fusion semantics are therefore verified by behaviour, not by reading source.

### 5.3 Figures as experiment artefacts

`figure_provenance_gate.py` in the paper pipeline is **enforcing**: any displayed
data-plot without a validated experiment render behind it is treated as
fabrication and stripped. The plotting-agent here is a local override that only
*selects* pre-rendered figures.

So `make_figures.py` renders into
`workspace/inputs/experiments/attention-robustness/figures/`, and Step 2 copies
byte-identically into `workspace/figures/`. This is the correct architecture
regardless of the gate.

**Palette was computed, not chosen.** Ran the dataviz six-check validator. First
candidate (Okabe-Ito subset) **failed**: deutan ΔE 7.6 and two colours under 3:1
contrast. Shipped palette `#0072B2 / #D55E00 / #009E73 / #7C3AED` (none/SE/BAM/CBAM,
fixed order, never cycled) passes all six. Marker shape and dash pattern duplicate
the colour encoding so figures survive greyscale print.

### 5.4 Paper pipeline (paper-orchestra, arXiv:2604.05018)

Ran Steps 1–4 and 6. **Step 5 (refinement loop) was NOT run** — stopped at a
complete, gate-clean draft for time.

- **Step 1 Outline** → `outline.json`, validates (5 figures, 3 clusters, 7
  sections incl. mandatory Limitations, 18 citation hints).
- **Step 3 Literature review — DEGRADED.** No Semantic Scholar or Exa key in the
  environment, so the parallel discovery pipeline could not run. Substituted
  title-verification against OpenAlex (no key needed) with fuzzy matching to
  reject wrong hits. 15/15 verified at match ratio 1.00. **Recorded as degraded
  in `research_brief.md` §2 rather than disguised.**
  - OpenAlex proved unreliable for canonical venue/year: it returned the CIFAR-10
    tech report **re-dated to 2024** and preprint records for several ICLR/CVPR
    papers. An audited `CANONICAL` override table supplies venue/year and every
    substitution is logged in `citation_pool.json`.
  - Author names normalised to ASCII — OpenAlex returns U+2010 hyphens that T1
    font encoding cannot typeset.
- **Step 4 Section writing** → one call producing the full manuscript.
- **Step 6 Compile** → tectonic 0.17.0, 9 pages, 0 undefined refs.

**`tex_profile.json` was corrected by hand.** `check_tex_packages.py` probes only
for `pdflatex`, found none, and wrote a fallback declaring *every* package
missing — including `booktabs`, which the guidelines require for tables. This
machine has tectonic, which fetches packages on demand. Verified empirically by
compiling a document loading all of them.

### 5.5 Gate results (all on `workspace/final/paper.tex`)

| Gate | Result |
|---|---|
| `orphan_cite` | PASS — 15/15 keys cited |
| `anti_leakage` | PASS |
| `claim_evidence` | PASS — 11/11 numeric claims corroborated |
| `figure_provenance` | PASS — 2 reused + 1 schematic, 0 generated data-plots |
| `figure_dupe` | PASS |
| `metric_consistency` | PASS |
| `claim_vs_code` | PASS — 3/3 method claims corroborated by code |
| `portable_paths` | OK |
| `latex_sanity` | **False positive** — reports an unmatched brace; a per-line balance check returns to depth 0 and the document compiles. Its counter mishandles escaped braces. |

---

## 6. Bugs found and fixed (all real, all in the pre-existing code)

### 6.1 Resume was completely broken — `RNG state must be a torch.ByteTensor`

**Symptom:** ten consecutive driver iterations advanced **zero** epochs while
printing "continuing BAM seed 3" each time. Five cells were stuck.

**Cause:** `torch.load(state_path, map_location=device)` moved the saved RNG
state onto CUDA; `torch.set_rng_state` accepts only a **CPU ByteTensor**.

**Why it hid so long:** before the memory kills, every run finished inside a
single installment, so the resume branch had never executed. The entire
`--time-limit-min` installment feature was untested until something interrupted
it.

**Fix:** load checkpoints on CPU (`load_state_dict` copies into the already-on-device
model; the optimizer relocates its own state), plus defensive coercion so
checkpoints written before the fix still resume.

**Lesson for next session:** the driver's grep filter showed "continuing…" and
looked like progress. **Always confirm advancement with `run_next.py --status`,
not with log lines.**

### 6.2 `run_next.py` allocated ~1.6 GB per invocation to read an integer

`completed_epochs()` did a full `torch.load` of every cell's state file (model +
optimizer + scheduler + RNG ≈ 135 MB × 12) just to read `state["epoch"]`.
Replaced with a `<run>_progress.json` sidecar written by `train.py`.

### 6.3 The seed-noise guard compared against the wrong denominator

It compared the top-two gap against the **largest per-cell std anywhere in the
grid** (3.09 pp, belonging to the noisy baseline/BAM arms) instead of the spread
of the two arms being compared. Uncorrected, it would have declared the mCA
difference unresolved and made the paper **under-claim a real finding**. Now a
per-pair test.

### 6.4 A wrong number in the draft

BAM's final train accuracy was written as 83.5% — that was a single seed. The
3-seed mean is **82.79%**. Caught because `claim_evidence_gate` flagged
ungrounded values; the fix also added per-corruption and train-accuracy figures
to the generated log so the claims are grounded.

### 6.5 Environment gotcha — the Bash heredoc mangles backslashes

Twice, a `<< 'EOF'` heredoc collapsed `\\` to `\`, once producing malformed
LaTeX (which I initially misread as a missing-package failure) and once an
invalid regex. **Write Python/LaTeX with the file tool, not shell heredocs.**
Also: `pathlib.write_text()` on Windows defaults to cp1252 and will truncate a
file mid-write on a non-ASCII character — always pass `encoding="utf-8"`.

---

## 7. Compute: what was measured and how time was reduced

### 7.1 The binding constraint was never the GPU

| Bottleneck found | Measurement | Fix | Saving |
|---|---|---|---|
| `cs.toronto.edu` (torchvision's CIFAR-10 host) | **~300 B/s** — 25 min for 52/170 MB | switched to HuggingFace (**3.3 MB/s**) | days → seconds |
| Compressed `.npz` caches | decompressed per access | uncompressed `.npy` + `mmap_mode='r'`, pages shared across processes | eliminated repeated decompression of 30 MB × 20 splits × 12 models |
| Dataloader workers | 4 procs × ~636 MB = **2.5 GB** | `num_workers=0` | **2.5 GB, zero speed cost** |
| `run_next.py` checkpoint churn | ~1.6 GB per invocation | progress sidecar | 1.6 GB |
| **Foreign ViT sweep** | **5.7 GB** (later 2.3 GB) | stopped, with authorisation | 6.6 GB |

### 7.2 Benchmarks actually measured

- **CBAM, AMP, workers=4: 52.2 s/epoch**
- **Baseline, AMP, workers=4: 44.4 s/epoch (median)**
- **Baseline, AMP, workers=0: 44.3 s/epoch** ← identical, so workers were pure
  memory overhead. **This is the single most useful measured fact for the next
  session.**
- Under memory pressure epochs degraded to 86–96 s (swapping) — a reliable
  early-warning signal.
- Full grid: 120 epochs ≈ **90 min of GPU time**; wall-clock was longer only
  because of the interruptions.
- Evaluation: 12 models × 21 conditions ≈ **10 min** total.

### 7.3 Trade-offs deliberately taken

| Decision | Chosen | Cost | Reasoning |
|---|---|---|---|
| **Epoch budget** | **10**, not 100 | low absolute accuracy; BAM's deficit unresolved | The ranking claim needs the budget *matched*, not large. 100 epochs = ~15 h and missed the deadline. |
| **Backbone** | **ResNet-18 (11.17M)**, not the plan's 1–2M | fails M2's parameter criterion | User's hardware analysis specified ResNet-18. Recorded as a documented deviation in `PROVENANCE.md` §3.2, not hidden. |
| **Corruption types** | **4**, not 15 | narrower external validity | Extension is inference-only; deferred, not abandoned. Data for all 15 is already cached. |
| **Seeds** | **3** | wide CIs on the noisy arms | Needed to test resolvability at all; fewer would make the ranking unfalsifiable. |
| **Baseline arm** | **added** (plan had 3 methods) | 9 → 12 runs, +33% compute | Without it you cannot tell whether attention helps *at all* under corruption, only which variant degrades least. |
| **`num_workers`** | **0 → 4** after memory freed | none | Measured identical; used 4 once RAM allowed. |
| **AMP** | **on for training, off for validation** | slightly slower validation | Checkpoint selection and final evaluation must share one numeric path. |
| **Two figures cut** | removed for the page limit | 5 → 3 figures | Both plotted numbers already tabulated **exactly**. Removing duplicated data is right editorially, not just for length. |
| **Step 5 refinement** | **skipped** | draft not reviewer-polished | Time. Draft is complete and gate-clean without it. |

---

## 8. Environment facts worth carrying forward

- **Hardware:** Dell Precision 3650, i7-11700 (8C/16T), **RTX A2000 6 GB**, 16.9 GB RAM.
- **Software:** Python 3.10, torch 2.5.1+cu121, CUDA 12.1, matplotlib 3.10.8,
  datasets 5.0.1, jsonschema (installed this session).
- **LaTeX:** **no pdflatex.** `tectonic 0.17.0` at `C:\Users\RCC\bin\tectonic.exe`
  (installed this session). Add to PATH: `export PATH="/c/Users/RCC/bin:$PATH"`.
- **Git auth:** SSH key generated this session, `origin` switched to SSH, GitHub
  host keys pinned. Pushes work unattended.
- **No API keys** for Semantic Scholar or Exa. OpenAlex and Crossref need none.
- ⚠️ **A ViT sweep** (`scripts/run_all.py --only vit`, chained across datasets and
  seeds) shares this machine and will respawn. It consumed 5.7 GB and killed the
  training grid three times. Check for it before long runs.

---

## 9. How to reproduce / continue

```bash
cd E:/H/Papers/alpha-attention-robustness-audit
export PATH="/c/Users/RCC/bin:$PATH"          # tectonic

python verify_milestones.py                   # 54 checks — run before training
python run_next.py --epochs 10 --time-limit-min 25 --num-workers 4   # one cell
python run_next.py --epochs 10 --time-limit-min 1 --status           # grid state
python run_track_b.py                         # evaluate → aggregate → log → figures
cd workspace/final && tectonic -X compile paper.tex
```

To rerun at a longer budget, **use a fresh `--out-dir`** — `--epochs` is frozen
per run at the first installment, so an existing `runs/` will not extend:

```bash
python run_next.py --epochs 100 --time-limit-min 45 --out-dir ./runs_e100
```

---

## 10. Next steps, prioritised

### Cheap, strengthens claims already made

1. **Training-curve figure (~15 min).** The BAM underfitting claim currently rests
   on one number. Per-epoch curves exist in `runs/*.csv` and would *show* BAM
   still climbing at epoch 10 while SE/CBAM flatten. This is the paper's most
   contestable claim and the cheapest fix.
2. **Replace the sd-sum heuristic with a real test (~20 min).** The resolvability
   check adds two standard deviations. Honest but ad hoc; Welch's t-test or 95%
   CIs would survive review. The conclusion probably will not change.
3. **Appendix (~30 min).** Per-seed numbers, the two cut figures, the full
   type × severity grid. Appendices do not count against the 9 pages.
4. **Reproducibility statement.** The repo has the verifier, provenance and exact
   commands; the paper does not say so.

### Moderate

5. **Extend 4 → 15 corruption types (~1–2 h, inference only).** Removes the biggest
   external-validity objection. ⚠️ Verify the actual HF config list first —
   `cifar10c.py::CORRUPTIONS_15` includes `snow`, which the probe did not
   explicitly name.
6. **Run Step 5 refinement.** The one pipeline stage not yet exercised.

### Expensive, highest scientific value

7. **Rerun the identical grid at 50–100 epochs (~8–17 h).** The real weakness. It
   would settle whether BAM's deficit is underfitting, and raise absolute
   accuracies. Needs the machine free of the ViT sweep. This is the experiment the
   paper's own Limitations section names as most valuable.

---

## 11. Filing the tracker result — important wording caveat

Attach `results/findings.json`.

The tracker's own summary said the results *"confirm your hypothesis that
corruption testing can reveal performance separations that clean evaluation
misses."* **Do not repeat that.** The pre-registered hypothesis was that spatial
gates would prove *more fragile* — and it was **refuted**. The
mCA-resolves-more-than-clean result is a genuine but **post-hoc** observation.
Filing it as a confirmed prediction would be HARKing.

Suggested wording:

> Pre-registered hypothesis refuted: the ranking did not compress or invert; the
> spatial-gate arm degraded most slowly. Separately, mCA resolved a top-two
> difference that clean accuracy could not (0.92 pp vs 0.49 pp combined seed
> spread) — reported as a secondary, post-hoc observation.

---

## 12. Commit log for this session

| Time (IST) | Commit | Summary |
|---|---|---|
| 11:35 | `d16d592` | Milestone 1: read & digest premise test |
| 11:35 | `c14c140` | Milestone 4: port BAM/CBAM from official repo |
| 12:14 | `03f60e9` | Milestone 5–9 pipeline: AMP, seed grid, HF data path, aggregation |
| 12:16 | `eeb0a62` | Paper draft scaffold (later removed) |
| 12:29 | `9818dd7` | Track A: workspace, idea.md, log generator |
| 12:40 | `2dcdd41` | Track A complete: outline, verified bibliography, intro + related work |
| 12:59 | `6e74fd0` | Figure rendering as experiment code |
| 13:02 | `f4be6e4` | Memory-map the CIFAR-10 cache |
| 13:36 | `39fe965` | Progress sidecar instead of loading checkpoints |
| 13:38 | `e8aa545` | Track B driver + memory-map the corruption cache |
| 14:08 | `298e9b9` | **Fix resume: RNG state must be a CPU ByteTensor** |
| 14:27 | `1cbde6c` | Correct the TeX profile |
| 14:56 | `c504eda` | Milestones 6–9: evaluation, aggregation, ranking result |
| 15:02 | `6b68d44` | Step 4: manuscript drafted, gates pass |
| 15:05 | `f08fa2b` | First draft complete: 9-page manuscript |
| 15:10 | `75f8272` | Remove superseded scaffold; one draft only |

Session start `945881e` → handover `75f8272`: **16 commits**.

---

*End of handover v1 — 2026-09-09 16:01 IST.*

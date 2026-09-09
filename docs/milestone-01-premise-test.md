# Milestone 1 — Read & Digest (Premise Test)

**Project:** Do Published Attention-Block Gains Survive Common Image Corruptions?
A Matched-Budget CIFAR-10 / CIFAR-10-C Stress Test of SE, BAM, and CBAM
**Repo:** `hanvithalla/alpha-attention-robustness-audit` @ `945881e`
**Date:** 2026-09-09

---

## 1. The premise, restated

The SE/BAM/CBAM sub-literature ranks attention blocks by a single number: **clean
top-1 accuracy**. CBAM positions itself as an improvement over SE-style channel
attention on that basis, and the applied papers that bolt these blocks onto CNNs
inherit the same metric. Nobody in this family reports what happens to that
ranking under distribution shift.

The premise of this project is that **the ranking is untested, not confirmed** —
and that a practitioner choosing SE vs BAM vs CBAM for a deployed classifier is
choosing on a metric that may not survive the corruption their deployment
actually sees.

The test is deliberately cheap and decisive: train all three blocks into **one
shared backbone** at a **matched budget** on clean CIFAR-10, then re-rank them by
degradation across CIFAR-10-C severities. Because CIFAR-10-C shares CIFAR-10's
exact label space and resolution, the corrupted evaluation is pure inference — no
retraining, no new labels, no architecture change.

---

## 2. Premise claims, independently verified

The premise rests on five factual claims. Each was checked directly rather than
taken from the scan record.

| # | Claim | How checked | Verdict |
|---|---|---|---|
| 1 | SE's official repo is Caffe-only, not Colab-installable — justifying SE as the one allowed reimplementation | Fetched `github.com/hujie-frank/SENet` | **Confirmed.** `.cpp`/`.cu` layers, `.prototxt` models, Apache-2.0. No Python/PyTorch in-repo; the PyTorch ports listed are third-party. |
| 2 | BAM and CBAM module definitions exist in `Jongchan/attention-module`, MIT | Fetched `MODELS/cbam.py` in full; `MODELS/bam.py` structure | **Confirmed.** Both present and extractable as standalone `nn.Module` code. |
| 3 | CIFAR-10-C is ungated, CC-BY-4.0, directly downloadable | Fetched `huggingface.co/datasets/WNJXYK/TTA-CIFAR-10-C` | **Confirmed.** CC-BY-4.0, not gated, no login. |
| 4 | SE and CBAM report clean top-1/top-5 (and mAP) only — no robustness metric | Scan record, `papers[]` metrics fields | **Inherited from scan**, consistent with the papers as read. |
| 5 | No attention-block paper in the field corpus reports a corruption-robustness metric | Scan record, `used_in_papers: 0`, three lit_search sweeps | **Inherited from scan.** Closest hits (AR2 2025; Multi-Scale Push-Pull 2025) propose *new* methods validated on corruption benchmarks — neither audits the existing SE/BAM/CBAM family under one matched protocol. |

**Verdict: the premise holds.** The gap is real, the code is obtainable under
permissive licenses, and the data is free and ungated.

*Record correction:* the reading-list entry attached to BAM is titled
"Squeeze-and-Excitation Networks" but points at `arxiv.org/abs/1807.06514`. The
URL is correct — that is BAM's own paper — and only the title field is
mislabeled, as the method table already notes.

---

## 3. The experiment, digested

**Varied axis — corruption severity, 6 levels:** clean (severity 0) plus
severities 1–5, averaged over four corruption types (`brightness`, `contrast`,
`defocus_blur`, `elastic_transform`).

This axis isolates the one thing the study is about: the *rate* at which a fixed,
already-trained model degrades. Architecture, training data, and training budget
are frozen; the only thing changing across levels is the corruption intensity
applied to the test images at evaluation time.

**Held fixed (4 factors):**

1. One shared backbone, attention inserted at the identical point in every arm.
2. Identical optimizer, LR schedule, batch size, epoch count, augmentation, seed set.
3. Identical test-time preprocessing, applied uniformly to clean and corrupted images.
4. The same corruption types averaged at every severity, for every method.

**Runs:** train once per (method, seed); corruption is inference-only.

**Metrics:**

- *Clean top-1* — the metric the field already ranks on; the reference point.
- *Mean corruption accuracy per severity* — Hendrycks & Dietterich framing; answers whether the **ranking** holds, not merely whether accuracy drops.
- *Relative robustness drop* `(clean − corrupted) / clean` — normalises out each method's different starting point, so a method that starts lower is not misread as "more robust" simply because it had less to lose.

The third metric is load-bearing. Without it, absolute mCA would reward whichever
arm happened to train worse on clean data.

---

## 4. Pre-registered predictions

1. **Clean:** BAM/CBAM (spatial + channel) edge out SE (channel-only) by a small margin, consistent with CBAM's own framing.
2. **Corrupted:** the ranking compresses or inverts — SE's simpler channel-only gating may degrade less steeply than BAM/CBAM's spatial masks, which can mislocalise under blur and low contrast.
3. **Null:** the clean ranking is preserved at every severity.

**What would falsify prediction 2:** the clean-accuracy ordering holding, within
seed spread, at all five severities and on the relative-drop metric. That is
outcome 3, and it is reported exactly as prominently as an inversion would be.

**What would invalidate the study itself** (as opposed to producing a null):
seed spread exceeding between-method gaps at every severity. In that case the
design cannot resolve any ranking, and the honest report is that three seeds at
this budget are underpowered for the effect size — a result about the
measurement, not about the methods.

---

## 5. Why a null is publishable

The question is *"is the field's single metric misleading for this method set?"*
"No" answers it as completely as "yes". A preserved ranking is a positive
statement — practitioners can keep using clean top-1 to choose among these three
blocks — and it is only obtainable by running the experiment. This is
pre-committed here so that a null cannot later be reframed as a failed study.

---

## 6. Confounds, and how the design neutralises them

| Confound | Neutralisation |
|---|---|
| BAM and CBAM come from the same repo and share an insertion convention that an independently reimplemented SE would not inherit | All three are inserted at the identical point in one backbone **we** define; the repo's own `model_resnet.py` (ImageNet-scale ResNet-50) is not used |
| A reimplemented SE arm could take a compute shortcut (e.g. a pretrained checkpoint) the others don't | All arms trained from scratch, by us, at the identical budget. No found or pretrained weights anywhere |
| Averaging only a subset of corruption types could flatter or punish spatial attention | Per-type breakdown reported alongside the mean, so a skewed average is visible rather than hidden; generality explicitly scoped to the types run |

---

## 7. Scope limits, stated up front

- **Three methods.** Supports "the ranking held/changed at severity X" as a descriptive, per-severity observation. It does **not** support a general claim that attention *type* predicts corruption robustness.
- **Corruption subset.** See §8.1 — this limit is now looser than the plan assumed.
- **SE reduction ratio.** A design-space point we choose and record, not a re-derivation of the paper's tuned value.
- **One dataset, one resolution.** CIFAR-scale only.

---

## 8. Premise-test findings that change the plan

Reading the plan against the repo and the live sources surfaced four items. None
threaten the premise; three block later milestones.

### 8.1 The 4-corruption scope limit can be lifted

The risk register scopes the study to four corruption types "unless the student
re-probes and extends". **This submission is that re-probe.** The HF dataset page
lists **15 corruption types × 5 severities** (750,000 rows, 1.68 GB), all
ungated — `gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`,
`glass_blur`, `motion_blur`, `zoom_blur`, `frost`, `fog`, `brightness`,
`contrast`, `pixelate`, `elastic_transform`, `jpeg_compression`.

Extending from 4 to 15 types costs **inference only** and removes the single
biggest external-validity objection a reviewer will raise. Recommended, pending a
compute re-estimate (evaluation cost scales ~3.75×; training cost unchanged).

### 8.2 The data loader targets the wrong source — blocks Milestone 7

`evaluate.py` expects the Zenodo `.npy` layout (`50000×32×32×3` per corruption,
five severities stacked, shared `labels.npy`). The confirmed-ungated source is HF
**Parquet**. A loader must be written.

### 8.3 BAM and CBAM are reimplemented, not extracted — blocks Milestone 4

Milestone 4 requires extracting the modules from `Jongchan/attention-module`;
`attention.py` hand-writes them. Diffed against the official source:

- **BAM's channel gate is CBAM's channel gate.** Both arms share one `ChannelGate` class (avg+max pooled shared MLP). Official BAM uses **average pooling only**, with `Linear → BatchNorm1d → ReLU → Linear`.
- **BAM's fusion is additive** (`sigmoid(ch + sp)`); the official repo is **multiplicative** (`1 + sigmoid(ch * sp)`). The BAM *paper* specifies additive — a genuine paper/repo discrepancy, worth documenting in the write-up rather than silently resolving.
- **CBAM's spatial conv is missing its BatchNorm** (official `BasicConv` carries `BatchNorm2d`, eps=1e-5, momentum=0.01).
- Minor: bias settings and hidden-width floors differ from `c // r`; `ChannelPool` concatenation order is reversed.

The first item is the serious one. The study's central contrast is channel-only
gating (SE) versus channel+spatial gating (BAM/CBAM); that contrast is partly
collapsed while two arms share a gate neither original uses in that form.

### 8.4 The backbone is 5–10× the specified size — blocks Milestone 2

Spec: ~1–2M parameters. Measured: **11.17M** (baseline), 11.26M (SE), 11.36M
(BAM), 11.26M (CBAM) — full ResNet-18. This invalidates the 15 min/run and 2 GB
VRAM figures that the 3.85 GPU-hour feasibility case rests on.

Coupled to §8.3: official BAM/CBAM use `channels // reduction_ratio` with **no
floor**, so a narrow stage-1 at `r=16` yields a hidden dim of 1. Width and
reduction ratio must be chosen together, and both recorded.

---

## 9. Verdict

**Premise confirmed; proceed.** The gap the project targets is real, the code and
data are obtainable under permissive licenses, and the design's confound handling
is sound.

Two corrections are required before Milestone 5 (training), because both change
the architecture and would invalidate any checkpoint trained first: restore
BAM/CBAM fidelity (§8.3) and resize the backbone (§8.4). One extension is
recommended: 4 → 15 corruption types (§8.1).

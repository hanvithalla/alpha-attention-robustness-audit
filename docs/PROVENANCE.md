# Provenance and deviations

Where every attention module came from, what was adapted, and every point at
which this implementation departs from the plan or from an upstream source.
Verified mechanically by `python verify_milestones.py`.

---

## 1. Module sources

| Module | Source | Licence | Status |
|---|---|---|---|
| SE | Hu et al., *Squeeze-and-Excitation Networks*, [arXiv:1709.01507](https://arxiv.org/abs/1709.01507), Eq. 2–3 | Apache-2.0 (official repo) | **Reimplemented** — the one reimplementation the protocol allows |
| BAM | `MODELS/bam.py`, [Jongchan/attention-module](https://github.com/Jongchan/attention-module) | MIT | **Ported** |
| CBAM | `MODELS/cbam.py`, [Jongchan/attention-module](https://github.com/Jongchan/attention-module) | MIT | **Ported** |

The upstream repo's own `model_resnet.py` and `train_imagenet.py` are **not**
used: they target ImageNet-scale ResNet-50. Only the module definitions are
taken, and they are inserted into our own shared backbone (`model.py`).

**Why SE is reimplemented.** Verified 2026-09-09 by fetching
`github.com/hujie-frank/SENet`: the official repo is Caffe-only — `.cpp`/`.cu`
custom layers and `.prototxt` model definitions, no Python or PyTorch anywhere.
The PyTorch ports it links are third-party. SE's block is fully specified by two
equations, so reimplementation is unambiguous.

---

## 2. Adaptations made during the port

Kept faithful: layer types, ordering, pooling choices, bias settings, BatchNorm
placement and hyperparameters (`eps=1e-5`, `momentum=0.01` in CBAM's
`BasicConv`), `channels // reduction_ratio` bottleneck widths with no floor,
`ChannelPool` concatenation order (`max`, then `mean`), the 7×7 spatial kernel,
and BAM's `dilation=4` × 2 dilated-conv stack.

Changed, and why:

| Change | Reason |
|---|---|
| `F.sigmoid` → `torch.sigmoid` | `F.sigmoid` was removed from modern PyTorch. Numerically identical. The upstream repo has no commits since 2023-03 and ships no `requirements.txt`; this is the incompatibility the plan's risk register anticipated. |
| Classes renamed (`ChannelGate` → `BAMChannelGate` / `CBAMChannelGate`, etc.) | Both files define a class named `ChannelGate` with **different** semantics. Namespacing them in one module prevents the collision described in §4.1. |
| `Flatten` → shared `_Flatten` | Byte-identical definition in both upstream files. |

---

## 3. Protocol deviations (intentional, required by the study design)

### 3.1 Insertion point

Official BAM sits *between* network stages (at bottlenecks); official CBAM sits
*inside* each block. Here **all three** modules are inserted at the identical
point — inside every `BasicBlock`, after `bn2`, before the residual addition,
with the identity shortcut bypassing the module.

This is deliberate and is the study's central confound control: BAM and CBAM
share a repo and an insertion convention that an independently reimplemented SE
would not inherit. Holding the insertion point fixed means any measured
difference is attributable to the *module*, not to where it was placed. The cost
is that no arm reproduces its own paper's placement — so absolute accuracies are
not comparable to published numbers, and none are claimed. Only the
*within-study ranking* is.

### 3.2 Backbone scale

Plan: "~1–2M params". Actual: **11.17M** (ResNet-18 scale).

| Arm | Params |
|---|---|
| none | 11.17M |
| SE | 11.26M |
| BAM | 11.36M |
| CBAM | 11.26M |

Accepted on hardware grounds: training runs on an RTX A2000 (6 GB), where
ResNet-18 at 32×32 fits comfortably. The consequence is that the plan's
15 min/run and 2 GB VRAM figures no longer hold and the compute budget is
re-estimated (~24 h for 12 runs × 100 epochs). Recorded here because the
parameter count is a milestone-2 acceptance criterion, and this fails it.

### 3.3 Reduction ratio

`r = 16` for all three modules — the default in all three sources. Recorded as
one point in the design space, not a re-derivation of any paper's tuned value.
Bottleneck widths are therefore 4 / 8 / 16 / 32 across the four stages; the
verifier asserts none collapses below 2.

### 3.4 Baseline arm

The plan specifies 3 methods; this implementation adds a fourth `none` arm (no
attention). Without it there is no way to tell whether attention helps *at all*
under corruption — only which attention variant is least bad. This raises the
run count from 9 to 12.

---

## 4. Upstream discrepancies worth reporting in the paper

### 4.1 BAM's and CBAM's channel gates are not the same mechanism

They are easy to conflate — both files call the class `ChannelGate` — but:

| | BAM (`bam.py`) | CBAM (`cbam.py`) |
|---|---|---|
| Pooling | average **only** | average **and** max |
| MLP | `Linear → BatchNorm1d → ReLU → Linear` | `Linear → ReLU → Linear`, shared across both pools |
| Output | broadcast logits, fused with the spatial gate | sigmoid-scaled in place |

Sharing one gate class between the two arms would partially collapse the
channel-only-vs-channel+spatial contrast this study exists to measure.
`verify_milestones.py` asserts `BAMChannelGate is not CBAMChannelGate`.

### 4.2 BAM: paper says additive, repo multiplies

Park et al. specify `M(F) = σ(Mc(F) + Ms(F))`. The official `bam.py` computes
`att = 1 + sigmoid(Mc(F) * Ms(F))`, then `att * F`.

Milestone 4 requires the **repo's** definition, so the multiplicative form is
what runs here. The discrepancy is reported in the write-up rather than silently
resolved — an implementation the field has cited 2,000+ times differs from its
own paper on the fusion operator, which is worth a paragraph.

---

## 5. Data sources

| Dataset | Source | Licence | Verified |
|---|---|---|---|
| CIFAR-10 | `torchvision.datasets.CIFAR10` | no upstream licence declared | — |
| CIFAR-10-C | [`WNJXYK/TTA-CIFAR-10-C`](https://huggingface.co/datasets/WNJXYK/TTA-CIFAR-10-C) | CC-BY-4.0 | 2026-09-09: ungated, no login, **Parquet** format, 15 corruption types × 5 severities, 750k rows, 1.68 GB |

**Open issue:** `evaluate.py` currently expects the Zenodo `.npy` layout
(`50000×32×32×3` per corruption + shared `labels.npy`), not HF Parquet. A loader
is required before Milestone 7. Tracked in
`docs/milestone-01-premise-test.md` §8.2.

The study is scoped to 4 corruption types (`brightness`, `contrast`,
`defocus_blur`, `elastic_transform`). The 2026-09-09 re-probe confirms all 15
are available ungated, so extending is inference-only and costs no retraining.

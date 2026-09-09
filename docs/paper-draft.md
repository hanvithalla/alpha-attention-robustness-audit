# Do Published Attention-Block Gains Survive Common Image Corruptions?

### A Matched-Budget CIFAR-10 / CIFAR-10-C Stress Test of SE, BAM, and CBAM

**Draft v0.1** — 2026-09-09

---

## Abstract

Channel and spatial attention blocks — Squeeze-and-Excitation (SE), the
Bottleneck Attention Module (BAM), and the Convolutional Block Attention Module
(CBAM) — are compared in the literature almost exclusively on clean test
accuracy. CBAM in particular positions itself as an improvement over SE-style
channel attention on that basis. Whether the resulting ranking survives the
distribution shift a deployed classifier actually encounters has not, to our
knowledge, been tested for this method family under a matched protocol.

We train all three blocks, plus an attention-free baseline, into a single shared
CIFAR-scale ResNet backbone at an identical training budget, inserting each
module at the identical point so that architecture, data, optimizer, schedule
and seed set are held fixed and only the attention mechanism varies. We then
evaluate the trained models, without retraining, on CIFAR-10-C at five
corruption severities, and re-rank them by mean corruption accuracy and by
relative robustness drop rather than by clean top-1.

`[RESULTS SENTENCE — filled from results/findings.json]`

`[VERDICT SENTENCE — filled from results/findings.json]`

We release the shared backbone, the module ports with their provenance record,
and the full per-seed result grid.

---

## 1. Introduction

Attention blocks are among the cheapest architectural additions in computer
vision: a handful of extra parameters inserted into an existing backbone,
reported to buy a consistent accuracy gain. SE [1], BAM [2] and CBAM [3] are the
canonical examples, and the applied literature bolts them onto CNNs for image
and medical classification on the strength of the accuracy numbers those three
papers report.

Those numbers are all clean-test numbers. A practitioner choosing between SE,
BAM and CBAM for a deployed classifier is choosing on a metric collected under
conditions their deployment will not reproduce — sensors blur, lighting drifts,
contrast collapses, images get recompressed. The relevant question is not
whether these blocks help on a clean benchmark, but whether the *ordering* they
induce on a clean benchmark is the ordering that holds under corruption.

That question is cheap to answer and, as far as we can tell, unanswered.
CIFAR-10-C [4] shares CIFAR-10's exact label space and resolution, so corrupted
evaluation requires no retraining, no relabelling and no architecture change —
only inference. Yet no paper in the attention-block family we surveyed reports
a corruption-robustness metric.

This paper runs that test. Our contributions:

1. **A matched-budget comparison.** All four arms (none, SE, BAM, CBAM) share
   one backbone, one insertion point, one optimizer, one schedule, one seed set.
   Any difference is attributable to the module.
2. **A ranking-stability result.** We report whether the clean-accuracy ranking
   is preserved at each corruption severity, using mean corruption accuracy and
   a relative drop that normalises out each arm's clean starting point.
3. **A provenance record.** BAM and CBAM are ported from the official
   implementation rather than reimplemented, and we document a discrepancy
   between BAM's paper and its official code that, to our knowledge, has not
   been noted (§3.2).

We pre-committed to treating a null result — the clean ranking holding at every
severity — as equally reportable, since it answers the question as completely as
an inversion would.

---

## 2. Related Work

**Attention blocks.** SE [1] introduced channel-only recalibration: global
average pooling produces a channel descriptor, a two-layer bottleneck MLP maps
it to per-channel gates, and the features are rescaled. BAM [2] adds a parallel
spatial branch built from dilated convolutions and fuses the two gates. CBAM [3]
applies channel and spatial attention sequentially, with the channel gate
pooling both average and max statistics through a shared MLP. All three report
top-1/top-5 accuracy (and mAP for detection); none reports a robustness metric.

**Corruption robustness.** Hendrycks and Dietterich [4] introduced
CIFAR-10-C/ImageNet-C and the practice of reporting accuracy across corruption
types and severities. The benchmark is now standard for robustness claims, but
is typically used to validate a *newly proposed* method rather than to audit an
existing family.

**Closest work.** Recent papers use corruption benchmarks alongside attention
mechanisms — AR2 [5] repairs CNN robustness under common corruptions using
attention guidance, and Multi-Scale Unrectified Push-Pull with Channel Attention
[6] proposes a corruption-robust channel-attention variant. Both propose new
methods validated on CIFAR-10-C-style benchmarks. Neither is a head-to-head
audit of the existing SE/BAM/CBAM family under a single matched training
protocol, which is the gap this paper fills.

---

## 3. Method

### 3.1 Shared backbone and a single insertion point

All arms use one CIFAR-adapted ResNet-18: a 3×3 stride-1 stem with no initial
max-pool (preserving 32×32 spatial detail), four stages of two basic blocks at
widths 64/128/256/512, global average pooling, and a linear classifier. The
attention module is inserted inside every basic block, after the second batch
normalisation and before the residual addition; the identity shortcut never
passes through it. Setting the module to `None` recovers the baseline exactly.

This shared-slot design is the study's primary confound control. BAM and CBAM
originate from the same repository and share an insertion convention that an
independently reimplemented SE would not inherit; had we used each paper's own
architecture, any measured difference would be confounded with placement. Fixing
the slot means the only thing varying across arms is the module itself.

The cost is that no arm reproduces its own paper's placement — official BAM sits
*between* stages rather than inside blocks. Absolute accuracies are therefore
not comparable to published numbers, and we claim none. Only the within-study
ranking is claimed.

### 3.2 Attention modules and their provenance

**SE** is reimplemented from Eq. 2–3 of [1]: global average pool → `Linear(C,
C/r)` → ReLU → `Linear(C/r, C)` → sigmoid → channel-wise rescale. This is the
study's one reimplementation, and it is necessary: the official SENet repository
is Caffe-only (`.prototxt` model definitions with custom `.cpp`/`.cu` layers)
and contains no Python. SE's block is fully specified by two equations, so the
reimplementation is unambiguous. We use r = 16, the paper's default, and record
it as one point in the design space rather than a re-derivation of a tuned value.

**BAM** and **CBAM** are ported directly from `MODELS/bam.py` and
`MODELS/cbam.py` of the official implementation [7] (MIT). We take only the
module definitions; the repository's own `model_resnet.py` targets ImageNet-scale
ResNet-50 and is not used. Layer types, ordering, pooling choices, bias
settings, batch-norm placement and hyperparameters (`eps=1e-5`, `momentum=0.01`
in CBAM's spatial convolution), `C // r` bottleneck widths, channel-pool
concatenation order and BAM's dilation-4 convolution stack are all preserved.
The only change is `F.sigmoid` → `torch.sigmoid`, required because the former
was removed from modern PyTorch; the repository has had no commits since March
2023 and ships no dependency pins.

Two details deserve emphasis, because conflating them would silently weaken the
comparison:

**BAM's and CBAM's channel gates are different mechanisms.** Both files define a
class named `ChannelGate`, but BAM's pools average statistics only and places
`BatchNorm1d` between its MLP layers, while CBAM's pools average *and* max
through a shared MLP with no normalisation. Sharing one implementation between
the two arms would partially collapse the channel-only-versus-channel+spatial
contrast this study exists to measure. Our verification suite asserts the two
classes are distinct.

**BAM's paper and its official code disagree on the fusion operator.** Park et
al. [2] specify additive fusion, `M(F) = σ(Mc(F) + Ms(F))`. The official
`bam.py` instead computes `att = 1 + sigmoid(Mc(F) * Ms(F))` and returns
`att * F` — a multiplicative fusion with a residual offset. Since the widely
used artefact in this literature is the code rather than the equation, we follow
the repository, and we report the discrepancy rather than silently resolving it.

### 3.3 Verification

Because the comparison's validity rests on the modules being what they claim to
be, module fidelity is checked mechanically rather than by inspection. A
verification suite asserts structural properties (BAM's average-only pooling and
`BatchNorm1d`; CBAM's dual pooling and batch-normalised 7×7 spatial convolution;
SE's pooling, bottleneck width and sigmoid gate) and, critically, *behavioural*
ones: BAM's output-to-input ratio must lie in (1, 2), as `1 + sigmoid(·)`
requires, and CBAM's in (0, 1), as a pure multiplicative gate requires. Fusion
semantics are thus verified by behaviour, not by reading the source. All checks
run before every training installment.

---

## 4. Experimental Setup

**Data.** CIFAR-10 (50,000 train / 10,000 test, 32×32, 10 classes) for training
and the clean reference point. CIFAR-10-C [4] (CC-BY-4.0) for corrupted
evaluation, restricted to four corruption types — `brightness`, `contrast`,
`defocus_blur`, `elastic_transform` — at severities 1–5, 10,000 images each.
Test-time preprocessing (normalisation by CIFAR-10 channel statistics) is
identical for clean and corrupted images.

**Training.** SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4, batch
size 128, cosine annealing over the full run, standard augmentation (random
32×32 crop with 4-pixel padding, random horizontal flip). Mixed precision is
used for training; validation runs in fp32 so that checkpoint selection and
final evaluation share a numeric path. Every arm receives an identical budget of
`[EPOCHS]` epochs and the identical seed set {1, 2, 3}, giving
`[N_RUNS]` runs. The best-validation-accuracy checkpoint per run is evaluated.

**Metrics.** (i) *Clean top-1* — the metric the field currently ranks on.
(ii) *Mean corruption accuracy (mCA)* at each severity, averaged over corruption
types. (iii) *Relative robustness drop*, `(clean − corrupted) / clean`, which
normalises out each arm's clean starting point so an arm that begins lower is
not mistaken for a robust one merely because it had less accuracy to lose. All
results are averaged over seeds and reported with the across-seed standard
deviation.

**Hardware.** A single NVIDIA RTX A2000 (6 GB), PyTorch 2.5.1 / CUDA 12.1.

---

## 5. Results

`[TO FILL from results/ranking_table.md — per-severity accuracy table, ranking
table, per-corruption-type breakdown]`

---

## 6. Discussion

`[TO FILL]`

---

## 7. Limitations

**Three methods, one baseline.** The design supports per-severity, descriptive
statements — "the ranking held/changed at severity *s*" — and not a general
claim that attention *type* predicts corruption robustness. Four points do not
characterise a design space.

**Four of nineteen corruption types.** Averaging over `brightness`, `contrast`,
`defocus_blur` and `elastic_transform` could flatter or punish spatial attention
relative to the full benchmark. We report the per-type breakdown alongside the
mean so that a skewed average is visible rather than hidden, and we scope the
finding to these four types. All fifteen types in the source are ungated, so
extension is inference-only and costs no retraining.

**Training budget.** The budget is matched across arms, which is what the
ranking claim requires, but it is short. Longer training could change absolute
accuracies and, in principle, the ordering. The seed-spread guard in §5
indicates whether the measured gaps are resolvable at this budget at all.

**Insertion point.** As noted in §3.1, holding the slot fixed trades
reproduction of each paper's own architecture for a clean attribution of
differences to the module. Absolute numbers are not comparable to published
results.

**Single dataset and resolution.** CIFAR-scale only.

---

## 8. Conclusion

`[TO FILL]`

---

## References

[1] J. Hu, L. Shen, G. Sun. *Squeeze-and-Excitation Networks.* arXiv:1709.01507.

[2] J. Park, S. Woo, J.-Y. Lee, I. S. Kweon. *BAM: Bottleneck Attention Module.*
arXiv:1807.06514.

[3] S. Woo, J. Park, J.-Y. Lee, I. S. Kweon. *CBAM: Convolutional Block Attention
Module.* ECCV 2018.

[4] D. Hendrycks, T. Dietterich. *Benchmarking Neural Network Robustness to
Common Corruptions and Perturbations.* ICLR 2019.

[5] *AR2: Attention-Guided Repair for the Robustness of CNNs Against Common
Corruptions.* 2025.

[6] *Multi-Scale Unrectified Push-Pull with Channel Attention for Enhanced
Corruption Robustness.* 2025.

[7] Jongchan Park. `attention-module`. https://github.com/Jongchan/attention-module (MIT).

---

## Appendix A — Provenance and deviations

Full record in `docs/PROVENANCE.md`: module sources and licences, every
adaptation made during porting, and three intentional protocol deviations
(shared insertion point, backbone scale, and the added attention-free baseline
arm, which the original three-method plan did not include but without which one
cannot tell whether attention helps at all under corruption).

## Appendix B — Reproduction

```bash
python verify_milestones.py                      # 54 module-fidelity checks
python run_next.py --epochs E --time-limit-min T # advances the least-finished cell
python evaluate.py --model CBAM --seed 1         # clean + CIFAR-10-C
python aggregate.py                              # seed averaging + ranking table
```

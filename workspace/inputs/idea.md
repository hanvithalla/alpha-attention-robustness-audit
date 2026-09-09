# Do Published Attention-Block Gains Survive Common Image Corruptions?

## Problem Statement

Channel and spatial attention blocks are compared in the literature almost
exclusively on clean test accuracy. Squeeze-and-Excitation (SE), the Bottleneck
Attention Module (BAM), and the Convolutional Block Attention Module (CBAM) each
report top-1/top-5 accuracy, and CBAM positions itself as an improvement over
SE-style channel attention on precisely that basis. The applied literature that
inserts these blocks into CNNs for image and medical classification inherits the
same single metric.

A practitioner selecting among these blocks for a deployed classifier is
therefore selecting on a metric collected under conditions the deployment will
not reproduce: sensors blur, illumination drifts, contrast collapses, images are
recompressed. The unanswered question is not whether attention blocks improve
clean accuracy, but whether the **ordering** they induce on clean data is the
ordering that survives corruption.

Two confounds make the existing literature unable to answer this. First, each
paper evaluates its own block in its own backbone at its own budget, so
cross-paper accuracy differences conflate the module with the architecture and
the training recipe. Second, BAM and CBAM originate from a single shared
repository and inherit a common insertion convention that an independently
reimplemented SE does not, so even a same-codebase comparison confounds module
identity with placement.

## Core Hypothesis

If the clean-accuracy ranking of SE, BAM and CBAM were a reliable selection
criterion, it would be invariant to corruption severity. We hypothesise it may
not be: SE applies a channel-only gate computed from a global average-pooled
descriptor, whereas BAM and CBAM additionally compute **spatial** attention maps
from local feature statistics. Spatial masks depend on where informative
structure lies in the image, and corruptions such as defocus blur and contrast
reduction attack precisely that spatial structure. A spatial gate that
mislocalises under corruption could therefore degrade faster than a channel-only
gate, compressing or inverting the clean ranking.

The competing outcome is equally informative and is treated as equally
reportable: if the clean ranking is preserved at every severity, the field's
single clean-accuracy metric was not misleading for this method set, which is a
positive finding obtainable only by running the experiment.

## Proposed Methodology (Detailed Technical Approach)

**Shared backbone with one insertion point.** We define a single CIFAR-adapted
ResNet-18: a 3x3 stride-1 stem with no initial max-pool (preserving 32x32
spatial detail), four stages of two basic blocks at widths 64/128/256/512,
global average pooling, and a linear classifier. Within each basic block the
attention module is applied to the residual branch after the second batch
normalisation and before the residual addition:

    out = relu( A(BN2(conv2(relu(BN1(conv1(x)))))) + shortcut(x) )

where A is the attention module under test, or the identity for the baseline
arm. The identity shortcut never passes through A. Because A is the only term
that varies, any measured difference is attributable to the module rather than
to its placement, its host architecture, or its training recipe. This is the
study's primary confound control, and it deliberately overrides each paper's own
placement convention (official BAM sits between stages rather than inside
blocks), at the cost that absolute accuracies are not comparable to published
numbers.

**Modules.** SE is reimplemented from Eq. 2-3 of its paper: global average pool
to z in R^C, then s = sigma(W2 delta(W1 z)) with W1 in R^{(C/r) x C}, then
channel-wise rescale by s. This is the study's single reimplementation and is
necessary because the official SENet release is Caffe-only and contains no
Python. BAM and CBAM are ported directly from the module definitions of the
official public implementation, preserving layer types, pooling choices, bias
settings, batch-norm placement and hyperparameters, and bottleneck widths C/r.
Reduction ratio r = 16 throughout, recorded as one point in the design space.

Two module details are load-bearing and must not be conflated. BAM's channel
gate pools **average statistics only** and places BatchNorm1d between its MLP
layers; CBAM's pools **average and max** through a shared MLP without
normalisation. Sharing one implementation between the two arms would partially
collapse the channel-only-versus-channel-plus-spatial contrast the study
measures. Separately, BAM's paper specifies additive fusion of its two gates
while its official code computes a multiplicative fusion with a residual offset;
we follow the code, since the code is the artefact the field actually reuses,
and report the discrepancy.

**Module verification.** Because the comparison's validity rests on the modules
being what they claim to be, fidelity is asserted mechanically rather than by
inspection. A verification suite checks structural properties and, critically,
behavioural ones: BAM's output-to-input ratio must lie in (1,2) as its residual
multiplicative form requires, and CBAM's in (0,1) as a pure gate requires.

**Matched-budget training.** Four arms (none, SE, BAM, CBAM) are trained from
scratch on the CIFAR-10 training split under an identical recipe: SGD, learning
rate 0.1, momentum 0.9, weight decay 5e-4, batch size 128, cosine annealing,
random 32x32 crop with 4-pixel padding and random horizontal flip. Every arm
receives the identical epoch budget and the identical seed set, and no arm uses
a pretrained or externally obtained checkpoint. Mixed precision is used for
training; validation runs in full precision so that checkpoint selection and
final evaluation share one numeric path.

**Corruption evaluation.** The trained checkpoints are evaluated without
retraining on CIFAR-10-C, which shares CIFAR-10's exact label space and
resolution, across corruption types at severities 1 through 5. Test-time
preprocessing is identical for clean and corrupted images.

**Metrics.** (i) Clean top-1 accuracy, the metric the field currently ranks on.
(ii) Mean corruption accuracy at each severity, averaged over corruption types.
(iii) Relative robustness drop, (clean - corrupted) / clean, which normalises
out each arm's clean starting point so that an arm beginning lower is not
mistaken for a robust one merely because it had less accuracy to lose. Results
are averaged over seeds and reported with across-seed standard deviation, and
the analysis explicitly checks whether the gap between the top two arms exceeds
seed noise before any ranking claim is made.

## Expected Contribution

1. **A matched-budget, single-backbone comparison** of SE, BAM and CBAM against
   an attention-free baseline, in which architecture, insertion point, optimizer,
   schedule, augmentation and seed set are all held fixed.

2. **A ranking-stability result**: whether the clean-accuracy ordering is
   preserved at each corruption severity, reported on mean corruption accuracy
   and on a starting-point-normalised relative drop, with the per-corruption-type
   breakdown shown alongside the mean so that a skewed average is visible rather
   than hidden.

3. **A provenance record** for the compared implementations, including a
   discrepancy between BAM's published fusion equation and the fusion its
   official code computes, which to our knowledge has not been reported.

4. **Practical guidance**: whether the clean-accuracy ranking a practitioner
   would read off the original papers is the ranking that should drive the
   choice, or whether corruption robustness reorders it.

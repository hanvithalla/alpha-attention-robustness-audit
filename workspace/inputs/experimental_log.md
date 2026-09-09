## Orchestration Decisions
Before any training was run, we audited the module implementations against the official public release rather than trusting a hand-written reproduction. The audit found that the two spatial-attention arms had been sharing a single channel-gate implementation, which would have partially collapsed the channel-only versus channel-plus-spatial contrast the study exists to measure. We replaced both modules with faithful ports before training, and added a mechanical verification suite that asserts module fidelity behaviourally, so the defect could not silently return.
We added a fourth, attention-free arm that the original three-method plan did not include. Without it the study could only report which attention variant degrades least, not whether attention helps at all under corruption.
We fixed the attention insertion point across all arms rather than reproducing each method's own placement convention. This was a deliberate trade: it forfeits comparability with published absolute accuracies in exchange for attributing any measured difference to the module rather than to its position in the network.

## Experiment 1: Matched-budget attention-block corruption audit (complete, objective: test whether the clean-accuracy ranking of SE, BAM and CBAM survives common image corruptions)
**Setup**
- Dataset (training and clean evaluation): CIFAR-10, 50,000 training images and 10,000 test images at 32x32 resolution across 10 classes.
- Dataset (corrupted evaluation): CIFAR-10-C, restricted to 4 corruption types (brightness, contrast, defocus_blur, elastic_transform) at severities 1 through 5, 10,000 images per type and severity. Inference only, no retraining.
- Backbone: one shared CIFAR-adapted ResNet-18 with a 3x3 stride-1 stem and no initial max-pool, four stages of two basic blocks at widths 64, 128, 256 and 512.
- Arms: none (attention-free baseline), SE, BAM, CBAM. The attention module is applied to the residual branch after the second batch normalisation and before the residual addition; the identity shortcut never passes through it.
- Parameter counts: none 11.17M, SE 11.26M, BAM 11.36M, CBAM 11.26M.
- Reduction ratio r = 16 for all three attention modules.
- Training: SGD with learning rate 0.1, momentum 0.9, weight decay 5e-4, batch size 128, cosine annealing, random 32x32 crop with 4-pixel padding and random horizontal flip. Budget of 10 epochs per run, identical across arms.
- Seeds: [1, 2, 3], giving 12 training runs.
- Precision: mixed precision for training, full precision for validation so that checkpoint selection and final evaluation share one numeric path.
- Hardware: a single NVIDIA RTX A2000 with 6GB of memory, PyTorch 2.5.1 with CUDA 12.1.
- Hypothesis: the clean-accuracy ranking is not invariant to corruption severity, because spatial attention maps depend on local structure that blur and contrast reduction attack directly.

**Results**
Clean top-1 accuracy, percent, mean over 3 seeds with across-seed standard deviation:
- none: 86.09 +/- 2.10
- SE: 89.21 +/- 0.18
- BAM: 81.56 +/- 2.47
- CBAM: 89.32 +/- 0.08

Mean corruption accuracy by severity, percent, averaged over 4 corruption types, mean over seeds with standard deviation:
- none: severity 1 83.96 +/- 2.25, severity 2 80.99 +/- 2.93, severity 3 75.53 +/- 3.00, severity 4 67.12 +/- 2.84, severity 5 55.01 +/- 0.99
- SE: severity 1 87.29 +/- 0.10, severity 2 85.05 +/- 0.05, severity 3 80.09 +/- 0.23, severity 4 71.98 +/- 0.06, severity 5 57.16 +/- 0.05
- BAM: severity 1 79.28 +/- 2.71, severity 2 75.51 +/- 3.09, severity 3 70.08 +/- 3.08, severity 4 62.83 +/- 2.67, severity 5 52.91 +/- 2.20
- CBAM: severity 1 87.68 +/- 0.22, severity 2 85.58 +/- 0.28, severity 3 80.94 +/- 0.67, severity 4 73.13 +/- 0.49, severity 5 58.87 +/- 0.77

Mean corruption accuracy across all severities (mCA), percent:
- none: 72.52 +/- 2.31
- SE: 76.31 +/- 0.04
- BAM: 68.12 +/- 2.54
- CBAM: 77.24 +/- 0.45

Mean accuracy per corruption type, percent, averaged over severities 1 through 5 and over seeds:
- none: brightness 83.76, contrast 59.66, defocus_blur 72.28, elastic_transform 74.38
- SE: brightness 86.83, contrast 67.06, defocus_blur 74.63, elastic_transform 76.75
- BAM: brightness 78.66, contrast 53.27, defocus_blur 69.40, elastic_transform 71.16
- CBAM: brightness 87.33, contrast 67.53, defocus_blur 76.05, elastic_transform 78.05

Final-epoch training accuracy, percent, mean over seeds (reported because it separates underfitting from a worse optimum):
- none: 87.90
- SE: 91.88
- BAM: 82.79
- CBAM: 92.48

Relative robustness drop, (clean - corrupted) / clean, percent:
- none: 15.77 +/- 0.74
- SE: 14.45 +/- 0.21
- BAM: 16.49 +/- 0.98
- CBAM: 13.52 +/- 0.46

Ranking by clean accuracy, best first: CBAM > SE > none > BAM
Ranking at severity 1, best first: CBAM > SE > none > BAM
Ranking at severity 2, best first: CBAM > SE > none > BAM
Ranking at severity 3, best first: CBAM > SE > none > BAM
Ranking at severity 4, best first: CBAM > SE > none > BAM
Ranking at severity 5, best first: CBAM > SE > none > BAM
Ranking by mCA, best first: CBAM > SE > none > BAM
Ranking by relative robustness drop, smallest drop first: CBAM > SE > none > BAM
Largest across-seed standard deviation observed in any cell: 3.09 percentage points
Clean-accuracy gap between the two best arms: 0.11 percentage points

**Baselines**
- none (attention-free ResNet-18): trained by us from scratch under the identical budget. No externally obtained or pretrained checkpoint was used for any arm.
- SE, BAM and CBAM: all trained by us from scratch under the identical budget. Published accuracies for these modules are not comparable to ours, because we hold the insertion point fixed rather than reproducing each method's own placement, and we therefore neither cite nor compare against them numerically.

**Figures**
- accuracy_vs_severity: mean corruption accuracy on the vertical axis against corruption severity 0 through 5 on the horizontal axis, one line per arm, error bars showing across-seed standard deviation. Renders the per-severity accuracy metrics.
- relative_drop_by_arm: relative robustness drop per arm as a bar chart with across-seed standard deviation. Renders the relative robustness drop metric.
- per_corruption_breakdown: grouped bars of mean accuracy per corruption type per arm, averaged over severities and seeds. Renders the per-corruption-type metrics and makes a skewed average visible.
- ranking_stability: the ordering of arms at each severity, shown so that a reordering between clean and any severity is directly readable. Renders the ranking metrics.

**Notes & Limitations**
- Only three attention methods plus one baseline were compared. The design supports per-severity descriptive statements about whether the ranking held or changed, and does not support a general claim that attention type predicts corruption robustness. Four points do not characterise a design space.
- Only 4 of the benchmark's corruption types were used. Averaging over them could flatter or punish spatial attention relative to the full benchmark, so the per-type breakdown is reported alongside the mean and the finding is scoped to these types. All types in the source are ungated, so extension is inference-only and requires no retraining.
- The training budget of 10 epochs is matched across arms, which is what the ranking claim requires, but it is short. Longer training could change absolute accuracies and in principle the ordering.
- Holding the insertion point fixed means no arm reproduces its own paper's placement, so absolute accuracies are not comparable to published results and none is claimed.
- A single dataset at a single resolution was used.
- The clean-accuracy gap between the two best arms is 0.11 percentage points against a combined across-seed standard deviation of 0.26 percentage points. On clean data those two arms are therefore not separated by more than seed noise, and their relative order should be read as unresolved rather than measured.
- On mean corruption accuracy the same two arms are separated by 0.92 percentage points against a combined across-seed standard deviation of 0.49 percentage points, which does exceed seed noise. The corruption measurement therefore resolves a difference between these two arms that the clean measurement cannot.

**Decisions**
- We fixed one insertion point across all arms rather than reproducing each method's own convention, so that a measured difference is attributable to the module and not to its placement.
- We ported BAM and CBAM from the official public implementation and reimplemented only SE, whose official release is Caffe-only and contains no Python, and whose block is fully specified by two equations.
- We kept BAM's and CBAM's channel gates as separate implementations, because the official versions differ: BAM pools average statistics only and normalises between its MLP layers, while CBAM pools average and max through a shared MLP.
- We followed the official BAM code's multiplicative gate fusion rather than the additive fusion its paper specifies, because the code is the artefact the field reuses, and we report the discrepancy rather than resolving it silently.
- We verified module fidelity behaviourally as well as structurally, asserting that BAM's output-to-input ratio lies between 1 and 2 and CBAM's between 0 and 1, so that fusion semantics are checked by behaviour rather than by reading source code.
- We report a relative robustness drop alongside absolute corruption accuracy, so that an arm starting from lower clean accuracy is not mistaken for a robust one.

-> implementation: experiments/attention-robustness/code/

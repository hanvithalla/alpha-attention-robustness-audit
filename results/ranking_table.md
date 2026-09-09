# Results (3 seeds: [1, 2, 3]; corruptions: brightness, contrast, defocus_blur, elastic_transform)


## Accuracy by severity (mean +/- std over seeds)

| Model | Clean | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 | mCA | Rel. drop |
|---|---|---|---|---|---|---|---|---|
| none | 86.09 ± 2.10 | 83.96 ± 2.25 | 80.99 ± 2.93 | 75.53 ± 3.00 | 67.12 ± 2.84 | 55.01 ± 0.99 | 72.52 ± 2.31 | 15.77% ± 0.74 |
| SE | 89.21 ± 0.18 | 87.29 ± 0.10 | 85.05 ± 0.05 | 80.09 ± 0.23 | 71.98 ± 0.06 | 57.16 ± 0.05 | 76.31 ± 0.04 | 14.45% ± 0.21 |
| BAM | 81.56 ± 2.47 | 79.28 ± 2.71 | 75.51 ± 3.09 | 70.08 ± 3.08 | 62.83 ± 2.67 | 52.91 ± 2.20 | 68.12 ± 2.54 | 16.49% ± 0.98 |
| CBAM | 89.32 ± 0.08 | 87.68 ± 0.22 | 85.58 ± 0.28 | 80.94 ± 0.67 | 73.13 ± 0.49 | 58.87 ± 0.77 | 77.24 ± 0.45 | 13.52% ± 0.46 |

## Ranking by severity

| Condition | Ranking (best first) | Same as clean? |
|---|---|---|
| Clean (sev 0) | CBAM > SE > none > BAM | -- |
| Severity 1 | CBAM > SE > none > BAM | yes |
| Severity 2 | CBAM > SE > none > BAM | yes |
| Severity 3 | CBAM > SE > none > BAM | yes |
| Severity 4 | CBAM > SE > none > BAM | yes |
| Severity 5 | CBAM > SE > none > BAM | yes |
| mCA (all sev) | CBAM > SE > none > BAM | yes |
| Relative drop | CBAM > SE > none > BAM | yes |

## Per-corruption-type mean accuracy (averaged over severities and seeds)

| Model | brightness | contrast | defocus_blur | elastic_transform |
|---|---|---|---|---|
| none | 83.76 | 59.66 | 72.28 | 74.38 |
| SE | 86.83 | 67.06 | 74.63 | 76.75 |
| BAM | 78.66 | 53.27 | 69.40 | 71.16 |
| CBAM | 87.33 | 67.53 | 76.05 | 78.05 |

## Verdict

The clean-accuracy ranking is **preserved at every severity** and on mCA. For this method set, clean top-1 was not a misleading basis for choosing between these attention blocks.

## Is the top-two gap resolvable at this budget?

| Metric | Top two | Gap (pp) | Sum of their seed sds | Resolved? |
|---|---|---|---|---|
| Clean top-1 | CBAM vs SE | 0.11 | 0.26 | no |
| mCA | CBAM vs SE | 0.92 | 0.49 | **yes** |

Largest per-cell seed std anywhere in the grid: 3.09 pp (from the higher-variance none/BAM arms). That figure is reported for completeness but is not the right denominator for a two-arm comparison.

**The two metrics disagree about what is resolvable.** Clean accuracy cannot separate CBAM from SE (0.11 pp, inside their combined seed spread of 0.26 pp), but mean corruption accuracy can (0.92 pp against 0.49 pp). On this method set the corruption axis is the more discriminative measurement, not the noisier one.

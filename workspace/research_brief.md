# Research Brief

## §1 · Core Claim and Narrative
_Written by: outline-agent, Step 1_

**Core claim:** The clean-accuracy ranking of SE, BAM and CBAM — the only
number this sub-literature reports — is re-tested under common image
corruptions in a matched-budget, single-backbone, single-insertion-point
setting, and we report whether that ranking survives.

**Narrative tension:** Practitioners select attention blocks on a metric
measured under conditions their deployment never reproduces, and the existing
literature cannot resolve the question because per-paper backbones confound the
module with its host, and because two of the three modules share a repository
and insertion convention the third does not.

**Key novelty framing:** Not a new method. An audit of published components
under one controlled protocol, using a benchmark that is free, ungated, and
inference-only for this method family.

**Outline decisions:**
- Plotting plan: 5 figures (1 diagram, 4 plots)
- Related Work clusters: Channel and Spatial Attention Blocks; Corruption
  Robustness Benchmarking; Attention Mechanisms Under Distribution Shift
- Section structure: Introduction; Related Work; Method; Experimental Setup;
  Results and Analysis; Limitations; Conclusion

**Potential weaknesses flagged at outline stage:**
- The training budget is short. If the arms separate by less than seed noise,
  the ranking question is unresolved rather than answered, and the paper must
  say so instead of ranking noise. The Results section and the Limitations
  section both carry an explicit instruction to check this.
- Only four of the benchmark's corruption types are used, so the aggregate
  could be skewed; the per-type breakdown is mandated alongside it.
- Absolute accuracies are not comparable to published numbers because the
  insertion point is held fixed rather than reproducing each paper's own
  placement. The paper must claim only the within-study ranking.

## §2 · Literature Grounding
_Written by: literature-review-agent, Step 3_

**Discovery mode:** Degraded. No Semantic Scholar or Exa API key is available
in this environment, so the skill's parallel candidate-discovery and
S2-verification pipeline was not run. Instead the outline's citation_hints were
resolved by title against OpenAlex (no key required), fuzzy-matched to reject
wrong hits, and only confidently matched records were written to refs.bib.

**Verified:** 15/15 references, all at match ratio 1.00.

**Known caveat:** OpenAlex reliably confirms existence and title but not
canonical venue/year — it frequently returns the preprint record, and returned
the CIFAR-10 technical report re-dated to 2024. A canonical venue/year override
table is applied on top and every substitution is logged in
`citation_pool.json` under `canonical_override`, so the bibliography is correct
and the correction is auditable.

**Consequence for the paper:** The reference list is deliberately small and
hand-checked (15 entries) rather than the 30–50 the full pipeline would
assemble. This is stated rather than disguised.

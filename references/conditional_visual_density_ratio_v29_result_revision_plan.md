# Conditional Visual Density Ratio V29 Result Revision Plan

Date: 2026-08-13

Status: completed result-publication revision; created after the fixed
development audit and before any V29 manuscript edit, then closed after source,
PDF, visual, and test verification

## Revision Stage

Internal major result revision. There is no reviewer response, supplement, or
submission package in this repository. The active manuscript is
`publication/ilm-image-native/ilm-image-native.tex`; the matching active PDF is
`publication/ilm-image-native/ilm-image-native.pdf`. No preserved TeX baseline
or redline exists, so source diff and compiled-PDF inspection will provide the
trace. Existing active filenames must remain stable.

## Problem

The repository now has a completed preregistered V29 development experiment,
but the public result narrative stops at V28. The fixed evidence rejects V29 as
a selected language mechanism while localizing the remaining failure:

- full 1,024-way top-1 is `0.0234375`, above image unigram `0.013671875`
  but below symbolic bigram `0.138671875` and trigram `0.20263671875`;
- full target log probability is `0.1794459149` nat better than suffix-4 and
  `3.1024791263` nat better than suffix-preserving shuffled context, but
  `1.5166222926` nat worse than the symbolic bigram;
- matched-suffix incremental arm accuracy is `0.507080078125`, both-correct is
  `0.08984375`, and ordered-minus-shuffled arm accuracy is only
  `0.011474609375`;
- incremental mean-margin gain over shuffled is `0.0031665079`;
- full-score arm accuracy is `0.497314453125`, with `0.0380859375`
  both-correct;
- suffix-only arm accuracy is exactly `0.5`, suffix pixels and score rows are
  exact, and all candidate-permutation score errors are zero;
- raw-retina two-candidate identity is `0.99951171875`, and frozen-semantic
  1,024-way cross-font identity is `0.96435546875`;
- 8 of 14 mechanism gates and 2 of 6 language gates pass; and
- no frozen image was instantiated, while frozen evaluation and writer
  training remain unauthorized.

Training-batch metrics must not be substituted for the deterministic
development audit. In particular, the large ordered-versus-shuffled natural
log-probability effect cannot be described as candidate binding.

## Fixed Evidence

- Audit:
  `artifacts/conditional_visual_density_ratio_v29_evidence/development_audit.json`
- Audit SHA-256:
  `16645844dd0b9dd4fb1e5157edbdd20d6e13b34b201c1147c177ac52464a5108`
- Final checkpoint:
  `artifacts/conditional_visual_density_ratio_v29_evidence/checkpoint_final.pt`
- Checkpoint SHA-256:
  `a8ec991968b577518d801090f5953406de13c688552107f26ac400fc2d508b8a`
- Protocol SHA-256:
  `4cb17b793de051d858418d3ec0a4cb08b2928308c2d8e56530fc5655d9ffad0f`
- Corpus SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`
- V28 source checkpoint SHA-256:
  `22503464cf5f5e8ed2d6adebbd6c794f6bc9b2836f978872027cb51712c7f64f`
- Frozen retina SHA-256:
  `90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`
- Candidate canonical-pixel SHA-256:
  `fbcce3b2661b6eca7697d631772a84043f42495afd289d067d5ea2d7384d50ce`
- Candidate two-view image SHA-256:
  `3d536fdb06b795080e7e0c8814b8b155b37221dd3c7986a025e96badb003bc31`
- Fixed run: 8,000 BF16 AdamW updates, 20,080,961 total parameters,
  18,451,585 trainable parameters, 3.356465 GiB peak allocated CUDA memory,
  and 4,956.14 seconds for training plus audit on one RTX 4090.

The artifact directory is ignored by Git. The tracked result receipt, figure
generator, generated figure, protocol, and source code provide the durable
public trace.

## Allowed Files And Locations

1. Add `docs/conditional-visual-density-ratio-v29-result.md` as the canonical
   result receipt, including verdict, fixed system, natural audit, suffix
   intervention, algebraic diagnosis, bounded interpretation, and reproduction
   commands.
2. Add
   `publication/ilm-image-native/generate_v29_result_figure.py` and its generated
   `figures/conditional_visual_density_ratio_v29_result.png`. The generator must
   read and validate the fixed JSON evidence and exact hashes before drawing.
3. Update the current-result section and artifact index in `README.md`.
4. Update only V29/current-state passages in
   `docs/first-imagized-language-model-goal.md`.
5. Update `publication/ilm-image-native/ilm-image-native.tex` only in the
   abstract/current evidence summary, contributions, post-V28 results,
   discussion, conclusion, and immediately related roadmap text. Add the V29
   figure and tables. Keep the active filename stable and rebuild the matching
   PDF.
6. Update this plan with execution and verification status.

## Out Of Scope

- Opening or evaluating the frozen partition.
- Training or authorizing a writer.
- Changing V29 code, protocol, thresholds, data, checkpoint selection, or
  audit seed.
- Claiming that V29 is a useful language model, a Qwen-8B peer, a general ILM,
  or an efficiency improvement over token language models.
- Treating training-set order gain or pair accuracy as development evidence.
- Calling cross-font identity language understanding.
- Adding page geometry, a third spatial/depth axis, motion, historical-glyph
  generation, or etymology capability to V29.
- Rewriting prior V1--V28 result sections except where a transition sentence
  must identify V29 as the current result.
- Editing multilingual README translations in this bounded result unit.
- Implementing or preregistering V30 before the V29 result profile is fully
  published and verified.

## Claim Change

Before: V28 is the latest completed development test, and candidate-conditioned
prefix-increment scoring is the next controlled hypothesis.

After: V29 is the latest completed development test. It demonstrates a compact
image-only conditional scorer with exact counterfactual and leakage controls,
strong same-scope visual identity, and a large natural ordered-context
log-probability effect. It nevertheless fails both held-out candidate binding
and language retrieval. The density-ratio subtraction also cancels exactly in
the two-by-two paired assignment margin, so it cannot repair a full critic that
does not itself learn context-candidate interaction. V29 is rejected; frozen
and writer stages remain sealed. Any V30 proposal must change the interaction
mechanism rather than only the baseline, dimensionality, or scale.

## Figure Contract

The measured figure must show:

1. authoritative input `B x N x 1 x 32 x 32` and arbitrary candidate image;
2. frozen retina and semantic adapters, causal context field, candidate-query
   cross-attention, full score `F`, suffix score `B`, and centered increment
   `G = F - B` without a token vocabulary;
3. natural top-1 bars for full, unigram, bigram, and trigram;
4. pair bars for full, suffix, shuffled, increment, and shuffled increment arm
   accuracy, plus both-correct rates;
5. raw versus frozen-semantic same-scope identity;
6. the exact paired-margin cancellation statement;
7. gate counts and the explicit `REJECTED ON DEVELOPMENT` verdict; and
8. sealed frozen split and unauthorized writer.

No generated decorative image may replace these measured values.

## Verification

1. Regenerate the V29 figure from the fixed audit twice and confirm stable
   SHA-256.
2. Run the figure generator against a modified/hash-mismatched report and
   confirm it rejects the evidence.
3. Run `git diff --check` and focused static checks for the generator.
4. Build `publication/ilm-image-native/ilm-image-native.tex` with the repository
   LaTeX build script.
5. Confirm the PDF page count and extract text with `pdftotext -layout`.
6. Render and visually inspect every changed PDF page, including the V29
   figure, tables, abstract, discussion, and conclusion. Check clipping,
   overlap, tiny text, and unsupported wording.
7. Search README, goal, result receipt, TeX source, and extracted PDF for exact
   gate counts and key metrics.
8. Run the full Python test suite to ensure publication utilities did not
   disturb executable behavior.

## Response-Letter Impact

None. There is no response letter. The canonical result receipt and this plan
serve as the traceable internal rationale.

## Execution Record

Completed on 2026-08-13.

- The canonical result receipt was committed as `26549cf`; the measured figure
  generator and PNG were committed as `791e7ed`; and the README and project-goal
  verdict were committed as `4aa4074`.
- The figure was regenerated twice from the fixed audit with stable SHA-256
  `20960d6d33c060826fb166bd59aee10bd27649efef26323c4c86354c02d72a32`.
  A deliberately modified audit was rejected at the fixed evidence-hash check.
- `ruff check publication/ilm-image-native/generate_v29_result_figure.py` and
  `git diff --check` passed.
- The manuscript was built twice with `scripts/latex_build.sh`. The final TeX
  SHA-256 is
  `0031083b45614e85bd78a072393e580b98b2df93325949cadc17c81e5cda47f6`;
  the final PDF SHA-256 is
  `637198f4c430b400037887b289583cca0b211d8bc08df058d50367fa9ba2b3fb`.
  The PDF is 56 US-letter pages and 8,723,345 bytes.
- The retained LaTeX log contains no undefined-reference, undefined-citation,
  or overfull-box warning. Extracted PDF text contains the fixed V29 gate
  counts, natural and pair metrics, rejection, and suffix-cancellation result.
- Directly changed and downstream-reflow pages 1--6 and 36--56 were rendered
  and inspected. The V29 equations, tables, measured figure, captions,
  discussion, and conclusion have no clipping or overlap. The unchanged Figure
  23 bitmap retains a pre-existing footer line at its right boundary; repairing
  that unrelated asset remains outside this revision.
- `PYTHONPATH=. pytest -q` passed: 297 tests, with one upstream `fontTools`
  deprecation warning.
- Frozen evaluation remained unopened, no frozen image was instantiated, and
  writer training remained unauthorized throughout publication.

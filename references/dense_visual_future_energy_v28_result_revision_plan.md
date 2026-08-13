# Dense Visual Future Energy V28 Result Revision Plan

Date: 2026-08-13

Status: executable result-publication plan created after the fixed development
audit and before any V28 manuscript edit

## Revision Stage

Internal major result revision. There is no reviewer response, supplement, or
submission package in this repository. The active manuscript is
`publication/ilm-image-native/ilm-image-native.tex`; the matching active PDF is
`publication/ilm-image-native/ilm-image-native.pdf`. No preserved TeX baseline
or redline exists, so source diff and compiled-PDF inspection will provide the
trace. Existing active filenames must remain stable.

## Problem

The repository now has a completed preregistered V28 development experiment,
but the public result narrative stops at V27. The fixed evidence rejects V28 as
a selected language mechanism while recording useful localization:

- full 1,024-way top-1 is `0.01416015625`, below unigram
  `0.0185546875` and bigram `0.13134765625`;
- full target log probability is `0.030369` nat better than suffix-4 and
  `0.215112` nat better than shuffled context;
- matched-suffix full-context arm accuracy is `0.49560546875`, both-correct is
  `0.0966796875`, and full-minus-shuffled arm accuracy is `-0.00390625`;
- full-minus-shuffled mean margin is `+0.0220696771`, the one passing learned
  pair-relation gate;
- frozen V16 cross-font identity over the same 1,024-way scope is
  `0.92041015625`, while the EMA semantic route reaches `0.96435546875`;
- 9 of 14 mechanism gates and 2 of 6 language gates pass;
- no frozen image was instantiated, and frozen evaluation and writer training
  remain unauthorized.

The publication must state this negative result precisely. Training-batch
metrics must not be substituted for the deterministic development audit.

## Fixed Evidence

- Audit:
  `artifacts/dense_visual_future_energy_v28_evidence/development_audit.json`
- Audit SHA-256:
  `2cb73707a01fccb5bef750690014a2729b6235941a4de33ccdfdb600e8f0fb3d`
- Final checkpoint:
  `artifacts/dense_visual_future_energy_v28_evidence/checkpoint_final.pt`
- Checkpoint SHA-256:
  `22503464cf5f5e8ed2d6adebbd6c794f6bc9b2836f978872027cb51712c7f64f`
- Protocol SHA-256:
  `b8e515d27f619033f53a04d1afd3ff8d71ba0dd68484728f2f4c1b68a7780f7f`
- Corpus SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`
- Frozen retina SHA-256:
  `90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`
- Fixed run: 10,000 BF16 updates, 17,859,142 total parameters,
  16,377,990 trainable parameters, 1.144400 GiB peak allocated CUDA memory,
  7,134.82 seconds for training plus audit on one RTX 4090.

The artifact directory is ignored by Git. The tracked result receipt, figure
generator, generated figure, protocol, and source code provide the durable
public trace.

## Allowed Files And Locations

1. Add `docs/dense-visual-future-energy-v28-result.md` as the canonical result
   receipt, including verdict, fixed system, natural audit, suffix intervention,
   interpretation, next controlled question, and reproduction commands.
2. Add
   `publication/ilm-image-native/generate_v28_result_figure.py` and its generated
   `figures/dense_visual_future_energy_v28_result.png`. The generator must read
   and validate the fixed JSON evidence and exact hashes before drawing.
3. Update the current-result section and artifact index in `README.md`.
4. Update only V28/current-state passages in
   `docs/first-imagized-language-model-goal.md`.
5. Update `publication/ilm-image-native/ilm-image-native.tex` only in the
   abstract/current evidence summary, contributions, post-V27 results,
   discussion, conclusion, and immediately related roadmap text. Add the V28
   figure and tables. Keep the active filename stable and rebuild the matching
   PDF.
6. Update this plan with execution and verification status.

## Out Of Scope

- Opening or evaluating the frozen partition.
- Training or authorizing a writer.
- Changing V28 code, protocol, thresholds, data, or checkpoint selection.
- Claiming that V28 is a useful language model, a Qwen-8B peer, a general ILM,
  or an efficiency improvement over token LMs.
- Treating training-set top-1 or pair accuracy as development evidence.
- Calling the EMA semantic improvement language understanding; it is a
  same-scope cross-font identity result.
- Adding page, 3D, motion, historical-glyph, or etymology capabilities to V28.
- Rewriting prior V1--V27 result sections except where a transition sentence
  must identify V28 as the current result.
- Editing multilingual README translations in this bounded result unit.

## Claim Change

Before: V27 is the latest natural-language test, and dense ordered visual
future prediction is proposed as the next controlled experiment.

After: V28 is the latest completed development test. It demonstrates a compact
continuous image-stream implementation with strong same-scope cross-font
identity and measurable full-versus-control log-probability/margin effects, but
fails candidate binding and frequency baselines. It is rejected; frozen and
writer stages remain sealed. V29 must target transferable context-to-future
binding rather than add dimensionality or scale.

## Figure Contract

The measured figure must show:

1. authoritative input `B x 68 x 1 x 32 x 32` and continuous future horizons
   1/2/4;
2. frozen retina, online/EMA semantic routes, causal field, and four continuous
   hypotheses without a token vocabulary;
3. natural top-1 bars for full, unigram, bigram, and trigram;
4. pair bars for full, suffix-4, and shuffled arm accuracy plus both-correct;
5. raw versus EMA same-scope identity;
6. gate counts and the explicit `REJECTED ON DEVELOPMENT` verdict;
7. sealed frozen split and unauthorized writer.

No generated decorative image may replace these measured values.

## Verification

1. Regenerate the V28 figure from the fixed audit twice and confirm stable
   SHA-256.
2. Run the figure generator against a modified/hash-mismatched report and
   confirm it rejects the evidence.
3. Run `git diff --check` and focused static checks for the generator.
4. Build `publication/ilm-image-native/ilm-image-native.tex` with the repository
   LaTeX build script.
5. Confirm the PDF page count and extract text with `pdftotext -layout`.
6. Render and visually inspect every changed PDF page, including the V28 figure,
   tables, abstract, discussion, and conclusion. Check clipping, overlap, tiny
   text, and unsupported wording.
7. Search README, goal, result receipt, TeX source, and extracted PDF for exact
   gate counts and key metrics.
8. Run the full Python test suite to ensure documentation utilities did not
   disturb executable behavior.

## Response-Letter Impact

None. There is no response letter. The canonical result receipt and this plan
serve as the traceable internal rationale.

## Execution Record

Pending.

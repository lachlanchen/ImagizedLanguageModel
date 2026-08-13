# Conditional Visual Field Flow V31 Result Revision Plan

Date: 2026-08-13

Status: approved internal evidence-publication plan; created after the fixed
V31 development audit and before manuscript edits

## Problem

The preregistered V31 experiment has completed two matched 10,000-update arms
and its joint fixed development audit. The repository contains the protocol,
implementation, checkpoints, and ignored audit artifacts, but not a tracked
result receipt, evidence-derived figure, README summary, or manuscript account.
The result must be published before replacing the failed mechanism.

## Fixed Evidence

- protocol SHA-256:
  `92b6f70975dffe25723e332268b8929fa547b9d848a296f9ed80968cf798f8f7`;
- spatial checkpoint SHA-256:
  `9808e9966b02c2f200cc91d8c63e611c9dd52692903d7b51ab5131d0ee05859f`;
- global-control checkpoint SHA-256:
  `485011a4db854626b468bf7dba93962ce0ed22a98020282aa19c4ba06d999b7e`;
- development audit SHA-256:
  `530c21d3d0f14e67e6616780a63d061c02f0a3ca22cf78849c914a65b8630985`;
- comparison receipt SHA-256:
  `2e97b7324c6dcbbf004737b3d5b6d832145b90791ce1310fc2bcbbbc870f1859`;
- autonomous nearest-image sheet SHA-256:
  `d57656cbb66c33a8f6c3f07b8eecdee7a460d5d675836d0a1eeb645d1b8384d0`;
- decision: rejected on development;
- frozen evaluation authorized: false; and
- writer training authorized: false.

## Allowed Files And Locations

1. Add `docs/conditional-visual-field-flow-v31-result.md` as the tracked result
   and reproduction receipt.
2. Add `publication/ilm-image-native/generate_v31_result_figure.py` and its
   generated result and autonomous-nearest-image PNGs under
   `publication/ilm-image-native/figures/`.
3. Update the upper research summary, latest-result section, and key-link table
   in `README.md`.
4. Update `publication/ilm-image-native/ilm-image-native.tex` in the abstract,
   contribution list, natural-language experiment sequence, discussion, and
   conclusion so V31 is reported from measured evidence.
5. Update this plan with build and verification status after execution.

## Required Claims

- Both parameter-identical arms began from byte-identical initialized states
  and completed exactly 10,000 finite BF16 updates.
- V31 trains a conditional continuous field flow with coherent base noise and
  an eight-probe path score; generation uses eight-step Heun integration.
- Spatial natural path top-1 is `0.0977%`, below the global control (`1.8555%`),
  image unigram (`1.6113%`), symbolic bigram (`13.5254%`), and symbolic trigram
  (`20.9961%`).
- Spatial exact-suffix path assignment is `50.4883%`, below the global control
  (`51.4648%`) and effectively chance. Spatial autonomous assignment is
  `50.1953%`.
- The spatial arm passes `14/19` common gates, the control passes `6/6`
  integrity gates, matched arms pass `4/8` gates, and spatial language and
  generation pass `0/10` gates.
- The spatial route detects order in mean path log probability and reacts to
  spatial permutation, but those sensitivities do not identify the next glyph.
- V31 is rejected; frozen images remain uninstantiated; frozen evaluation and
  writer training remain unauthorized.
- Peak allocated memory near `1.00 GiB` is only a resource measurement for a
  rejected mechanism, not capability-normalized efficiency.
- The autonomous panel contains evaluator-nearest rendered glyphs to generated
  latent retinal fields. It is not direct pixel output and must be labelled as
  a diagnostic proxy.

## Forbidden Claims And Edits

- Do not call V31 a usable language model, general ILM, glyph generator, or
  successful proof of image-native language.
- Do not claim parity with Qwen, GPT, Llama, OCR systems, diffusion image
  generators, or token language models.
- Do not describe nearest evaluator glyphs as generated pixels.
- Do not open or report frozen-partition images.
- Do not authorize or train a writer under the V31 protocol.
- Do not alter the V31 protocol, audit artifacts, thresholds, model, trainer,
  evaluator, or prior result receipts.
- Do not modify unrelated untracked `.auto-readme-work/` content.

## Figure Requirements

The deterministic figure generator must refuse changed evidence by checking
the exact audit, protocol, checkpoint, comparison, and sample-sheet hashes. It
must show:

- the 64-cell visual context, QKV causal reader, conditional flow field, and
  fixed path/sample audit;
- natural path and sample top-1 against image unigram and symbolic bigram and
  trigram baselines;
- exact-suffix pair assignment and order/spatial interventions;
- matched-arm differences, gate counts, and resource receipts;
- a clearly bounded excerpt of the autonomous-nearest diagnostic; and
- the negative decision and the resulting direct-raster-block requirement.

## Verification

1. Run the figure generator against the fixed local audit and regenerate it
   byte-for-byte.
2. Run focused V31 tests and formatting/lint checks.
3. Build `publication/ilm-image-native/ilm-image-native.pdf` using the existing
   repository build workflow.
4. Render and inspect the README/result figure and the relevant compiled PDF
   pages for clipping, alignment, captions, and proxy labelling.
5. Confirm `git diff --check` and that `.auto-readme-work/` remains untouched.
6. Commit and push the bounded publication unit.

## Response-Letter Impact

None. This is an internal research manuscript, not a reviewer-response round.

## Execution Record

Completed on 2026-08-13.

### Evidence Publication

- Added the tracked V31 result receipt, deterministic evidence figure, and
  exact copy of the autonomous nearest-image diagnostic.
- The figure generator validates the fixed report, comparison, sample-sheet,
  protocol, and checkpoint SHA-256 values before drawing.
- The result figure regenerates byte-for-byte with SHA-256
  `cc24d56f20a56be44fbba5ee4f20e8da30b0665ac0a9aac0d507a9ee37a068e6`.
- The publication copy of the nearest-image diagnostic retains SHA-256
  `d57656cbb66c33a8f6c3f07b8eecdee7a460d5d675836d0a1eeb645d1b8384d0`.
- README and manuscript text consistently report rejection, `0/10` language
  and generation gates, sealed frozen images, and the absence of direct pixel
  output.

### Manuscript Revision

- Added the V31 conditional-flow equations, path score, autonomous ODE,
  matched resource receipts, natural and exact-suffix tables, gate decision,
  result figure, and standalone proxy diagnostic.
- Updated the abstract, contribution list, prior-art discussion, broader
  discussion, and conclusion. The next bounded test now separates causal QKV
  visual semantic planning from direct multi-glyph raster rendering.
- Added primary-source citations for Orthus, FlowAR, and UniDDT as architectural
  precedents while preserving the no-token student boundary.
- Corrected a caption count during PDF inspection: spatial path top-1 is
  `2/2,048`, and autonomous nearest-bank top-1 is `3/2,048`.

### Verification

- `PYTHONPATH=. pytest -q` on the four focused V31 test files: `24 passed` with
  one external `fontTools` deprecation warning. An initial bare `pytest`
  invocation omitted the repository's required `PYTHONPATH=.` and failed only
  at import collection; the corrected documented invocation passed.
- `ruff check`, `ruff format --check`, and `python -m py_compile` pass for the
  V31 figure generator.
- The figure generator was run twice and produced identical hashes.
- `latexmk` completed and produced a 66-page PDF with SHA-256
  `dcfb2dd0b98b6e4d6aaa2390f145e40817ea30def8627105428ef0129964af63`.
- The final log contains no LaTeX errors, undefined references or citations,
  or overfull boxes. Existing underfull table/reference warnings and four
  float-specifier normalizations remain outside this bounded revision.
- PDF pages 44--49 were rendered at 150 DPI and inspected. Equations and tables
  are readable; the measured figure is aligned; the autonomous diagnostic is
  legible; and both captions distinguish nearest evaluator glyphs from direct
  generated pixels.
- `git diff --check` passed, and unrelated untracked `.auto-readme-work/`
  content was not modified or staged.

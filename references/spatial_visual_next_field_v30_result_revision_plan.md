# Spatial Visual Next-Field V30 Result Revision Plan

Date: 2026-08-13

Status: approved internal evidence-publication plan; created after the fixed
V30 development audit and before manuscript edits

## Problem

The preregistered V30 experiment has completed both 8,000-update arms and its
joint development audit. The repository currently contains the protocol,
implementation, checkpoints, and ignored audit artifacts, but not a tracked
result receipt, evidence-derived figure, README summary, or manuscript account.

## Fixed Evidence

- protocol SHA-256:
  `81d2b2af1eb3a305b4acd1028c004ddddc607e826eea1d50b6d137d32ed180a5`;
- spatial checkpoint SHA-256:
  `11a3a7e9f13f1db932dcc913e1c79b0e0db49b95bd49f98a878897028bf86130`;
- global-control checkpoint SHA-256:
  `66378d4b972702490f6819d87d95c2576546e15e6fc74d10542307aaf4483411`;
- development audit SHA-256:
  `2d0a3a08e2f5d4b267e276b695448cc8311687a822776a33c0a883dc0a74fd8f`;
- comparison receipt SHA-256:
  `98865b1d25f80079b998dd9c07bdfb7ada0b5701d2814a2604d4911a89eb0419`;
- decision: rejected on development;
- frozen evaluation authorized: false; and
- writer training authorized: false.

## Allowed Files And Locations

1. Add `docs/spatial-visual-next-field-v30-result.md` as the complete tracked
   result and reproduction receipt.
2. Add
   `publication/ilm-image-native/generate_v30_result_figure.py` and its
   generated PNG under `publication/ilm-image-native/figures/`.
3. Update the upper research summary, latest-result section, and key-link table
   in `README.md`.
4. Update `publication/ilm-image-native/ilm-image-native.tex` in the abstract,
   contribution list if needed, the natural-language experiment sequence, and
   conclusion so V30 is reported from measured evidence.
5. Update this plan with build and verification status after execution.

## Required Claims

- Both parameter-identical arms began from byte-identical initialized states
  and completed exactly 8,000 finite BF16 updates.
- The spatial route predicts a candidate-independent `4 x 4 x 192` field and
  is causally sensitive to patch alignment, but that sensitivity is too weak
  to support conditional visual language.
- Spatial natural top-1 is `1.2695%`, below the global control (`2.4902%`),
  image unigram (`1.3184%`), and symbolic bigram (`11.7188%`).
- Spatial exact-suffix assignment is `50.0488%`, below the global control
  (`50.5859%`) and effectively chance.
- The spatial arm passes `12/18` common gates, the control passes `12/12`
  integrity gates, matched arms pass `5/9` gates, and spatial language passes
  `0/8` gates.
- V30 is rejected; frozen images remain uninstantiated; frozen evaluation and
  writer training remain unauthorized.
- Peak allocated memory near `1.60 GiB` is only a resource measurement for a
  rejected model, not evidence of capability-normalized efficiency.

## Forbidden Claims And Edits

- Do not call V30 a usable language model, a general ILM, or a success.
- Do not claim parity with Qwen, GPT, Llama, OCR systems, or token language
  models.
- Do not claim that spatial permutation sensitivity proves language.
- Do not open or report frozen-partition images.
- Do not authorize or train a writer.
- Do not alter the V30 protocol, audit artifacts, thresholds, model, trainer,
  evaluator, or prior result receipts.
- Do not add page folding, a third geometric axis, character movies, or
  historical-glyph answer generation as measured V30 capabilities.
- Do not modify unrelated untracked `.auto-readme-work/` content.

## Figure Requirements

The figure generator must refuse changed evidence by checking the exact audit,
protocol, and checkpoint hashes. It should show:

- the shared image stream and candidate-independent field predictor;
- aligned local scoring versus the repeated-global control;
- natural top-1 against suffix, shuffle, unigram, and symbolic controls;
- exact-suffix pair assignment and patch-permutation interventions;
- matched-arm differences and gate counts; and
- the negative decision with frozen and writer stages closed.

## Verification

1. Run the figure generator against the fixed local audit.
2. Run focused V30 tests and formatting/lint checks.
3. Build `publication/ilm-image-native/ilm-image-native.pdf` using the existing
   repository build script.
4. Inspect the generated figure dimensions and the compiled PDF text/pages;
   render relevant PDF pages when necessary to verify clipping and placement.
5. Confirm `git diff --check` and that `.auto-readme-work/` remains untouched.
6. Commit and push the bounded publication unit.

## Response-Letter Impact

None. This is an internal research manuscript, not a reviewer-response round.

## Execution Record

Pending.

# V35 Publication Revision Plan

Date planned: 2026-08-14

Revision stage: internal major scientific update after the preregistered V35
development decision

Baseline: Git commit `9c0d6f3` on `main`. The active manuscript is
`publication/ilm-image-native/ilm-image-native.tex`; Git history is the
immutable pre-V35 baseline because this repository does not maintain a separate
named baseline TeX file.

## Problem

The active manuscript and README stop at the rejected V31 experiment. They do
not describe the qualified V34 continuous glyph codec, the V35 causal
pixel-to-pixel student, its external PIXAR initialization, the fixed prompt
controls, or the measured V35 decision. The prior-art section also predates
MIXAR and must not imply that this project introduced pixel language modeling.

## Scientific Claim Discipline

The revision may claim only what the immutable V34 and V35 reports establish.

- V34 may be described as a qualified codebook-free `32 x 32` writing codec.
- V35 may be described using exactly one frozen status:
  `not-qualified`, `visual-causal-qualified`, or
  `semantic-raster-qualified`.
- A visual-causal result is not semantic language understanding.
- A failed development or sealed gate must be reported as a negative result.
- PIXEL, PIXAR, MIXAR, PixelGPT, and other relevant published precedents must be
  credited. The paper must not claim the first pixel language model.
- The potentially novel scoped contribution is the evaluated combination of a
  continuous non-quantized writing latent, a closed
  decode-threshold-reencode raster loop, historical-writing motivation, and a
  one-RTX-4090 training receipt. It is a project contribution, not a priority
  claim unless a complete literature audit supports that wording.
- No GPT, Gemini, Llama, or Qwen parity claim is allowed. No efficiency claim
  against token LLMs is allowed without a matched benchmark.
- Offline public foundations and evaluator tools must be named. The deployed
  student boundary must separately state what is absent at runtime.

## Allowed Files

- `publication/ilm-image-native/ilm-image-native.tex`
- `publication/ilm-image-native/generate_v35_result_figure.py`
- `publication/ilm-image-native/generate_v35_pipeline_figures.py`
- `publication/ilm-image-native/figures/causal_glyph_flow_v35_result.png`
- `publication/ilm-image-native/figures/causal_glyph_flow_v35_training.png`
- `publication/ilm-image-native/figures/causal_glyph_flow_v35_inference.png`
- `references/causal_glyph_flow_v35_result.md`

README, goal-document, multilingual-summary, and inference-documentation work
is a separate documentation unit and is not authorized by this manuscript
plan.

## Planned Manuscript Changes

1. Replace the oversized historical abstract with a concise abstract that
   states the problem, V34/V35 method, one-GPU receipt, measured decision, and
   limits.
2. Clarify the introduction's pain point: Unicode and tokens are useful codecs
   but incomplete interfaces to historical, handwritten, regional, damaged,
   or unencoded writing.
3. Update contributions so the list emphasizes falsifiable interfaces,
   preregistered controls, V34, and V35 instead of presenting every historical
   experiment as a coequal contribution.
4. Update prior art with primary-source descriptions of PIXEL, PIXAR, MIXAR,
   PixelGPT, and continuous image-generation precedents. Distinguish
   image-native language from OCR-free document understanding and from models
   that still use discrete text tokens.
5. Add a V34 method/result subsection with parameter count, data boundaries,
   development and sealed codec metrics, compute, and the explicit statement
   that reconstruction is not language evidence.
6. Add a V35 method subsection covering raster patches, V34 codec, PIXAR causal
   initialization, residual adapter, causal field, anchor and rectified-flow
   writers, visible feedback, objectives, data mixtures, and runtime boundary.
7. Add a V35 result subsection only after the development report is immutable.
   Include raw versus EMA diagnostics, anchor-versus-flow selection, correct,
   shuffled, blank, and final-quarter prompts, copy counterfactuals, public
   continuation, instruction and paraphrase results, training time, peak VRAM,
   and sealed transfer only if the protocol permits opening it.
8. Add one deterministic measured figure generated directly from the training
   metrics, report JSON, and autonomous PNG galleries. Concept illustrations
   remain labeled as concepts.
9. Add deterministic V35 training and inference diagrams that match the
   implemented codec, adapter, causal field, anchor/flow writer, stop head, and
   visible feedback loop. These diagrams describe architecture and boundary;
   they must not depict the word-origin target as a measured capability.
10. Revise the conclusion to state the strongest supported result and the next
   bounded experiment. Keep page, 3D, movie, speech, and broad word-origin
   answering as future work unless directly measured.

## Figure Requirements

The V35 result figure must be generated by repository code, not manually
painted or synthesized as evidence. It should contain:

- Stage B/C training curves with stage boundaries;
- teacher-forced visual metrics;
- autonomous OCR/readability under correct and control prompts;
- copy counterfactual target preference;
- representative target and generated raster strips copied from the immutable
  evaluator galleries; and
- explicit labels for development versus sealed evidence and selected writer.

Every displayed number must be recoverable from report JSON. The figure must
remain legible at one-column width, use a restrained multi-hue palette, avoid
decorative gradients, and never substitute OCR text for generated pixels.

The two V35 pipeline figures must be generated from repository code and label
the public PIXAR initialization, frozen V34 codec, and evaluator-only OCR
explicitly. The training figure distinguishes offline alignment from causal
student training. The inference figure accepts a rendered prompt strip or
writing image and returns generated raster patches through the actual
decode-threshold-reencode loop. Existing concept images remain in the repository
but are not reused as evidence.

## Out Of Scope

- changing V35 protocol thresholds, training data, splits, or model weights;
- opening sealed records before development qualification;
- adding unmeasured historical-glyph question answering;
- rewriting unrelated V6-V31 result sections;
- claiming biological equivalence to human reading; and
- producing a response letter or journal submission package.

## Verification

1. Confirm the final V35 checkpoint, development report, optional sealed report,
   and standalone checkpoint hashes.
2. Regenerate the V35 figure from the immutable reports and compare every
   plotted value to JSON.
3. Build with `cd publication && make ilm-image-native/ilm-image-native.pdf`.
4. Inspect the PDF visually and run `pdftotext -layout` to verify the abstract,
   V34/V35 sections, figure caption, conclusion, and reference rendering.
5. Search source and extracted PDF for unsupported phrases including
   `first pixel`, `on par`, `GPT parity`, `Qwen parity`, and unqualified uses of
   `understands`.
6. Record build status, PDF page count, unresolved warnings, and any deviation
   from this plan in the final V35 result note.

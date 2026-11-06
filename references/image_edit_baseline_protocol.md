# Image-conditioned generator: bounded baseline protocol

Specified before weight acquisition or model execution, 7 November 2026.
This is a candidate experiment after the current matched V82 comparison,
not a new claim that language generation works.

Use the Apache-2.0 FLUX.2-klein-4B image generator at source revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. Retain source attribution and
measure the actual transformer, VAE and runtime-condition sizes. The source
model card warns about distorted writing and factual unreliability. Its
pretraining is external; this baseline performs no local optimization.

The stages are separate. CPU inspection resolves the pinned model's explicit
configuration/weight allowlist without downloading weights or initializing a
model. CPU data preparation renders the fixed exploratory questions and
references as separate pages with a verified CJK font face, 32-pixel type,
32-pixel page margins and 48-pixel line pitch. For QA, extract display text
from the source context/question fields, omit HTML formatting tags and put
the question on a separate labelled line. Keep factual wording, including
source imperfections, and all reference answers unchanged. Confirm each QA
answer remains in its displayed passage. Preserve every display-text character
and keep Latin/Vietnamese word runs and numbers together across line breaks
when they fit the declared width. Do not place closing punctuation alone at
the start of a line.
and check actual ink bounds; fail on missing glyphs or clipping. Question
pages are 1024 by 1024 and reference answer pages are 512 by 512. Data layout
preparation does not run a model. Acquisition and all model stages are gated on a completed, verified
V82 comparison in which neither arm meets the existing absolute fit criterion
of 24/32 autonomous exact answers and foreground F1 at least 0.98. If either
arm fits, evaluate its generalization first instead of starting this fallback.
The GPU lock and normal workstation resource checks still apply. Do not run
a second heavy model job, replace the current trainer or alter its sources.

Offline condition preparation reuses the existing shared Z-Image installation's
Qwen3-4B text encoder and tokenizer. Verify its three declared weight hashes;
do not copy the model or assume that it is byte-identical to the separately
sharded text encoder in the FLUX release. Cache the same generic instruction
and the empty negative instruction once, using the installed Klein pipeline's
native hidden-state extraction at layers 9, 18 and 27. Record source weights,
configuration/tokenizer hashes, framework versions, tensor hashes and cost.
This fixed-condition composition is not an exact reproduction of the stock
FLUX text-prompt baseline. Its effects remain an empirical limitation.

The generic condition is: “Read and follow the instruction written in the
reference image. Produce only the requested answer as clearly legible black
writing on a plain white page. Match the language of the instruction. Preserve
the exact visual form when copying a glyph is requested.” All question content
and varying instructions must be in the reference image. No per-question
caption, token sequence, embedding, transcript or reference answer enters
generation. Keep the generic numeric condition identical across every case.

The fresh generator process loads only the pinned transformer/VAE and fixed
condition. Construct the real source pipeline with tokenizer and text encoder
absent, set offline loading, and verify the actual module tree. Use the source
image branch, scheduler and VAE decoding; retain raw output latents, raw RGB
tensors and unrepaired PNGs. Do not select a font, snap glyphs to an inventory,
OCR-replace text or call a teacher. Source integer spatial positions remain
image coordinates, not Unicode or language-vocabulary IDs.

Before language evaluation, run actual VAE encode/decode on every chosen input
and reference page at its declared geometry. Save original and decoded pixels.
Report thresholded foreground F1 as a high-contrast writing diagnostic, with
raw RGB preserved. This is supplied-image reconstruction, not a generated
answer or proof of language. Poor stroke preservation blocks training at that
geometry until the representation problem is resolved.

The initial language check uses all twelve existing literal-copy requests
and the first four paired passages (eight questions) in the original fixed
32-question cohort, in their existing order. This is 20 normal prompts,
with a matched blank-image control for each. It is exploratory reused-data
evaluation; no sealed test is opened. Share one seed within each original
two-question pair and both of its blank controls. Use
four source-distilled sampling steps, guidance one, and a fixed 512-by-512
answer canvas. Source pages must have dimensions divisible by 16, area at
most 1,048,576 pixels and no hidden resize or answer-aware crop. Retain all
40 outputs, including copied prompts, blank pages, distortions and mistakes.

Readability and answer correctness must be assessed separately from image
overlap because the generator can choose a different writing layout. Any OCR
or vision evaluator is offline, sees only the produced answer image and must
first be checked against the reference pages. Keep unparsed and ambiguous
outputs visible. A selected readable example, source codec success, question
dependence or failed blank control cannot qualify useful language learning.

This protocol currently permits preparing the runner, inspecting real source
metadata and exercising unfinished-experiment rejection. Successful checkpoint
loading, encoder-free model execution, codec fidelity and all 40 baseline
answers remain unverified until actually run. Training a learned ILM and its
broader language/generalization requirements remain the active objective.

Pre-model data review identifies a literal orphan `</div>` in the first
passage. The first source-faithful rendering is retained. Display-text
extraction is specified before any candidate model call and uses a fresh
preparation directory; no reference, question pair or outcome is selected anew.
Source-image review also identifies word/number splits under character-only
wrapping. Word-aware wrapping is fixed before any model call, with source
characters and targets unchanged. Preserve earlier preparation directories.

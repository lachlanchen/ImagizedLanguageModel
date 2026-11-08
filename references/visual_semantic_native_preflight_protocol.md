# Registered visual-semantic native preflight

This preflight tests a concrete pretrained image-to-image path that carries
page features through a visual-language body before image generation. It
asks whether the packaged native model can be run locally, whether constant
task compilation preserves its output, and whether unused lexical components
can be removed without changing that output. No weights are trained in this
preflight. No language-learning or single-GPU fit result is assumed.

## Source and execution

Use the complete DeepGen release at
`85c2ed257b9406d8531965c68e140dc7fdb81382`, including its bundled VLM,
connector, VAE and transformer. Verify all 23 source files, including the
five complete weight files. Reuse the project cache; do not substitute the
separate base Qwen release or duplicate model weights.

Construct the actual released component configurations on the meta device
first. Every stored component key and shape must match, allowing only the
VLM's unused output-head key to be absent. Explicitly check the legacy-to-current
VLM key names. Actual loading must subsequently report no missing semantic or
generative weights. An initialized unused head, if present, is recorded and
removed; it must never be called by generation.

The Python compatibility changes are explicit: property aliases for the
current nested VLM bodies, the checksum-pinned Transformers 4.56.1 rotary
position method, and SDPA attention. Use the source's native pipeline
constructor, transformer upgrade, image preparation, semantic feature path,
connector, sampling and pixel conversion. The first test uses the standard
all-on-GPU path, with no CPU-offload variation. Numerical replay through these
changes remains necessary; metadata inspection is insufficient.

Preserve the live FLUX QA coordinator and its entire 128-update/320-new-call
experiment. The new runner may wait for that exact owner, identified by PID,
start ticks and command. It may load the candidate only after the coordinator
exits with a complete assessment. Acquire the same exclusive project GPU0
lock and workstation resource checks. GPU1 belongs to another project. A
memory or compatibility failure is retained, with no automatic smaller-image
fallback or output-quality retry.

## Fixed input and sampler

Prepare an input-only JSON index with exactly `id`, `input`, `input_sha256`
and `seed`. The generator never loads a QA source, target image, accepted
answer, OCR model or question-specific text condition. The offline preparer
has access to the existing manifest; that distinction is disclosed.

Use the existing four-case full-page preflight QA index, both held-pair
question-erased pages and both held-pair passage-erased pages. Add the first
Chinese and first English validation copying pages from `pages_v2`. This is
ten complete 1024 × 1024 input pages. Preserve the eight QA/control seeds;
assign the two copy seeds by the fixed expression in the preparer. Do not
choose cases based on candidate outputs. These reused public development
cases do not constitute a new blinded generalization test.

Every generation uses the same generic task as the current writer, empty
negative task text, one reference page, 1024 × 1024 output, 50 native sampler
steps and guidance 4. All changing instructions and passages enter as pixels.
No page is cropped or resized to an answer strip. The native wrapper's equal
input/output geometry is retained. Seed the global Python, NumPy, CPU Torch
and selected CUDA RNG as well as the native denoiser generator: the source
reference-posterior sampling uses the global Torch RNG. Disable TF32 and
request deterministic algorithms.

## Replay and removal

Generate the first registered QA page through the native task preparation.
Keep its real decoded tensor, its native PNG and its exact pre-transformer
input fields. Install the compiled fixed task and replay that same image and
seed without changing weights. All pre-transformer fields, decoded pixels
and PNG pixels must be bit-identical. Stop and preserve both calls if they
differ.

Then remove the VLM word embedding table, lexical output head and tokenizer.
Retain its pretrained visual and language transformer bodies and the fixed
compiled task vectors. Replay the same page and seed once more. Require the
same exact fields and actual image output. Record parameter counts before
and after removal, whether the original table/head shared a parameter, and
the actual peak GPU allocation. Do not infer these measurements from the hub
summary or meta prototype.

If both replays pass, retain the vocabulary-removed first output as baseline
case zero and generate the remaining nine pages once. Thus there are twelve
actual generation calls: native, compiled and vocabulary-removed case zero,
then nine more vocabulary-removed cases. Authenticate every PNG against its
real decoder tensor using the source's own pixel-conversion formula. Save
all twelve outputs and their tensors; do not repair, crop, rerender or select
writing. The identical question-erased pair must also yield identical actual
decoded tensors.

## Interpretation

Complete execution would demonstrate local inference feasibility and
bounded native equivalence on these pages. Only inspection and independent
scoring of every original answer page can establish what the model actually
answered. This preflight has no success threshold that qualifies a useful
language model: ten reused development examples cannot support that claim.
Retained language-model pretraining is counted and disclosed. Subsequent
paired instruction training, broader article-disjoint assessment, reliable
writing, efficiency comparisons and historical-form generation remain
required for the project goal. Keep the manuscript's capability claims
unchanged until actual complete results justify a revision.

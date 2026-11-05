# V71: check visual capability through the imported image interface

Stage: registered diagnostic direction after V70, before implementation or
measurement. This is a bridge to repairing the independent image-language
actor, not a replacement for its pixel-output language requirement.

## Why this is the next step

V70's authenticated native source answers 13/16 Chinese image questions
exactly, while the independently trained pixel-output actor still fails.
The native source reads at least 1,024 merged visual positions per image.
The custom actor reads the first page at 224 positions and uses learned
queries in place of the native task/chat structure. Both differences require
measurement before selecting another output density or long adaptation.

V70's prompt-count reconstruction matches all 24 cases, but does not establish
numerical pixel or feature equivalence. The local native source implements
area-constrained, factor-32 image resizing and selects bilinear interpolation
for Qwen-VL. Source warnings about minimum resolution concern grounding tasks;
they do not prove a universal reading threshold or explain the ILM failure.

## Bounded implementation and test

Use the authenticated original source archive and the complete retained
Python vision/contextual body, initially with adapter outputs at zero. Do not
load the V68 adaptation in this first import check. Use the first eight Chinese
V70 cases, four article pairs, selected by original order rather than score.
These remain diagnostic development cases and cannot become training labels.

Implement a separate image-only contextual wrapper, without changing frozen
V68/V69 producer code. Its structural prefix/suffix are numeric vectors
compiled offline from fixed native chat/task controls. The wrapper consumes
physical images, inserts actual vision/DeepStack states, assigns sequential
causal positions and three-dimensional rotary coordinates, and returns the
context/cache. It has no tokenizer, character IDs, reference strings,
vocabulary embedding lookup, OCR or network client at inference. Attribute
the source-derived control-vector initialization explicitly.

For diagnostic reading only, a separate helper may temporarily use the source
tokenizer, source token rows and source output projection, following the
existing text-only compatibility check. Never attach those to the image
wrapper or an exported ILM. Do not save a deployed token decoder. Retain the
assertion that the contextual body's vocabulary module is absent. This helper
tests imported source capability, not final independent image generation.

Compare normal and original matched-layout blank pages under two pixel
presentations: original raw resolution, and the V70 native-budget geometry.
Use a declared bilinear implementation, preserve original physical data and
record transformed pixels, dimensions and hashes. Geometry matching alone
must not be described as exact native pixel-preprocessor reproduction.
Also check the eight matched source-text prompts inside the helper, so a
failure in temporary decoding or source initialization can be separated from
visual preparation. This yields 40 diagnostic generations: eight cases,
two image conditions, two resolutions, plus eight text controls.

Use greedy decoding with a uniform 64-source-token cap and authenticated
source end-of-generation IDs, all confined to the helper. This free-text
check differs from V70's native JSON-constrained responses and is not a claim
of full numerical engine equivalence. Preserve every response, score literal
reference agreement and record truncation. Do not repair output strings.

Before execution, freeze the exact implementation, numerical settings, data
identities and control-vector compiler; commit and push them. Load only one
complete model on physical GPU 0, measure peak allocation, and release it after
measurement. Reuse shared weights. Keep other projects' services untouched.

## Decision and subsequent learning

If the imported wrapped reader retains useful visual answers, train the
continuous writing path from those verified conditions and retest complete
autonomous answers. The source helper and vocabulary must remain absent from
the final actor. If only the higher-resolution route works, register a matched
reader-scale intervention with the same writer, data and budget. If text works
but both visual routes fail, audit visual weight conversion, positional
encoding and preprocessing before another language-training run. Failure on
text controls requires fixing source/interface compatibility first.

Do not apply a 1,024-position minimum independently to every generated 32-by-64
block in a future actor without measuring its cost. A prompt-scale observer
or another bounded reading policy would need its own causal and memory checks.
Upsampling adds no information to the original image and guarantees no
language improvement. The full completion requirement remains an independent,
useful image-input/image-output language model on one consumer GPU.

## Implementation freeze for the first execution

Use seed 20267100, four CPU threads, deterministic algorithms, no TF32,
deterministic cuDNN and the existing FP32 eager/math attention path. Preserve
the authenticated source model, BF16 contextual storage and FP32 visual
weights. Remove only the unused writer from this diagnostic instance. No
optimizer, V68 adaptation or source-model service is used.

Compile the fixed controls from authenticated source embedding rows:
`<|im_start|>user\n<|vision_start|>` before the actual visual features, and
`<|vision_end|>` followed by the V70 task instruction and
`<|im_end|>\n<|im_start|>assistant\n` afterward. The structural-state artifact
contains only the two FP32 vector arrays. Symbolic compiler metadata is a
separate offline artifact. For text controls use the same task plus two
newlines and the original source prompt inside the native user/assistant
frame. Compile exact source special IDs outside the image wrapper.

Use Pillow RGB bilinear resizing to the V70 reconstructed native dimensions;
record its version and preserve both source and transformed PNGs. This is a
declared geometry-matched presentation, not an assertion of exact C++ pixel
equivalence. Keep all 40 fixed generations and a 64-source-token cap. The
temporary source projection is stored in BF16 and contracted in FP32, as in
the earlier text compatibility check. Save every selected source ID, its
contextual hidden state and projection logit for CPU verification.

Report literal trimmed-reference exact agreement, bounded character edit
accuracy, truncation, and reference inclusion. Also report reference inclusion
without the paired question's reference, explicitly as a string-based
selectivity diagnostic rather than semantic correctness. This secondary
measure cannot qualify an image-language actor or credit full-page copying
as correct answering. Inspect all raw responses.

Independent CPU checks reconstruct all source/transformed pixels, compile the
numeric controls again, decode the retained helper IDs, and recompute the
output projection from retained hidden states. Report exact argmax agreement
and numerical ties separately; the retained selected logit must be within
1e-3 of its CPU replay and within 1e-4 of the CPU maximum. These tolerances
concern floating-point projection replay, not answer quality. The verifier
does not independently replay the whole vision/contextual body.

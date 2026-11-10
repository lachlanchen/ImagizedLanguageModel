# Training language continuation through the semantic glyph interface

The [glyph primitive protocol](semantic_glyph_loop_protocol.md) remains
frozen. Stage A's complete checkpoint must pass mean foreground F1 above
0.999 and minimum per-glyph F1 above 0.95 on both direct decoding and image
roundtrip. This is a rendering gate, not a language test. Its inherited input
anchors are training-only supervision; the runtime has no character table.

Stage B uses the 512 original training questions. A source character outside
the prepared anchor coverage makes that training case unsupported; record
it explicitly rather than substitute a character. Do not filter evaluation
cases. Use source strings only to prepare gold answer glyph sequences. All
development answers, helper outputs and hidden states remain excluded from
training. The initial experiment uses 508 supported questions, excluding four
training strings containing 褔, 砲 or 臙. This is not 512-case coverage.

Freeze the pretrained visual reader and contextual body, plus the primitive
encoder and decoder. Read each original training prompt image once. Cache
the unrounded FP32 final native assistant state and states after encoding
gold previous glyph **images**, using the primitive's continuous encoder.
The state before a target glyph must not see that glyph. Append the terminal
state as a stopping target. Cache images, bridge provenance, state counts and
hashes. On the first sequence, compare the first three bulk causal states
against one-glyph-at-a-time execution, requiring maximum error below 0.002.
This is a numerical causal check, not evidence of generated-answer quality.

Fit only G and stopping parameters for 6,000 batch-128 AdamW updates, learning
rate 2e-4 with cosine decay to one tenth and seed 202611102. Minimize semantic
anchor MSE, cosine discrepancy (weight 0.2), foreground/background-balanced
pixel BCE through the frozen decoder, and unweighted stop BCE. All glyph
events and terminal events are sampled uniformly from cached states. This
is a supervised reconstruction/continuation objective, not a normalized
multimodal language likelihood. Conditional mean regression may still blur
uncertain predictions: inspect it rather than assume the repair succeeds.
Use FP32 parameters and BF16 small-network autocast. A finished fit is
explicitly labelled as not yet demonstrating usable language.

Autonomous assessment uses eight fixed training controls and all sixteen
Chinese development questions with both normal and erased images. Preserve
all forty raw tensors and PNGs, including failures. Generate up to 64 glyphs
with a learned stop threshold 0.5; reference length never enters generation.
Each actual binary output glyph is encoded before the next contextual step.
The native pretrained body is present, but its lexical input/output tables,
tokenizer and source text decoder are absent. The canonical font is used only
in preparation, never as an answer renderer. A software flag allowing a
completed fit to run is not a scientific success criterion. Inspect complete
unseen answers and stopping before making any language capability claim.

Cache and assessment are separate frozen-model GPU0 jobs, with the exclusive
project lock and the diagnostic's 64 GiB host-entry, 24 GiB runtime RAM and
low-pressure admission. Cap allocated VRAM at 22 GiB. Each native contextual
forward during autonomous generation checks reserves. High pressure pauses
computation with at most 300 seconds total waiting per stage. Cache is capped
at 3,600 seconds, fitting at 1,200 seconds, and assessment at 1,200 seconds.
Fitting loads only the small bridge and uses its separate training admission.
Preserve failed stages, their outputs and actual update history. No original
completed training source is modified and no foreign job is stopped.

This is an externally pretrained 7.5B-body interface experiment. It neither
trains an equally capable small foundation from scratch nor establishes an
efficiency advantage over text decoding. Historical scripts, unseen glyph
identities and open-ended English remain unvalidated. Success requires a
clear improvement in autonomous unseen image-language answers, not merely a
lower teacher-forced training loss or sharp familiar glyphs.

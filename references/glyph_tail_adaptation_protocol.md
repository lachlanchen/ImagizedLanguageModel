# Train the sequence dynamics at glyph granularity

The completed native-step comparison gives 12/16 literal development matches
with native source units and 3/16 with exact character vectors. Better isolated
image copying also fails to improve complete answers. The next controlled
candidate therefore adapts the final four contextual attention layers jointly
with the existing glyph writer and stopping head. It does not add a new motor,
lexical classifier, vocabulary table or runtime teacher.

The inherited body has 32 layers. Layers 0–27, vision, numeric task controls,
the whole-glyph encoder E, decoder D and affine normalization stay frozen.
Only rank-8 q/v/o adapters in layers 28–31, G and stop train. The existing
zero-initialized adapter convention is retained. Start from the qualified
pixel-only bridge; do not import diagnostic trajectories or C3 labels.

Training expands from 508 questions to all available CMRC train questions in
the existing authenticated complete-writing cache. Record source cases missing
from that older cache and answers outside the existing glyph primitive, rather
than silently substituting glyphs. Each reference prepares a complete target
image sequence. The prompt API receives prompt pixels only. During teacher
forcing, a target image enters only after the state predicting that image.

Cache the frozen layer-28 input for each full prompt-plus-gold-image sequence.
The prefix cache includes positional coordinates and target positions, not a
precomputed prediction from the changing final layers. Final-layer self-attention
still processes the complete causal prefix during fitting. This is exact
factorization of a frozen prefix, not detached cached final states for a body
that is supposedly being trained. Store FP32 states without lossy quantization.

Before the long cache/fit, prepare four cache preflight cases. Require zero-adapter
tail outputs to match the full body below 0.002 maximum absolute error. Change
the last supplied gold glyph and check that earlier predictor states remain
unchanged below that tolerance. Then release the full body and perform two real
tail-only optimizer updates. Verify finite nonzero adapter-up gradients, actual
adapter/readout changes and unchanged frozen parameters. These updates are
discarded; the main fit restarts from the fixed checkpoint and zero adapters.

The main fit is fixed at 18,000 AdamW updates, seed 202611117, with three
question-sequence updates followed by one genuine-copy rehearsal update.
Use learning rates 2e-5 for body adapters and 1e-4 for G/stop, cosine decay to
10%, weight decay 0.01 and gradient clipping at 1. Pixel loss is balanced binary
cross entropy through frozen D. Stopping loss weights the terminal event and
mean nonterminal events equally. There is no source-anchor regression loss.
Rehearsal uses the previously fixed 8,172 genuine-copy fitting states only.
Its encoder/decoder targets remain physical images. Held-wording states are
excluded. No development checkpoint selection or early stopping is allowed.

Use all original eight training controls and sixteen normal/erased development
questions for continuity, with their explicitly reused status. Reserve another
32 source questions as sixteen hash-ordered, distinct development contexts with
two questions per context. Exclude contexts in the current development cohort
and training source. Select questions by ID hash without filtering on the answer,
its glyph coverage, length or a model result. Keep reference strings separate
from generation. Earlier source-model pretraining exposure is unknown, and the
entire project history is not claimed to have been exhaustively decontaminated.
Inspect all 32 actual new input images before the main fit. C3 is excluded and
is not reused to choose this candidate or its training schedule.

Qualification reports every training predictor's image F1 and learned stopping,
and all 256 fixed held-wording copy cases. Copy qualification must run the actual
input images through the full body with its learned adapters: the old cached
final states would not test the changed body. Rehearsal on those old states is
only a writer regularizer, not a guarantee of copy retention. A physical gate requires training
mean F1 at least 0.98, copy mean F1 at least 0.95 with at least 80% reaching
0.95, and at least 95% correct terminal/nonterminal decisions separately.
This gate is not language accuracy. Inspect its raw boards before autonomous
generation. Report complete stopped answers, caps, exact canonical matches and
output-first uncertain readings on all 72 normal/control streams. The new
paired-question comparisons test prompt dependence without changing context.
Do not select a checkpoint or alter the recipe based on these results.

All neural stages use the single GPU0 mutex. Full-reader caching is limited to
22 GiB allocation, 24 GiB available host RAM, a 300-second total pressure wait
and 14,400 seconds. Tail fitting loads only four source layers, never the whole
reader, with a 48 GiB host entry reserve, 32 GiB runtime reserve, 12 GiB allocation
cap and 10,800 seconds. Preserve failed and partial runs. Export only bridge
weights and the small contextual deltas; reuse the authenticated shared source
weights. Inference installs those deltas in the full image body and re-encodes
actual emitted pixels. No source lexical head, tokenizer, font or glyph bank
is permitted in that runtime. A successful fit or preflight is not the usable
language proof; that requires the complete autonomous assessment.

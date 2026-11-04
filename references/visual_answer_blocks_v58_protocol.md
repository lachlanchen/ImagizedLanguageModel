# V58: predict an answer as an observed image block

Frozen before implementation/training, 2026-11-05. V57's copy outputs have
already been inspected: 10/256 exact answers, 43 correct first images, and
14 complete reference prefixes followed by extra output. Its matched
no-copy audit is still finishing. This new experiment cannot alter V57.

Question interpretation and answer boundaries both remain weak. Test whether
selecting a complete variable-length image block is a better grounded-reading
head than independently choosing each next glyph. This is a paragraph-reading
component of ILM, not a replacement for the general free-writing objective.

## Model

Initialize both arms from the completed V57 copy EMA, checkpoint SHA-256
`44ad7c18047a3d5f564b911321b1b225a6bd113e1330457ea53e823ea1c841f7`.
Preserve its entire reader, image-copy law, stop hazard and free raster writer.
The new block arm adds separate 384-to-384 start/end query and key projections
and 16 learned length biases. Initialize both projection pairs from V57's copy
query/key; initialize length biases at zero. The matched glyph arm constructs
the same heads but freezes them and continues V57's original QA loss/decoder.

For fully observed prompt H with L image cells, candidate blocks are the
contiguous observed sequences X_i,...,X_j with 1<=j-i+1<=16. Start/end scores
are scaled query-key dot products from the final prompt state and source
states, plus the learned length bias. A softmax over valid (i,j) pairs defines
a normalized distribution over observed blocks. Repeated identical blocks
contribute summed mass. Training targets are reference image sequences and
their lengths, not Unicode identities or gold span offsets. Marginalize all
source blocks exactly equal to the target image sequence. No answer-aware
source masking is allowed: source candidates include all prompt positions.

The reading decoder emits the highest-marginal-probability observed block,
with first start position then shortest length breaking exact ties. It must
sum identical-block probabilities, not choose a position mode and call that
an image-sequence mode. Output length is learned through the joint block
distribution; the 16-cell maximum is the disclosed dataset bound. No reference
length, answer list, OCR, tokenizer, global glyph bank or external model enters
inference. Native aligned image prompts remain valid. This head cannot invent
an answer absent from its prompt; the retained V57 free-writing path remains
available separately, with its existing limitations.

## Matched training

Use unchanged V57 CMRC files/rendering and natural corpus, with their pinned
hashes. Both arms make 4,000 new updates, batch four, with seed 20265810 and
data seed 20265811. Three updates use QA and every fourth uses exactly the
V57 natural/motor objective. The block QA loss is total negative log reference
block mass divided by total reference cells plus terminal events, matching
V57's per-event scaling. The glyph control retains V57 event NLL. Both use
the same questions, references, natural positions and motor draws.

AdamW, rates, schedule, EMA, BF16, deterministic math SDPA, clipping, resource
limits, checkpoint/resume checks and no-development-until-both-finish rule
remain exactly as V57. Reader rate 3e-5, density 1e-4, output heads 3e-4;
200 warmup updates, cosine decay to .1 of peak, EMA .999. Both use final EMA.
No best-checkpoint selection. Unit tests must establish exact block equality,
duplicate marginalization, probability normalization, source bounds and finite
reader/head gradients. Run a four-update interrupted/resumed GPU smoke before
production. Production jobs must follow the completed V57 audit sequentially
on physical GPU 0; do not use GPU 1.

## Evaluation and limits

Reuse V57's exact 256 development IDs (seed 20265720). This is explicitly
reused development, not an independent final test. Compare raw block answers,
matched extra-trained glyph answers and blanked-question block answers.
Compute event/sequence likelihoods in their own defined units, without
equating block mass with a free image probability. Evaluate the same 393
paired questions and 256 supplied motor images. Report raw first audit rows,
answer exact match, multiset raster F1, paired article intervals (2,000 draws,
seed 20265919), stop/length/framing, timing, VRAM and every gate.

Retain the V57 bounded gates: image-only boundary; clean split/finite resource;
256/256 exact supplied motor images; >=5% primary exact answers; >=25% raster
F1; ordered-minus-blank F1 lower interval >0; >=2% both-correct question pairs.
Additionally report the paired block-minus-glyph answer gain and its interval;
do not claim improved learning from surface fidelity alone. Even a pass means
grounded paragraph extraction only. General language, novel free-form answers,
historical-script semantics and Qwen-level capability remain open goals.

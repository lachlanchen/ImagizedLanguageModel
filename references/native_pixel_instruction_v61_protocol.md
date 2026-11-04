# V61: full native pixel-language instruction adaptation

Registered 2026-11-05 before implementation, backward preflight or new training.
V58–V60 extraction probes have finished. Their limited question dependence
does not produce useful general language. V60's input-order intervention is
rejected; its exact answer count is 21/256 versus 27/256 matched context-first.

## Core experiment

Adapt the complete released MIXAR-477M pixel-language model end to end, instead
of attaching another extraction head. Every learned backbone and pixel-output
parameter receives answer-image likelihood gradients. The final model reads
prompt pixels and emits new binary image blocks, with no source-copy address,
symbol inventory, Unicode ID, OCR, tokenizer or external inference call.
It must write its answer rather than select a span. This is supervised
adaptation of external pretraining, not from-scratch single-GPU pretraining
or a claim to invent pixel language modelling.

Reuse the shared Apache-2.0 release revision
`c5d5bcfbad2f290296fd83810558f396c6a6333a`, weight SHA-256
`be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8`.
Retain its external-pretraining attribution. Do not duplicate the source
weights, Torch or CUDA installation. Use the existing Python 3.12 runtime.

Use the same pinned 3,211 train and 1,321 development CMRC questions, full
paragraphs, 480-cell prompt maximum and 16-cell answer maximum. The layout is
context-first, matching the retained reading interface, not selected anew on
this experiment's outputs. Render Noto CJK 26-pixel forms in 32-pixel binary
cells; explicitly disclose this shift from the author's native font renderer.
Append one blank cell to odd prompts and odd answers, solely to form complete
two-cell native 32-by-64 blocks. Append two all-white answer-termination blocks.
Cache losslessly packed image bits. Target strings and source offsets never
enter the actor. The first retained reference supervises training; scoring
accepts all references.

The model's native causal Bernoulli law predicts block t+1 from blocks <=t.
Optimize unweighted BCE over answer and two termination blocks only; the
prompt is context and contributes no label loss. Average each example over
its target pixels, then average eight accumulated microbatches. The causal
shift must have a regression test: neither current target nor later answer
pixels may enter its prediction context. Background is white=+1 at native
input and target white probability is one. Do not introduce text IDs, answer
length conditions, class weighting, output snapping or a source-image selector.

## Training and resources

Exactly 1,000 optimizer updates, microbatch one, accumulation eight, uniform
training questions with replacement. Seeds 20266110 (training) and 20266111
(data). All model parameters train in FP32 storage with BF16 forward/backward,
AdamW rate 2e-5, betas (.9,.95), decay .01, warmup 50 updates, cosine to .1,
norm clip one. EMA .99. Enable nonreentrant gradient checkpointing if supported;
do not compile a second model or launch a parallel GPU job. Deterministic
CUDA algorithms and math attention; record any unsupported operation before
production, not as a silent relaxation. Save optimizer/RNG/source hashes and
the final EMA, with no best-epoch selection.

Run a real full-backbone four-update interrupted/resumed preflight, using
microbatch one and accumulation two to bound smoke cost. Confirm finite
gradients in pixel input, first/last decoder layers and pixel output, real
optimizer updates and bit-exact resumed model/EMA/RNG before production.
Run sequentially on physical GPU 0 only, ceiling 20 GiB allocated VRAM and
four hours each for training and audit. Save safely on interrupt/resource
limit. Resource timing excludes external pretraining, cache preparation and
model loading, each reported separately. GPU 1 belongs to other work.

## Locked evaluation

Complete training before inspecting new development performance. Compare
final EMA and unchanged released base on the fixed 256 primary IDs. Test
normal and question-blanked prompt images, plus all 393 shared-paragraph pairs
for the trained model. Greedy native pixel decoding uses logit >0 as white,
feeds back the actual generated blocks and stops at the first exactly all-white
block. Maximum is ten generated blocks, fixed independently of references.
Discard only that detected termination block and trailing completely blank
32-pixel cells, identically for all conditions; this is a layout convention,
not glyph repair. A nonwhite malformed block remains in the output. Reaching
the maximum without blank termination is capped, not stopped.

Score exact stopped image sequences and exact-raster multiset F1, without OCR
or a persistent writing bank. This strict metric may reject nearly correct
rasters; preserve raw first eight outputs and report pixel foreground fidelity
as descriptive evidence, not a substitute for meaningful answers. Record
cap/blank rates, question dependence, repeated cells, paired both-correct
accuracy, all-development answer-only pixel NLL and resource use. Bootstrap
by source article, 2,000 draws, seed 20266219. Explicitly reused development.

A bounded free-reading pass requires exact answers >=10%, raster F1 >=25%,
positive lower 95% interval for trained-minus-base F1 and normal-minus-blank
F1, >=2% pairs with both answers exact, <20% capped outputs, and all
boundary/split/finite-resource/real-backward/replay checks. Otherwise report
failure. These gates are not high-quality general-language, historical-form,
new-font, Qwen-parity or from-scratch training claims. Native PNG inference and
weight-only replay must be verified. Broader licensed instruction data and a
new held-out final test are later work if this core free-writing path works.

# V62: matched whole-image density adaptation

Registered 2026-11-05 before implementing the production trainer or inspecting
development output from either arm. The preceding goal turn made progress:
it added a sourced research review and a mathematically tested eight-expert
prototype, with one real full-backbone GPU update. It did not establish usable
language. V61's failed audit remains unchanged.

## Question and scope

Does a conditional mixture over complete native image blocks improve coherent,
question-dependent free writing over a single product-Bernoulli head, under
matched end-to-end adaptation? This isolates the output density. It does not
resolve font shift, external pretraining, broad instruction coverage or general
language quality by assumption. Both arms use the same Noto CJK interface as
V61; do not change fonts between arms or select fonts from their development
scores. Author-font adaptation remains a separate future experiment.

The single-head arm has K=1; the correlated arm has K=8. Both start from the
same pinned MIXAR-477M release, revision
`c5d5bcfbad2f290296fd83810558f396c6a6333a`, SHA-256
`be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8`.
Use the identical linear physical-pixel projection implementation in both
arms; K=1 is the exact original Bernoulli probability law. K=8 repeats the
released projection and adds zero-mean expert perturbations of scale .01,
seed 20266210, with a zero-initialized gate. Capacity and output computation
increase with K; this is not a parameter-matched comparison or an efficiency
claim. Every contextual and rendering parameter trains.

## Fixed training

Reuse V61's lossless image cache and its source/data hashes: 3,211 training
questions and 1,321 development questions, article-disjoint, with complete
prompts <=480 cells and first-reference answers <=16 cells. No labels, Unicode
IDs, source-answer offsets or transcripts reach the actor. Native blocks have
32x64 physical binary pixels. Prompt padding, answer padding and two blank
termination blocks are exactly V61's convention.

For each arm, run 1,000 updates, microbatch one, accumulation eight, uniform
replacement training sampling. Use seeds 20266220 (training) and 20266221
(data) identically in both arms. AdamW 2e-5, betas (.9,.95), decay .01,
50-update warmup and cosine decay to .1, clip norm one, EMA .99. FP32
weights/optimizer/EMA and BF16 forward/backward, deterministic algorithms,
eager/math attention, nonreentrant activation checkpointing. Optimize the
exact marginalized joint block likelihood divided by 2,048 pixels, averaged
over answer/termination blocks per example and over accumulated examples.

Before production, perform four updates with accumulation two, uninterrupted
and as two plus resumed two, for each arm. Require bit-exact model, EMA,
optimizer and all RNG states, nonzero finite gradients through input, first
and last contextual layers and head (plus gate for K=8), and zero causal
effect of changing future answer pixels on earlier predictions. Verify K=1
physical projection equivalence and cached versus full-prefix inference.
Smoke weights do not qualify as a trained language model.

Run sequentially on physical GPU 0 only. GPU 1 belongs to other work. Each
training/audit has a 20 GiB allocated-memory and four-hour ceiling. Save
optimizer, RNG and source identity on interruption or resource limit. Report
loading, training through final update, and audit separately; external
pretraining and renderer preparation are not local foundation training.
Use final EMA only; no checkpoint or seed selection from development.

## Decoder and independent runtime

At every native step, threshold each expert's logits into one complete bitmap,
evaluate all those candidates under the full normalized mixture, and choose
the highest score. This is approximate MAP, not a proof of global MAP.
K=1 gives ordinary zero-threshold native decoding. Feed back the actual
selected image. Stop on the first exactly white 32x64 block, otherwise cap at
ten generated blocks. Remove only the detected terminal block and completely
blank trailing cells; preserve every malformed nonblank pixel.

Record candidate selection counts and gate entropy as descriptive collapse
diagnostics, including terminal decisions. Neither is a quality gate or a
reason to retune the completed experiment. Deploy learned weights plus local
architecture code without original release weights. Verify native PNG input,
bit-exact answer tensors/PNG/events in a fresh offline process with empty HF
caches. No tokenizer, OCR, stored glyph bank or external model call is allowed
inside the deployed actor.

## Locked evaluation and decisions

Finish both arms before examining their development performance. Use the
same fixed 256 primary IDs as V57--V61, normal and blank-question conditions,
all 393 shared-paragraph question pairs for both arms, and all 1,321
development examples for answer-image NLL. These are explicitly reused
development data, not an unscored final test. Preserve all raw outputs and the
first eight primary outputs without selection. Score stopped exact raster
answers, exact-raster multiset F1, descriptive aligned foreground F1,
blank/cap/repeat rates, paired both-correct accuracy and resources.

Bootstrap paired differences by source article, 2,000 draws, seed 20266229.
Report K8-minus-K1 exact answer, raster F1 and foreground F1, and normal-minus-
blank differences within each arm. A density-intervention pass requires a
strictly positive lower 95% interval for K8-minus-K1 raster F1 and for K8's
normal-minus-blank raster F1, with cap rate <20% and every boundary, resource,
backward/resume/causality and offline replay check passing.

A separate bounded free-reading pass retains V61's thresholds: exact answers
>=10%, raster F1 >=25%, question-dependent F1 lower interval >0, >=2% question
pairs both correct, cap rate <20%, and all implementation checks passing.
Even both passes would not establish high-quality general language, historical
script understanding, Qwen parity or from-scratch single-GPU pretraining.
No OCR or vision-judge score is introduced into these locked decisions.

Source files and this protocol are hashed before each run; reject resume or
evaluation when identities differ. Any revised hypothesis or training recipe
requires a new experiment rather than changing the gates after results.

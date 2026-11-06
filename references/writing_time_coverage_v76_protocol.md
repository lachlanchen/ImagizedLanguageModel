# Matched denoising-time refinement of the failed full-corpus writer

The completed V75 candidate fails language qualification. Its 32-case training
diagnosis also fails with correct prior answer codes and known answer length.
The fixed codec reconstructs supplied codes almost perfectly. A retained-code
probe shows that the denoiser changes otherwise exactly decodable near-clean
inputs: at t=31/32, original-block agreement falls from 32/32 to 6/32. This
minor stroke corruption does not explain the much larger free-answer failure
by itself. The logit-normal training law gives only 0.423 expected samples at
or above this time among 7,007,799 sampled blocks; the actual tail draws have
not been replayed.

Test the narrow hypothesis that insufficient high-signal supervision damages
the learned writing head. Preserve the original failed checkpoint, sources,
audit and requirements. This experiment is not a general language benchmark.

## Matched intervention

Start both arms from the completed V75 checkpoint
d0cb3aa7923db450dc904269c753e0a4157e704a6da1be487507b168d7612249.
Freeze every parameter except the existing 36,403,200-parameter flow head.
Reuse all 12,976 authenticated training records. No inherited reader, new
dataset, teacher, tokenizer, OCR, vocabulary, image inventory or architecture
is introduced. The causal body and stop head remain fixed. Save head deltas
as complete head states; retain the original checkpoint by reference.

Continue each arm for exactly 4,000 updates at batch 16 and at most 16 sampled
blocks per example, with equal per-example weighting. Both start a fresh
AdamW optimizer (betas .9/.95, decay .01, peak 1e-4, warmup 100, cosine decay
to one tenth, gradient clipping 1). Use the same shuffled-record seed 20267601
and CUDA noise seed 20267602. Draw the same standard normal time variate u:
control t=sigmoid(-.8+.8u); intervention t=Phi(u), using erf in FP32 and
clamping to [1e-6,1-1e-6]. Phi(u) is uniform before floating-point rounding
and clipping. Record selected-example, position, noise, and time-variate
digests at update 1 and every 500 updates and check equality across arms.

Retain direct clean-code prediction and the original error weighting
1/max(1-t,.05)^2. Thus both exposure and the effective integrated objective
change with the time law; this is not a pure numerical-integration test.
An identity skip, sampler modification, output repair and retrieval are absent.
Only the head is differentiated; causal conditions use frozen true histories
during supervised optimization. A short real-data execution check must verify
exact optimizer/noise/sampler resume and frozen-body preservation before the
two production arms. Its weights are discarded.

## Fixed assessment

Evaluate only the final update of each arm on exactly the 32 training questions
already fixed by V74 and independently examined in V75. Reuse their verified
prompt-derived memory, without loading the inherited reader. This is cached
image-feature inference, not a new actual-image reader call. Free generation
receives no target codes or length, uses seed 20267420 plus pair index, the
unchanged 32-step sampler, re-encoding of emitted pixels, learned stopping and
the original 128-block safety cap. Retain every output. Also repeat the retained
first-block noisy-code probe at t=31/32 and t=1 with seed 20267597 and the same
target/noise vectors as V75. These assisted probes receive no autonomous credit.

Report raw original-image exact counts, F1, learned stopping, first-block
scores and paired differences for both arms. The writing repair advances only
if the uniform arm reaches at least .999 near-clean first-block F1, at least
.80 free full-stream F1, at least 16/32 free exact answers and at least .05
free F1 gain over continued logit-normal training. These are training-fit
engineering criteria, not replacements for the original development gates.
Failure rejects this bounded repair; success still requires fresh actual-image
development assessment and new-copy transfer before any language claim.

Independent CPU scoring authenticates the source checkpoint, all final head
files and saved outputs, reconstructs references from the original image-only
training data, recomputes physical scores, validates frozen-body hashes and
the common-random-number trace, and re-decodes both sets of retained near-clean
predictions. It does not replay the entire neural sampler. Sealed data stays
unopened. Use one project GPU lock and GPU 0 only; no GUI stack.

The clean-prediction formulation is informed by Li and He,
[Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720).
Its ImageNet results do not establish this writing-code time law or language
quality. Pixel-language generation has prior art, including
[PIXAR](https://arxiv.org/abs/2401.03321); neither novelty nor LLM parity follows
from adopting image input and output.

# The full-corpus writer fails even with correct answer history

On the fixed 32 training questions used in the earlier small learning control,
the completed full-corpus writer produces **0/32 exact answers** and 45.771%
foreground F1. Supplying the correct preceding image-code history and the
correct answer length raises F1 to 58.045%, with **0/32 exact pixel streams**.
Free first-block F1 is 59.929%. All 32 free answers stop by the learned rule.
The fixed decoder given the correct codes produces 28/32 exact complete
original streams and 99.9953% F1. The failures therefore exist before long
free rollouts; the writing codec itself can preserve these targets.

These are seen training cases, not generalization evidence. The earlier
32-question fit had 2,000 passes over its small set; full-corpus training had
49 complete passes. Several conditions differ, so this comparison does not
identify one cause. All 32 actual prompt-derived reader memories exactly
match the previously cached features. Correct-question conditioning reduces
first-block clean-code error relative to the paired question at low noise
times, but that conditional sensitivity does not establish correct answers.

A separate CPU reconstruction probes retained predictions on constructed
near-clean target codes. At t=31/32 the direct noisy input decodes to 32/32
original first blocks exactly; passing it through the learned clean predictor
reduces this to 6/32 and 99.1349% foreground F1. At t=1 the clean predictor
still changes supplied codes. These are target-assisted probes, not actual
free sampler states. The small high-signal stroke errors do not account for
the entire free-generation failure.

The completed sampler metadata and authenticated answer lengths imply
7,007,799 sampled writing blocks in training. Under the declared logit-normal
time law, the probability of sampling t at or above 31/32 is 6.0333e-8,
giving **0.423 expected exposures** in that region. This is a theoretical
expectation, not the observed count; the full original time stream was not
replayed. A matched head-only refinement tests this possible defect while
preserving the failed original checkpoint and language requirements.

Independent CPU verification reconstructs references, authenticates all raw
files and recomputes free, assisted and supplied-code scores. It re-renders
all 32 supplied-code streams and checks retained latent errors. It does not
replay the full neural actor. The first two fixed cases are retained without
pixel repair under evidence/corpus_fit_v75_raw/; all 32 full raw cases remain
under artifacts/corpus_fit_v75_20261106/predictions/.

See the [machine-readable evidence](evidence/corpus_fit_v75_result.json),
[fixed diagnosis protocol](corpus_fit_v75_protocol.md),
[completed full-data audit](clean_corpus_v75_result.md), and
[matched repair protocol](writing_time_coverage_v76_protocol.md).
This candidate remains unusable as a general image-language model.

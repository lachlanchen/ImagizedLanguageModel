# Capture and diagnose the failed physical-pixel cache comparison

The preceding integration failed before optimizer updates at a fixed maximum
absolute FP32 logit discrepancy of 1e-4. Its actual discrepancy was not saved.
This follow-up retains the exact decoder weights, actual native writing-query
states and complete true-prefix traces before numerical assertions. It does
not silently retry the training run or relax its acceptance criterion.

Rebuild the same initialization and five owning PNG prefills, with the same
seed, checkpoint, transferred raster tensors and reset condition projection.
Keep full native KV. Retain all five native prefixes, the untrained actor state
and the two complete training-image encodings for reproducible diagnosis.
Use the first two contextual query states and their complete 256-piece target
blocks, as in the failed comparison. Capture all FP32 teacher logits through
both full and incremental paths, target-suffix causality, and both autonomous
greedy pixel trajectories. These two local block trajectories are numerical
controls; they are not autonomous language answers.

After the heavy model exits, compare both captured FP32 paths with independent
FP64 CPU equations for the same architecture and parameters. The reference
implements explicit query/key attention and softmax rather than production
SDPA. Record absolute, RMS and normwise errors, worst coordinates, teacher
argmax changes, log-probability error and total variation. Also compare full
and incremental FP64 teacher paths. Preserve original tensors for inspection.
No automatic tolerance change or language qualification follows from this
measurement. Any implementation repair or revised criterion requires a stated
numerical rationale and a separate registered validation.

The one GPU0 capture has a 600-second deadline and a 22 GiB allocation cap.
Require at least 24 GiB available host RAM, swap at most 75% and at least
20 GiB free GPU0 memory before loading. Keep GPU1 and foreign services intact.
No optimization, complete-answer generation or external source service runs.
Preparation and FP64 analysis use CPU with CUDA hidden. Bound source, protocol
and predecessor hashes before capture; preserve terminal failures without an
automatic rerun. This is a numerical diagnosis, not evidence of language
learning or an efficiency advantage.

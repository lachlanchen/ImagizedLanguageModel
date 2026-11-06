# Diagnose native-context writing on the existing fixed Chinese cohort

This diagnostic is specified after the 20,000-update training images and
before the final native checkpoint or its development outputs are available.
It changes no training, checkpoint selection, inference rule or original
quality gate. The purpose is to distinguish prediction errors that persist
under correct image history from errors introduced during free continuation.
It is a training-set diagnosis and provides no generalization qualification.

Reuse exactly the 32 Chinese questions in sixteen visible-passage pairs from
the existing V74 receipt, SHA-256
`86809a7b29bc57c84ee4991d10d320e04031cf56da2ad6d49724e693aa20f884`.
These are the same cases used in the V75 failure diagnosis and are included
in the native model's 3,193-record training set. Authenticate the original
physical image, complete target, cached image codes and prompt-memory digest
for every case. Keep every case and failure. The older and newer models have
different architectures and training histories; matching the cohort does not
make their comparison a single-factor causal experiment.

CPU-only preparation is allowed while training continues. GPU evaluation must
wait for actual completion of the original 40,000-update run, all 384 fixed
audit outputs, independent physical verification and the three standalone
CLI calls. Use the project GPU lock and an otherwise free physical GPU0.
Reuse shared frozen weights. Perform no optimization or external model calls.
Run this diagnostic and the auxiliary neural reader sequentially.

For each case, prefill the actual owning prompt pixels once. Call the unchanged
native writing model's `generate` method with that full prefix, seed 20267420
plus pair index, the registered 32 flow steps, learned stop and 128-block cap.
No target, reference length or cached answer code enters this free call.
Only after it finishes, load target codes and produce two separate controls:

1. **Correct-history writing.** Compute causal native states from the true
   preceding image codes, then draw one block at each supplied target position.
   Reuse the free call's seed and sequential one-block random draws. Preserve
   the predicted coordinates, physical images and stop probabilities under
   reference history, including the terminal state. Pixel agreement is measured
   at a supplied length; this path receives no autonomous-answer credit.
2. **Supplied-code reconstruction.** Decode the authenticated target codes
   through the unchanged fixed codec and retain the resulting physical images.

Check equality of the first stop probability and, whenever the free call emits
a first block, exact equality of that block between free and correct-history
paths. They share the same initial state and noise, before history differs.
Check the prompt-memory digest and initial state against the authenticated
training cache. Do not retain a second full layerwise prefix cache on disk.

After all 32 free calls, reuse the earlier first-block conditioning probe:
times 0, 0.1, 0.3, 0.5, 0.9, 31/32 and 1; one Gaussian noise vector per case
from seed 20267597, shared across times and owning/paired-question conditions.
Keep the target and noisy input fixed while replacing the initial reader state
with its same-passage partner's state. Preserve every predicted coordinate and
report MSE by time. These constructed target-assisted states are not free
sampler trajectories. Sensitivity to the condition does not establish correct
reasoning or identify a unique cause of failure.

Report free, correct-history and supplied-code pixel agreement, first-block
quality, free stopping and reference-history stopping separately. Retain all
raw tensors, unaltered images, source bindings and timings. Independently
reconstruct source rasters and prompt layouts on CPU, validate stop traces,
recompute scores and latent errors, and replay all 32 supplied-code decodes.
Verify retained correct-history coordinates through the fixed decoder too.
This does not independently replay the complete inherited reader, contextual
body or neural sampler. Failed execution, prediction and verification remain
visible and never qualify the general ILM goal.

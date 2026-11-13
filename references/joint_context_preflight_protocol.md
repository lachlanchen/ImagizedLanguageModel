# Complete-source preflight for joint image/context gradients

The new intervention is selected in
[the context-learning decision](joint_writer_context_learning_decision.md).
The completed head-only control misses its physical prerequisite and its exact
probability probe rejects the registered search-only criterion. No earlier
physical reading, gate or language score is changed by this preflight.

Initialize the actual final point-reader continuation state and the identical
fresh image-head draw used by the head-only control, with its training-only
normalization. The native body, prompt memory, image-feedback encoder, affine
buffers and stop parameters are frozen. Only the existing continuation residuals,
initial query and image head receive gradients: 5,005,760 parameters in the
specified construction. Recount the actual complete deployed model as well.
Do not infer its size from the trainable subset.

Choose the first eligible training QA with an odd complete target length in
5..63 and the first book with maximal complete target length. The current
manifest gives source indices 4762 (five QA glyphs) and 342 (513 book glyphs).
Use both fixed and proportional prompt layouts. This choice is based on shape
and complete-source coverage, not observed prediction quality. No target is
truncated. The source exclusions and entire training manifest are unchanged.

For each complete QA, compare ordinary unsplit autograd with the exact state-VJP
implementation using two-image microbatches, including the final one-image
remainder. Compare every trainable tensor. Require relative gradient L2 error
at most 0.01 or maximum absolute error at most 1e-5, and always require maximum
absolute error at most 1e-5+0.01 times the reference tensor's maximum absolute
gradient. Require loss agreement within 2e-5. These fixed tolerances account for
floating-point reduction; the chain-rule construction is mathematically exact.
Perturb one supplied image and require its own and earlier predictors unchanged,
with later states changed, in the actual new constructor. No cached predictor
state may replace the differentiable continuation computation.

Each complete long book must produce finite, nonzero gradients for every
continuation/head trainable, within 22 GiB allocated GPU memory. Use 64-image head
microbatches, one balanced stop loss with weight 0.1, gradient norm clipping at
one, FP32 AdamW at 4e-7 and weight decay 0.01. This is the full recipe's first
warm-up learning rate. Verify one actual optimizer step and counters for all
trainables, actual weight changes and unchanged frozen parameters. Reset all
trainables exactly to the registered initialization after each step. Both
preflight updates are counted and may never enter the fresh full fit.

Use the existing exclusive GPU0 mutex. Entry requires 48 GiB available host RAM
and 20 GiB free device memory; runtime retains 24 GiB host memory with a 22 GiB
allocated-device ceiling. Keep two CPU threads, at most 1,800 total instrumented
seconds including imports/validation and at most 300 seconds of pressure waits.
Do not start another project heavy job. There is one attempt, no automatic
retry and no autonomous image generation. Preserve any failure before proposing
a correction; passing a computational preflight does not establish language.

Commit source before CPU-only registration, commit/push the registration, then
run `scripts/preflight_joint_context_learning.py run` with physical GPU0 and the
documented deterministic CUDA environment. A pass permits preparing the separately
registered complete 41,298-update trial and its full output evaluation. A failure
requires repairing the observed computational or resource issue, preserving the
attempt and its cost. The ILM, compactness, independence, historical-form and
manuscript requirements stay open in either case.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/preflight_joint_context_learning.py register
# Commit and push the generated public registration before the next command.
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  python -u scripts/preflight_joint_context_learning.py run
```

# V62 matched training preflight

Both real four-update runs pass bit-exact interrupted/resumed model, EMA,
optimizer, Python/NumPy/CPU/CUDA and data RNG checks. Input projection, first
and last contextual layers, physical-image heads and the K=8 gate receive
nonzero finite gradients. Changing future answer pixels has zero effect on
earlier predictions. Training-only examples are used; no development quality
is inspected.

Peak allocated memory is 9.898 GiB for K=1 and 10.169 GiB for K=8, with FP32
model/optimizer/EMA storage and BF16 training. These short runs establish
full-backbone feasibility, not production speed or language quality.

The initial cache-equivalence check at a 2e-4 tolerance fails under TF32:
K=1's maximum logit difference is 0.08578, with no thresholded-pixel changes
in the checked source. A diagnostic isolates shape-dependent TF32 numerical
error: strict FP32 reduces the maximum difference to 0.00002098 for K=1 and
0.00001526 for K=8. The verifier preserves fast-kernel discrepancies separately
and uses strict FP32 for cache semantics. The original training settings and
the registered language gates remain unchanged.

The immutable receipt is
`artifacts/correlated_native_density_v62_20261105/preflight/report.json`.
Source identities are fixed before production. Both 1,000-update arms and the
locked language audit remain required; the overall ILM goal is active.

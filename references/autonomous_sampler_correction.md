# Match native input encoding before differentiable sampling

The first autonomous sampler preflight at commit `d388371` fails before any
optimizer. The native call reproduces the previously retained final tensor
exactly. The custom call using the historical paired input cache differs at
step 0 (maximum latent difference .015625) and at step 49 (.07421875).
The exact-match gate is unchanged.

A separate VAE-only execution compares the actual source input path with the
historical cache. Encoding outside autocast reproduces that cache bit-exactly.
Encoding the same input under the native pipeline's outer BF16 autocast gives
maximum difference .0625; only .185341 of feature elements remain equal.
Frozen BF16 weights alone do not reproduce an autocast computation.

The corrected sampler encodes each reference through the same native
preprocessing, posterior mode, batch normalization and BF16 autocast as the
real source pipeline. It does not reuse historical outside-autocast features
as if they were exact native inference features. The cache remains unchanged
and its page bindings are still checked. The reference encoder is frozen.

Repeat the original two-language, all-50-step and final-pixel comparison before
any optimization. Keep the original failed preflight. This is an implementation
correction under the original requirement, not a relaxed numerical bound or
evidence that this feature difference caused the prior model's content errors.

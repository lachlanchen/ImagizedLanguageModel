# Conditional whole-image mixture: feasibility result

The proposed eight-expert actor has 489,998,208 parameters, versus 477,145,984
in the original native model. A real full-backbone optimizer update, including
FP32 model/optimizer/EMA storage and BF16 forward/backward, completes on
physical GPU 0. All gradients are finite; the image projection, first and last
contextual layers, expert projection and mixture gate have nonzero gradients.
The optimizer changes the weights. Peak allocated memory is 9.270 GiB and the
measured update takes 0.849 s, excluding loading and setup.

This is one training example, not matched throughput against V61, a production
run or language-quality evidence. Context lengths and accumulation differ
from V61, so the lower allocation number is not an efficiency improvement.
No development data or OCR labels enter the preflight.

Three mathematical tests pass: exhaustive small-space normalization,
single-expert Bernoulli equality, whole-form mode behavior, native pixel
projection layout, finite gradients and exact scoring of generated candidates.
Candidate mode enumeration is approximate MAP. The head uses conditional
neural functions over physical pixels; it stores no character images or
symbol/codepoint table. A shared latent expert allows pixel correlations,
but expert collapse and poor semantic selection remain possible.

The receipt is `artifacts/correlated_native_density_v62_preflight.json`.
[Plan](correlated_native_density_v62_preflight_plan.md),
[implementation](../ilm/visual_lm/correlated_native_density.py),
[real GPU preflight](../scripts/preflight_correlated_native_density_v62.py).
Matched training and language evaluation require a new registered protocol.

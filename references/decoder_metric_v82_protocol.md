# Matched physical-metric test after the failed readout repair

Prospective protocol, 7 November 2026. The completed V81 adapted model has
0/32 exact autonomous answers and 72.748% image foreground F1 on the fixed
training cohort; correct-history F1 remains 79.629%. The frozen codec has
99.9953% supplied-code F1. Decoder-only CPU/GPU derivative preflights pass.
This motivates testing physical sensitivity in optimization, without assuming
that it resolves conditional prediction or language learning.

Both arms restore the same completed V81 adapted checkpoint, SHA-256
`7da31d229f4c5cc5bb51a5f5776a2c6e61e444f171ba412de547165911cb532c`.
Keep all 32 existing training questions, sixteen same-passage pairs, their
original order and actual prompt-only caches. Both use the identical query,
rank-8 suffix adapters, image input/flow/stop tensors and frozen codec/base
weights. Complete/trainable counts remain 7,583,225,971/53,096,961. This is a
local adaptation of externally pretrained weights, not a lightweight model
trained from scratch.

The only experimental change is the per-block clean-code regression error.
Control: mean(v²). Metric: mean(v²)+0.001 mean((J_R(x)v)²), where v=C(x,t,h)-z,
x=(1-t)epsilon+t*z, and R is the frozen decoder-logit map including the
actual per-tile layer normalization. Detach the metric point x, use no hidden
target-dependent metric weights, and retain the existing per-case weighting
and division by max(1-t,0.05)². The positive code term makes the metric
positive definite. Its state-only conditional-risk optimum is unchanged;
finite-model optimization and autonomous image quality are empirical tests.
No semantic feature, OCR, character vocabulary, teacher service or output
lookup is added. The autonomous sampler and physical-pixel feedback are
unchanged and do not call the new loss.

Use fresh AdamW (0.9,0.95), weight decay 0.01, gradient clip one, batch four,
2,000 updates per arm, peak learning rate 1e-4, 100-update warm-up and cosine
decay to 1e-5. Seed 20268210; case sampler +1 and CUDA noise +2. Both receive
identical indices and noise draws. Keep uniform time sampling, all block
selection rules, 32 sampling steps, 0.5 learned stop and 128-block cap. Each
question has 250 additional exposures per arm. No early stopping or best
checkpoint selection from generated answers is allowed.

Before this training, an actual complete-actor preflight on original case
positions [0,2,3,17] must demonstrate: strength-zero loss equals the original
adapted objective exactly under the same RNG; finite nonzero gradients in
query/input/flow/stop/adapter groups; exact third-update model, Adam, sampler
and noise resume; unchanged frozen parameters and all prompt KV; exact future
exclusion; the existing V81r2 normwise padding/incremental criterion, with
original absolute-criterion outcomes also retained; and actual generated-pixel
feedback replay. Reset all preflight optimization before both arms. Peak
allocated memory must stay below 24 GiB on one GPU. Preserve every failure.

Retain before/after autonomous images for every case. Initial generations in
both arms must replay all 32 V81 adapted autonomous outputs exactly. At final
update, retain incremental correct-history/known-length and supplied-code
controls, using the common incremental start so first stop/pixels are exactly
shared with free generation. No reference enters an autonomous call. CPU
verification must score all 64 final answers independently, validate both
initial replays, decode all control coordinates and supplied codes, and bind
sources, checkpoints, targets, pixels and PNGs.

The original provisional training-fit criterion is unchanged: metric free
exact >=24/32 and free F1 >=0.98, with both exceeding the control. Report all
paired case scores and every failed criterion. A pass would only justify
new-prompt/copy and original development evaluation; it would not complete
the general ILM goal. A failed comparison blocks scaling this recipe without
a new diagnosis. Neither decoder derivatives nor training-fit alone qualify
language understanding, historical-script generation or efficiency parity.

Use `scripts/train_decoder_metric_v82.py --stage preflight`, then `--stage
train` only after that complete-actor preflight passes. GPU invocations use
the existing compatible Python 3.12 environment, explicit GPU0 and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Run `--stage verify` with CUDA explicitly
hidden after both arms finish. One project GPU job may be live; source files
and this protocol must be committed before execution. Outputs default to
`artifacts/decoder_metric_v82_20261107`; completed earlier runs stay unchanged.

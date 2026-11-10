# Spatial semantic motor: prospective image-flow test

The168.4M dense semantic-coordinate flow fails both primitive gates. It gives
mean selected F10.906284 on clean conditions and0.715449 on45%mixtures,
with zero exact glyphs. All fixed64-triplet sheets were inspected. The run's
129.67s fit and85.10s assessment do not establish successful motor learning;
no language assessment uses it. Preserve the original verified model.

This experiment reuses the existing `FovealInkFlow` spatial U-Net instead of
expanding that dense flow. Inputs are a32x32noisy glyph canvas, a4096-dimensional
uncertain semantic vector and the preserved neural decoder's32x32continuous
ink plan. The spatial plan exposes stroke geometry. Base channels32,
context dimension256, condition dimension4096. D,E,G,stop and the source body
remain unchanged. The new CNN has no character inventory or lexical projection.
No language question/answer or development state enters training.

Sample the same training-only4,087 forms uniformly. Conditions are60%mixtures
with a uniformly different unit-RMS anchor (weight0.25–0.45, plus0.05Gaussian
noise),20%Gaussian-corrupted anchors (sigma0–0.8),20%clean. Normalize conditions
to RMS one. Compute the spatial ink plan as2sigmoid(D(c))-1, with positive ink.
Canonical target is2ink-1 with independent0.001Gaussian smoothing. Use the
existing path x_tau=(1-tau)target+tau*epsilon, target velocity epsilon-target,
uniform tau, and *unweighted* velocity MSE. Integrate from tau1noise to tau0data.
This is the reversed-time form of conditional flow matching; finite training
and sampling are not correctness guarantees. The spatial CNN's output spans
pixels through shared convolutions rather than a narrow fixed dense projection.

Train8,000batch128 AdamW updates, lr2e-4, betas(0.9,0.99), weight decay0.01,
clip1,200-stepwarmup, cosine decay to10%, BF16 compute/FP32 weights and loss.
EMA min(0.999,(1+step)/(10+step)); use only the final EMA. Seed202611109.

Assess every form under clean conditions and the fixed55/45mixture with
distractor offset1049, seed202611110. Generate four images with16Heun intervals
(last step Euler), no classifier-free guidance. Re-encode every actual binary
sample with preserved E and select by cosine to the condition. No runtime
prototype/character bank, reference, source output head or font may select or
repair the samples. Retain all candidates, selections, scores and pixels.
Reference-best candidate F1 is an explicitly oracle diagnostic only.

Keep the same prospective gate: clean selected meanF1>=0.99 and95%of forms
>=0.95; mixture selected meanF1>=0.95 and80%>=0.95. Inspect the fixed64-triplet
boards. A pass qualifies only a known-form image motor, not general language.
Only a separately specified autonomous forty-stream language assessment can
establish whether the resulting sampled image history improves answers.

Use sole GPU0 lock;48GiBhostentry/32GiBruntime,8GiBallocation,1,800s perstage,
PSI<=0.5/0.1 and<=300spressurewait. Preserve foreign services and all completed
sources, datasets, checkpoints and failed results. No physics/novelty priority,
from-scratch language learning or matched-quality efficiency claim follows.

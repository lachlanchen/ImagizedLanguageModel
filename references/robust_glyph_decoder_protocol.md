# Decoder robustness before further language fitting

The preserved language checkpoint's first-state diagnostic finds the reference
first form ranked highest by semantic cosine on nine of sixteen normal Chinese
questions, but only one exact first image. This is an evaluator-only rank,
not a deployed nearest-form repair or semantic accuracy. The original decoder
is robust to small isotropic noise but degrades on mixtures. Both broader
readout compilers were rejected. This new intervention freezes their entire
language path at the *original* verified baseline and trains D alone.

Use all 4,087 existing training-only anchor/image pairs. No question, answer,
development state or teacher output enters this fit. Normalize each anchor to
RMS one for corruption, preserving its canonical direction. Uniformly draw a
target form and a different distractor. Sixty percent of draws add independent
Gaussian noise, with per-example standard deviation uniform in [0,1]. Thirty
percent mix the unit-RMS target with one distractor, total distractor weight
uniform in [0,0.45], then add Gaussian noise of standard deviation 0.05. Ten
percent retain the original exact anchor. The target is always the original
canonical glyph. Corruptions are continuous training vectors, not new scripts
or language knowledge.

Start from the preserved complete language checkpoint, not either rejected
readout compiler. Train decoder_input and decoder only for 6,000 batch-128
AdamW updates, learning rate 2e-4 with the original cosine schedule, balanced
pixel BCE and gradient clipping at one. Seed 202611105. Keep the image encoder,
semantic readout, affine buffers and stop predictor exactly unchanged. Runtime
remains the same continuous neural G->D->actual pixels->E loop, without
nearest-anchor selection, prototype bank, source vocabulary or font rendering.

Compare the original and candidate on every anchor under three fixed input
routes: clean original codes; independent Gaussian noise sigma0.6 on unit-RMS
codes; and a fixed 0.55/0.45 mixture with distractor offset1049. Reset the
independent assessment seed202611106 for each model and verify exact input
hashes across arms. Preserve every raw output and per-form metric. This tests
new corruptions of known forms, not unseen identities. Require mean foreground
F1>=0.999 on clean codes, >=0.99 on Gaussian codes (at least95% of forms with
F1>=0.95), and >=0.95 on mixtures (at least80% with F1>=0.95). Inspect all
three fixed64-triplet sheets before interpreting the result.

Only after this primitive gate passes, run all eight original training controls,
sixteen normal Chinese questions and sixteen erased controls through autonomous
actual-image feedback. Preserve the original64-glyph cap and learned stop.
No target length, character identity, source head or reference is supplied to
inference. Review raw images before canonical scoring. These are reused
exploratory development cases, not a sealed semantic benchmark. Sharp glyphs
alone do not establish a correct complete answer. Retain the earlier date
checkpoint even if the new decoder passes rendering.

Use the same small-stage48GiB host entry/32GiB runtime reserve,8GiB VRAM cap,
1,800-second stage cap, exclusive GPU0 mutex and PSI/wait limits as the compiler.
Language assessment uses the unchanged original reader admission and22GiB
runtime allocation cap. Preserve foreign jobs and all completed experiment
sources. Report resource failures or quality failures without weakening the
prospective gate after seeing the result.

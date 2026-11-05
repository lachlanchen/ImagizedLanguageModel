# Sampling precision after the fixed-image control

The completed 4,000-update comparison gives 0/64 exact renders for the
narrow flow and 63/64 for the full-dimensional flow. The latter has mean
foreground F1 0.999539, but minimum F1 0.970526. It fails the original
requirement that every seed reach 0.99. That result remains a failure.

This is a post-hoc diagnosis of the remaining error, with one separately
specified fresh-noise confirmation. No weights, target image, output
threshold, LayerNorm or quality requirement change. It is not a language
experiment.

Run the trained full-dimensional flow on all original 64 noise draws under
this fixed order of numerical settings:

1. BF16 contractions, 32 midpoint steps: the original sampler.
2. FP32 contractions, 32 midpoint steps.
3. BF16 contractions, 64 midpoint steps.
4. FP32 contractions, 64 midpoint steps.

The original setting must reproduce all saved noise, endpoints, codes and
pixels exactly. Retain every setting's raw values, physical outputs and
scores. These 64 draws now serve as calibration data; they are not a fresh
test after the failed control has been inspected.

Select the first setting in the stated order that meets all three original
image requirements: mean original F1 at least 0.995, every original F1 at
least 0.99, and at least 90% exact supplied-code renders. If none passes,
stop without a confirmation or a full language run.

If a setting passes calibration, evaluate that one setting once on 128 new
noise draws with generator seed 20267303. The trained weights remain
unchanged. Apply the same quality requirements, retain every output and
independently score/redecode all physical images on CPU. Do not select a
second setting using these confirmation results.

The original control's failed outcome must not be replaced. A successful
confirmation would show repeatable generation of one known image under
the stated numerical setting. It would not demonstrate prompt reading,
semantic correctness, unseen glyph generation or a general ILM.

# Spatial 64-value codec comparison

Use the exact images, partition, form-balanced sampling, source indices, loss,
6,000-update optimizer schedule, precision, replay, evaluation and resource
limits of the completed [equal-form comparison](balanced_codec_comparison_protocol_20261118.md).
Do not rerun or modify that control. No new corpus or confirmation record is used.

Change only the two bottleneck maps as specified in the
[spatial design](spatial_binary_codec_direction_20261118.md). Keep the plan at
64 values/bits, the 32/64/96/128 convolutional channels, four features at each
4-by-4 site, raster site order, and every shared convolutional initial tensor.
The new pair has 1,767,365 parameters per codec; the saved global pair has
2,030,465. This tests the architecture including its parameter difference,
not an isolated equal-parameter locality theorem. Both new modes share exact
initial tensors; only hard sign versus continuous tanh differs within the pair.

Bind the saved global initial checkpoint, result and source order. Require exact
equality of all 1,766,209 common convolutional values and of the actual sampled
index order. Keep the whole-source weighted gradient/update/reset preflight,
all 64,041 training cells and 8,629 uniform sampling representatives, and all
68,948 source occurrences in initial/final evaluation. The validation partition
still has 4,907 cells/952 forms and is reused physical development.

Keep all original absolute thresholds: .98 training/validation pooled F1, .98
validation macro F1, .80 macro exactness, .005 control retention, continuous
validation ceiling .98, no false blanks/ink and source plan-plus-correction
mean at most .60 of the existing 97.30248883216336 direct-tree events. Require
also a .02 validation-F1 gain over the saved equal-form global binary .8205924536268252.
Do not enlarge the plan, change sampling or weaken a condition during execution.
Retain only final checkpoints for qualification and preserve every raw image,
physical correction count, update/replay receipt, failed condition and old model.

Use the same one-owner project mutex and GPU0 resource admission: 48 GiB host,
20 GiB free GPU, 48 GiB disk and low entry pressure; retain 24 GiB host, at most
8 GiB allocated GPU and 8 GiB process RSS, two CPU threads, 2,400 seconds
including at most 300 seconds pressure wait. External timeout is 2,460 seconds
plus 30 seconds termination grace. Preserve foreign services and all failures.
No automatic retry, conditional language fit or goal completion follows. A full
support pass only justifies separately registering conditional image generation;
the complete language and independent confirmation requirements remain unchanged.

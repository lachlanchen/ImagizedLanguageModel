# Equal-form codec comparison with unchanged natural-frequency controls

Use the exact source, physical partition, codecs, initial CPU seed 202611195,
64-dimensional/64-bit bottleneck, 2,030,465 parameters per mode, loss, FP32,
6,000-update AdamW schedule, 128-image batch, clipping, optimizer replay,
all-source initial/final evaluation and resource limits of the completed
[codec comparison](binary_codec_comparison_protocol_20261118.md). Preserve that
failed run, all output bytes and its final models. No old control is rerun.

The [observed frequency imbalance](form_balanced_codec_direction_20261118.md)
motivates one change: uniform physical-form sampling for codec learning. Select
the first original training index for each exact mask group, sort representatives
by source index, permute with CPU seed 202611196 and cycle that order without
dropping any form. There are 8,629 representatives and still 768,000 retained
draws per arm. Preserve all 64,041 training and 4,907 validation source cells,
their aliases and original byte order; evaluation covers all 68,948 occurrences.
The hash groups are not decoder inputs or a glyph bank used at inference.

Complete-source preflight still covers all 64,041 training cells, using weights
N/(G n_g) to give each form equal objective mass. Use the original one actual
maximum-LR update, finite-gradient checks, complete-objective measurement and
exact reset before fitting. Main minibatches use the uniformly sampled complete
forms without an additional inverse-frequency weight. Two CPU gradient tests
qualify this equality. No held mask enters either training route.

Register both new continuous and binary arms from the same original initial
tensor hash. Keep the reconstruction objective, warmup/decay, clipping, update-
three exact model/optimizer replay, final-only qualification and all raw outputs.
Only the sampling/objective distribution changes. Losses across different
sampling distributions are not directly comparable quality measures. Quantify
the new binary/continuous difference and the differences from both saved natural-
frequency arms using the same physical metrics. This is reused physical
development, not an untouched confirmation set or a language experiment.

Retain every original absolute fidelity, macro exactness, continuous-ceiling,
control-retention, false-blank/ink and source-event-reduction condition. Add at
least .02 validation foreground-F1 gain over the saved natural-frequency binary
control (.8104007561580353). No condition is weakened. Even a gain can fail the
absolute conditions. A failure remains terminal for this recipe, with no extra
epochs or snapshot selection; a pass only permits separately registering a
conditional prior/correction test. No full language fit or autonomous score follows.

Use one serial GPU0 owner and the same mutex. Require 48 GiB available host,
20 GiB free GPU, 48 GiB disk and low pressure at entry; retain 24 GiB host, at
most 8 GiB GPU allocation and 8 GiB process RSS, two CPU threads, 2,400 seconds
including at most 300 seconds waiting. External timeout is 2,460 seconds plus
30 seconds termination grace. Preserve foreign services, failed executions,
original language readings and reserved confirmation data. No automatic retry.

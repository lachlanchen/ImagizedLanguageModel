# Spatial 256-value capacity comparison

Use all source images, the exact partition, uniform physical-form sampling,
complete-source weighted gradient/update/reset preflight, initial CNN seed,
source order, optimizer replay, 6,000-update schedule, FP32, all-source evaluation
and resource limits of the completed
[spatial-64 comparison](spatial_codec_comparison_protocol_20261118.md).
Preserve the original final checkpoints and output arrays; no old control is rerun.

Change only features per spatial site, from four to sixteen on the fixed 4-by-4
grid. Both new modes have 1,770,449 parameters and retain all 1,766,209 common
CNN initial values. Control uses 256 continuous tanh values; candidate uses
256 hard bits with the existing straight-through gradient. Both new modes
share exact initial tensors. Keep the same equal-form batch indices, losses,
warmup/decay, clipping and final-only qualification. Do not add noise, change
loss weights, source multiplicity or stopping rules.

Bind the saved spatial-64 common initial checkpoint, source-order file and
completed result. Verify actual shared CNN values and source order. Preserve
every original source occurrence in initial and final evaluation, every new
256-bit plan/continuous value and every XOR correction event count. Final
checkpoints record the spatial codec architecture, sixteen features per site
and 256-bit width. No target encoder enters generation of a future image.

Keep the original absolute conditions: binary training and validation pooled
F1 .98, validation macro F1 .98, macro exactness .80, no false blanks/ink,
continuous validation ceiling .98 and .005 retention. Add .02 validation-F1
gain over the saved spatial-64 binary .8473440743579628. Plan overhead is
**32** eight-bit events. Require their mean plus actual correction-tree events
to remain at most .60 of the existing 97.30248883216336 direct-tree source mean.
No threshold is weakened to accommodate the wider plan. Report extra capacity
and cost even if it improves fidelity. Event counts are not bytes or GPU speed.

The same 4,907 cells/952 physical forms are reused development and never enter
fitting. No reserved DRCD or other language confirmation is opened. A failed
recipe is terminal, with no extra epochs or checkpoint selection. A complete
support pass only permits separately registering conditional-image learning;
supplied-image reconstruction is not language or an autonomous answer.

Use one GPU0 owner under the project mutex. Entry requires 48 GiB host, 20 GiB
free GPU, 48 GiB disk and low pressure. Retain 24 GiB host, at most 8 GiB GPU
allocation and 8 GiB process RSS, two CPU threads, 2,400 seconds including at
most 300 seconds pressure wait. External timeout is 2,460 seconds plus 30 seconds
termination grace. Preserve foreign services, source bindings, failed executions
and original language results. No automatic retry or full language fit is admitted.

# Check positive-branch sampling before another training change

The completed final normal cohort fails both seen and held-out answering.
Training optimized the positive fixed-task velocity only. Registered image
assessment instead uses the native guidance-4 combination

\[
v_4=v_-+4(v_+-v_-)=4v_+-3v_-.
\]

Writing changes in the unoptimized negative branch can therefore affect that
sampler. This is a possible contributor, not an established explanation for
the failed answers. Neither positive-branch field alignment nor the following
small sampling diagnostic establishes language learning.

## Fixed comparison

Select positions 0, 1, 32 and 33 from the original normal assessment index:
the first complete seen passage pair and first held-out passage pair. Their
guidance-4 failures are already known. The selection is by original index,
with no search over image appearance. These are reused diagnostic cases,
not a new blinded test or a success-qualified language benchmark.

Use the same final export and the same initial-source arm with zero-output
adapters. For each arm, first replay original position 0 at guidance 4. All
captured reader-input tensors, actual decoder pixels and saved PNG pixels
must match that arm's existing original assessment generation exactly.
Preserve the replay image even if the equality fails. Stop on failure; do
not reroll or choose another case.

Then generate all four fixed cases at guidance exactly 1, preserving each
original image, seed, 1024-square geometry, 50-step scheduler and input-only
condition. The released native pipeline uses only the positive denoiser
branch at guidance 1. Keep that source implementation unchanged. Record
actual denoiser batch sizes: two for the guidance-4 replay and one for each
guidance-1 generation. The native reader still computes its two task-prefix
conditions; do not report a pruned reader or a new runtime parameter count.
Reader-input tensors for every case must equal its original guidance-4
counterpart. No expected answer, target image, teacher, OCR engine or
reference manifest enters the generation function.

There are ten new autonomous images: eight guidance-1 diagnostic answers
and two guidance-4 execution replays. There are 500 logical denoiser calls
and 600 denoiser batch elements across both arms. These are scheduled counts;
actual calls, time, complete runtime parameters and peak GPU memory must be
measured. No optimizer update or weight download is part of this diagnostic.

## Scoring and decision

Calibrate the unchanged offline OCR engine on all four exact reference pages,
then score every guidance-1 output in separate CPU processes. Retain raw
transcripts, canonical/source-accepted counts and both-correct pairs. Compare
with the same four original guidance-4 outputs from the complete assessment.
Authenticate every actual decoder tensor and PNG. Inspect all ten new original
pages separately; preparation or an OCR count does not complete raw review.

There is no checkpoint selection, guidance sweep or success threshold in this
diagnostic. A visible gain would motivate a subsequent complete matched
assessment; it would not replace the failed registered guidance-4 result.
Failure at guidance 1 would leave conditioning, representation, optimization
and sampling errors open. Do not claim that guidance caused the failure
merely because the two sampling paths differ.

## Admission and preservation

CPU preparation may run during the existing model work. Freeze this script
and protocol before any new neural call. The lightweight coordinator waits
for the exact observed condition-probe PID/start/command identity. Admit an
arm only after that probe has completed all 144 field forwards and the
original 384-image assessment has completed successfully. Neither code path
may bypass these prerequisites. GPU0's shared lock and RAM/swap checks still
apply; GPU1 and other projects are preserved.

Preserve every partial attempt and log. Use fresh directories, with no
automatic retry, resume, source patch, alternate noise or reduced canvas.
No publication renderer may race the queued diagnostic for GPU admission.
This experiment remains separate from the full ILM objective.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_flow_positive_sampling_probe.py \
  --stage prepare --training-run artifacts/native_visual_flow_learning_v1 \
  --assessment artifacts/native_visual_flow_assessment_v1 \
  --condition-probe artifacts/native_flow_condition_probe_v1 \
  --output artifacts/native_flow_positive_sampling_probe_v1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_flow_positive_sampling_probe.py \
  --stage run --output artifacts/native_flow_positive_sampling_probe_v1 \
  --wait-for-probe-pid <actual-live-condition-probe-pid>
```

Use the existing verified model Python. CPU preparation and a queued process
are not an observed guidance-1 result. The original full assessment and
condition-probe implementations remain frozen.

# Additional old-objective control for the completed image-writing pilot

This control is specified after the final-image pilot has been assessed and
before its own weights or outputs are produced. It is a retrospective
comparison with a newly specified control, not a prospectively randomized
two-arm experiment. The pilot's 15/32 seen and 13/32 held-out OCR observations
are already known. Do not tune this control after seeing its outputs.

## Fixed comparison

Start from the same recognition-treatment checkpoint, SHA256
`23332544624497e12b1c9b18f21b65a0543bea025a44eb18fa933caa5a5723af`.
Reuse the exact undistilled FLUX.2-klein-base-4B, VAE, rank-16 adapter
structure, constant generic condition, admitted frozen GOT reader and the
first 32 copying pairs. No QA examples or held-out strings are optimized.

Run 32 updates of the existing `loss_with_reader` from
`scripts/train_recognition_guided_writer.py`, without modifying that source:

    z_s = (1-s)y + s epsilon, s ~ Uniform[0,1]
    L = mean((v_theta(z_s,x,s) - (epsilon-y))**2)
        + .05 1[s<=.5] CE(reader(native_decode(z_s-s v_theta)), text+EOS).

Retain the existing cached target features and decode/preprocessing function.
Replace only the input features with `encode_reference` from the verified
native-autocast path, matching the completed final-image pilot. This holds the
identified input-precision correction common to both continuations. It does
not isolate all differences between their composite objectives.

Reset model RNG to 20268401. Restore all adapter tensors from the declared
checkpoint and initialize fresh FP32 AdamW state, matching the pilot:
learning rate 1e-5, weight decay .01 and global clipping one. Do not restore
the earlier optimizer. Use the same CPU example and noise-seed streams
(20268402 and 20268403); reproduce all 32 entries of the retained pilot
journal exactly. For each example, initialize the control's device noise/time
generator with that update's declared noise seed. Its latent noise and time
draws follow the old objective. Equal seed keys do not make these draws equal
to the full sampler's realized initial noise or stochastic process.

The budget matches optimizer updates and example exposure, not FLOPs, number
of transformer evaluations, wall time or total loss weighting. The pilot
differentiates through 50 native steps and uses different loss coefficients.
Report costs separately; do not call this a compute-matched efficiency test
or attribute any difference uniquely to trajectory differentiation.

## Execution checks

Before optimization, reproduce the first Chinese and English starting outputs
with the unchanged native inference helper, original inputs/seeds and 50 steps.
Require equality with their saved actual PT tensors, not just OCR equality.
These two checks do not constitute a complete independent replay.

Use GPU 0 alone and the existing project lock. Freeze base, VAE and teacher
weights; require finite nonzero student gradients and no gradients on frozen
weights. Observe the input-feature gradient on the first update. Require at
least one active recognition update and changed adapters. Fail on nonfinite
values or 22 GiB allocated memory. Preserve all updates and source hashes.
Do not alter the historical pilot, teacher admission, evaluator or outputs.

## Complete assessment and decision

Use a fresh process running the unchanged `infer_paired_image_cohort.py` on all
64 original input-only records, original seeds, 50 steps, guidance one and a
512x256 answer canvas. It must load no reader, target image, labels or varying
text condition. Save actual PT pixels and unchanged PNGs. Reuse the exact
clean-reference calibration and independent CPU evaluator. Verify every PNG
against its quantized neural tensor; compare every original-order output to
the initial state and completed pilot. Review all raw images, including both
language groups, failures and regressions. Never reroll or select outputs.

Exploratory support for the final-image objective bundle requires at least
three additional held-out OCR-exact copies over this control and no lower
aggregate seen count. Otherwise no advantage over this update-matched control
is established. This rule is fixed before observing the control, but after
knowing the pilot; it is not a confirmatory significance threshold. Report
paired wins/losses and language-specific counts regardless of this decision.

Writing admission remains 30/32 correct, legible requested strings alone in
each split. Raw obvious failures can reject it independently of imperfect
OCR. Neither a comparative gain nor passing copying establishes language
understanding. A favourable comparison would require replication and reliable
unseen Chinese writing before later question-dependent language assessment.
The full independent ILM goal remains unchanged and unachieved.

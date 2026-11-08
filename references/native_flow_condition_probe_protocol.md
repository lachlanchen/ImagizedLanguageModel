# A target-free field diagnostic after the complete image assessment

Register this diagnostic before computing any paired field. It changes no
training, complete assessment cohort, sampler, score or qualification rule.
Its purpose is to distinguish a measurable conditional-field question from
the separate demand to sample a correct answer image.

## Fixed cases and information boundary

Use the first four complete normal-question passage pairs in each split of
the existing assessment index: eight seen and eight article-disjoint held-out
questions. Include the first held-out question-erased pair as an exact-zero
execution control. Select by index, never by output quality. Each normal pair
has disjoint accepted answer sets. Retain complete original input images.

Run the exported final model, then a fresh source control with zero-output
adapters and no learned adapter weights loaded. Preserve the compiled generic
condition and removed vocabulary/head/tokenizer. Use only the positive fixed
task branch, the branch optimized in training. This does not measure the
guidance-4 sampled field or replace its full native image assessment.

For each case, compute frozen features and reference latents independently
from its own input and the other question's complete input image. Matching
feature RNG seeds within a passage pair hold reference-codec randomness
common. Only then encode the reference answer as offline diagnostic supervision.
A fixed native noise draw is shared within each passage pair and across arms.
Use native training-grid indices 0 and 499, with native BF16 sigmas and
timesteps. Index 0 must have sigma exactly one. No field receives a transcript,
answer embedding, answer length or candidate identity as conditioning.

At sigma one, assert that the actual noisy field is exactly the independent
noise tensor. The target is absent from that model input. At index 499 the
noisy target contains answer information; report this secondary condition
separately and do not call it target-free.

## Primary continuous assignment statistic

For shared noise e and the two reference-coded native velocity targets
u0 and u1, obtain p0 = v(e, I0) and p1 = v(e, I1). Define M(a,b) as mean
squared coordinate error. The sum of the two correct-versus-swapped loss
advantages is

\[
\Delta=M(p_1,u_0)-M(p_0,u_0)+M(p_0,u_1)-M(p_1,u_1).
\]

Normalize by the distance between the actual native targets:

\[
A=\frac{\Delta}{2M(u_1,u_0)}
 =\frac{\langle u_1-u_0,p_1-p_0\rangle}
        {\|u_1-u_0\|^2}.
\]

The coordinate averaging cancels in the right expression. Perfect matching
gives A=1, a question-independent field gives A=0 and exactly swapped fields
give A=-1. A is unbounded and is not accuracy or a probability. Use the actual
BF16 native velocity targets converted to float64 for offline arithmetic;
do not silently replace their rounding by an idealized real-valued target.

For both directions at sigma one, require the predictions to swap bit-exactly
when the input question is swapped: the same field, condition and RNG must
not depend on which reference supplies scoring. Question-erased own/paired
fields must also be bit-identical. Save actual predictions, noise, noisy frame,
target velocities and codec latents, with complete source/seed/image bindings.
At both indices retain per-case native MSE advantages, not just averages.

This small diagnostic has no success threshold and selects no checkpoint.
Even positive held-out alignment would not establish readable sampling,
general language understanding, efficiency or historical-script capability.
Layout, partial lexical overlap and inherited pretraining remain possible
contributors. The complete paired image assessment remains primary.

## Execution order and cost

Prepare all 18 cases, hashes, seeds and the final deployment binding on CPU.
Freeze this script/protocol before model calls. Wait for the exact live
complete-assessment coordinator to exit successfully with all 384 outputs
authenticated. Its raw review remains separate. Then run one GPU0 child at
a time through the shared project lock. Preserve GPU1 and other projects.

There are 18 cases × two noise levels × two question conditions × two arms:
144 denoiser forwards. These are field probes, not 144 autonomous images.
Record complete runtime parameters, actual time and peak memory. Save all
attempts; do not automatically retry, reroll or change cases after results.
No new optimizer update, model download or language-quality claim is admitted.

The CPU metric check uses the actual saved preflight tensor geometry but
constructs perfect, ignored and swapped fields to check the algebra. Those
constructed fields are arithmetic controls, not new model observations.

The [completed arithmetic check](evidence/native_flow_condition_probe_math.json)
reproduces those values on the real saved tensor geometry without loading a
model. Source and implementation hashes are retained.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_flow_condition_probe.py \
  --stage prepare --training-run artifacts/native_visual_flow_learning_v1 \
  --assessment artifacts/native_visual_flow_assessment_v1 \
  --output artifacts/native_flow_condition_probe_v1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_flow_condition_probe.py \
  --stage run --output artifacts/native_flow_condition_probe_v1 \
  --wait-for-assessment-pid <actual-live-coordinator-pid>
```

Use the existing verified model Python. The parent has CUDA hidden and waits
for an exact observed PID/start/command identity. It admits model children only
after successful completion of the complete 384-image assessment. The parent
and its two sequential children preserve their logs and actual field tensors.
The arithmetic check has a separate `--stage math --output <new-folder>` entry.

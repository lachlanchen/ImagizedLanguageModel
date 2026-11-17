# Predictive teacher transfer fails the bounded writing preflight

Replacing teacher-state cosine by complete predictive KL does not qualify a
successor visual language model. The candidate fails all three prospective
support checks. Across all 552 target glyph positions in the three complete
training records, canonical foreground F1 falls from **0.350346** at
initialization to **0.307754**, below the unchanged cosine control's **0.336480**.
False blanks increase from zero to five; the control produces two. Lower
prediction KL on the two QA records does not compensate for worse book KL or
malformed physical writing.

| Complete training record | Positions written | Initial KL | Control KL | Candidate KL | Initial pixel F1 | Control F1 | Candidate F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 513-glyph book | 512 | 6.805294 | 5.529593 | 14.433991 | .350779 | .335282 | .307385 |
| 846-glyph QA | 5 | 9.671794 | 3.933223 | 2.085354 | .339549 | .475655 | .471154 |
| 1,042-glyph longest-native QA | 35 | 27.357561 | 25.217896 | 7.911338 | .344555 | .340211 | .295420 |

The positions written exclude the initial book context and QA prompts. All
three *complete* source records enter each arm twice, including their full
prompt/history and terminal event. The physical assay supplies correct source-
image history and next-span lengths, with the same eight-step sampler and
per-span noises in all states. It writes every next-span target, not a selected
set of easy glyphs. These F1 values measure agreement with source pixels, not
readability, semantic correctness or complete autonomous answers. Autonomous
language is unmeasured in this preflight; the preceding 792-stream failed
language result remains unchanged.

## Controlled intervention and preservation

The [protocol](predictive_state_distillation_protocol_20261118.md) and
[registration](evidence/predictive_state_distillation_registration_20261118.json)
precede execution. Both arms start from the exact final learned-context actor
and saved projection, use identical complete record order/seeds, fresh ordinary
FP32 AdamW moments and six real updates. Only the transfer term differs. An
RNG-matched check verifies the other five losses exactly unchanged at
initialization. All twelve finite optimizer updates execute, including the
full contextual body, reader, writer and training-only projection. No old
optimizer update is replayed and no development or reserved DRCD source is fit.

The frozen 151,936-by-4,096 source readout remains on CPU and is streamed for
complete-vocabulary FP32 KL and gradients. Its 622,329,856 coefficients and
training cost are additional to the 1,465,381,142-parameter deployed visual
model and its training-only projection. No teacher body or new teacher forward
is used; existing causal teacher states supply supervision. The readout and
token identities are not deployed. This is inherited text-knowledge transfer,
not language pretraining from images alone.

Both model and projection are restored exactly after each arm. The restored
model tensor hash is
`d8b7177213b4f67c9a7738739c49101a5eecd8284e31365337edd552311e5b95`,
matching the earlier final actor. The frozen readout also remains unchanged.
No successor checkpoint is retained. All owner processes exit successfully,
the project mutex is free and GPU0 returns to 16 MiB used. Foreign services are
unchanged.

The measured execution lasts **468.186 seconds**, including **180.174 seconds**
waiting for workstation memory pressure; peak allocated VRAM is **11.255 GiB**
and minimum available host RAM is **40.602 GiB**. The outer supervisor lasts
480.489 seconds; these are nested scopes and must not be added. Costs exclude
inherited pretraining, prior source extraction, the original full fit and CPU
verification. They do not establish useful matched-quality efficiency.

The [public result](evidence/predictive_state_distillation_result_20261118.json)
binds all nine complete raw output arrays, twelve update records, reset checks,
source shards and execution receipt. A separate CPU exporter recounts all
1,656 generated tiles against authenticated training-source pixels in 2.349
seconds. Pooled integer counts and F1 are separately verified from all individual
case counts. Neural KL values come from the bound execution; they are not a
second independent neural replay. Four prior CPU tests qualify the streaming
probability/gradient implementation without establishing language quality.

## Decision

Do not start a full fit, extend unchanged epochs, tune weights or select a
checkpoint from this result. This six-update adaptation is insufficient support
for the proposed replacement. It does not prove that every predictive
distillation schedule would fail, identify a unique cause of the original
language failure or prove that image-based language is impossible.

The next intervention must return to a measured limitation. The existing
[reader analysis](compact_visual_span_diagnosis_result.md) proves an affine
native-coordinate restriction in the same 384-wide feedback design; changing
the writer or the state-transfer loss does not remove that restriction. A
nonlinear feedback residual is a candidate that would remove that particular
affine-family bound while preserving the original function at zero
initialization. Its feasibility, source-target identifiability and effect on
conditional writing remain untested. It is not selected as a working repair,
and no further neural execution or full fit is admitted by this result. Reuse
completed controls and preserve the original language acceptance criteria.

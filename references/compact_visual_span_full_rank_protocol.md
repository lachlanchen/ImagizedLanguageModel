# Controlled test of a full-dimensional visual writer

The compact V1 fit completed all 8,172 updates but produced no credited
readable complete response in its 372-page evaluation. This comparison tests
one architectural change: widen the conditional flow writer from 384 to
1,024 channels while retaining the 768-dimensional codec. Preserve the
[failed control](compact_visual_span_learning_result.md) and its loader.

## Hypothesis and unchanged components

The narrow writer has a fixed velocity subspace and cannot contract initial
Gaussian noise in at least 384 codec directions. Increasing its width gives
the velocity a path through every codec dimension. This is a necessary capacity
repair to test, not an established explanation for every failed answer.

Keep the contextual foundation, codec, 384-channel feedback encoder, native
alignment, source partitions and offline teacher cache unchanged. Use the same
13,785 sequences, 108 shards, 2,380,620 causal state targets, one-epoch order,
1,024-position padded batch budget, six equally weighted objectives, AdamW
learning rate 0.0002, weight decay 0.01 and gradient norm cap 1.0. The complete
pass schedules 8,172 updates. The entire optimizer loop and source/target
assembly are AST-identical to the preserved V1 trainer.

Initialize the original compact model and training-only teacher projection
in their original order with seed 202611152, then replace the writer. Construct
the new writer in an isolated CPU random stream with that seed. Copy the
original length head exactly. Restore the caller's CPU random state and verify
that the GPU noise/time random state is unchanged. The trainer checks its
encoder, length head and new writer against the registered initial hashes
before the first update. Encoder and length hashes come from the actual V1
initial snapshot; the teacher projection's initializer and position in the
random stream are unchanged.

The writer grows from 6,900,113 to 42,575,633 parameters. Image parameters grow
from 12,872,213 to 48,547,733. The training-only projection adds 8,388,608,
giving 56,936,341 trainable parameters. Report the additional compute and
memory. The 1.409B contextual foundation and project-trained codec stay frozen;
this remains visual adaptation of inherited language knowledge.

## Capacity check, including normalization

The writer's final LayerNorm matters. With width \(m\), its output has the form

\[
v=W\operatorname{diag}(\gamma)
\frac{C h}{\sqrt{\|Ch\|^2/m+\epsilon}}+W\beta+b,
\qquad C=I-\mathbf1\mathbf1^\top/m.
\]

Its varying part lies in the columns of
\(W\operatorname{diag}(\gamma)C\), whose rank is at most \(m-1\).
Thus setting the hidden width equal to 768 would still restrict the output
even if the bare projection matrix had rank 768. The chosen width is 1,024.

Before fitting, the actual initialized projection and the LayerNorm-centered
map both have rank 768, with smallest singular values 0.076381 and 0.076379.
The single-glyph velocity Jacobian has rank 768 at the fixed test times
0, 0.5 and 0.875. Its smallest singular values are 7.94e-5, 6.12e-5 and 2.45e-5.
The audit first matches the actual FP32 velocity expression exactly, then
promotes the same weights and expression to float64 for the Jacobians and
matrix calculations. It does not prove global invertibility, a BF16 hardware
sampler identity, trainability or language quality. Repeat capacity measurements
on the retained trained weights before interpreting the comparison.

The final registration audit took 5.29 CPU seconds at 1.354 GiB peak process
RSS and initialized no CUDA context. The retained preliminary audit adds
4.20 CPU seconds. See the
[registration evidence](evidence/compact_visual_span_full_rank_registration_20261111.json).

## Fixed evaluation and decision

Use the unchanged [development cohort](compact_visual_span_evaluation_protocol.md):
32 questions in sixteen passage groups and twelve prefixes from two held-out
books. Save the initial snapshot and four shard-exposure milestones. Generate
all 120 cases at both endpoints and the 44 normal prompts at intermediate
steps, for 372 pages. Use seed 202611153, eight flow steps, a 128-glyph budget,
learned span lengths and stopping, and actual emitted-pixel feedback. Retain
blank and capped streams in every denominator. Keep the same first QA pair and
first book prefix as the fixed visual examples.

The primary comparison uses the final checkpoint's complete exact QA count
out of 32 and both-correct pairs out of sixteen, alongside readable complete
outputs and normal-minus-question-erased/passage-erased contrasts. The other
checkpoints describe the trajectory; selecting a favorable step cannot replace
the final result. Score book coherence and natural stopping separately. Freeze
opaque visual readings before opening their mappings and references. Disclose
the model review and its lack of independent human adjudication. Use the
existing paired passage-group uncertainty analysis and preserve its small-cohort
limits. A trained cold replay must reproduce the fixed book case without
training data, references or a teacher service in the answer path.

If no readable conditional complete answers are credited, this repair has not
solved the writing/language failure. Diagnose visual access, conditional writing
and generated-history drift before choosing another change. A development
improvement needs confirmation on new untouched source groups. This cohort is
already a development set; it cannot establish broad generalization or a useful
language model by itself. Same-quality efficiency and Qwen parity remain unproved.

## Execution and resource limits

Reuse the existing corpus, weights and teacher shards. The CPU registration
binds their identities, the control, code and evaluation. The trainer verifies
teacher-state hashes and offsets as each shard is loaded. Every GPU stage uses
the existing project mutex and owned-process audit. Fit admission requires
48 GiB available host RAM, low memory pressure and 20 GiB free device memory;
runtime reserves 24 GiB host RAM and caps allocated VRAM at 22 GiB. Each fit
attempt has a 10,800-second deadline and 1,800-second pressure-wait allowance.
Only an existing resource deadline with a newly committed shard permits the
same resumable attempt. Preserve optimizer/RNG state and report all attempts.
Inference retains its existing admission and the fixed generation budgets.
Run one heavy project job at a time; do not alter another project's service.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/prepare_full_rank_visual_span.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
  python scripts/run_full_rank_visual_span_pipeline.py
```

The actual registration is
`artifacts/compact_visual_span_full_rank_v1/registration.json`; completed
registration is not completed training. Fit artifacts go to
`artifacts/compact_visual_span_full_rank_fit_v1`, and the serial pipeline and
opaque review go to `artifacts/compact_visual_span_full_rank_pipeline_v1`.
Standalone deployment uses `scripts/infer_full_rank_visual_span.py` and only
the authenticated foundation, codec, input image and image-head snapshot.

The registered fit has started on GPU0. An independent inspection of one
atomically opened live checkpoint verified 419 AdamW updates over six shards,
with all optimizer counters equal to 419. Forty-four encoder tensors and
fifty-five writer tensors had changed; the feedback normalization buffers
were unchanged. All six saved objectives and parameters were finite. The
peak allocated GPU memory recorded in those shards was 4.222 GiB. Initial
encoder and writer hashes matched registration, and the runtime's GPU RNG
preservation check passed before its first snapshot. This verifies training
execution, not language quality or full-pass completion. See the
[saved-state evidence](evidence/compact_visual_span_full_rank_start_20261111.json).

After the fixed suite is complete, use the wider writer's traced replay:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/verify_full_rank_visual_span_cold.py \
  --run artifacts/compact_visual_span_full_rank_pipeline_v1/image_heads_step_0008172 \
  --checkpoint artifacts/compact_visual_span_full_rank_fit_v1/image_heads_step_0008172.pt \
  --output artifacts/compact_visual_span_full_rank_cold_v1
```

It uses the same fixed book input, syscall access audit and exact pixel/PNG,
stopping and feedback comparison as V1. The wider-writer replay is not yet run.

## Complete fit, before language scoring

The registered pass has completed all 8,172 optimizer updates and 108 unique
shards, with all 13,785 sequence exposures and 2,380,620 causal state targets
accounted for. Final optimizer counters all equal 8,172; the six objectives
and saved image parameters are finite. Its five snapshot steps remain
0, 1,956, 4,049, 6,152 and 8,172. The final image-head SHA-256 is
`c54e6ac462c112eac1ba36f5b0d51b7b05d86fbedf53da6702a0f82976533f85`.

Elapsed fit time is 1,945.56 seconds, including 150.14 seconds of resource
waiting. Peak allocated GPU memory is 4.889 GiB. The actual trained projection,
LayerNorm-centered map and three fixed local velocity Jacobians retain rank
768 in the same float64 mathematical audit. Those checks remove the identified
fixed dimensional restriction; they do not establish readable conditional
writing, global invertibility or a hardware sampler guarantee. See the
[complete fit receipt](evidence/compact_visual_span_full_rank_completed_20261111.json).
The five-checkpoint image evaluation is running; language scoring and trained
cold replay remain pending.

All five generation runs have subsequently completed: 120/44/44/44/120 pages
at the registered checkpoints, with their full controls and original streams.
A fresh traced child replayed the fixed final book case with 128 glyphs and
128 actual emitted-pixel feedback events. Pixels, PNG, termination and feedback
hashes match the suite; no forbidden training/reference paths or IP connections
were observed. The wrapper session ended with code 143 before saving its
verification receipt, while every retained child trace ends with exit zero.
CPU-only post-processing used the unchanged access audit and comparison
functions to recover the receipt from those original completed outputs.
Inference was not rerun. The wrapper's termination cause is not established.
See the [generation and cold receipt](evidence/compact_visual_span_full_rank_cold_20261111.json).
Opaque review and language scoring remain pending; cold reproducibility does
not establish a correct answer.

# Actual-context motor fit completed; language evaluation pending

The registered actual-context motor completes all 12,000 AdamW updates on
23,262 authentic cached training positions: 3,176 first positions and 20,086
later positions from 3,176 training questions. Each pool supplies 768,000
sampled target exposures. No development answer, native-helper state or
synthetic semantic mixture supplies training data.

The head has 24,477,696 trainable parameters. Its completed fit takes
109.485 seconds with no pressure wait, at 0.989 GiB peak allocated GPU memory
and at least 54.628 GiB available host RAM. The inherited reader is not loaded
during this head fit; its parameters, adapters, glyph encoder/decoder and
stopping remain fixed. Existing reader acquisition, pretraining and authentic
cache construction are excluded from this small-head cost, not from the
method's total cost.

The first launch fails before model loading or any update because the
required deterministic CUBLAS environment setting is omitted. Its terminal
status, console log and admission receipt are retained. It lasts 0.299 seconds
after stage creation; its complete validation wall time is unmeasured. The
successful launch uses the same registered code, data, seed and exposure,
with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. This is an execution correction, not
a changed training recipe. The
[recovery receipt](evidence/glyph_context_motor_admission_recovery_20261111.json)
preserves that distinction.

The [independent CPU audit](../scripts/audit_glyph_context_motor_fit.py) checks
the actual final and previous checkpoints, all optimizer parameter steps,
both source label orderings, every changed parameter tensor, finite fitted
and EMA parameters, exact training-only normalization and the unchanged
parent checkpoint. Final EMA weights alone are eligible for generation.
The [complete execution evidence](evidence/glyph_context_motor_fit_20261111.json)
binds the training/study registrations, checkpoint and history hashes.

These are optimization and execution results. Language accuracy, readable
complete answers, retention, paired-question dependence, book continuation
and cold runtime reproduction remain unmeasured for this candidate. The
unchanged actor's 6/32 proportional exact answers are still the current
language result. Finish the registered 216-stream comparison before choosing
another model change or drawing a learning-quality trend in the manuscript.

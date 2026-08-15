# V48 Development-Audit Erratum

Date: 2026-08-15

Status: frozen before the amended audit rerun

## Scope

The first strict V48 development audit completed model scoring and every
substantive integrity check, but stopped before writing its report. The cause
was a polarity error in the final integrity aggregation:

```python
"frozen_partition_opened": False
all(integrity.values())
```

The `False` value is the required state—the frozen partition remained closed—
but `all(...)` interpreted it as an integrity failure. A read-only post-mortem
rerun recovered the exception-frame dictionary and showed that every other
integrity value was `True`.

## Authorized Correction

The evaluator may make exactly one semantic correction:

```python
"frozen_partition_remained_closed": True
```

No model, checkpoint, corpus, partition, renderer, evaluator bank, audit
window, pair, baseline, metric, threshold, gate, or generation rule may
change. The final report must retain `frozen_partition_opened: false` as a
plain evidence field if that improves clarity, but the aggregated integrity
tree must use the positive invariant above.

## Source Receipt

The production checkpoint is unchanged:

```text
path    artifacts/visual_future_block_language_v48_20260815/checkpoint_final.pt
SHA-256 d281f8c8403d07b2662bc6d091145287f218727b3c8df2f2ea87da04c70165f3
update  10000
```

Its registered evaluator source is:

```text
scripts/eval_visual_future_block_language_v48.py
SHA-256 d7c2e865362715eb5eda7f23b2726031a648acafdba2a03b9b739e3d2e7e446d
```

The amended evaluator must verify that every other registered V48 source file
still matches the checkpoint exactly, that the checkpoint's original
evaluator digest equals the value above, and that the evaluator is the only
registered source mismatch. The report must record both evaluator digests and
this erratum's digest.

## Rerun Rule

Run the complete strict development audit once after committing this erratum.
Do not reuse metrics from the failed report, change a gate, select another
checkpoint, or open the frozen partition. The amended run is valid regardless
of how many scientific gates pass.


# Native continuous writing density: complete-source execution passes

Both the one-component control and eight-component candidate pass all three
complete-record update/reset checks. The fixed book has 513 images; the QA
records have 846 and 1,042. Each arm scores 552 actual next-image targets and
555 stop decisions, makes three real AdamW trial updates and resets all learned
tensors exactly. Common reader/local initial weights match, the codec remains
unchanged, and all selected input, first/last reader, local-density and stop
gradients are finite and nonzero.

| Actual whole system | One component | Eight components |
|---|---:|---:|
| Total parameters | 23,519,395 | 23,549,194 |
| Trainable parameters | 16,096,034 | 16,125,833 |
| Frozen, previously trained codec | 7,423,361 | 7,423,361 |
| Retained trial updates | 0 | 0 |

The [registered execution](evidence/continuous_autoregressive_preflight_result_20261118.json)
takes 11.962 seconds, with 1.336 GiB peak GPU allocation and 1.732 GiB process
RSS. An independent [saved-array audit](evidence/continuous_autoregressive_preflight_audit_20261118.json)
recounts all 2,208 original initial/updated output images and agrees with every
pixel count and complete-case receipt. It does not rerun the neural model.
The public export omits only the private process-command inventory; its original
report is preserved privately and bound by the audit's SHA-256.

Next-code NLL decreases in all six cases, but only slightly: the candidate's
book changes from 1.418950 to 1.418495 nats per coordinate. Most of the total
objective reduction is stop BCE. Updated supplied-history foreground F1 remains
about .22–.23, with no exact source images and ink at all four book spaces in
each arm. These are outputs after a single reset trial, not trained quality.
The density is defined over continuous coordinates, not exact image likelihood.

This result qualifies numerical/source execution of the new comparison. It
does not establish readable autonomous answers, faster language learning,
generalization, historical-form knowledge or 1,000-fold efficiency. No inherited
language parameters or teacher states are loaded; the codec's previous training
and all corpus preparation remain part of whole-system cost. The independent
ILM goal stays active. A complete-corpus comparison and its autonomous evaluation
require a separate committed binding before training.

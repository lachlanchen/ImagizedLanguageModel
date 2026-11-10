# Much of the first-form signal is present before image writing

The retrospective [source-head audit](evidence/glyph_first_policy_audit_v1/report.json)
projects every saved first-answer state in both reserved cohorts through the
original source lexical matrix. No reader forward call, optimizer update or
new answer image is produced. These are already scored questions, not new
held-out evidence. The source projection is an offline diagnostic and is not
added to the ILM runtime.

| Actual saved-state condition | Source first character matches a reference | First emitted image F1 ≥ 0.95 | Canonical exact first image |
|---|---:|---:|---:|
| First cohort, contextual adaptation | 30/32 | 14/32 | 5/32 |
| First cohort, readout only | 31/32 | 11/32 | 3/32 |
| Second cohort, unchanged contextual parent | 29/32 | 16/32 | 5/32 |
| Second cohort, balanced writer | 29/32 | 17/32 | 6/32 |

The last two conditions have bit-identical first states, as required for an
unchanged reader. With the balanced writer, twelve cases have a matching
source first character but fail the first-image F1 threshold. In the first
contextual trial that joint count is sixteen. A source first-character match
does not establish a complete correct answer; physical F1 does not establish
a Unicode identity. Both distinctions remain in the raw joint records.

The matrix is rounded BF16 then FP32 as in the earlier source diagnostic,
but this audit uses CPU FP32 accumulation. The smallest recorded top-two logit
margin is 0.15937. CPU/GPU winner equivalence has not been independently proved.
For adapted conditions this is the unmodified lexical head acting on an
adapted body state; it is not a complete unmodified-source-model run. Partial
or unsupported source winners and reference-form coverage remain explicit.

The result localizes a substantial first-form loss between the actual initial
state and emitted pixels. It does not show that language selection is solved,
that the lexical projection may be retained in deployment, or that the current
writer can be repaired just by more fitting of the same states.

An actionable next hypothesis is broader **function transfer across multi-character
source-unit regions**, with physical glyph targets and no semantic-anchor MSE.
The earlier compiler covered only single-character source-unit anchors and
failed its physical gate. A new recipe must demonstrate generalization across
actual source-state regions before a new complete-answer test; it cannot
inherit success from the diagnostic head or turn current evaluation states
into training examples. No such new fit is claimed in this result.

# Complete-sequence adaptation fails the complete-language pilot

The final continuation-only model gives **7/32 strict complete proportional
answers, 0/16 both-correct passage pairs and 14/32 readable answers**. None of
the twelve normal book prefixes produces a credited coherent complete
continuation, at either the 128- or 640-glyph cap. The original pilot and the
independent audit of all four QA-erasure arms both fail. Useful visual language
and the full ILM objective remain unproved.

The fixed trial completes 41,298 updates over three epochs of 13,766 eligible
sequences: 9,004 complete QA records and 4,762 book records, with 6,108,741
scheduled target glyph positions. Every source receives one fixed-cell and two
proportional presentations. Complete targets extend to 513 glyphs. The
[training completion audit](visual_continuation_completion_result.md) verifies
the schedule, ledger, snapshots and final optimizer states.

All 36 answer layers receive rank-eight residual adaptation, alongside a learned
continuous query, image writer and stop predictor. The prompt reader and image
interfaces stay frozen by construction. There are 20,212,225 trainable and
7,578,390,003 total deployed parameters, including 7,522,463,984 inherited visual
and contextual parameters. The actual initial checkpoint is the comparison
control; it differs from the earlier balanced actor because the continuation
model initializes fresh answer-layer residuals and a fresh query. The smaller
trainable count does not establish a small foundation model.

| Update | Fixed exact / readable / pairs | Proportional exact / readable / pairs |
|---:|---:|---:|
| 0 | 0 / 0 / 0 | 4 / 5 / 0 |
| 10,324 | 1 / 1 / 0 | 4 / 8 / 0 |
| 20,649 | 2 / 3 / 0 | 6 / 13 / 0 |
| 30,973 | 2 / 4 / 1 | 9 / 16 / 1 |
| 41,298 | 2 / 4 / 1 | 7 / 14 / 0 |

Each layout has 32 normal questions in sixteen passage pairs. A complete exact
answer must satisfy the unchanged canonical matching and natural-stop rule.
The three-quarter result is reported as a fixed intermediate measurement;
it does not replace the prescribed final checkpoint. The final model improves
some counts relative to initialization, but the improvement is not monotonic.
Against the earlier balanced actor's 6/32 exact and 16/32 readable proportional
answers, the final result has one additional exact answer and two fewer readable
answers on the reused development cohort. This is not an isolated architectural
effect or an untouched generalization result.

The final model retains both required fixed answers and six of eight union
proportional answers. It loses `qa_07_a_normal` and `qa_09_a_normal`. Final
question-erased QA scores zero exact in both layouts; passage-erased QA scores
one exact in each layout. The proportional normal-minus-best-erased difference
is six, below the registered eight. Only the per-erasure maximum requirement
passes; the exact, paired, readable, book, dependence-gap and retention gates
fail. Sparse erased successes alone do not establish useful dependence.

All twelve final 128-cap normal book streams reach the cap. At cap 640, eight
final candidate streams stop and four cap, with zero readable or coherent
complete continuations. The matched balanced parent stops in seven of twelve
normal long-book cases, also with zero readable or coherent complete
continuations. Erased long-book prefixes all stop without readable continuation.
The corpus uses record boundaries; a stop at a record boundary does not establish
a natural whole-document ending. Here, lack of readable continuations already
prevents a book-language claim.

All 732 streams, 183 opaque boards and 96 visible normal-book prefixes were
inspected before new mapping/reference scoring. Whole readings, unknown glyphs
and coherence judgments were frozen first. Nearest-neighbor enlargements preserve
the original pixels. One model reviewer had prior cohort familiarity; there is
no independent human adjudication. Canonical mismatch alone does not establish
semantic falsity. The original scorer is unchanged; the independent protocol
audit verifies every reviewed image hash and preserves its scores. A separate
CPU export recounts all 660 core records against their original aggregates.

Cumulative training child time is 75,209.326 seconds (20.89 hours), including
the preserved earlier failed execution; peak allocated VRAM is 18.568 GiB.
Cumulative candidate evaluation child time is 6,338.799 seconds and parent
long-book evaluation is 609.626 seconds, including preserved failed child
attempts. Successful evaluator reports give peaks of 17.794 and 17.786 GiB.
Training pressure waiting is 1,771.685 seconds; the successful candidate
evaluator records another 30.028 seconds. Admission queues, offline preparation,
CPU/review work and inherited pretraining costs remain separate. The full-reader
host entry requirement is 48 GiB, distinct from the single 24-GiB GPU limit.
These costs do not establish matched-quality efficiency over a local LLM.

The [public aggregate evidence](evidence/visual_continuation_learning_20261113.json)
binds the complete scores, four-arm audit, review freeze, generation reports and
model account by SHA-256. Original images, prefixes and detailed readings remain
in the local run's `review/` directory. The reproducible CPU export check is:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/summarize_visual_continuation_learning.py --verify
```

The [fixed protocol](visual_continuation_full_protocol.md) is complete and this
bounded recipe is rejected. The [next decision](visual_continuation_failure_decision.md)
preserves the full visual-language objective and requires a failure-specific
intervention rather than more epochs of the same recipe.

# Full contextual adaptation did not produce usable visual language

Both arms of the registered complete-corpus comparison fail. The final trained
context and matched frozen context each give **0/32 strict complete answers,
0/16 both-correct question pairs, 0/32 readable complete answers and 0/12 credited
coherent complete long-book continuations**. All 792 output streams have actual
visual review receipts. Readings were frozen before reference scoring.

The [public evidence](evidence/trainable_visual_context_result_20261117.json)
binds the original scores, immutable readings, 180 native-pixel boards, review
receipts, fit audits and both cold checks. The
[protocol](trainable_visual_context_comparison_protocol_20261116.md) and all its
original acceptance rules remain unchanged. The complete six-objective recipe is
rejected; no further unchanged epoch or preferred intermediate follows.

## Final behavior

| Final checkpoint, update 8,172 | Learned context | Frozen context | Required treatment floor |
|---|---:|---:|---:|
| Strict complete normal QA | 0/32 | 0/32 | 12/32 |
| Both questions correct in a passage pair | 0/16 | 0/16 | 4/16 |
| Readable complete normal QA | 0/32 | 0/32 | 24/32 |
| Credited coherent complete long books | 0/12 | 0/12 | 4/12 |
| Strict QA with question erased | 0/32 | 0/32 | At most 2/32 |
| Strict QA with passage erased | 0/32 | 0/32 | At most 2/32 |
| Normal-minus-best-erased exact count | 0 | 0 | At least 8 |
| Natural stops, normal long books | 12/12 | 2/12 | Separate from coherence |
| Capped normal long books | 0/12 | 10/12 | Retained as censored |

All five prescribed snapshots in both arms have zero strict, paired and readable
normal QA scores. Intermediate snapshots did not generate the separate long-book
cohort; their empty long-book counts are not observations of failed books.
The final learned-context QA outputs all stop naturally and contain 4–97 emitted
glyph positions. Its twelve normal long-book outputs stop after 209–415 positions.
The frozen control has 24/32 natural QA stops and eight caps. Its two naturally
stopped normal long-book outputs are blank; the other ten reach 640 positions.
Stopping changed without producing readable complete language.

Across all 396 treatment streams, 367 readings are uncertain and 29 are blank.
The control has 374 uncertain and 22 blank readings. Overall, 741 streams contain
uncertain writing and 51 are blank. A blank answer has an empty transcription,
not readability credit. The symbol `□` denotes an unreadable region rather than
one counted glyph. Isolated recognizable shapes do not establish a reliable
complete answer. Coherence is uncredited when the continuation cannot be read;
this does not identify what latent semantics the model might have represented.

Review used one OpenAI model reviewer with prior development familiarity and no
independent human adjudication. Answer images had opaque identifiers; normal
books additionally displayed their actual visible prefixes. All earlier batch
readings remain unchanged in the final freeze. Neither source weights nor
reference answers were used to repair uncertain transcriptions.

## Training and execution

Each arm completes one epoch over 13,785 real sequences and 8,172 updates. The
source account contains 2,576,239 target glyph positions and 2,380,620 saved
teacher states per arm; these are not independently recounted conditional loss
exposures or unique facts. The same fresh reader, full-rank conditional flow
writer, fixed codec, six equally weighted losses and source order are retained.
Only treatment makes all 1,409,410,048 inherited contextual parameters trainable.
The body inherits text pretraining; this is visual adaptation rather than
language learned from images alone.

| Fit measurement | Learned context | Frozen context |
|---|---:|---:|
| Whole deployed parameters | 1,465,381,142 | 1,465,381,142 |
| Trainables including training-only projection | 1,466,346,389 | 56,936,341 |
| Host optimizer moment bytes | 11,730,771,112 | 455,490,728 |
| Monitored fit seconds | 60,127.268 | 4,309.029 |
| Resource-wait seconds within fit scope | 1,421.354 | 30.029 |
| Peak allocated VRAM, GiB | 11.647 | 7.200 |
| Minimum available host RAM, GiB | 38.464 | 39.625 |

The training-only projection is not deployed, explaining why treatment trainables
can exceed whole deployed parameters. The monitored fit scopes do not include
inherited pretraining, earlier teacher preparation, all generation or failed
executions. There is no useful matched-quality efficiency result.

Both original CPU source/state audits pass. They verify source order, final
optimizer counters, moment sizes and complete deployment state against the
saved training state; they do not independently recompute the full optimizer
trajectory. Both fresh strict cold replays pass on the fixed final book case,
reproducing pixels, stopping and emitted-image feedback without IP connections,
font/tokenizer/corpus access or undeclared project-artifact reads. Each checks
one case outside the 792 language streams. These execution successes do not
qualify language. The earlier failed admission and cold attempts are preserved.

The first and last shard flow losses fall from 1.962 to .533 in treatment and
1.964 to .522 in control. These are different source batches, not a fixed-set
learning curve, and are not evidence of complete-answer improvement. The frozen
visual review is the behavioral measurement.

## Decision

Reject the complete recipe under the registered development criteria. Full
contextual adaptation was insufficient here. The result does not prove that
visual language learning, flow matching or trainable context is impossible.
It also does not establish the cause of the malformed conditional writing.

The [next failure investigation](trainable_visual_context_failure_decision_20261117.md)
concerns the learning signal at the conditional writer and the interaction of
its loss with auxiliary objectives. It admits no new fit. Proportional-layout
retention, untouched-source confirmation, historical or unencoded forms,
etymology and matched-quality savings remain unproved. The 58-page manuscript
and its existing review/redline PDFs predate this result; their earlier pending
language statements must not be read as current execution status.

To verify the public export against the retained local evidence, without model
calls or changing any readings:

```bash
python scripts/summarize_visual_context_result.py --verify
```

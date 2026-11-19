# Finding an unnamed participant in a real passage

We selected DRCD questions containing 誰, 谁, 哪位, 哪一位 or 何人 where every annotated answer is absent from the question. An independent bitmap audit confirmed that no accepted answer appears as a contiguous glyph sequence in any selected question. These are human-authored extractive questions with a surface cue for a person; the answers can also be titles, organizations or descriptions. The passage supplies answer evidence.

The training partition contains 2,508 questions on 1,895 passages and 652 within-passage pairs with different possible answers. The development partition contains 387 questions on 276 passages, including 131 qualifying pairs. Train and development have zero identical article IDs, full passage strings or full question strings across partitions. The development data have previously been used for evaluation, and earlier passage training used the DRCD training partition. [Cohort definition](evidence/natural_role_questions_20261119/cohort.json) · [DRCD and licence](https://github.com/DRCKnowledgeTeam/DRCD)

| Writer | Exact stopped answers / 387 | Both questions correct / 131 |
|---|---:|---:|
| Writer before participant exchange | 2 | 0 |
| Randomized participant training | 3 | 0 |
| Matched participant training | 4 | 0 |
| Deployed passage reader | 95 | 7 |

Scoring used each system's stored answer pixels on available cases. For the three role-study checkpoints, 268 records per model already existed and 119 further cases were decoded. A second count rendered the original human answers and matched the saved outputs directly. All eight counts agreed. The passage reader provides a stronger comparison for future studies that ask for an unnamed participant. [Measured baseline](evidence/natural_role_questions_20261119/baseline.json) · [Raw-answer recount](evidence/natural_role_questions_20261119/audit.json) · [Crossed participant writing](visual_role_writer_result.md)

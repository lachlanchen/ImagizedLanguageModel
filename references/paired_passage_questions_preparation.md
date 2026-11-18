# Writing two answers from a pictured passage

The source questions and answers in this corpus come from CMRC 2018, in which human annotators asked questions about Chinese Wikipedia paragraphs. Each new prompt asks two distinct questions about the same paragraph. Each target joins their two recorded answers with a printed semicolon. Reversing the questions preserves the complete inventory of glyphs in the prompt and changes the required order of the answer. The instruction and joining convention are authored additions. [Cui et al., 2019](https://aclanthology.org/D19-1600/) · [CMRC source and licence](https://github.com/ymcui/cmrc2018)

| Quantity | Training | Development |
|---|---:|---:|
| Disjoint source passages in the original partition | 1,702 | 425 |
| Original questions | 7,235 | 1,788 |
| Distinct retained pairs of questions | 12,423 | 3,043 |
| Requests including both orders | 24,846 | 6,086 |
| Complete targets occurring as a single passage span | 3 | 2 |

Both component answers occur in the passage. Pairs are excluded when the questions have identical pixels, one answer contains the other or exchanging the two answers leaves the target unchanged. The complete target occurs as a contiguous visible span for only two of the 6,086 development requests. This upper bound is for a writer restricted to copying a single passage span under exact-reference scoring. A reader with the actual division between the questions can run once on each. Joining the earlier reader’s two archived, independently stopped answers gives **440/6,086** exact images (220 pairs in both orders). This number benchmarks a system supplied the question split and the output composition rule. No model received the combined prompts in this preparation. The development passages are the established internal partition of the original CMRC training corpus and have appeared in earlier evaluations. The official development and reserved confirmation sources are outside this preparation. [Manifest](evidence/paired_passage_preparation_20261119.json) · [Independent corpus recount](evidence/paired_passage_preparation_audit_20261119.json) · [Original reader](mixed_ink_reading_result.md)

The packed corpus is under `data/paired_passage_questions_v1` on the training workstation. Each split stores the original passage and question pixels once and references them from an ordered pair table. The reference and input interfaces are separate. To inspect the first complete example, run

```bash
python -m scripts.read_paired_passage_questions --split development --row 0 --output /tmp/ilm-paired-passage
```

A subsequent writer can learn from the natural passage questions and the ordered composite images. Score both complete answers, both orderings, individual-question retention, and interventions that exchange one question while preserving the other. Separate the benefit of a supplied split or a deterministic join from learning to select and sequence the evidence through the deployed visual writer.

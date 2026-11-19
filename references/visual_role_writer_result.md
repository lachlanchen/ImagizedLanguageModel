# Which participant does the pictured statement describe?

A 694,152-parameter visual writer learned to answer different questions about the same two participants. Training used 1,035 predicate–complement pairs taken from Chinese claims and rendered into crossed positive and negative clauses. The output is a stopped sequence of 32 × 32 bitmaps. It can include historical forms copied directly from a pictured source. [Figure 84](../publication/ilm-nature/figures/Figure_84_visual_role_writer.png) · [Experiment protocol](visual_role_writer_protocol.md)

| Exact answer | Parent | Randomized assignments | Matched assignments |
|---|---:|---:|---:|
| Reordered statements, new names and complements, familiar layout / 404 | 0 | 186 | 368 |
| All four input combinations in that task / 101 | 0 | 0 | 80 |
| Reordered statements, new layout and wording / 404 | 0 | 161 | 297 |
| Earlier independently written judgments, both source orders / 48 | 0 | 0 | 5 |
| New independently written judgments, both source orders / 32 | 0 | 0 | 1 |
| Human-authored OCNLI judgments / 2,947 | 1,610 | 1,551 | 1,533 |
| Missing phrases in the same claims / 2,947 | 524 | 521 | 512 |
| Printed descriptions about Qin and Chu / 32 | 0 | 16 | 32 |
| Bronze descriptions about Qin and Chu / 32 | 0 | 16 | 24 |
| Return the bronze participant from its property / 32 | 0 | 0 | 0 |

Matched training raised both-order statement accuracy from 54 to 179 of 202 pairs on the familiar layout and from 36 to 123 on the new layout. The two tests use the same 101 new predicate–complement pairs. Their object strings and participant names were excluded from the role-training pool. On properly paired unfamiliar situations the measured gains were much smaller. The 95% interval over eight new situations included zero for the one-pair gain. The model still gave identical judgments on 42/48 earlier and 30/32 new pairs. [Scores and output archive hashes by cohort](evidence/visual_role_writer_20261119/result.json) · [Independent recount](evidence/visual_role_writer_20261119/audit.json)

## A correct predicate with the wrong name

The first illustrated passage gives Qin a bronze mirror and denies one to Chu. Asked about Qin, the writer gives the right affirmative predicate but copies the Chu inscription. In the second passage the names change places. The same pictured request produces the correct negative predicate and the Qin inscription. Across eight invented objects, the two authenticated bronze bitmaps were absent from the fixed writing memory, and none of the object strings had been used as a role-training complement. The matched model recovered 24 of 32 complete bronze descriptions and both orders in eight of 16 pairs. The two scripts test the same facts. The task requires grounding in the supplied images and confers no evidence that the network recognized these states historically or supplied an etymology. [Source identifiers and complete results](evidence/visual_role_writer_20261119/historical.json)

When the question asked only which participant had the property, the model gave zero exact names in either script. A follow-up scored each of the two source names followed by stop. The matched model preferred the correct complete name in 21/32 cases in each script, with both orders correct on nine printed pairs and eight bronze pairs. The twelve-of-sixteen criterion failed in both scripts. Restricting output to the two names leaves a substantial association error. In a second diagnostic, changing an independently written affirmative claim to the original source wording left exactly the same 4/40 pairs correct. Explicit negative wording for the second participant reduced the result to 1/40. Half the original claims already used the literal affirmative wording. [Name-candidate likelihoods](evidence/visual_role_writer_20261119/candidate_scores.json) · [Controlled wording changes](evidence/visual_role_writer_20261119/wording.json)

## Natural language and cost

On the 105 official examples previously compared with local Qwen3-8B, the adapted writer made 57 correct judgments and 22 exact restorations. The earlier text model scored 73 and 18; it had received the original text and no task-specific examples. Independent English dictionary writing fell from 48 to 29 of 96, Chinese from five to one, and no complete reference was recovered on 571 DuReader prompts. Historical examples and the positive training result therefore describe a specialized capability. The natural-inference count also remains below the sparse visual baseline of 1,689. [Original benchmark and Qwen comparison](visual_inference_writer_result.md)

Both fits began at the same earlier paired checkpoint and made 2,615 updates, seeing the same natural judgments, constructed squares, masks and written answers. Complete matched and randomized adaptations took 126.5 and 123.4 seconds on a single RTX 4090 D. Peak allocated training memory was below 0.2 GiB per arm, excluding the framework context and inherited training. Re-evaluating the unchanged parent reproduced the four original natural exact counts; two individual emissions in each natural decision and restoration archive differed across executions. The independent recount used the actual new archives. The results do not quantify end-to-end efficiency against an LLM. [Fit and data bindings](evidence/visual_role_writer_20261119/registration.json)

## Reproduce and run

The writer can run on CPU outside the training repository. The portable ZIP contains the complete fitted weights, the 7,715-shape memory, Python inference source and sample input/output images. The packed binary outputs can contain symbols outside the installed fonts. The original fit and corpus are stored locally. [Portable inference instructions](../docs/visual-role-writer.md)

```bash
python -m scripts.export_visual_role_writer --output /tmp/visual-role-writer.zip
unzip /tmp/visual-role-writer.zip -d /tmp/ilm-role
cd /tmp/ilm-role
python read.py --passage-image examples/bronze1_premise.png \
  --question-image examples/bronze0_query.png --output my-answer
```

From the original prepared corpus and inherited checkpoint, the following reruns both fits into a new directory and evaluates their final weights.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m scripts.train_visual_role_writer --output /path/to/new-role-run
```

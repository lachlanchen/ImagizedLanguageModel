# Question-first reading: V60 result

The registered input-order intervention fails. Question-first reading gives
21/256 exact answers (8.203%) and 23.765% multiset raster F1, versus 27/256
(10.547%) and 26.908% for matched continued context-first training. Its F1
difference is −3.143 percentage points, source-article bootstrap 95% interval
[−6.544, 0.446]; the exact-answer difference is −2.344 [−5.490, 0.810]. Neither
resolves zero. This does not establish that one order is universally better.

Normal-minus-blank-question F1 remains positive: 14.667 [10.873, 18.809] points
question-first and 19.336 [14.825, 23.650] context-first. The respective blank
exact counts are 2 and 3. Both questions are correct in 5/393 question-first
pairs versus 16/393 context-first pairs. The intervention passes four of eight
gates and its decision remains **reject_question_first_intervention**.

Both 19,597,363-parameter models start from the same completed V58 block EMA
and make 4,000 matched updates, batch four, with the same QA/natural-motor
schedule. Only the host-side prompt order changes; the visible characters,
full paragraph, answer images, parameter count and training budget agree.
Question-first passage states can depend on the question through causal
attention. Correct blanking positions and source preservation have unit tests.
Neither references, their lengths nor answer offsets enter decoding.

Context-first training takes 214.699 s, question-first 225.207 s; each peaks
at 0.966 GiB allocated CUDA memory on physical GPU 0, an RTX 4090 D. Costs
exclude the shared prior foundation. A four-update two-plus-two replay
matches uninterrupted model/EMA weights and every recorded RNG exactly.
Both production arms finish before audit. Both retain exact copying of all
256 fixed supplied motor images. The audit takes 192.723 s at 0.184 GiB.

The fixed 256 development IDs and all 393 paragraph pairs are explicitly
reused. Bootstrap groups by source article, 2,000 draws, seed 20266119.
One seed and a context-first-trained foundation limit interpretation. Image
block answers are still extractive. No general assistant, historical writing,
independent seed replication, free language generation or large-text-model
parity is established.

Checkpoints, full reports, raw answer PNGs/tensors and standalone replay live
under `artifacts/question_first_reading_v60_20261105/`.
The continued context checkpoint SHA-256 is
`84cd0721a01d8de7e8d1428d54f6feb95595a2cf7e16db1c2c2343d368ed66d6`.
The retained inference-only native PNG route uses
`scripts/infer_visual_answer_blocks.py`; output is an image block, with no
runtime tokenizer, Unicode IDs, OCR, glyph inventory or external model call.

[Protocol](question_first_reading_v60_protocol.md),
[training](../scripts/train_question_first_reading_v60.py),
[audit](../scripts/eval_question_first_reading_v60.py).

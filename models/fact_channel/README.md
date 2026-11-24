# Reading a fact and writing an answer from pixels

The input is a pictured question and one or more pictured source regions. The fact reader scores contiguous strips of ink in those regions. A separate writer uses the question and the selected pixels to generate a stopped answer image. Both models run locally. Inference requires the released weights and bitmap bank and has no external language model dependency. The input format uses aligned 32 × 32 cells. Spelling draws from a bank of 4,333 entries including two reserved positions. A glyph outside the bank can be copied from the input.

From the repository root, run the first English paper example with an unused output directory.

~~~bash
python -m scripts.write_fact_channel \
  --input models/fact_channel/examples/statements-1/input.json \
  --output /tmp/ilm-fact-answer
~~~

The directory contains the actual answer PNG, the intermediate fact PNG and a receipt with pixel hashes, span offset and stop flag. Use the other saved inputs to reproduce the Chinese examples and the requests for the oracle and bronze forms of 言. A typed question and one or more passage arguments are rendered into the same cell representation. The full-page control uses the direct arm. The human arm was trained on human evidence and uses its own reader at inference. The predicted arm was trained on the reader's predictions plus their surrounding region.

## Training and evaluation

Use the repository's pinned SQuAD, QA2D, E2E, CrossWOZ, CMRC and DRCD source files. The two earlier phrase confirmation reservations are inputs to the exclusion list. Regenerating the training corpus checks its contents against the release receipt.

~~~bash
python -m scripts.prepare_fact_channel_sources
python -m scripts.prepare_region_writer
python -m scripts.prepare_phrase_confirmation
python -m scripts.prepare_marginal_confirmation
python -m scripts.prepare_fact_channel
python -m scripts.audit_fact_channel_data
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_fact_channel reader direct human
CUDA_VISIBLE_DEVICES=0 python -m scripts.fit_predicted_fact_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_fact_channel
CUDA_VISIBLE_DEVICES=0 python -m scripts.transfer_fact_channel
~~~

The predicted-writer entry point prepares the train-time fact cache before resetting the writer seed so every arm starts from identical weights. Completed checkpoints are verified and reused. To train from scratch, use a separate checkout and move the three writer and reader fit directories to a backup outside that checkout. Retain the palette and readback files. All three writers see 101,364 human answer wordings four times. The auxiliary reader sees 39,057 questions from disjoint pages and queries. The held material contains 192 English and 192 Chinese queries, in 96 pairs per language. Physical input identities were recounted independently. [Corpus](../../references/evidence/fact_channel_v1/corpus.json) · [Split verification](../../references/evidence/fact_channel_v1/physical-split-audit.json) · [Study design](../../references/fact_channel_protocol.md)

## What the released weights can do

On 192 unseen questions in each language, the full-page, fact-only and fact-plus-region systems produced 4, 10 and 9 exact stopped English statements and 2, 29 and 13 exact Chinese short answers. The Chinese reader alone returned 29 accepted answers. For a new pictured input, provide the query and source regions in the example manifest format. Each image contains aligned 32 × 32 cells. The output directory includes the independently generated answer image even when it is incorrect. The released examples use the fact-plus-region arm by default. Select `--arm human` to run the strongest fact-only pipeline with autonomous reading. The systems failed the bilingual promotion rule and transfer to book continuation. [Held scores, limitations and controls](../../references/fact_channel_result.md)

# Choosing a visible span and writing an answer

The two released heads reuse the pictured-page reader, bitmap bank and answer writer in `models/fact_channel`. Each head has 70,144 learned parameters. The whole-span pipeline uses 1,052,296 parameters including the fixed reader and writer. It reads aligned 32 × 32 cells, identifies a span within a source region, then generates a stream of answer glyphs and a stop decision. It can copy arbitrary input ink and spell with the learned writer's existing bitmap bank.

From the repository root, run either released PNG example into a new directory.

~~~bash
python -m scripts.write_fact_channel --arm whole \
  --input models/fact_span_ranking/examples/exam/input.json \
  --output /tmp/ilm-bus-answer
python -m scripts.write_fact_channel --arm whole \
  --input models/fact_span_ranking/examples/statements/input.json \
  --output /tmp/ilm-reading-answer
~~~

The output includes the selected fact image, generated answer image and physical hashes. The Chinese example recovers the date of a bus service change. The English example identifies the affected person and writes a sentence that reverses the intended role. Four more manifests exercise both period requests for 言 in each language. Select `--arm endpoints` to use the additive control. Select `--arm human` to run the original fact-only system with its own reader. Typed input uses `--question` and one or more `--passage` arguments. [Measured results and calibration](../../references/fact_span_ranking_result.md)

## Training and measurement

Prepare the pinned bilingual reader corpus as described in the fact-channel release. The ranking preparation restores the hashed QAMR source where necessary. Only training questions contribute gradients. Keep a fresh private artifact directory for an independent reproduction. Released fits are checked and reused. To refit in a separate checkout, move the `endpoints` and `whole` fit folders and the private epoch-resume files aside before training. The frozen reader, writer and bank are required at deployment. The quantized comparator is only needed for the last benchmark. [Base weights and data preparation](../fact_channel/README.md)

~~~bash
python -m scripts.prepare_fact_ranking_confirmation
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_fact_ranking endpoints whole
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_fact_ranking
CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect_fact_ranking_frames
python -m scripts.figure_fact_ranking
python -m scripts.audit_fact_ranking
CUDA_VISIBLE_DEVICES=0 python -m scripts.compare_fact_ranking
~~~

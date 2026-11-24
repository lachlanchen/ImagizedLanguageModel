# Comparing candidate answers on a pictured page

The independent and competing models accept question and passage PNGs made of ordered 32 × 32 cells. A fixed reader proposes 32 spans and the fitted scorer selects one. The independent image writer draws a reply and may stop. The previous signed model remains the general baseline. The release includes positive examples, a wrong answer to a story question, and a birthplace question answered with a profession. [Measured results](../../references/optical_competition_result.md)

~~~bash
python -m scripts.write_optical_competition --device cpu --mode competing \
  --input models/optical_competition/examples/story-question-1/input.json \
  --output /tmp/ilm-coral-reply
~~~

The output includes the answer PNG, the selected evidence PNG and the stop/bitmap receipt. Use a new output directory for each invocation. Change the manifest to `story-question-2/input.json` for the other question on the same story. The CLI also accepts `--question` and repeated `--passage` arguments for typed input. Select `--mode independent` for the equal-size control. Both depend on the released directional reader, shared fact-channel writer and whole-span ranker. [Parent weights and inputs](../directional_reading/README.md)

Use the pinned parent corpus and earlier reservations, the official MLQA/DROP sources and the archived FairytaleQA commit identified in the reservation script. Reproduction uses a separate experiment checkout to avoid writing private caches over the original training bank. The matched comparison and CPU inference work without an external teacher. The semantic benchmark requires the pinned Qwen vision and judging weights.

~~~bash
python -m scripts.prepare_optical_competition
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_optical_competition
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_optical_competition evaluate intervene judge
CUDA_VISIBLE_DEVICES=0 python -m scripts.inspect_optical_answer_ceiling enumerate judge
python -m scripts.audit_competition_split
python -m scripts.figure_optical_competition
~~~

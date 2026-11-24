# Testing a joint score for the start and end of an answer

Does a learned interaction between the two ends of a written fact help the reader select that fact? For a fixed page and question the previous score adds a start, an end and a learned length bias. After length correction its mixed difference across any four legal intervals is zero. The rank-eight product readout can express the missing interaction. A matched additive control has an identical parameter count and receives the same schedule, initial shared weights, visual inputs and loss. [Lean proof and counterexample](../../formal/span_list/CompetingFacts.lean) · [Prespecified experiment](../../references/endpoint_interaction_protocol.md)

Both new models solved both human questions on the same 18/128 new English and Chinese pages. Their parent solved 20. Each individually answered 100/256 correctly, up from 92 for the parent. The product and additive models emitted identical ink on 237 questions and different ink on 19. Two correct answers were gained and two were lost relative to the additive control. Given the question about movie categories the product reader omitted “second and”. On the Chinese title question, its additional final mark completed 《大目犍連變文》. Both cases were replayed from question and page PNGs on CPU. [All outputs and paired results](../../references/evidence/endpoint_interaction_v1) · [Input and output images](examples)

Each fit drew 128,000 pictured questions and updated 237,349 parameters. The learned scalar multiplying the endpoint correction reached 0.0612 for the additive model and 0.3941 for the product model. The weights differ, and the comparison failed its criterion for model selection. The product fit took 41.42 seconds and 6.30 kJ of device energy on a single RTX 4090 D. These costs cover the additional fit. The shared upstream writer's cost is reported separately. [Weights and optimizer measurements](../../references/evidence/endpoint_interaction_v1/product.json)

Run the title example from the repository root.

```bash
p=models/endpoint_interaction/examples/title
python -m scripts.write_paired_evidence --arm product \
  --question-image "$p/question.png" --question-cells 15 \
  --passage-image "$p/page.png" --passage-cells 516 \
  --output /tmp/ilm-title
```

The verified reply is **《大目犍連變文》**. Pixel generation also preserves forms absent from the writing palette. The optional text readout recognizes 29,458 uniquely rendered Han glyphs in the bundled font index. Ambiguous or unfamiliar input shapes can remain readable in the image while their text transcription contains a replacement character.

The experimental endpoint options are `additive` and `product`. To reproduce the comparison, first prepare the competing-facts corpus and its upstream splits, then use the commands below. Completed checkpoints are verified and reused. During a new fit the optimizer, model, random state and exact schedule position are checkpointed locally every thousand steps.

```bash
python -m scripts.reserve_endpoint_pages
python -m scripts.train_endpoint_interaction
python -m scripts.replay_endpoint_interaction
```

# Writing a visual fact list with two latent readings

These research checkpoints add a 10,434-parameter head to the released 308,422-parameter contrast reader. Both inputs are images of aligned 32 × 32 cells. The output concatenates selected source ink with a visible separator. Run the following command from the repository root with a fresh output directory.

```bash
python -m scripts.write_fact_binding --arm mixture --question-image models/fact_binding/examples/187689/question.png --question-cells 5 --page-image models/fact_binding/examples/187689/page.png --page-cells 557 --output /tmp/ilm-binding-example
```

The directory contains the written PNG, packed bits and the selected source intervals. Select `--arm linear` for the matched additive correction and `--arm parent` for the frozen reference reader. The new head improved pairing with supplied candidates and lost autonomous accuracy. [Held results](../../references/fact_binding_result.md)

With the original public corpora prepared as described in the [multi-part reader](../multiple_ink_runs/README.md), run the previous complete-set preparation, witness audit and binding reservation before fitting. The released fits use exactly the registered order. Use a separate output directory for a new fit.

```bash
python -m scripts.prepare_complete_fact_sets
python -m scripts.audit_fact_set_binding
python -m scripts.prepare_fact_binding
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_fact_binding mixture --output-dir artifacts/my-binding-fit
```

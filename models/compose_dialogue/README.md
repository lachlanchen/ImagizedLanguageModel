# Writing answers from pictured words

This 547,125-parameter English writer accepts a passage PNG, a question PNG and optional pictured history. It writes its answer to a PNG. It can stop, copy a visible word with a boundary edit, or spell from its learned letter-image palette. Inference runs independently on CPU or CUDA. Inputs use the training typeface, aligned 32-pixel letter cells and spaces. The limits are 448 passage words, 32 question words, 48 history words, 16 output words and 24 cells in a spelled word.

From the repository root, try the visual checkpoint on one GPU. The CLI renders typed fields before the visual model sees them. On the example below the released model answers **a compass**, omitting the notebook. `--passage`, `--question` and `--history` accept image paths in place of their text counterparts. Omit `--device cuda` to run on CPU.

```bash
scripts/with_ilm_gpu python -m scripts.answer_dialogue_surface \
  --model models/compose_dialogue/ink/lists --device cuda \
  --passage-text 'Mina packed a compass and a notebook. Omar carried the rope.' \
  --question-text 'What did Mina pack?' --output /tmp/answer.png
```

The default decoder chooses one action and word or letter at a time. `--marginal-copy` combines the probabilities of all locations yielding the same visible word. `--copy-search` proposes stopped sequences of visible words using eight live prefixes and emits the best sequence. It has no spelling route. The original writer spelled successfully elsewhere; all five words in the illustrated multi-span answer came from copying. [Measured outcomes](../../references/compose_dialogue_result.md)

## Reproduce the four fits

Install PyTorch with CUDA, NumPy, PyArrow, Pillow and safetensors. Prepare the pinned CoQA sources and official development scorer, then obtain the MultiSpanQA archive from its authors. Preparation checks the archive SHA-256 and builds story-disjoint cohorts. This model uses the released cold parents. The uniform continuation controls reside under `models/dialogue_evidence/{ink,byte}/plain`. [CoQA preparation](../dialogue_ink/README.md) · [Cold parent training](../dialogue_surfaces/README.md) · [MultiSpanQA dataset and CC BY-SA 4.0 terms](https://multi-span.github.io/)

```bash
mkdir -p data/raw/multispanqa
curl -fL https://multi-span.github.io/data/MultiSpanQA_data.zip -o data/raw/multispanqa/release.zip
python -m scripts.prepare_compose_dialogue
scripts/with_ilm_gpu python -m scripts.train_compose_dialogue --output artifacts/my-composition-models
scripts/with_ilm_gpu python -m scripts.evaluate_compose_dialogue \
  --models artifacts/my-composition-models --output artifacts/my-composition-evaluation
```

The `ink` and `byte` directories each contain `assembly` and `lists` checkpoints. Each fit used 48,000 presentations. `weights.json` records the parent, data and batch sequence hashes, training trace, preparation and optimizer costs. Use a new evaluation directory for each run. The following commands score the released weights and reproduce the illustrated example from the original pictured fields.

```bash
scripts/with_ilm_gpu python -m scripts.evaluate_compose_dialogue --output artifacts/my-released-evaluation
scripts/with_ilm_gpu python -m scripts.test_compose_copy_decoder --control artifacts/my-released-evaluation --output artifacts/my-marginal
scripts/with_ilm_gpu python -m scripts.score_compose_answer_pairs --output artifacts/my-pairs
scripts/with_ilm_gpu python -m scripts.check_compose_answer_source --original artifacts/my-pairs --output artifacts/my-exchange
scripts/with_ilm_gpu python -m scripts.search_composed_answers --control artifacts/my-released-evaluation --output artifacts/my-search
python -m scripts.audit_composed_answers --render-example artifacts/my-example
scripts/with_ilm_gpu python -m scripts.answer_dialogue_surface --device cuda \
  --model models/compose_dialogue/ink/lists --passage artifacts/my-example/examples/list-composition-passage.png \
  --question artifacts/my-example/examples/list-composition-question.png --output /tmp/list-answer.png
cmp /tmp/list-answer.png references/evidence/compose_dialogue_v1/list-composition-lists.png
```

# Answering from a blank image strip

The masked visual writer reads a pictured passage, question and optional preceding exchange. It predicts the number of answer words, fills the most confident blank slots, then uses their ink to fill the remaining slots. After four passes it returns a PNG and the pass in which it committed each word. It can copy visible word images or spell a word one physical letter cell at a time. The matching causal writer generates its reply from left to right. Each checkpoint runs independently of a text teacher or external service. The current geometry is aligned English print in the training font, up to 448 passage words, 32 question words, 48 history words and 16 output words.

## Run the released model

The following example uses existing pictured inputs. `--passage-text`, `--question-text` and `--history-text` render typed fields before visual inference. The default device is CPU. The project launcher also supports CUDA and NVML on the shared 4090 D workstation.

```bash
example=references/evidence/dialogue_ink_v1/examples
scripts/with_ilm_gpu python -m scripts.answer_dialogue_surface --device cuda \
  --model models/word_canvas/ink/masked --passage "$example/energy-passage.png" \
  --question "$example/energy-question.png" --history "$example/energy-history.png" \
  --output /tmp/answer.png
```

Use `ink/causal` for the matched continuation or `byte/{masked,causal}` for the ASCII baselines. The masked writer has 574,021 parameters and the causal writer 547,125. The JSON next to each checkpoint records SHA-256, the parent, seed, schedule, costs and distinct training turns.

## Reproduce the comparison

Install PyTorch with CUDA, Pillow, PyArrow, NumPy and safetensors. Prepare the preceding CoQA and MultiSpanQA split. `prepare_word_canvas` reserves 200 additional stories from the fixed ordering and checks overlap with the earlier reservation. The run fits both representations and both writing methods, serially. Each receives the same 102,400 presentations. [Data setup and original list-trained models](../compose_dialogue/README.md) · [Registered experiment](../../references/word_canvas_protocol.md)

```bash
python -m scripts.prepare_word_canvas
scripts/with_ilm_gpu python -m scripts.train_word_canvas --output artifacts/my-canvas-models
scripts/with_ilm_gpu python -m scripts.evaluate_word_canvas --models artifacts/my-canvas-models --output artifacts/my-canvas-evaluation
scripts/with_ilm_gpu python -m scripts.diagnose_word_canvas --models artifacts/my-canvas-models \
  --evaluation artifacts/my-canvas-evaluation --output artifacts/my-canvas-diagnosis
```

To recount the released scores, render the original example inputs and reproduce the answer in Figure 132, use the following commands. The image masking check also changes the future visible ink to verify that answer slots can use right context.

```bash
python -m scripts.audit_word_canvas --output artifacts/my-canvas-replay --inputs artifacts/my-canvas-inputs
scripts/with_ilm_gpu python -m scripts.check_word_canvas
scripts/with_ilm_gpu python -m scripts.answer_dialogue_surface --device cuda \
  --model models/word_canvas/ink/masked --passage artifacts/my-canvas-inputs/draft-passage.png \
  --question artifacts/my-canvas-inputs/draft-question.png --history artifacts/my-canvas-inputs/draft-history.png \
  --output /tmp/researcher.png
cmp /tmp/researcher.png references/evidence/word_canvas_v1/draft.png
```

The model wrote **Amanda Lenhart**, committing the names on passes one and three. Across the reserved stories it matched 693/3,155 human answers. It matched zero of 443 references requiring new phrasing. One left-to-right visual answer combined three pictured words into the previously unfitted wording **he killed himself**. [Results and limitations](../../references/word_canvas_result.md)

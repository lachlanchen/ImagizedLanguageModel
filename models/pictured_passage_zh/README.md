# Reading Chinese questions in pixels

The default checkpoint reads a pictured passage and question and writes its answer in source pixels. It learned from 21,934 additional traditional-Chinese questions. Asked when nearly 50,000 people in India could speak Sanskrit, it returned **1991年**. Asked which region began scientific study of Sanskrit, it chose a scholar instead of Europe. Both outputs are preserved. [Exact answer](traditional_examples/gained/answer.png) · [Wrong answer](traditional_examples/missed/answer.png) · [Reference for the error](traditional_examples/missed/reference.png)

Run from the repository root with PyTorch, NumPy, Pillow and safetensors installed. For a PNG, provide the occupied glyph count when the final row includes layout padding.

```bash
PYTHONPATH=. python scripts/answer_pictured_passage.py \
  --passage-image models/pictured_passage_zh/traditional_examples/gained/passage.png --passage-cells 392 \
  --question-image models/pictured_passage_zh/traditional_examples/gained/question.png --question-cells 26 \
  --output /tmp/ilm-census-answer
```

The command writes `answer.png`, packed bits in `answer.bin`, and a receipt recording the cell count and whether the model stopped. The default device is CPU. Either input can also be supplied as text with `--passage-text` or `--question-text`, which rasterizes it before inference. Image inputs use a known reading order and 32 × 32 binary glyph cells. The checkpoint contains all learned tensors and its visual landmarks. Inference requires no remote service, character dictionary or font for image inputs.

The planner scores reading actions on the prompt. A five-state dynamic program permits an instruction prefix, contiguous questions, copied separators and a learned stop. The passage network chooses the start and end of each answer. It scores ink patterns and matching runs in the pictured inputs. The emitter copies the selected passage spans and prompt punctuation. There are 101,144 stored model parameters. Source copying preserves visible shapes but restricts which answers the model can produce.

The mixed-evidence reader answered 1,270 of 3,505 individual DRCD questions and 1,111 of 3,170 official CMRC questions exactly. The preceding checkpoint answered 500 and 999. The traditional examples derive from Delta Research Center's DRCD corpus, adapted from Wikipedia (CC BY-SA 3.0); the CMRC railway examples retain CC BY-SA 4.0. [Measured transfer and limits](../../references/traditional_passage_training_result.md) · [DRCD source and terms](https://github.com/DRCKnowledgeTeam/DRCD#copyright-notice-%E7%89%88%E6%AC%8A%E8%81%B2%E6%98%8E) · [CMRC source](https://github.com/ymcui/cmrc2018)

The preceding checkpoint, `varied_planner.safetensors`, remains available through `--weights`. Its first successful short-instruction exchanged pair concerns the Guangmao Railway. [Original question](examples/first/question.png) · [Measured output](examples/first/answer.png) · [Reversed answer](examples/swapped/answer.png) · [Preceding comparison](../../references/paired_passage_result.md)

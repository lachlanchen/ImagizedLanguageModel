# Reading paired Chinese questions in pixels

The checkpoint accepts a passage and a complete prompt. Its output is a PNG assembled from the input ink. In the first successful exchanged pair under the shorter instruction, it read two questions about the Guangmao Railway and wrote “364.6公里；三茂铁路股份有限公司”. Reversing the questions exchanged the company and the length. The examples directory includes both PNG requests, both measured outputs and the first failure. [First question](examples/first/question.png) · [Generated answer](examples/first/answer.png) · [Reversed answer](examples/swapped/answer.png) · [Failed answer](examples/missed/answer.png)

Run from the repository root with PyTorch, NumPy, Pillow and safetensors installed. For a PNG, provide the occupied glyph count when the final row includes layout padding.

```bash
PYTHONPATH=. python scripts/answer_pictured_passage.py \
  --passage-image models/pictured_passage_zh/examples/first/passage.png --passage-cells 438 \
  --question-image models/pictured_passage_zh/examples/first/question.png --question-cells 31 \
  --output /tmp/ilm-railway-answer
```

The command writes `answer.png`, packed bits in `answer.bin`, and a receipt recording the cell count and whether the model stopped. The default device is CPU. Either input can also be supplied as text with `--passage-text` or `--question-text`, which rasterizes it before inference. Image inputs use a known reading order and 32 × 32 binary glyph cells. The checkpoint contains all learned tensors and its visual landmarks. Inference requires no remote service, character dictionary or font for image inputs.

The planner scores reading actions on the prompt. A five-state dynamic program permits an instruction prefix, contiguous questions, copied separators and a learned stop. The passage network chooses the start and end of each answer. It scores ink patterns and matching runs in the pictured inputs. The emitter copies the selected passage spans and prompt punctuation. There are 101,144 stored model parameters. Source copying preserves visible shapes but restricts which answers the model can produce.

On 836 official CMRC development passages, 505 of 4,800 distinct pairs had complete correct images in both orders under the shorter instruction. On individual questions, 999 of 3,170 had a complete accepted answer. Free composition, new fonts and reliable open-ended reading remain unproved. The Chinese question and passage images derive from CMRC 2018 (Cui et al.), rasterized and paired for this experiment. The examples retain the corpus's CC BY-SA 4.0 terms. [Experiment](../../references/paired_passage_result.md) · [Source corpus](https://github.com/ymcui/cmrc2018) · [License](https://creativecommons.org/licenses/by-sa/4.0/)

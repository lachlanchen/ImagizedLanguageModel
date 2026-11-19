# Answer a pictured Chinese “who” question

The inverse participant writer takes two inputs, a passage and a question, and emits its answer as a PNG. It can copy an unfamiliar inscription appearing in the passage. The model is a research checkpoint for answering short questions from supplied evidence. It often gives a fluent-looking name from the wrong part of a passage.

Unpack the archive, install the packages in `requirements.txt` if needed, and run from the unpacked directory. This example asks who has a bronze mirror. The two names in the PNG are attested bronze forms of Qin and Chu. The passage itself is a constructed reading exercise.

```bash
python read.py --passage-image examples/bronze0_premise.png \
  --question-image examples/bronze_query0.png --output mirror
```

Open `mirror/answer.png`. The writer returns the pictured Qin glyph. Asking who lacks the mirror with `examples/bronze_query1.png` returns Qin again, although the passage assigns that property to Chu.

For a longer passage, `--evidence overlap` selects the sentence sharing the largest proportion of distinct pictured glyphs with the question. The receipt records its half-open range of input cells. Whole-passage reading is the default. On 387 human Chinese questions, sentence selection raised exact whole answers from 62 to 116. Questions requiring multiple sentences can lose essential evidence when this option is enabled.

```bash
python read.py --passage-text '秦有铜镜，楚没有铜镜。' \
  --question-text '根据材料，谁有铜镜？请只写名称。' \
  --evidence overlap --output printed
```

Inputs can independently be text or PNG. Typed input uses the Noto Sans CJK Regular collection, face 2. Supply its path with `--font` if needed. PNG inputs use 32 × 32 monochrome cells in reading order. Use `--passage-cells` and `--question-cells` to exclude padding on the last row. Each run needs a fresh output directory. It writes the selected response to `answer.png`, packed pixels to `answer.bin` and the stop flag to `receipt.json`. CPU is the default. CUDA is available with `--device cuda:0`.

Inference runs locally using the included weights, writing memory and source code. Training included DRCD and OCNLI. OCNLI is distributed under CC BY-NC 2.0. [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) · [OCNLI](https://github.com/CLUEbenchmark/OCNLI)

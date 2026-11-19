# Choose an answer from pictured evidence

This research checkpoint takes a pictured Chinese question and passage and returns the answer as a PNG. Its 88,019-parameter reading head scores runs of up to 64 glyphs in the passage. A learned decision chooses between copying one run and composing a response with the original 694,152-parameter writer. Both components run locally. Typed inputs are rendered before inference. An inscription that has no font encoding can be supplied directly as image cells.

Unpack the archive and install the dependencies in `requirements.txt`. Run from the unpacked directory. The first example contains 41 occupied cells on a wrapped page. The second question asks whom Wang Mang blamed. The output records the model's own choice.

```bash
python read.py --passage-image examples/pair0_passage.png --passage-cells 41 \
  --question-image examples/pair0_q1.png --output answer
```

Open `answer/answer.png`. The same directory contains the original packed output and a receipt with the stop decision. For a typed passage, Noto Sans CJK Regular face 2 must be installed. Supply its path with `--font` if necessary.

```bash
python read.py --passage-text '木匠交給林青一把鑰匙，讓她交給館長。' \
  --question-text '誰從木匠手裡接過鑰匙？' --output key
```

Either input can independently be PNG or text. PNG cells are 32 by 32 pixels, arranged in reading order. For wrapped inputs use the occupied cell count to exclude padding. Add `--evidence overlap` to first select a single sentence by the distinct shapes it shares with the question. Whole-passage reading is the default. A question requiring several sentences may lose evidence with this option. The program uses CPU by default and accepts `--device cuda:0`. Choose a fresh output directory for each response.

This checkpoint improves short extractive answers. It frequently picks the wrong participant on unfamiliar questions, even when both names and their roles appear explicitly in a sentence. Training included DRCD and OCNLI. OCNLI is distributed under CC BY-NC 2.0. [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) · [OCNLI](https://github.com/CLUEbenchmark/OCNLI)

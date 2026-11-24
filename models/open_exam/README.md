# Chinese pictured-question writer

The reader takes images of a passage and a question and writes a stopped glyph image. Both released fits are experimental. The matched fit wrote 22 of 384 exact held exam answers and completed zero of 128 two-question pages. The six authored examples include successes and failures. The two Chinese travel pages change only the destination. The matched model writes 北京 and 广州. [Evaluation](../../references/open_exam_result.md) · [Original pages and outputs](examples/)

Run this command from the repository root with a fresh output directory. CPU inference uses the released exam weights, the shared optical prior and the glyph palette. Use `--device cuda` for the project's single-GPU path. The image interface takes row-major grids of 32 × 32 cells. The occupied cell counts exclude padding. The question supports 160 cells, the passage 600, and generation stops at the learned stop or a cap of 64 decisions.

```bash
python -m scripts.write_open_exam \
  --passage-image models/open_exam/examples/travel/original/passage.png --passage-cells 26 \
  --question-image models/open_exam/examples/travel/original/question.png --question-cells 7 \
  --output /tmp/ilm-travel
```

Read `answer.png` for the model output, `answer.bin` for the packed original cells and `receipt.json` for the emitted length and stop flag. The alternate arguments `--question` and `--passage` render typed input into the same image stream. `--arm exchanged` selects the shuffled-question control. The shared prior resides in [visual_discourse](../visual_discourse/). Rendering requires Noto Sans CJK. Corpus preparation also uses DejaVu Sans. The weights and exam evaluation are released for non-commercial research according to the source data terms. Raw exam pages remain in the private local cache. [Source and license](https://github.com/nlpdata/c3/tree/1f41682ebb684d63afe7c583c8f38e137fc8a826)

To rebuild the pinned corpus, refit in a separate directory and score that fit, use the following commands. The fit resumes from its optimizer checkpoint. Corpus preparation authenticates the regenerated data against the original receipt. The evaluator authenticates each checkpoint against its own fitting receipt. [Protocol and controls](../../references/open_exam_protocol.md)

```bash
python -m scripts.prepare_open_exam
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_open_exam matched --run-dir artifacts/open_exam_replication/matched
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_open_exam --arm matched --model-dir artifacts/open_exam_replication --output artifacts/open_exam_replication/scores
```

# A pictured question, a page and a written reply

This experimental writer reads two aligned glyph grids and emits a bitmap for each answer cell until it selects stop. On the original closed-museum page the copper mirror was in the east cabinet and the rubbing in the west. Exchanging the cabinets changed its replies to both questions. On natural Chinese examination questions it answered 23/288 exactly and completed 1/80 paired pages. English reading regressed after training. The earlier deployed bilingual reader remains available. [Measurements and limitations](../../references/branch_ink_result.md) · [All eight authored examples](examples/) · [Historical specimens and outputs](examples/history/)

From the repository root, run the packaged checkpoint on a CPU or use `--device cuda` on the project GPU. Use a new output directory. Each cell occupies 32 × 32 pixels. The cell count omits trailing grid padding. The model reads the passage and question pixels directly. Its independent shared prior and palette are in [visual_discourse](../visual_discourse/).

```bash
python -m scripts.write_open_exam --arm branch --model-dir models/branch_ink \
  --passage-image models/branch_ink/examples/shelf/0/1/passage.png --passage-cells 23 \
  --question-image models/branch_ink/examples/shelf/0/1/question.png --question-cells 11 \
  --output /tmp/ilm-cabinet
```

For this example the stopped answer is 西柜. The packed pixels in `answer.bin` have SHA-256 `7128629e1a1477cda45b9a6062fe4795657cb05220790dddd62e62555230c4d8`. View `answer.png` directly. The CLI also accepts `--question` and `--passage` for typed input. The reader uses Noto Sans CJK to render it.

The following commands reconstruct both exam cohorts, resume an independent fit and evaluate the resulting model on the original and fresh development questions and crossed historical cards. Training uses the released initial optical reader. Place the C³ corpus as described in the [original preparation](../../scripts/prepare_open_exam.py). Research use follows the [source data terms](https://github.com/nlpdata/c3/tree/1f41682ebb684d63afe7c583c8f38e137fc8a826).

```bash
python -m scripts.prepare_open_exam
python -m scripts.prepare_branch_ink
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_branch_ink --run-dir artifacts/branch_replication/branch
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_branch_ink --model-dir artifacts/branch_replication --output artifacts/branch_replication/scores
```

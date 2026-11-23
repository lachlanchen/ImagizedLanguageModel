# Training an image writer to copy and stop

This checkpoint takes a pictured passage and pictured question and returns a strip of 32-pixel glyph cells. A separate reading network supplied source-position probabilities during training. The standalone writer runs locally with its own 86.6 million parameters. Its fixed image inventory and pointer can combine modern glyphs with forms present on the source card.

Extract `interval-distilled-writer.zip` into an empty directory. With Python, PyTorch, Transformers, safetensors, NumPy, Pillow and fontTools installed, run from that directory.

```bash
python -m scripts.respond_pointing_glyphs --model writer \
  --passage examples/human_page.png --question examples/human_question.png \
  --output answer.png
```

Use `yan_oracle_page.png` and `yan_oracle_question.png` for the historical forms of 言. Use `yan_bronze_question.png` for the opposite order. The optional `--device cuda` enables GPU inference. For typed input, use `--passage-text` and `--question-text` with Noto CJK installed. Input strips require explicit cell boundaries.

To reproduce the matched training study with the repository's archived parent, source splits and local weights, run the following from a clean experiment directory layout.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.interval_teacher_corpus
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_interval_teacher replay aligned distilled
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_interval_teacher parent teacher replay aligned distilled
python -m scripts.audit_interval_teacher
```

The preparation command preserves existing experiment directories. The evaluation stores complete returned pixels, offsets and terminal decisions. The released checkpoint includes inherited text-pretrained parameters. [Measured comparisons](https://github.com/lachlanchen/ImagizedLanguageModel/blob/main/references/interval_teacher_result.md)

On 98 further CMRC passages, the writer gave 72 exact answers to 196 questions and answered both questions on 11 passages. Ordinary replay answered 63 questions and passed 11 pairs. This checkpoint is available for reproducing the experiment. The earlier mixed model remains the default. All 192 familiar instructions for returning a supplied historical form succeeded. Among 300 open answers requiring content glyphs absent from the input, zero matched the recorded reference exactly.

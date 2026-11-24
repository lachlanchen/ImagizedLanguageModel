# Read a classical Chinese character in context

Given a Chinese passage and a marked character, this 1.18-million-parameter model writes its English dictionary gloss as a PNG. For 有是言也, it writes **speech** for 言. For the final 言 in 言及之而不言, it writes **speak**. Both outputs came from a recurrent image writer trained from scratch. [Actual speech image](../../references/evidence/visual_word_senses_20261124/examples/KR1h0004_yan_1_answer.png) · [Actual speak image](../../references/evidence/visual_word_senses_20261124/examples/KR1h0004_yan_0_answer.png) · [Measured performance and limitations](../../references/visual_word_sense_result.md)

From the repository root, with Python, PyTorch, NumPy, Pillow, safetensors and fonttools installed:

```bash
python -m scripts.write_visual_gloss --page-text '有是言也' --word 言 \
  --output speech.png --transcribe
python -m scripts.write_visual_gloss --page-text '言及之而不言' --index 5 \
  --output speak.png --transcribe
```

For a scanned or rendered **grid** of 32 × 32 glyphs, give the cell position in row-major reading order. Positions start at zero. This pictured example has eight leading cells before the queried position.

```bash
sample=references/evidence/visual_word_senses_20261124/examples
python -m scripts.write_visual_gloss --page "$sample/yan_input.png" \
  --index 8 --output speech.png
cmp speech.png "$sample/KR1h0004_yan_1_answer.png"
```

The page reader uses eight cells on each side of the marked cell. Use `--page-cells N` for a grid with padding at the end. Typed mode renders each character to ink using Noto Sans CJK regular, font index 2 by default. `--word` locates a unique character in typed text. For multiple occurrences, select the intended one with `--index`. CPU is the default. Use `--device cuda` for GPU inference. The release contains the complete learned state. It runs offline without an external LLM, tokenizer, pretrained language encoder or Unicode output table. Optional `--transcribe` matches the already emitted pixels to the output font and prints a readable version. Without that option, pictured inputs and outputs need no installed font.

**Other input faces.** Add `--match-ink` to compare the input shapes with a 1.38 MB bank of 10,765 physical training images before reading them. This option also requires SciPy. It improved the number of exact serif-input answers from 2,615 to 10,839 on the same 14,160 passages. It can substitute a familiar shape for an unseen one. UKai, a printed brush-style font, remained substantially harder. The output face stays fixed. [Typeface experiment](../../references/visual_sense_ink_matching_protocol.md) · [Model and bank hashes](weights.json)

## Reproduce the experiment

The sources are the public Kyoto UD v2.16 Classical Chinese training file and the locally installed Noto CJK regular and bold fonts. The comparison held two works out of the fit. Source and split hashes, pair indices and secondary aliases are recorded. Set `ILM_KYOTO_CONLLU` if the pinned file lives outside `data/raw/ud_classical_kyoto_r216/`. The code reuses the project GPU lock for GPU 0. Use a fresh experiment workspace when reproducing training to preserve the original checkpoint directories. [Upstream Kyoto release](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/r2.16) · [Source and font hashes](../../references/evidence/visual_word_senses_20261124/split.json) · [Registered protocol](../../references/visual_word_sense_protocol.md)

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
export ILM_WORD_SENSE_RUN_DIR="$PWD/artifacts/my-gloss-fit"
python -m scripts.prepare_visual_word_senses
python -m scripts.train_visual_word_senses motor
python -m scripts.train_visual_word_senses aligned
python -m scripts.train_visual_word_senses shuffled
for condition in aligned shuffled head_only frozen; do
  python -m scripts.evaluate_visual_word_senses "$condition"
done
python -m scripts.audit_visual_word_senses
```

The output target is an occurrence-specific English gloss, up to 24 printed cells in training. This model takes an explicitly marked word. It has yet to demonstrate conversational answers, reliable production of new English vocabulary, or the ability to explain a character's historical origin. For paired images of attested historical forms, the project has a separate source-grounded editor. [Historical image editor](../grounded_ink_editor/README.md)

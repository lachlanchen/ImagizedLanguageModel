# Ask about a classical Chinese passage in pictures

With the passage 執大象 from chapter 35 of the *Dao De Jing*, ask for the main verb and the model draws **hold**. Ask for its direct object and it draws **image**. Both inputs can be pictures, and the native response is a strip of English glyph images. The model selects the source location itself. [Passage](../../references/evidence/asked_gloss_v1/hold_page.png) · [Pictured question](../../references/evidence/asked_gloss_v1/order_question1.png) · [Actual reply](../../references/evidence/asked_gloss_v1/hold_free1.png)

From the repository root, using Python with PyTorch, Pillow, NumPy, safetensors and fonttools:

```bash
samples=references/evidence/asked_gloss_v1
python -m scripts.answer_visual_gloss --page "$samples/hold_page.png" \
  --question "$samples/order_question1.png" --output image.png
cmp image.png "$samples/hold_free1.png"
```

For typed input the front end renders the text in Noto Sans CJK. The reported source cell is zero-based. The default device is CPU. Add `--device cuda` for GPU. To print an optional transcription of the answer pixels:

```bash
python -m scripts.answer_visual_gloss --page-text 執大象 \
  --question-text 'Translate the main verb in this line and ignore the direct object in this line' \
  --output hold.png --transcribe
```

**Available output modes.** By default the model draws an unrestricted sequence of characters in its learned English output face. `--known-glosses` keeps the output within the dictionary task's 2,193 attested training glosses. This option keeps a freely written member that ends correctly. Otherwise it evaluates every training glyph sequence including its stop and draws the highest-probability one. It can still select a wrong word. For 知其雄 the object changes from *malloun* to *male*. For 守其雌 the change is from *troust* to *turn*; the reference is *female*. [Original image comparison](../../publication/ilm-nature/figures/Figure_120_asked_gloss.png)

The input image format is a grid of 32 × 32 cells in row-major order. For a padded last row, specify `--page-cells` or `--question-cells`. The released reader was trained on clauses of 3–32 characters in one printed source face. Its two grammatical questions concern the main verb and its direct object. It has substantial difficulty with everyday paraphrases. The two required weight files total 1,345,457 learned parameters. The 164,480-parameter focus reader is in this directory; its English writer is shared with the marked-character model. Neither model calls an external LLM. [Contextual writer](../visual_word_senses/README.md) · [Held-out scores and limits](../../references/asked_gloss_result.md)

## Train and evaluate

Use the pinned Kyoto UD 2.16 train, development and test CoNLL-U files. The training frontend also uses the visual writer's authenticated split and Noto CJK fonts. All six role-question fits run on one 4090 D. Fitting the shared English writer from scratch has its own commands. The commands below use a fresh run directory. [Kyoto release](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/r2.16) · [Writer training](../visual_word_senses/README.md) · [Focus protocol](../../references/asked_gloss_protocol.md) · [Matched order comparison](../../references/ordered_gloss_question_protocol.md)

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
export ILM_KYOTO_DIR="$PWD/data/raw/ud_classical_kyoto_r216"
export ILM_KYOTO_CONLLU="$ILM_KYOTO_DIR/lzh_kyoto-ud-train.conllu"
export ILM_ASKED_GLOSS_DIR="$PWD/artifacts/my-question-run"
export ILM_ASKED_GLOSS_RESULTS="$ILM_ASKED_GLOSS_DIR/evidence"
export ILM_ASKED_GLOSS_RELEASE="$ILM_ASKED_GLOSS_DIR/release"
export ILM_WORD_SENSE_RUN_DIR="$ILM_ASKED_GLOSS_DIR/writer-split"

python -m scripts.prepare_visual_word_senses
python -m scripts.prepare_asked_gloss
python -m scripts.train_asked_gloss all
python -m scripts.evaluate_asked_gloss
python -m scripts.train_ordered_gloss all
python -m scripts.export_asked_gloss "$ILM_ASKED_GLOSS_RELEASE"
python -m scripts.evaluate_ordered_gloss
python -m scripts.probe_asked_gloss_paraphrases
python -m scripts.evaluate_attested_gloss
```

The three original fits compare answer supervision, supplied training locations and exchanged answers. The three order comparisons use pooled or recurrent questions and a recurrent exchanged-answer control. The cached image likelihoods are reused across fits.

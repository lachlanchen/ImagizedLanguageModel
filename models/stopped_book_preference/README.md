# Running the visual book-answer comparison

The writer accepts a Chinese passage and a Chinese or English question as two black-on-white horizontal strips of 32 × 32 glyphs. It writes a third strip and predicts its own stop. A printed-glyph inventory and a pointer to the original page supply the output images. The causal body inherits text pretraining. Qwen reads training examples during label preparation and runs separately as a text comparison.

From the repository root, with the local training artifacts present, run the complete image inference path on CPU:

```bash
export PYTHONPATH=.
sample=references/evidence/stopped_book_preference_20261124/examples
python -m scripts.respond_pointing_glyphs \
  --model artifacts/stopped_book_preference_v1/preference/page \
  --passage "$sample/success_page.png" \
  --question "$sample/success_question.png" \
  --output /tmp/ilm-book-answer.png --device cpu
```

The example folder includes the complete passage, question, actual emitted strip and accepted reference. Replace `success` with `failure` to examine an incorrect book answer. Pass `--device cuda` to use an available GPU. The ordinary model is under `artifacts/bilingual_novel_v1/bilingual/page` and the matched control under `artifacts/stopped_book_preference_v1/replay/page`. The released bilingual model also has an offline package. [Released writer](../bilingual_novel_writer/README.md)

## Training and evaluation

The training inputs are the earlier bilingual writer, its 49,283 rendered supervision records, and the human-annotated novel training assignments. The corpus builder verifies the training digest. Dataset preparation and the inherited checkpoint are documented with the bilingual model. For a fresh experiment, use a separate working copy with those prerequisites and a fresh `artifacts/stopped_book_preference_v1` output directory. On this workstation the training programs take the project lock on GPU 0. [Bilingual data and training](../../references/bilingual_novel_protocol.md)

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
python -m scripts.mine_stopped_book_replies
python -m scripts.select_book_rivals
python -m scripts.review_stopped_book_labels
python -m scripts.review_stopped_book_labels --apply
python -m scripts.train_stopped_book_preference preference replay
python -m scripts.evaluate_stopped_book_preference
python -m scripts.audit_stopped_book_pairs
python -m scripts.audit_book_question_interaction
python -m scripts.benchmark_stopped_book_preference
python -m scripts.paired_book_transfer
python -m scripts.figure_stopped_book_preference
```

The committed review decisions correspond to the archived sample. A different sample needs a new passage-by-passage review before fitting. The mining and timed text-comparison commands use the local services and checkpoint paths identified by their scripts. Image inference needs only the writer checkpoint and its Python dependencies. The deployable interface expects segmented glyphs. Raw page segmentation, free drawing of a previously unseen glyph absent from the source, and unrestricted prose generation are separate capabilities. [Protocol](../../references/stopped_book_preference_protocol.md) · [Results](../../references/stopped_book_preference_result.md)

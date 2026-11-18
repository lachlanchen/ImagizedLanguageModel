# Classical Chinese visual editor

## Ordered source copying

The experimental neural writer reads a pictured classical Chinese strip and emits modern Chinese glyph images. Four complete examples, including two errors, appear in [Figure 66](../../publication/ilm-nature/figures/Figure_66_anchor_gap.png). The packaged model scored two faithful answers on 24 reviewed passages. A finite-gap version scored two on the same cohort. Both accept the same image interface. Output directories contain the PNG, packed pixels and a receipt.

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method anchor-gaps --image references/evidence/anchor_gap_20261119/45_source.png --output /tmp/ilm-neural-gap-answer
PYTHONPATH=. python scripts/infer_visual_translation.py --method finite-gaps --text '虽欲得请之，邻国不与也。' --output /tmp/ilm-finite-gap-answer
```

Training uses the prepared image pairs and ordered source correspondences. The two fit commands consume the archive produced by `scripts/prepare_ordered_copy_anchors.py`. The finite comparison also requires the frozen control and review files. Experiment scripts keep immutable outputs in fresh run directories. [Methods, data and results](../../references/anchor_gap_result.md)


## Source-only translation

The compact phrase composer reads a classical Chinese glyph strip and returns an answer image. It runs on CPU. On held-out books it can modernize short expressions, but it frequently mistranslates prose. The 32,768-pair checkpoint generated 六月，郑悼公去世。 from 六月，郑悼公卒。 The second checkpoint adds 23,104 training pairs. A neural writer is also available for the paired experiment. [Evaluation and limitations](../../references/source_only_visual_translation.md)

From the repository root, choose a new output directory.

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --text '六月，郑悼公卒。' --output /tmp/ilm-gloss
PYTHONPATH=. python scripts/infer_visual_translation.py --image references/evidence/block_translation_20261118/gain_source.png --output /tmp/ilm-from-ink
PYTHONPATH=. python scripts/infer_visual_translation.py --weights models/visual_edit_zh/composer_55k.msgpack.gz --text '冬，公孙归父如齐。' --output /tmp/ilm-large-gloss
```

Each directory contains `answer.png`, the unpadded bit-packed stream in `answer.bin`, and a receipt with cell counts and hashes. Input images contain 32 × 32 black-on-white cells in row-major order. Use `--cells N` to exclude trailing grid padding. Typed input uses the project's Noto CJK renderer. Image-mode composition requires Python 3.10+, NumPy, Pillow and msgpack. The neural comparison additionally requires Torch and safetensors and runs with `--method neural`. Both methods produce the answer without a text teacher.

Train a new composer on UTF-8 JSONL using one aligned pair per line. Either side can contain text or a path to a pictured grid relative to the JSONL file, for example `{"source":"六月，郑悼公卒。","target":"六月，郑悼公去世。"}` or `{"source_image":"old.png","target_image":"modern.png"}`. Optional `source_cells` and `target_cells` exclude grid padding. Training also uses RapidFuzz. Typed rendering uses fontTools and the project's Noto CJK font.

```bash
PYTHONPATH=. python scripts/fit_visual_translator.py paired.jsonl --output /tmp/ilm-fitted.msgpack.gz
PYTHONPATH=. python scripts/infer_visual_translation.py --weights /tmp/ilm-fitted.msgpack.gz --image pictured-source.png --output /tmp/ilm-fitted-answer
```

## Earlier visual editor

This research checkpoint accepts a pictured classical sentence and a pictured request to quote or translate. Each output cell is a 32 × 32 bitmap. Its 2,026,237 parameters include the source and question encoders, recurrent editor, glyph renderer and a predictor for the number of insertions allowed at each source position. The output generator runs independently on CPU and reads no corpus, tokenizer, glyph table or font.

From the repository root, with PyTorch, NumPy, Pillow and safetensors installed, reproduce one of the exact held-out edits using only a picture of the source and a typed instruction. Typed input uses Noto Sans CJK and the repository's rasterizer.

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --source-image references/evidence/visual_edit_20261118/death_source.png \
  --question-text '请翻译成白话文' --output /tmp/ilm-classical-death
```

Open `answer.png` for the writing. The output directory also contains the packed bitmaps and the action and stop receipt. Supply a new output directory for each invocation. Use `--source-text` for typed passages and `--question-image` for a pictured instruction. Each input PNG must consist of ordered 32-pixel cells. Set `--source-cells` or `--question-cells` to exclude unused final padding. The model was trained on a single upright typeface and aligned cells. The supported tasks are quotation and classical-to-modern editing. Use `请原样抄录古文` for quotation. The optional `--joint-choice` setting evaluates the alternative action scorer.

On confirmation chapters the default decoder reproduced 2,045/2,045 renderable quotations and 30/2,048 reference translations. Three translations longer than eight glyphs combined visible source writing with a newly drawn Han glyph. The simpler local-edit control had lower overall edit distance. The checkpoint is available for reproducing these measurements. [Experiment, controls and training commands](../../references/visual_edit_translation_result.md)

SHA-256 of `editor.safetensors` is `705f020c14a1d2b81f55c4b4fc2650cff94e9710b5ded2ffd7251c618a0e025e`.

## Reproduce a measured change of wording

The optional likelihood-trained decoder contains the 242,901 updated parameters from the matched training experiment. With the pictured question to translate, it changed 何谓气？ into 什么是气？ on a previously unused chapter. It matched four complete references on 2,048 requests in that set. Add the following arguments to the invocation above, replacing its source image and output directory. [Original images](../../publication/ilm-nature/figures/Figure_49_edit_contrast.png) · [Results and controls](../../references/edit_contrast_result.md)

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --decoder-update models/visual_edit_zh/likelihood_decoder.safetensors \
  --source-image references/evidence/edit_contrast_20261118/short_source.png \
  --question-text '请翻译成白话文' --output /tmp/ilm-classical-question
```

### Reproduce the sentence-diversity experiment

Figure 50 shows the larger-corpus editor on a successful calendrical expression and on a command whose meaning it changes. This research update contains the 242,901 decoder parameters trained on 192,000 distinct additional translation pairs. On two chapter-disjoint cohorts it gave 32 and 20 exact matches per 2,048 requests. Its aggregate glyph error was slightly higher than the repeated-passage control. [Experiment](../../references/edit_diversity_result.md)

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --decoder-update models/visual_edit_zh/diverse_decoder.safetensors \
  --source-image references/evidence/edit_diversity_20261118/success_source.png \
  --question-text 请翻译成白话文 --output /tmp/ilm-solar-eclipse
```

### Generate from the visual span transducer

The research update `span_update.safetensors` holds the trainable parameters of a writer that sums over alternative alignments of pictured glyphs. On four constructed role-swap pairs the writer preferred the correct joint assignments. Its two independent responses never reproduced both recorded modern answers in a pair. The PNGs in Figure 51 show a one-glyph modernization and a failure that reached 192 output cells. [Study](../../references/visual_span_result.md)

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --decoder-update models/visual_edit_zh/span_update.safetensors \
  --span-decoder action \
  --source-image references/evidence/visual_span_20261118/changed_source.png \
  --question-text 请翻译成白话文 --output /tmp/ilm-span-modernization
```

Use `--span-decoder prefix` and a fresh output directory to sum over compatible source positions as the image is written. The receipt gives the probability of the observed prefix (including termination if reached), its maximum probability-conservation residual and the number of decisions for which a discarded within-glyph prefix could still have beaten the selected output.

### Draw a missing modern phrase at a supplied boundary

The optional phrase writer reads a classical source, the number of source cells preceding the insertion, and any available modern prefix. It generates a new 32-pixel-high strip and a stop receipt. Use the published base editor with `phrase_update.safetensors` (SHA-256 `d02cd89140ece2a741841353c0aa3341e99117096695ba6ba69a6a84b24ee8eb`). The update includes the original pictured translation instruction. This specific task was trained with aligned, nonempty insertions and a correct preceding modern prefix.

From the repository root, the following CPU example returns the two pictured glyphs 没有 and stops. The source, prefix and answer can be inspected directly in the figure.

```bash
PYTHONPATH=. python scripts/infer_visual_phrase.py \
  --source-image references/evidence/visual_phrase_20261118/writer_source.png \
  --cursor 7 \
  --prefix-image references/evidence/visual_phrase_20261118/writer_prefix.png \
  --output /tmp/ilm-phrase-example
```

Use a new output directory for each run. `--source-text` and `--prefix-text` rasterize typed input in the training typeface. Omit the prefix when the insertion begins the modern sentence. On 10,789 sites the model produced 605 exact stopped phrases, including 20 multi-glyph insertions. A local training-count rule recovered 1,676. [Study and limits](../../references/visual_phrase_result.md) · [Original output images](../../publication/ilm-nature/figures/Figure_52_visual_phrase.png)

### Generate a phrase using neighboring glyphs

The optional local-ink model estimates four-pixel patterns from nearby classical and modern glyphs and feeds back its own drawings. This CPU command writes the two pictured glyphs 我想 and stops. It takes the marked insertion boundary and the preceding modern text as inputs.

```bash
PYTHONPATH=. python scripts/infer_visual_phrase.py \
  --local-ink stream \
  --source-image references/evidence/local_ink_20261118/success_source.png \
  --prefix-image references/evidence/local_ink_20261118/success_prefix.png \
  --cursor 4 --output /tmp/ilm-local-phrase
```

Open `answer.png` in the new output directory. Use `--local-ink onset` to apply the correction only to the first glyph. The full model adds 18.95 MB of compressed training counts to the existing base editor and phrase update. It recovered 405 of 5,987 held-out insertions, against 187 for the earlier writer and 651 for a whole-phrase count baseline. It is an experimental insertion model with a supplied boundary. [Results, examples and limitations](../../references/local_ink_result.md)

### Condition on the previous strokes

This optional 200,892-byte checkpoint changes the distribution of each four-pixel pattern using the preceding pattern in the same glyph. This CPU example reproduces the three pictured glyphs 所以来 in Figure 54. The full preceding modern text and the insertion boundary are supplied.

```bash
PYTHONPATH=. python scripts/infer_visual_phrase.py \
  --local-ink stream --ink-interaction models/visual_edit_zh/conditional_ink.safetensors \
  --source-text '长男曰： 固也，弟今且赦，故辞去。' --cursor 13 \
  --prefix-text '长男说： 是啊。我弟弟很幸运在今天碰上楚王大赦，' \
  --output /tmp/ilm-conditional-phrase
```

The result is in `answer.png`. On 6,435 new held-out insertions this update recovered 455 exact phrases; the parent recovered 440 and local whole-phrase counts recovered 670. Both visual writers received 13 acceptable judgments out of 48 sampled insertions. [Figure 54](../../publication/ilm-nature/figures/Figure_54_conditional_ink_aligned.png) · [All 48 reviewed examples](../../references/evidence/conditional_ink_20261118/reading_gallery.pdf)

### Conditioning the writer on the source sentence

The two 198,164-byte residuals read the pictured source at the insertion boundary and the pooled state of its complete sentence. The example below uses the assignment-trained residual. With the supplied preceding modern text, it draws 一定 and stops. The following modern text begins with 会. The output is generated with the frozen 32-by-32 writer and its own pixel feedback.

```bash
PYTHONPATH=. python scripts/infer_visual_phrase.py \
  --local-ink stream --ink-interaction models/visual_edit_zh/conditional_ink.safetensors \
  --sentence-boundary models/visual_edit_zh/sentence_assignment.safetensors \
  --source-text '齐破晋强，秦王畏晋之强也，必重君以取晋。' --cursor 13 \
  --prefix-text '齐国败而晋国强大，秦王惧怕魏国的强大，' \
  --output /tmp/ilm-sentence-answer
```

Open `answer.png`. To run the matched likelihood control, use `sentence_likelihood.safetensors` with the same inputs. Image inputs use `--source-image` and `--prefix-image`. The learned model remains an experimental insertion writer. [Complete results and failure examples](../../references/sentence_choice_result.md)

### Learn from the first wrong glyph

This CPU example takes two images and writes the insertion 国的. The classical sentence is 今魏之辞倨而币薄。 and the supplied modern prefix is 现在魏.

```bash
PYTHONPATH=. python scripts/infer_visual_phrase.py \
  --local-ink stream --ink-interaction models/visual_edit_zh/conditional_ink.safetensors \
  --first-emission models/visual_edit_zh/first_preference.safetensors \
  --source-image references/evidence/first_emission_20261118/gained_source.png \
  --prefix-image references/evidence/first_emission_20261118/gained_prefix.png \
  --cursor 2 --output /tmp/ilm-first-writing
```

The output directory contains the glyph strip in `answer.png`, its packed pixels and the stopping decision. The matched likelihood weights are `first_likelihood.safetensors`. Text input uses the existing `--source-text` and `--prefix-text` options. The preference update recovered 2,280 exact insertions in 24,048 held-out cases. [Measured successes and failures](../../references/first_emission_result.md)

### Run an autonomous translation

This mode starts with an empty modern page and decides where and how much to write. It is available to reproduce the complete-sentence experiment. The 84-sentence evaluation yielded zero exact translations and the output frequently repeats. This example reproduces the original eight-glyph output in Figure 56d.

```bash
PYTHONPATH=. python scripts/infer_visual_phrase.py --whole-sentence \
  --source-text '少阴，少阳。' --local-ink stream \
  --ink-interaction models/visual_edit_zh/conditional_ink.safetensors \
  --first-emission models/visual_edit_zh/first_preference.safetensors \
  --output /tmp/ilm-free-sentence
```

The optional first-emission weights enable the preference writer. Omitting them runs its frozen control. The receipt distinguishes a learned stop, the 32-glyph phrase limit and the 192-glyph sentence limit. [All autonomous scores](../../references/evidence/first_emission_20261118/sentences.json)

## Reproduce the causal visual writer

This research checkpoint reads the classical source and its own emitted glyphs. To reproduce the first example of Figure 59 on CPU, run the following command from the repository root.

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method feedback --text '尝攻襄州，为孔勍所败。' --output /tmp/ilm-xiangzhou
```

The answer image reads 曾经进攻襄州，被孔勍打败。 The same example can be supplied with `--image references/evidence/visual_feedback_20261118/accepted_copy.png`. On 1,024 new book passages this writer reached its 96-cell limit 350 times. Its receipt records whether it learned to stop. Its single acceptable answer in an eighteen-passage reading was shared with the clean-history control. The shipped weights have SHA-256 `23b86fcb1b644ff1c0c5e2f9df9305f4b37144c86bc8ca4f958d230d0895900b`. [Complete results](../../references/pictured_clause_and_feedback_result.md)

## Revise a pictured first draft

The optional two-stage editor reads a classical source, makes a draft with the 55,872-pair composer, then rereads the source to make bounded edits. To reproduce the gained exact gloss in Figure 60, use a new output directory.

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method revision --text '脱因，蒙古人也。' --output /tmp/ilm-mongol-gloss
```

The result reads 脱因是蒙古人。 Use `--image references/evidence/pictured_revision_20261118/loss_source.png` to reproduce the damaged calendar date. The two files required for inference are `composer_55k.msgpack.gz` and `pictured_reviser.safetensors`. Together they occupy 6.39 MB. On 1,024 new passages the model made 49 fewer glyph edits than the draft. In the eighteen changed passages reviewed for meaning it lost two previously acceptable drafts. [Scores, limits and training commands](../../references/pictured_revision_result.md)

## Alternative glosses and pooled image probability

The two experimental methods below use the same 2,251,925-parameter checkpoint. The first reproduces the supplemented student's original decision. The second combines copy and exemplar probabilities wherever they produce identical ink. Both take real glyph images and write a PNG, packed image bytes and a stopping receipt. The image strips below are part of Figure 61.

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method paired --image references/evidence/paired_student_20261119/gain_source.png --output /tmp/ilm-paired-answer
PYTHONPATH=. python scripts/infer_visual_translation.py --method pooled --image references/evidence/paired_student_20261119/gain_source.png --output /tmp/ilm-pooled-answer
```

Pooling reduces full-stream edit distance, but both students scored zero faithful modern translations on the eighteen reviewed passages. [Results and reproduced images](../../references/paired_student_result.md)

## Run the whole-glyph likelihood writer

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method marginal --image references/evidence/visible_likelihood_20261119/speech_3_source.png --output /tmp/ilm-king-refusal
```

The supplied image reads 王曰： 不可。 The checkpoint writes 王说： 不可以。 in `answer.png` and saves the actual packed output in `answer.bin`. An optional `--text` argument rasterizes a typed source before inference. The experiments use a single printing font and fixed cells. The model has 2,251,925 parameters and runs independently from the training data, Qwen and the phrase composer. It reproduced four changed full archive glosses among 1,024 broad passages. On a separate eight-utterance probe it preserved speaker and polarity eight times. [Results](../../references/visible_likelihood_result.md) · [Independent masked readings](../../references/evidence/visible_likelihood_20261119)

### Translate an event written with historical names

The optional `bound-event` method takes a strip of aligned 32 × 32 glyph cells. This example uses the original bronze exemplars of Qin and Chu. The resulting `answer.png` says that Qin surrounded Chu, retaining both input name images. Use `predicate_5_source.png` to reverse the participants. The autonomous model runs on CPU with the packaged weights and the frozen neighbourhood table. [Measured results and limitations](../../references/visual_event_roles_result.md)

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method bound-event \
  --image references/evidence/event_roles_20261119/predicate_4_source.png \
  --output artifacts/qin_surrounds_chu
```

### Translate a pictured verb with more than one meaning

The experimental `context-phrases` writer reads a grid of 32 × 32 glyph images and writes a modern answer image. This held-out example uses 去 for distance and produces 距离寿春三十里。 Substituting `leave_source.png` gives a different reading of the same pictured verb. The method recognizes two trained centre images and uses the existing phrase composer for other material. It requires the packaged visual encoder, composer and learned phrase file. [Complete outputs and limitations](../../references/predicate_context_result.md)

```bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method context-phrases \
  --image references/evidence/predicate_context_20261119/distance_source.png \
  --output artifacts/shouchun_distance
```

## Experimental source-span writer

Use the packaged phrase-boundary weights to run the four-verb experiment on a CPU. The input is a strip of aligned 32 × 32 glyph cells. The output directory receives the answer PNG, packed ink and weight hashes.

~~~bash
PYTHONPATH=. python scripts/infer_visual_translation.py --image references/evidence/phrase_boundary_20261119/camps_source.png --method phrase-boundaries --output /tmp/ilm-camps-boundary
~~~

The example produces 两营相距九里。 The joint-span and single-cell writers each translated ten of the 29 fresh archive clauses faithfully. Five figure examples reproduce byte for byte, including two incorrect translations. [Measured results](../../references/phrase_boundary_result.md)

## Reviewed translations

Run the released 2.33-million-parameter visual writer on the changed month from Figure 67. The command takes a pictured source and writes the generated glyphs to `answer.png` in a fresh directory.

~~~bash
PYTHONPATH=. python scripts/infer_visual_translation.py --method anchor-gaps \
  --weights models/visual_edit_zh/screened_targets/screened.safetensors \
  --image references/evidence/screened_targets_20261119/changed_source.png \
  --output /tmp/ilm-shaanxi-fifth-month
~~~

This input yields 五月辛亥日，赈济陕西的灾荒。 The adjacent `book.safetensors` checkpoint uses the original translations in the matched fit. Both are experimental. [Complete results](../../references/screened_visual_targets_result.md)

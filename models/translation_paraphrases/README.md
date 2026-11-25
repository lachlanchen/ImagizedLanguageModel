# Image-to-image translation of short Chinese and English sentences

The varied-source writer reads up to 96 occupied glyph cells, including spaces, and emits a stopped answer image of up to 128 cells. The network has 1,688,480 parameters. It reads local ink densities with a learned optical encoder and writes using a fixed inventory of training bitmaps. Inference runs locally on CPU or CUDA. Typing a sentence renders it into the same 32 × 32 cells used by the PNG interface.

From the repository root, run either example with a new output directory.

```bash
python -m scripts.write_ordered_translation --model paraphrase-varied --to zh \
  --text 'The doctor was invited by the nurse.' --output /tmp/ilm-passive-zh
python -m scripts.write_ordered_translation --model paraphrase-varied --to zh \
  --image models/translation_paraphrases/examples/item-55/source.png --cells 36 \
  --output /tmp/ilm-reversed-zh
```

The first example produces **护士邀请了医生。** The second exchanges the doctor and nurse and produces **医生邀请了护士。** Each command writes the answer PNG and a receipt with the occupied length, stop decision and digest of the packed ink. The buttons, order, coordinates and spacing of arbitrary page scans require a separate layout reader. The demonstrated font and sentence lengths define this checkpoint's tested input range. [Original inputs, references and outputs](examples/) · [Training, comparisons and limitations](../../references/visual_proposition_translation_result.md)

## Cross-language image retrieval

The whole-sentence alignment model can rank a gallery of pictured sentences by similarity to a query image. A gallery is a JSON array of objects with an `image` path relative to the manifest and an occupied `cells` count. The reported cosine is a ranking score. This interface returns existing images. The generation command above writes a new sequence.

```bash
python -m scripts.retrieve_bilingual_images \
  --image models/translation_paraphrases/examples/item-54/source.png --cells 36 \
  --gallery models/translation_paraphrases/examples/gallery.json --top 4
```

In a separate test with 128 OPUS-100 images per language, projected cosine similarity identified the paired sentence on 109 English queries and 112 Chinese queries. In the four-image example above, the correct Chinese answer ranks second (cosine 0.698). The role-reversed answer ranks first (0.727). [PNG replay and measured margins](../../references/evidence/bilingual_ink_v1/retrieval_margin.json)

## Reproducing the training comparisons

Start with the ordered translation palette, prior checkpoints and prepared conversational/editorial corpora described in the earlier model release. Run the following modules in order from the repository root. The preparers check pinned dataset hashes. Training writes resumable local state under `artifacts/` and final weights and receipts under `models/`. Existing completed fits are checked against their bindings. The two-order semantic benchmark also requires the project's local quantized Qwen generator and judge. [Preceding corpus and model](../ordered_translation/README.md)

```bash
export CUDA_VISIBLE_DEVICES=0
python -m scripts.prepare_proposition_translation
python -m scripts.train_proposition_translation
python -m scripts.evaluate_proposition_translation
python -m scripts.prepare_bilingual_ink
python -m scripts.train_bilingual_ink
python -m scripts.evaluate_bilingual_ink
python -m scripts.prepare_translation_paraphrases
python -m scripts.train_translation_paraphrases
python -m scripts.evaluate_translation_paraphrases
python -m scripts.evaluate_voice_transfer
python -m scripts.audit_bilingual_retrieval
```

Use `--model proposition-likelihood` or `proposition-contrast` for the controlled event comparison, `bilingual-likelihood` or `bilingual-aligned` for the OPUS comparison, and `paraphrase-canonical` or `paraphrase-varied` for the wording comparison. The aligned model also supports `--decode forward` and `--decode roundtrip` on the same bounded candidate list. All six fits, including the controls, use the ordinary visual writer architecture.

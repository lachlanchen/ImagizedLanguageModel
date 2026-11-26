# Bilingual state prediction from glyph images

Each 3,222,529-parameter reader converts a pictured sentence into a continuous field of 512-dimensional vectors and log attention weights. The image contains aligned 32 × 32 glyph cells. At inference the model sees the pixel values and their sequence length. Its source side has no text vocabulary. The English-output model reads Chinese; the Chinese-output model reads English. Training used 28,342 real passages per direction and contextual supervision from the corresponding frozen OPUS-MT source encoder.

The following CPU command reads the supplied image of 汤姆问道。 and saves the field. It uses only the released visual reader for the selected direction.

~~~bash
python -m scripts.read_optical_states --image models/optical_state/examples/tom.png \
  --cells 5 --to en --output /tmp/ilm-tom-states
~~~

To measure the meaning in that field, add `--readout` and use a new output directory. The optional readout uses a pretrained OPUS-MT text-piece decoder and returned “Tom asked.” on this example. The independent ILM writes its own bitmap answers through [the visual-writer interface](../library_roles). The state experiment achieved 3/96 source-only complete meanings through the pretrained decoder, compared with 35/96 from that decoder's native source encoder. [Experiment and controls](../../references/optical_state_learning_result.md)

The readout and training runs used Transformers 4.57.6, PyTorch and SentencePiece. Cache the two pinned public teaching models once in the shared Hugging Face cache. The tools use that local cache on subsequent runs.

~~~python
from huggingface_hub import snapshot_download

sources = {
    "zh-en": "cf109095479db38d6df799875e34039d4938aaa6",
    "en-zh": "408d9bc410a388e1d9aef112a2daba955b945255",
}
files = ["config.json", "generation_config.json", "vocab.json", "source.spm",
         "target.spm", "tokenizer_config.json", "pytorch_model.bin", "README.md"]
for direction, revision in sources.items():
    snapshot_download("Helsinki-NLP/opus-mt-" + direction, revision=revision,
                      allow_patterns=files)
~~~

The experiment entry points are `scripts.check_glyph_measure`, `scripts.train_optical_state_reader`, `scripts.evaluate_optical_state_reader` and `scripts.audit_optical_state_readout`. Run them as Python modules on the project workstation with `CUDA_VISIBLE_DEVICES=0`. Training and assessment reuse the private aligned book images and frozen panels identified by the public receipts. [Chinese–English teacher](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) · [English–Chinese teacher](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)

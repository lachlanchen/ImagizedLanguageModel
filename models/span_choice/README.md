# Choose a pictured answer from a Chinese passage

The four span readers each identify a strip of glyphs on the input page. The trained selector pools confidence for strips with identical pixels. With the local reader and factual model installed, run the command below from the repository root. Give each PNG one horizontal row of 32 × 32 glyphs.

```sh
python -m scripts.answer_visual_fact --passage page.png --question question.png \
  --span-choice models/span_choice/choice.safetensors --direct --output answer.png
```

For an ordinary typed prompt, substitute `--passage-text` and `--question-text` for the two PNG arguments. To generate a factual statement, replace `--direct` with `--writer artifacts/role_binding_v1/writer --word-order object`. The passage and question together can contain up to 509 glyphs. All inference runs locally.

For a portable installation, extract `visual-fact-reader-writer.zip` and the new `span-choice-update.zip` into the same directory. Run from that directory with `--reader reader --model model --span-choice model/choice.safetensors --span-experts model --relation-head model/pictured-relation.npz`. The update includes the two participant scorers, pictured relation head and selector. The pretrained reader and image writer are in the base archive.

The selector matched the generic reader on 285/384 fresh direct answers and passed 63/128 complete name-exchange boards. The prior likelihood-only scorer passed 79/128 boards. These weights are an optional experimental model. [Evaluation and limitations](../../references/span_choice_result.md)


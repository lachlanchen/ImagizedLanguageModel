# Matched glyph and byte dialogue writers

Both released writers train from random initialization on the same batches and generate words by pointing to their input or spelling a sequence and stopping. Each has 547,125 trainable parameters. The visual checkpoint reads packed images of letters. The byte checkpoint reads ASCII. Both assume known word boundaries and saw the same printed English conversations. The visual model has no text vocabulary and requires no pretrained reader.

```bash
example=references/evidence/dialogue_ink_v1/examples
python -m scripts.answer_dialogue_surface --model models/dialogue_surfaces/ink \
  --passage "$example/energy-passage.png" --question "$example/energy-question.png" \
  --history "$example/energy-history.png" --output /tmp/cold-reply.png
python -m scripts.answer_dialogue_surface --model models/dialogue_surfaces/byte \
  --passage-text 'A lamp is on the table.' --question-text 'Where is the lamp?' \
  --output /tmp/byte-reply.txt
```

The CLI accepts typed input for the visual arm and renders it to glyphs. The byte arm writes raw text. For strict reproduction use the same font and 32-pixel image cells as training. The models have a 320-word passage limit, 32-word question limit and 48-word previous-exchange limit. Output can be wrong or fail to stop.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dialogue_surface \
  --output artifacts/cold-dialogue-refit/models --work artifacts/cold-dialogue-refit/measurements
```

Preparation needs the scored dialogue split and visual control decisions. The trainer retains the final weights with the model and the three intermediate weights in the measurement directory. It evaluates both splits after training each branch and reuses the same word indices and exchange assignments at every checkpoint. The comparison was registered before either fit. [Protocol](../../references/dialogue_representation_protocol.md) · [Data preparation](../dialogue_ink/README.md) · [Results and energy](../../references/dialogue_efficiency_result.md)

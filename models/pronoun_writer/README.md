# Writing two meanings of 之

The same printed 之 can be a verb of motion or a third-person pronoun. This trained residual lets the shared visual writer draw **go** for 子之武城 and **it** for 不以其道得之. It also draws **it** for 之一邦, where the registered reading is **go**. The model reads the surrounding glyph images each time. The source form and part of speech are supplied as annotations during corpus preparation. The inference network receives the visual context.

## Reproduce the images

Run from the repository root with the environment used by the visual gloss writer. In the supplied padded window the central character is at zero-based index 8.

```bash
p=references/evidence/pronoun_writer_v1/samples
python -m scripts.write_visual_gloss --writer-adapter models/pronoun_writer \
  --page "$p/held_book-motion-correct.png" --index 8 --output go.png --transcribe
cmp go.png "$p/held_book-motion-correct-aligned.png"
python -m scripts.write_visual_gloss --writer-adapter models/pronoun_writer \
  --page "$p/held_book-object_person-correct.png" --index 8 --output it.png --transcribe
cmp it.png "$p/held_book-object_person-correct-aligned.png"
```

To have the pictured question select the source character as well, use both releases together:

```bash
p=references/evidence/everyday_gloss_v1/samples
python -m scripts.answer_visual_gloss --reader-dir models/everyday_gloss \
  --writer-adapter models/pronoun_writer --page "$p/page26.png" \
  --question "$p/question1.png" --output pearl.png --transcribe
```

The complete path has 1,396,977 parameters. The 51,520-parameter residual runs before the shared English speller. Both examples above use free writing. The runtime requires the released writer weights and a CPU or compatible PyTorch device.

## Training and evaluation

The preparation code verifies the pinned Kyoto UD 2.16 corpus and earlier writer-training records. It reserves physical windows from the evaluation books. Run the older writer and classroom data preparation first on a clean checkout, following their release instructions.

```bash
python -m scripts.prepare_pronoun_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_pronoun_writer both
python -m scripts.evaluate_pronoun_writer
```

The aligned and matched control checkpoints are produced under `artifacts/pronoun_writer_v1/`. The registered split, actual scored responses and image examples accompany the report. A standard third-person training reply of *it* tests the grammatical distinction. Translating a specific person may require *him* or *her*, which also requires identifying the referent. [Results and limitations](../../references/pronoun_writer_result.md) · [Physical outputs](../../references/evidence/pronoun_writer_v1/samples)

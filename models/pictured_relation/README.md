# Reading the relation stated in the question

This update adds a small question reader to the local pictured fact model. The question “《秋潮》的作曲人是谁？” asks for the composer. The original reader selected the right name and described that person as the producer. The updated image reader recognizes the requested relation and writes 沈河是《秋潮》的作曲人。 It still makes errors in deciding which name to copy.

Extract the full visual fact model into a directory. Extract the small question-reader update into the same directory. It adds the relation weights and module and updates the inference command. The original archive preserves the previous version. From this directory run

```sh
python -m scripts.answer_visual_fact --reader reader --model model --writer writer \
  --passage examples/song_0_passage.png --question examples/song_0_question.png \
  --word-order object --output answer.png
```

The model detects the additional weights in its model directory. The command defaults to CPU. Use `--device cuda` for GPU inference. Typed inputs can replace the two PNGs via `--passage-text` and `--question-text`. The new examples include the same two-artist song before and after exchanging their names. The model reads exact local glyph shapes in the question, so performance on unfamiliar fonts needs separate testing.

When running from the source repository with the previous local weights, specify `--relation-head models/pictured_relation/pictured-relation.npz`. The original data, proof and fit procedures are documented in [the main model training notes](../role_binding/TRAINING.md).

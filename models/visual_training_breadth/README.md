# English–Chinese visual writers trained at matched exposure

Both checkpoints accept 1–96 aligned glyph images or a typed sentence rendered into the same 32×32 print cells. They emit their own ordered glyphs until the learned stop. Inference uses local numerical weights, the parent editor's printed repertoire, source marker and bilingual phrase tables. No translation teacher runs at inference.

```sh
PYTHONPATH=. python -m scripts.write_visual_training_breadth --text '我理解你的痛苦。' --to en --output answer.png
PYTHONPATH=. python -m scripts.write_visual_training_breadth --image question.png --cells 8 --to en --arm narrow --output answer.png
```

The first command reproduced the archived held answer **I understand your pain.** pixel for pixel on CPU. Use `--arm broad` for six disjoint cohorts, `--arm narrow` for six passes on one cohort, or `--arm parent` for the unmodified editor. The broad fit earned 6/96 complete-meaning credits, the narrow fit 4/96 and the parent 5/96 on the same four new books. Both trained weights can be run without the large pair or draft arrays used during fitting. The fit receipts and their public source bindings must be present. [Raw comparison](../../references/visual_training_breadth_result.md) · [Parent and dependencies](../glyph_draft_editor/README.md)

To reproduce the training comparison on a workstation containing the source corpus and local book editions, run the preparation, single-card training, evaluation and CPU audit in order. The evaluation uses the locally installed comparator and grader listed in the assessment. The public output archive can be inspected independently of these installations.

```sh
export PYTHONPATH=.
python -m scripts.prepare_visual_training_breadth
scripts/with_ilm_gpu python -m scripts.train_visual_training_breadth
scripts/with_ilm_gpu python -m scripts.evaluate_visual_training_breadth
python -m scripts.audit_visual_training_breadth
```

To compare the separate pretrained routes on an archived printed source, run `python -m scripts.write_optical_span --input source.png --cells N --to en --reader path --output translated.png` with the real occupied cell count. Use `--reader phase` for the experimental position-specific feedback. Both require the original pinned local OPUS-MT weights. The phase variant changed 31 book answers and lost more meaning credits than it gained. [Pretrained comparison and actual outputs](../../references/visual_training_breadth_result.md)

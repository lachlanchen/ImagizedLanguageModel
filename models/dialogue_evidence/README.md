# Reading a printed question and writing an answer

The visual writer takes images of an English passage, the current question and an optional previous exchange. It returns an image of its answer. Each step can stop, copy an observed word image with a boundary edit, or spell a word with the learned glyph head. The standalone checkpoint has 547,125 parameters and runs on CPU. Its reading geometry uses aligned 32-pixel cells in the training typeface and spaces between words. The released CLI accepts up to 448 passage words, 32 question words and 48 words of history. It can generate up to 16 words and 24 cells per spelled word.

From the repository root, run the released checkpoint on the supplied images.

```bash
example=references/evidence/dialogue_ink_v1/examples
python -m scripts.answer_dialogue_surface --model models/dialogue_evidence/ink/human \
  --passage "$example/energy-passage.png" --question "$example/energy-question.png" \
  --history "$example/energy-history.png" --output /tmp/visual-answer.png
```

For typed questions, use `--passage-text`, `--question-text` and optional `--history-text`. The visual interface renders each input before reading. The ordinary decoder is the released default. The optional `--predicted-interval` uses the writer's own two endpoint distributions. Its measured gain was uncertain. [Decoder experiment](../../references/evidence/dialogue_teacher_v1/predicted_interval.json)

## Training

The four releases are under `ink/{plain,human}` and `byte/{plain,human}`. Each pair continues its matching cold parent. Both arms receive human answers and endpoint supervision. The `human` arm also rewards assigning pointer mass to correct word images inside human-marked evidence. The byte arm accepts ASCII and writes text. Every checkpoint includes SHA-256, parent hash, seed, training trace, cost and the number of eligible training positions in `weights.json`.

After preparing the pinned CoQA split and installing the official development data and scorer, the following trains both visual and byte arms on one GPU. The released teacher qualification receipts select the human-supervised comparison. No teacher installation or teacher requests are needed to train these arms or run the checkpoints. NVML measures GPU energy. Set `ILM_NVML_RUNNER` if your workstation requires a compatible environment wrapper. Choose an empty output path for a new training run. [Dataset and preparation](../dialogue_ink/README.md) · [Cold parents](../dialogue_surfaces/README.md)

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dialogue_evidence --output artifacts/my-dialogue-models
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_dialogue_evidence --model-root artifacts/my-dialogue-models --output artifacts/my-dialogue-evaluation
CUDA_VISIBLE_DEVICES=0 python -m scripts.replicate_dialogue_evidence --output artifacts/my-dialogue-replications
python -m scripts.audit_dialogue_evidence
python -m scripts.check_matched_reading_cost --lean /path/to/lean
```

On 758 turns from 47 fresh stories the originally released visual pair obtained 41.08% and 41.84% word F1. Across four seeds the mean difference was +0.39 percentage points with a 95% interval of −0.23 to +1.06. The eight visual fits collectively yielded zero exact answers on 27 difficult questions across both length groups. [Complete results](../../references/dialogue_evidence_result.md)

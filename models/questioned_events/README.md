# Ask about two actions in a single pictured line

For the line 舉杯邀明月, the reader writes *cup* when asked what is raised and *moon* when asked who or what is invited. The question changes which two cells the model reads as verb and object. This verse was present in the inherited training material. The release includes a separate evaluation on 971 held multi-event clauses and ten literary extracts.

```bash
python -m scripts.answer_questioned_event --line-text '舉杯邀明月' \
  --question-text 'What is it that they invite here?' \
  --output artifacts/moon.png --transcribe
```

For image input, replace `--line-text` and `--question-text` with `--line passage.png` and `--question question.png`. Use a strip of 32 × 32 pixel cells, one character per cell in the training typeface. The source can contain 2–32 cells. For the example above the zero-based positions are 2 (邀) and 4 (月). The reply is a generated strip of English glyphs. Optional transcription compares the emitted pixels against a known alphabet.

The default head was trained on literal action names. Add `--reader models/questioned_events/described/aligned` to run the version trained with an additional description of each action. Both heads use the released 3.53-million-parameter visual action reader and writer in this repository. Each adds 65,427 parameters. Inference requires only the saved networks and input pixels.

```bash
python -m scripts.reproduce_questioned_events --output artifacts/event-replay
python -m scripts.reproduce_questioned_events --train --output artifacts/event-refit
```

Use a fresh output directory. Replay evaluates the four published heads on CPU and decodes their emitted bitmaps to recount the scores. Refitting reconstructs the exact 28,148-page Kyoto training split from public data and trains the four heads serially on GPU 0. The original two-line reader remains available separately.

The literal reader correctly located all annotated verbs and objects and wrote all their English meanings on 247 of 971 held clauses. The description-trained reader completed five of 112 pages with natural questions and zero of 21 where all complete questions were previously unseen. Neither wrote every answer on any of ten Pu Songling extracts. [Results and actual images](../../references/questioned_events_result.md)

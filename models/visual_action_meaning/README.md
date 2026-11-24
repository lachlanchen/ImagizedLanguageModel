# Reading pictured English descriptions of Chinese actions

Give the model two classical Chinese lines and an English question asking for the object of one action. It draws a short English answer and reports the selected row and the zero-based object cell. The aligned model has 3,526,604 parameters including the inherited Chinese reader and recurrent image writer. It runs locally on CPU with the released weights. The English dictionary and numerical word vectors are used during training.

From the repository root, reproduce the pictured reading in Figure 125.

```bash
python -m scripts.answer_pictured_action \
  --reader-dir models/visual_action_meaning/aligned \
  --lines '王謹受教' '命取桃子' \
  --question-text 'What is it that they give admittance to here?' \
  --output artifacts/teaching.png --transcribe
```

The model writes *teaching*. Asking “What is it that they seize with the hands here?” selects the second line but produces *you*. With the position of 桃 supplied, the same writer produces *peach*. Both normal requests accept PNG input through `--page` and `--question`. The page consists of two rows of 32-pixel cells. The optional transcription decodes known output bitmaps after generation.

```bash
python -m scripts.reproduce_visual_action_meaning --output artifacts/meaning-replay
python -m scripts.reproduce_visual_action_meaning --train --output artifacts/meaning-refit
```

Choose a new output directory for each run. The first command evaluates both released readers and the parent on CPU, including all 480 Pu Songling requests. The second rebuilds the pinned Kyoto training split and lexical targets, fits the optical word encoder and both comparators on CUDA device 0, and runs the same evaluation. Inference uses the shared model dependencies under this repository's `models/` directory. The exchanged control stores its trained tensors as a delta against the aligned reader.

The aligned reader completed 44 of 268 definition pairs, compared with 11 for the parent and six for exchanged supervision. All three completed zero of ten screened Pu Songling pairs. The previous reader remains the default. [Evidence and scope](../../references/visual_action_meaning_result.md)

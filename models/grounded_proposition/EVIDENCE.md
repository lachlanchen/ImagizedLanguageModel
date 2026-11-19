# Writing with an extracted evidence span

This model reads a pictured question and a pictured passage. A visual reader selects a span of the passage. A second model reads that span with the original input and writes an answer as an image. Both stages run locally. The models transfer language knowledge from the CKIP Chinese reader and UER Chinese GPT-2.

Extract the archive and install PyTorch, Transformers, safetensors, NumPy, Pillow and fontTools. From the extracted directory, run

```sh
python -m scripts.answer_proposition_with_evidence --reader reader --model writer \
  --passage examples/note_0_passage.png --question examples/note_0_question.png --output answer.png
```

The four examples exchange two people's names and ask about two different film credits. Each includes the image actually emitted by the model. It can repeat the wrong name or return a quotation. CPU is the default. Use `--device cuda` for GPU inference. A printed input consists of a single strip of 32 × 32 cells. For typed inputs use `--passage-text` and `--question-text` and a font that covers the requested characters.

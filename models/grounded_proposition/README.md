# Pictured Chinese proposition writer

This local model takes a printed passage and a printed question and returns a strip of answer glyphs. A learned planner can direct the same writer to produce two ordered statements. It reads the headword and the requested roles from the pictured question. The writer can draw from a finite bank of printed shapes or copy a glyph from the supplied passage. Its Chinese language prior comes from UER Chinese GPT-2. The archive includes the model and eight worked examples.

After extracting the archive, use Python with PyTorch, Transformers, safetensors, NumPy, Pillow and fontTools installed.

```sh
python -m scripts.answer_grounded_proposition --model model \
  --passage examples/compound_0_passage.png --question examples/compound_0_question.png --output answer.png
```

The three other compound requests reverse the order of the two descriptions, exchange the names in the passage, or change both. To submit ordinary text, substitute `--passage-text` or `--question-text` for the corresponding image argument. Text is rendered on the host before the model runs. For image input, use one horizontal line of 32-pixel square cells. CPU is the default, and `--device cuda` selects the GPU.

The reusable Python interface is `PlannedGlyphResponder(folder, device).answer_many(pairs)`. A pair consists of two nonempty uint8 tensors with one 128-byte row per packed 32 × 32 input cell. The call returns a packed answer tensor and a stop flag for each pair. Reuse one responder across requests to retain the visual alphabet in memory. Output is limited to 48 glyphs and up to two writing calls. Answers about open topics or unfamiliar syntax require verification. The research fits use Baidu's DuIE 2.0 under its noncommercial research terms. Source corpus files are obtained separately from the organizers.

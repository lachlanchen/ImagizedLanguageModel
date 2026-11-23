# Answering a pictured question with source ink

The reader accepts a Chinese passage and a Chinese or English question and returns a PNG containing one to eight consecutive source glyphs. It processes a horizontal row of 32-pixel cells. The optional detector proposes up to sixteen expressions referring to people. The two included neural networks contain 100,618 parameters in total and run on CPU or a single GPU. They need no text model or network service at inference.

From the repository root with PyTorch, NumPy, Pillow, fontTools, transformers and safetensors installed, run the following.

```bash
PYTHONPATH=. python -m scripts.read_book_participant \
  --page references/evidence/book_participants_20261124/example/page.png \
  --question references/evidence/book_participants_20261124/example/question0.png \
  --person-detector models/book_participant_reader/person_detector.safetensors \
  --output reply.png
```

Use `--page-text` or `--question-text` for the host-side print renderer. Omit `--person-detector` to score every distinct source strip. Add `--device cuda` for GPU inference. The example reproduces the recorded model output, including errors. Both files of trained weights are included here. [Measured performance](../../references/book_participant_reader_result.md)

The original matched comparison used the project's pinned BiPaR and DuIE research data. The following commands reproduce the data preparation and the aligned reader from a fresh training directory. The training driver preserves existing fits.

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.
python -m scripts.prepare_book_participant_questions
python -m scripts.train_book_participant_reader aligned fit
python -m scripts.train_book_participant_reader aligned reserved
python -m scripts.train_book_person_proposals prepare
python -m scripts.train_book_person_proposals train
```

The comparative evaluation also fits `independent` and evaluates the `literal` control. The detector evaluation requires both fitted question readers. On the shared workstation the verified device-counter wrapper can be supplied through `ILM_NVML_RUNNER`.

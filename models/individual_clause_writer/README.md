# Write a clause from a pictured question

The writer reads a Chinese passage and one or more questions in Chinese or English. Each question receives a separately generated Chinese glyph strip. The runner concatenates those strips in the supplied order and reports the length and stop flag for each. The model can copy visible source ink and choose from its learned print alphabet. A separate stop flag permits inspection of incomplete answers.

From a trained checkout, supply a pictured page and two typed questions. The input adapter renders the questions before inference. The PNG interface accepts a horizontal row of 32-by-32-pixel cells.

```bash
PYTHONPATH=. python -m scripts.write_pictured_clauses \
  --page references/evidence/action_binding_20261124/examples/page0.png \
  --question-text '木偶的动作由谁控制？' '木偶说话的声音来自谁？' \
  --output reply.png
```

Use `--page-text` to render a typed passage, or `--questions q0.png q1.png` for image-only inference. CPU is the default. The same command accepts `--device cuda`. Each clause can contain up to 48 generated cells. The stop flags describe sequence completion; factual accuracy is measured separately.

The model update in the local publication package applies directly to the earlier bilingual writer. Extract `bilingual-novel-writer.zip` in a directory named `parent` and `individual-clause-update.zip` in a sibling directory. From the extracted update directory run the following commands.

```bash
python scripts/install_frame_update.py --base ../parent/writer --update . --output restored
PYTHONPATH=. python -m scripts.write_pictured_clauses --writer restored \
  --page examples/page1.png --questions examples/q0.png examples/q1.png --output edited.png
```

The installer verifies the parent, change archive and reconstructed weights. For training on the project workstation, prepare the earlier paired role corpus, then run the following against its fixed parent. The teacher preparation reuses the original book targets. Each fit creates a fresh output directory.

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.
python -m scripts.extend_book_clause_targets
python -m scripts.train_individual_clause_writer independent
python -m scripts.train_individual_clause_writer aligned
python -m scripts.evaluate_individual_clause_writer parent
python -m scripts.evaluate_individual_clause_writer independent
python -m scripts.evaluate_individual_clause_writer aligned
```

The packaged image runner operates offline once its Python dependencies and weights are installed. [Study design](../../references/individual_clause_writer_protocol.md) · [Measured results](../../references/individual_clause_writer_result.md)

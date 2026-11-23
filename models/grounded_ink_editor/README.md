# Write a reply from two pictured questions

Give the model a Chinese passage, two questions and a short writing instruction. It returns the reply as a PNG. On a page reading 三藏上馬，行者引路, asking first who led the way and then who mounted the horse produces **行者引路，而三藏上馬。** The writer takes the two clauses from different places on the page, puts them in question order, draws the connective and stops. [Actual output](../../references/evidence/grounded_ink_editor_20261124/written0_1.png) · [A failed output](../../references/evidence/grounded_ink_editor_20261124/example/failure_output.png) · [Its reference](../../references/evidence/grounded_ink_editor_20261124/example/failure_reference.png)

From the repository root, run

```bash
python -m scripts.respond_grounded_ink \
  --page-text '三藏上馬，行者引路' \
  --first-text '誰在前面帶路？' --second-text '誰騎上了馬？' \
  --instruction-text '分別敘述兩件事' --output reply.png
```

Each input also accepts an image. This command reproduces the same reply byte for byte on the training workstation.

```bash
book=references/evidence/grounded_ink_editor_20261124
python -m scripts.respond_grounded_ink --page "$book/page0.png" \
  --first "$book/question1.png" --second "$book/question0.png" \
  --instruction "$book/example/instruction1.png" --output reply.png
cmp reply.png "$book/written0_1.png"
```

For a historical page, use the original optical reader. This example returns two attested shapes for 言, joined by 與 and followed by 。. Swapping the first and second question images reverses the requested stages. [Oracle then bronze](../../references/evidence/grounded_ink_editor_20261124/historical/wording0_0.png) · [Bronze then oracle](../../references/evidence/grounded_ink_editor_20261124/historical/wording0_1.png)

```bash
plate=references/evidence/yan_page_responses_20261119
python -m scripts.respond_grounded_ink --reader original \
  --page "$plate/plate.png" --page-cells 34 \
  --first "$plate/oracle_question.png" --second "$plate/bronze_question.png" \
  --instruction-text '列出兩種字形' --output yan.png
```

**Inputs and weights.** Images use row-major grids of 32 × 32 cells, black ink on white. For padded page grids, supply the number of occupied cells. Each question and page together must fit within 509 cells. The short names instruction used for training was 列出兩個人名. Typed inputs require the selected font to cover the text. Direct PNG input preserves historical ink outside the font repertoire. The editor learned two response formats. A full etymological essay, arbitrary book scan or unseen instruction needs further validation. [Held-out books and character families](../../references/grounded_ink_writing_result.md)

Use Python with PyTorch, Transformers 4.x, safetensors, NumPy, Pillow and fonttools. CPU is the default. Use `--device cuda` for GPU inference. The released editor and adapted reader are in Git. Four additional base files are in `artifacts/font_invariant_reader_v1` on the training workstation. Elsewhere, extract the `reader/` directory of the project's existing `visual-page-responder.zip` and pass `--base /path/to/reader`. The CLI checks the weight hashes. The adapted model has 86,220,389 parameters, including 220,035 in the new editor, plus 11,799,814 reader buffer elements. It runs locally without an external language model or OCR service. The optical reader inherits a pretrained contextual encoder. [Base model and export](../../docs/visual-page-responder.md) · [Reader weight hashes](../optical_role_reader/weights.json) · [Editor hash](weights.json)

## Reproduce training and evaluation

With the original optical reader, prepared Kyoto and BiPaR sources and the two ZhJpBook editions restored, use a fresh experiment workspace. Both fits used identical samples, initialization and compute budgets. The second fit randomizes the association between the retrieved regions and the questions. The full audit also expects the registered local Qwen comparison. [Corpus and split](../../references/evidence/grounded_ink_editor_20261124/corpus.json) · [Training protocol and prerequisites](../../references/grounded_ink_editor_protocol.md)

```bash
python -m scripts.prepare_grounded_ink_editor
python -m scripts.train_grounded_ink_editor aligned
python -m scripts.train_grounded_ink_editor independent
python -m scripts.evaluate_grounded_ink_editor locate
python -m scripts.evaluate_grounded_ink_editor aligned
python -m scripts.evaluate_grounded_ink_editor independent
python -m scripts.benchmark_grounded_ink_text
python -m scripts.audit_grounded_ink_editor
```

The public novel passages and output archive also support a CPU replay using only the released inference weights and font. All 384 autonomous replies matched the GPU pixels and stopping decisions. [Replay receipt](../../references/evidence/grounded_ink_editor_20261124/release_cpu.json)

```bash
python -m scripts.verify_grounded_ink_release --output replay.json
```

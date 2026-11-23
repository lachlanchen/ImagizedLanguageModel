# Optical reader for questions about people

Give this reader an image of a Chinese passage and an image of a question. It returns a PNG containing the selected source glyphs. The current model reads one horizontal stream of 32 × 32 cells per input and can return one to eight contiguous cells. On 14 reserved classical excerpts it gave all four answers for four excerpts in Chinese and none in English. [Evaluation and limitations](../../references/optical_role_adapter_result.md)

From the repository root, with Python, PyTorch, Transformers 4.x, safetensors, NumPy, Pillow and fonttools installed, run

```bash
sample=references/evidence/optical_role_adapter_20261124/example
python -m scripts.read_optical_role --page "$sample/page0.png" \
  --question "$sample/question0.png" --output answer.png
cmp answer.png "$sample/answer0_0.png"
```

The pictured question asks 誰隨行？ (“Who accompanied him?”). The reader returns 密康公 on the original page and 恭王 on the page with the names exchanged. To try the latter, use `page1.png`. For typed input use `--page-text` and `--question-text`. These options render the text before the model sees it. PNG input needs no font. CPU is the default. `--device cuda` selects a GPU. [Example and selection rule](../../references/evidence/optical_role_adapter_20261124/example/selection.json)

The adapter in this directory requires four files from the previously fitted visual page responder. The CLI finds them in `artifacts/font_invariant_reader_v1/` on the training workstation. If using the existing `visual-page-responder.zip`, extract its `reader/` directory and pass `--base /path/to/reader`. The CLI validates all four hashes and the adapter. A fresh Git checkout contains the adapter only. The complete tensor state occupies 391 MB. [Expected weights](weights.json) · [Base reader, export and prerequisites](../../docs/visual-page-responder.md)

## Training and evaluation

The comparison uses the project's prepared BiPaR question pairs, the Kyoto release 2.16 training annotations and the verified editions under the neighbouring ZhJpBook checkout. In a new experiment workspace with the original corpus and parent reader restored, prepare the inputs, fit the two arms serially and evaluate them with

```bash
python -m scripts.prepare_role_question_binding
python -m scripts.prepare_classical_argument_reader
python -m scripts.train_optical_role_adapter prepare
python -m scripts.train_optical_role_adapter aligned
python -m scripts.train_optical_role_adapter independent
for arm in parent aligned independent generic text; do
  python -m scripts.evaluate_optical_role_adapter "$arm"
done
python -m scripts.audit_optical_role_adapter
```

The text-reference arm needs the configured local Qwen3-8B and the two earlier classical baseline response files. The parent replay check uses the earlier optical benchmark. The trained image CLI uses neither. The checked-in receipts identify the exact data and model. [Training protocol and source requirements](../../references/optical_role_adapter_protocol.md) · [Training implementation](../../scripts/train_optical_role_adapter.py) · [Earlier classical baseline](../../scripts/benchmark_classical_context.py)

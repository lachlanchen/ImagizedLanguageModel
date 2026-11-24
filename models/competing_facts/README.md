# Reading and writing from pictured evidence

The model accepts a pictured question and a pictured page, emits the fact it selected as ink, and can write an English statement from the two. The modern Chinese and English short-answer reader has 2.44M parameters including its shared book model. The optional English sentence writer brings the whole pipeline to 2.79M. Inference runs locally on CPU or CUDA. The public test includes a research-networking article, a dinosaur alias, a monarch's name and a historical period. [Results and baseline comparisons](../../references/competing_facts_result.md) · [Original inputs and answers](examples)

Run this held English example from the repository root with the project's Python dependencies installed.

```bash
p=models/competing_facts/examples/squad/gain/1
python -m scripts.write_visual_statements --reader contrast \
  --question-image "$p/question.png" --question-cells 23 \
  --passage-image "$p/page.png" --passage-cells 600 \
  --output /tmp/ilm-network-answer
```

The two images in the output directory are `read_fact.png` and `answer.png`. The latter reads **NSFNET promoted research and education networking.** The optional machine-readable transcript, stop flags and packed bits are saved alongside it. Inputs use aligned 32 × 32 cells. For typed input, use `--question` and `--passage`. To produce only a short fact in either language use `python -m scripts.write_paired_evidence --help`. The `evidence` reader remains the default. The two experimental optical readers are `likelihood` and `contrast`.

## Training and evaluation

Prepare the upstream four-corpus training archive and earlier held splits as described in the paired evidence, recurrent evidence and visual statement releases. Use the project environment with PyTorch, NumPy, Numba, SciPy, Pillow, PyArrow, fontTools and safetensors. Then run:

```bash
python -m scripts.prepare_competing_facts
python -m scripts.train_competing_facts --run-dir artifacts/competing-refit
python -m scripts.evaluate_competing_facts
python -m scripts.replay_competing_facts
```

The separate fit writes new weights and training receipts to the requested directory. The last two commands evaluate or replay the published weights. Archived raw results are compressed JSON under `references/evidence/competing_facts_v1`. The CLI reads model weights and input images without access to the training pages or Qwen. The separate quantized comparison uses this workstation's LocalLLM installation. [Upstream corpus](../paired_evidence/README.md) · [English sentence writer](../visual_statements/README.md) · [Formal model](../../formal/span_list/CompetingFacts.lean)

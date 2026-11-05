# Clean image writing: results and execution

The completed clean writer learns 32 seen CMRC questions and emits full
Chinese answer images. It generates 29 original images exactly and 31
frozen-codec reference images exactly, with 99.9967% original-image
foreground F1. All 32 answers stop correctly. The
[result and raw comparison](../references/clean_learning_v74_result.md)
specify the evidence and limits: these questions receive 2,000 training-set
passes, so unseen language competence is not established.

The [older velocity writer](semantic-image-language-v72.md) fails its
development audit with 0/128 exact Chinese answers. Its checkpoint and
inference command reproduce that failed candidate. The clean-aware command
below loads the separately trained clean model and checks the checkpoint's
purpose and completed training budget.

## Generate an answer from a PNG

From the repository root, reproduce the first seen-case answer:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/infer_clean_ilm_v74.py \
  --prompt artifacts/clean_learning_v74_20261106/audit/predictions/00/prompt.png \
  --checkpoint artifacts/clean_learning_v74_20261106/training/checkpoint_latest.pt \
  --out artifacts/clean_question_demo --seed 20267420
```

Choose a fresh output directory. The command retains the actual prepared
input under input/, the binary image stream and stopping trace in answer.pt,
the full answer raster in answer.png when nonempty, and provenance in
report.json. Page mode preserves grayscale pixels and pads to a 32-pixel
boundary. Explicit --layout strip folds at white columns. It does not use
OCR or character segmentation. Prompts have a fixed 512-cell input budget.
An empty answer has no placeholder PNG. Arbitrary new prompts can still fail;
this checkpoint is the fixed learning control.

A fresh-process replay through this standalone PNG command reproduces the
first registered answer's pixels, stopping probabilities and feedback hashes
exactly. This is one repeated seen case, not an additional language benchmark.

## Train and evaluate on the full corpus

The [full-corpus protocol](../references/clean_corpus_v75_protocol.md) starts
a fresh 64.52M-parameter writer on all 12,976 admitted training examples.
It reuses the verified image-feature cache and frozen writing codec, with an
externally pretrained visual/contextual reader. Neither the 32-question
weights nor the failed velocity weights initialize training.

```bash
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/run_clean_corpus_v75.py --run artifacts/clean_corpus_v75_20261106
```

The coordinator performs execution checks, 40,000 updates, the original
384 normal/question-erased generations and independent CPU verification.
Quality requirements remain unchanged. Runtime inference uses no tokenizer,
character-ID embedding, language output head, persistent glyph inventory,
OCR or teacher service. External language/vision pretraining remains part of
the provenance; these local costs do not represent training the entire
foundation from scratch.

A separate [data-only audit](../references/evidence/clean_data_v75_result.json)
authenticates all 12,976 training images and 192 evaluation images. It finds
zero shared record IDs and zero exact prompt rasters across those sets.
Eleven evaluation cases have a reference answer raster also present among
training answers: ten Chinese and one English. Shared short answers do not
establish comprehension, and this audit does not assess near-duplicate
passages or contamination in external foundation pretraining.

The Chinese subset has 3,193 examples, with a median answer length of three
32-by-64 blocks and a maximum of seven. The 9,783 English examples have a
median of 26 answer blocks and a maximum of 128. Consequently even a bounded
Chinese reading pass would not establish Chinese long-form writing. Reproduce
the exact-data check without a neural model or GPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python scripts/audit_clean_data_v75.py \
  --out artifacts/clean_data_recheck.json
```

The [literal-writing transfer diagnostic](../references/copy_transfer_v75_protocol.md)
contains 12 new requests in Chinese, English and digits. CPU preparation
preserves all marks and confirms that none of the prompt rasters occurs in
the admitted training corpus. This is prepared evaluation data, with zero
model calls or performance claims. A separate lightweight coordinator can
wait for the exact live full-corpus process and then generate and independently
score these images after the original audit finishes:

```bash
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/run_copy_transfer_after_v75.py
```

It starts no GPU work while the main coordinator is alive, verifies the main
run's terminal receipt, and uses the same shared GPU lock for its model stage.
An interrupted or failed original run does not trigger a replacement training
job. Copy-probe scores are descriptive and do not replace the original gates.

| Location under the full run | Contents |
| --- | --- |
| cache/ | Link to the existing authenticated V72 cache |
| pipeline_status.json | Current phase, owned PIDs and completed phases |
| preflight/preflight.json | Execution checks, with no language credit |
| training/status.json | Latest update, loss, time and allocation |
| training/checkpoint_latest.pt | Resumable model/optimizer/RNG/cursor |
| training/training_report.json | Completed fixed-budget training receipt |
| development/predictions/ | Unmodified autonomous outputs and source headers |
| development/report.json | Full original audit decisions |
| verification.json | Independent physical scoring and provenance checks |

The pipeline uses physical GPU 0 and a shared project GPU lock. It creates
no GUI service. A completed pipeline can still fail the language requirements;
consult the explicit bounded_chinese_development_pass field. The general
ILM goal remains open, and sealed examples stay unopened in this pipeline.

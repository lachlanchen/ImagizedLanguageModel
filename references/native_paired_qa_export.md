# Exporting the complete native QA learning experiment

`scripts/export_native_paired_qa.py` is an offline verifier and exporter. It
never launches a neural model or changes the registered experiment. The full
export is gated on actual completion; it cannot turn a training prefix into
a completed QA result.

## Verifying a live training prefix

With CUDA hidden, `--stage inspect-training` reads one atomic checkpoint
snapshot. It verifies the exact registered schedule prefix, all optimizer
steps and settings, source/supervision hashes, finite gradients and every
retained pre-update PNG against its actual neural tensor. It compares adapters
with the fixed initial weights and independently compares the retained first
native and differentiable outputs. The snapshot is an observation of the live
file, which is replaced as training advances; the audit does not retain an
extra weight copy.

The executed check authenticated all 29 completed training pages in its
observed checkpoint, with 206 changed adapter tensors. This confirms actual
execution and saved-image integrity, not successful language learning. An
additional visual inspection of the first 24 pre-update pages still finds
corrupted source fragments, retained labels and faint writing. No final or
held-out answer score is assigned from those training images.

Run from the repository root in the established model Python:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$ILM_PYTHON" \
  scripts/export_native_paired_qa.py --stage inspect-training \
  --run artifacts/native_paired_qa_learning_v2 \
  --output artifacts/native_paired_qa_prefix_recheck.json
```

`ILM_PYTHON` denotes the existing image-generation interpreter; use the normal
local NVIDIA-library wrapper if that environment requires it. Output paths
must be new so earlier observations are preserved.

## Export after the full assessment

`--stage export` requires the completed 128-update checkpoint and report,
all 256 new QA/control calls, all 64 new final copying calls, and the 64
previously authenticated initial copying outputs. It independently rebuilds
canonical/source-accepted exact counts, both-correct passage-pair counts and
all six prospective conditions from the bound OCR observations. It checks
identical erased-question input/seed pairs against actual output tensors.
Targets and reports are post-processing inputs, never generation inputs.

The package retains every one of the 384 initial/final answer PNGs and all
128 actual pre-update training PNGs, alongside original inputs, references,
source reports, complete tables and original-order galleries. Generated
answers are never cropped, resized, repaired, replaced or selected. Input
previews alone are resized to fit contact sheets; their original PNGs remain
verbatim. The pinned CMRC source licence and source/transformation attribution
are included.

The contact renderer has been exercised on actual earlier copying strips
and complete natural-question pages. Both layouts were visually inspected;
checks compare pasted evidence regions with their source pixels. These layout
checks are not new model evaluations. Full export validation remains pending
until the active experiment finishes.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$ILM_PYTHON" \
  scripts/export_native_paired_qa.py --stage export \
  --run artifacts/native_paired_qa_learning_v2 \
  --output artifacts/native_paired_qa_complete_export
```

The CPU completion worker also accepts `--export-result`. It first prepares
the opaque-ID transcription packet, then authenticates and exports the full
result to a separate local folder. It waits for the exact currently live
coordinator and its terminal complete-assessment receipt. It does not launch
or retry inference, publish artifacts, fill visual annotations or qualify a
language model. Raw review remains pending even if every numerical condition
passes. No current manuscript claim changes before the complete result is
assessed.

# V37 Visual Semantic Distillation Result Revision Plan

Date: 2026-08-14

Revision stage: internal major evidence update after the frozen V37 run

## Problem

V37 has completed its hash-pinned 8,000-update training run and both frozen
development evaluations, but the README and image-native manuscript still end
at V36. The new result is materially better at visual reading and answer-plan
retrieval, yet it fails the complete semantic gate. The record must therefore
preserve both the advance and the rejection before a successor experiment
changes the implementation.

## Allowed files and locations

- `docs/visual-semantic-distillation-v37-result.md`: complete measured result,
  interpretation, and next bounded experiment.
- `publication/ilm-image-native/evidence/v37/`: immutable small JSON receipts
  and reports copied from the ignored run directory; no weights or target
  tensors.
- `publication/ilm-image-native/generate_v37_result_figure.py`: deterministic,
  hash-pinned result-figure generator.
- `publication/ilm-image-native/figures/visual_semantic_distillation_v37_result.png`:
  generated result figure.
- `README.md`: make V37 the current evidence and retain V36 as prior evidence.
- `publication/ilm-image-native/ilm-image-native.tex`: update the abstract,
  contribution list, measured-result sequence, conclusion, and figure/table
  references.
- This plan file: record verification and delivery status.

## Claims to add

- V37 is an 89,768,706-parameter image-only reader/planner initialized from the
  exactly attributed Pixel-Linguist-v0 visual encoder. It receives only prompt
  rasters and clean visual masks at runtime and emits continuous prompt states,
  answer plans, and visual length.
- BGE-M3 is an offline target builder only. Exact provenance is pinned, and no
  BGE call, text, token/Unicode ID, OCR result, candidate bank, or target tensor
  is present in the deployed runtime or standalone EMA artifact.
- Training completed 8,000/8,000 finite updates in 3,695.48 seconds on GPU 0
  of one RTX 4090, with 2,951,057,920 bytes peak allocated CUDA memory. The
  final deterministic answer-plan effective rank is 36.44.
- EMA development prompt-state top-1/top-5/MRR is
  47.45/77.55/61.54 percent. Answer-plan top-1/top-5/MRR is
  20.41/45.41/33.77 percent. Raw weights do not repair the result.
- Counterfactual assignment reaches 93.88 percent and the 30-way paraphrase
  top-5 reaches 73.33 percent. Prompt reading, counterfactual assignment,
  paraphrase top-5, and plan-rank conditions pass.
- The complete EMA gate passes only 20/33 conditions. Prompt paired cosine,
  absolute answer retrieval, answer cosine/margin controls, held-font
  consistency, paraphrase consistency, and length miss their frozen bounds.
- The exact decision is `not-qualified`; sealed evaluation stays unopened and
  V37-R remains forbidden.
- The result supports a narrower claim than a language model: end-to-end
  semantic distillation makes useful visual reading and measurable
  candidate-free answer planning feasible in about one hour on one 4090, but
  does not yet qualify autonomous semantic planning or image generation.

## Claims to remove or soften

- Do not describe evaluator retrieval as deployed retrieval, answer text,
  generated language, or raster output.
- Do not describe 47.45 percent prompt retrieval as proof of general language
  understanding; it is a fixed 196-way development reading probe.
- Do not imply that passing counterfactual assignment or paraphrase top-5 can
  override the conjunctive gate.
- Do not claim sealed generalization, renderer capability, Qwen/GPT parity,
  training-from-scratch, or efficiency superiority over token LLMs.
- Do not claim Pixel-Linguist or BGE-M3 as project work. Attribute their exact
  initialization/offline-teacher roles and preserve the local-research-only
  redistribution boundary caused by unstated Pixel-Linguist weight licensing.

## Out of scope

- No V37 model, data, trainer, evaluator, protocol, threshold, checkpoint, or
  target-bank changes.
- No sealed-data access.
- No renderer implementation, target construction, or training.
- No checkpoint, target tensor, similarity tensor, teacher weight, or local
  model cache committed to Git.
- No paper-wide stylistic rewrite or unsupported biological/priority claim.

## Evidence anchors

- `artifacts/visual_semantic_distillation_v37_20260814/run_receipt.json`
- `artifacts/visual_semantic_distillation_v37_20260814/training_summary.json`
- `artifacts/visual_semantic_distillation_v37_20260814/development_report_ema_v37.json`
- `artifacts/visual_semantic_distillation_v37_20260814/development_report_raw_v37.json`
- `references/visual_semantic_distillation_v37_protocol.md`
- Git commit history is the manuscript baseline; no separate formal redline
  baseline exists, so `latexdiff` is outside this revision.

## Verification

1. Copy the four small JSON evidence files and record their SHA-256 values.
2. Hash-pin every tracked evidence file consumed by the V37 figure generator.
3. Generate and visually inspect the PNG dimensions, typography, alignment,
   and decision labels.
4. Run Ruff and the complete V37 test subset.
5. Build `publication/ilm-image-native/ilm-image-native.tex` with
   `scripts/latex_build.sh`.
6. Inspect the compiled PDF with `pdfinfo`, `pdftotext -layout`, and rendered
   pages containing the V37 section, figure, and conclusion.
7. Audit wording for `not-qualified`, sealed, renderer, candidate-only
   evaluation, and external-work attribution boundaries.
8. Commit and push this bounded revision.

## Status

- Plan frozen before manuscript edits: yes
- Evidence copy: complete
- Source edits: complete
- Figure generation and inspection: complete
- Tests and PDF build: complete
- PDF inspection: complete
- Commit and push: complete in the containing atomic revision

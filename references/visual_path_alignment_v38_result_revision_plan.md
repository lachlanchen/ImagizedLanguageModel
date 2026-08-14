# V38 Visual Path Alignment Result Revision Plan

Date: 2026-08-14

Revision stage: internal major evidence update after the frozen V38 run

## Problem

V38 completed its hash-pinned 8,000-update training run and both frozen
development evaluations. The result fixes several V37 failure modes, including
font-path consistency and near-identity prompt-to-answer transitions, but it
still fails the complete answer-semantic gate. The repository must preserve
both the measured advance and the rejection before a successor changes data or
architecture.

## Allowed files and locations

- `docs/visual-path-alignment-v38-result.md`: complete measured result,
  interpretation, limitations, and next bounded experiment.
- `publication/ilm-image-native/evidence/v38/`: immutable small JSON receipts
  and reports copied from the ignored run directory; no weights or tensors.
- `publication/ilm-image-native/generate_v38_result_figure.py`: deterministic,
  hash-pinned result-figure generator.
- `publication/ilm-image-native/figures/visual_path_alignment_v38_result.png`:
  generated measured-result figure.
- `publication/ilm-image-native/generate_figures.py` and
  `publication/ilm-image-native/figures/ilm_v_yan_readme_hero.png`: update the
  stale V30 status text discovered during PDF inspection; preserve the concept
  image and regenerate only its deterministic overlay.
- `README.md`: make V38 the current evidence and retain V37 as prior evidence.
- `references/visual_path_alignment_v38_research.md`: change pre-run status to
  completed and append the frozen result and diagnosis.
- `docs/first-imagized-language-model-goal.md`: update only the current-evidence
  snapshot and next bounded proof.
- `publication/ilm-image-native/ilm-image-native.tex`: update the abstract,
  hero cross-reference, contribution list, measured-result sequence,
  conclusion, and bibliography; add one V38 result subsection, table, and
  figure.
- This plan file: record verification and delivery status.

## Claims to add

- V38 is a 90,753,281-parameter image-only visual reader and
  prompt-conditioned answer-state model. The deployed tensor boundary accepts
  a prompt raster and clean patch mask and emits continuous prompt state,
  answer state, and visual length. It contains no strings, token/Unicode IDs,
  OCR, visual codebook, candidate bank, BGE, Qwen, teacher, or network client.
- Pixel-Linguist-v0 is exactly attributed external initialization. BGE-M3 and
  Qwen-family models are offline target/data preparation tools only; none is a
  runtime dependency or project contribution.
- The frozen run completed 8,000/8,000 finite BF16 updates in 5,982.31 seconds
  on GPU 0 of one RTX 4090, with 3,188,168,704 bytes peak allocated CUDA
  memory. Final deterministic prompt/answer rank probes are 28.00/30.45.
- EMA development prompt top-1/top-5/MRR is
  60.71/86.73/72.36 percent. Prompt-conditioned answer-state
  top-1/top-5/MRR is 21.94/49.49/34.60 percent. Raw weights are materially the
  same and do not repair the result.
- V38 improves V37's prompt paired cosine from 0.2374 to 0.3789, canonical to
  held-font prompt/answer consistency from 0.4130/0.4089 to 0.7925/0.7323,
  and prompt-answer transition cosine from near identity (0.9972) to 0.5769.
  Transition-direction cosine is 0.3401 and answer-state effective rank is
  49.02.
- The complete EMA gate passes only 25/39 conditions. Absolute answer
  retrieval/cosine, answer cyclic win, counterfactual assignment by a narrow
  margin, held-font answer transfer, paraphrase consistency, and length miss
  their frozen bounds.
- The exact decision is `not-qualified`; zero sealed rows were rendered and no
  raster renderer is authorized.
- The result supports a narrower claim than a language model: paired visual
  paths and a full answer map improve visual reading, font invariance, and
  transition geometry on one 4090, but a small 5,822-pair curriculum and a
  single answer vector do not generalize the answer relation.

## Claims to remove or soften

- Do not call evaluator retrieval deployed retrieval, generated text,
  generated language, or raster output.
- Do not infer broad language understanding from a fixed 196-way development
  probe or high training-batch retrieval.
- Do not let font, rank, transition, or paraphrase top-5 passes override the
  conjunctive gate.
- Do not claim sealed generalization, renderer capability, Qwen/GPT parity,
  from-scratch training, or efficiency superiority over token LLMs.
- Do not claim Pixel-Linguist, BGE-M3, Qwen, PIXEL, PixelGPT, or SPIRAL as
  project work. State their exact roles and retain the local-research-only
  redistribution boundary caused by unstated Pixel-Linguist weight licensing.

## Out of scope

- No V38 model, data, trainer, evaluator, protocol, threshold, checkpoint, or
  target-bank changes.
- No sealed-data access and no renderer implementation or training.
- No checkpoint, standalone weight, target tensor, similarity tensor, teacher
  weight, local model cache, or raw private book data committed to Git.
- No paper-wide stylistic rewrite, multilingual README rewrite, or unsupported
  biological, priority, parity, or superiority claim.

## Evidence anchors

- `artifacts/visual_path_alignment_v38_20260814/run_receipt.json`
- `artifacts/visual_path_alignment_v38_20260814/training_summary.json`
- `artifacts/visual_path_alignment_v38_20260814/development_report_ema_v38.json`
- `artifacts/visual_path_alignment_v38_20260814/development_report_raw_v38.json`
- `references/visual_path_alignment_v38_protocol.md`
- Git history is the manuscript baseline. No separate formal TeX baseline
  exists, so this revision uses source diff plus compiled-PDF inspection rather
  than inventing a latexdiff baseline.

## Verification

1. Copy the four small JSON evidence files and record their SHA-256 values.
2. Hash-pin every tracked evidence file consumed by the V38 figure generator.
3. Generate and visually inspect the PNG dimensions, typography, alignment,
   decision labels, and sealed/renderer boundary.
4. Run Ruff and the complete focused V38 tests.
5. Build `publication/ilm-image-native/ilm-image-native.tex` with the repository
   LaTeX build script or Makefile.
6. Inspect the compiled PDF with `pdfinfo`, `pdftotext -layout`, and rendered
   pages containing the V38 section, figure, table, and conclusion.
7. Audit wording for `not-qualified`, zero sealed rows, renderer closure,
   evaluator-only retrieval, and external-work attribution.
8. Confirm `.auto-readme-work/` and all large ignored artifacts remain
   untracked, then commit and push this bounded revision.

## Status

- Plan frozen before manuscript edits: yes
- Evidence copy: complete; all four tracked JSON hashes match the source
  artifacts.
- Source edits: complete; claims are limited to the measured continuous-state
  result and external roles are explicit.
- Figure generation and inspection: complete; the hash-pinned V38 figure is
  `1800 x 1120`, and the stale V30 concept-hero labels were regenerated from
  source as V38 labels.
- Tests and PDF build: complete; 40 focused V38 tests pass, both figure
  generators pass Ruff, `git diff --check` passes, and the 76-page PDF builds
  successfully in two passes.
- PDF inspection: complete; `pdfinfo`, `pdftotext -layout`, log audit, and
  rendered-page inspection cover the abstract, V38 equations, table, figure,
  decision, and conclusion. No undefined reference, overfull box, missing math
  delimiter, or fatal error remains.
- Commit and push: pending

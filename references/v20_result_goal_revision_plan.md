# V20 Result and Goal Revision Plan

Date: 2026-08-12

## Problem

The project documents still describe V20 as a future proposal. The completed
paired development experiment changes the evidence: the field route became
causally necessary and spatially local, but neither arm passed checkpoint
selection and the candidate did not beat its exact-capacity control by the
preregistered quality margin. The durable goal must distinguish this causal
advance from a readable-writer or language-model success.

## Scope

This revision may change:

- `docs/first-imagized-language-model-goal.md`;
- `README.md`;
- `references/continuous_sensory_language_scan_2026.md`;
- a new measured V20 result receipt under `docs/`;
- `publication/ilm-image-native/ilm-image-native.tex`;
- a new reproducible V20 figure generator and its generated PNG; and
- the compiled paper PDF.

## Claims to Add

1. V20 used two exactly matched 506,448-parameter writers and a fixed salted
   development protocol on one RTX 4090.
2. The candidate's correct local field improved dense F1 by `+0.12177` over a
   shuffled field and `+0.36128` over a zero field. A quadrant occlusion changed
   only its matching output quadrant (`1.0` locality). This is evidence of
   causal, topographic use of a continuous retinal field.
3. The candidate remains rejected: overall F1 was `0.63608`, target cosine was
   `0.81519`, and the numerical detail-invariant tolerance was missed. It also
   improved dense F1 over the equal-capacity control by only `+0.01791`, below
   the fixed `>0.03` paired margin. No checkpoint was selected, no human review
   was authorized, and frozen images remained uninstantiated.
4. V21 should test a field-complete writer: local continuous fields determine
   both coarse occupancy and fine topology, while global state supplies only
   spatially uniform modulation. A fixed zero-mean basis should make the
   decomposition invariant algebraic rather than tolerance-sensitive.
5. The first complete proof remains image-in/image-out Chinese language on a
   small corpus, followed by causal future-field prediction and autonomous
   write--reread generation. Scale, historical answers, and a Visual
   Word-Origin Book follow only after those gates pass.
6. Spatiotemporal 3D writing is a future extension: a rendered/captured 3D
   Chinese or English form may enter as a visual field and the primary output
   may be a short 3D character/writing movie. It is not part of the first 2D
   proof or its compute claim.

## Claims to Remove or Soften

- Remove wording that presents V20 as unrun future work.
- Do not call V20 a successful writer, autonomous model, language learner, or
  efficiency win.
- Do not infer frozen performance from development measurements.
- Do not claim biological equivalence or Qwen parity.

## Figure

Create a measured-result figure from the final candidate and control sample
grids. It must show correct, shuffled-field, zero-field, and local-occlusion
outputs; fixed gates; exact parameter matching; and the rejected verdict. It
must not use frozen examples or imply that a checkpoint was selected.

## Verification

1. Regenerate the V20 result PNG from local development artifacts.
2. Run the figure generator a second time and verify a stable SHA-256.
3. Compile the LaTeX paper with the repository build target.
4. Extract the compiled PDF text and confirm the V20 verdict and V21 direction.
5. Run targeted tests for the V20 model/training/evaluator and run Ruff on the
   new figure script.
6. Inspect the final diff, commit atomically, and push `main`.

## Out of Scope

- Changing the completed V20 protocol, metrics, model, or artifacts.
- Opening the V20 frozen partition.
- Implementing or training V21 in this revision.
- Building the 3D/movie extension before the 2D causal language loop passes.

## Execution Record

Status: completed on 2026-08-12.

- Added the measured V20 receipt and deterministic result figure.
- Updated the engineering goal, README, 2026 decision scan, TeX manuscript,
  and compiled PDF.
- Added the Visual Language Stream contract with explicit sequence/time and
  optional depth axes while retaining a 2D-first acceptance order.
- Regenerated the result figure twice with stable SHA-256
  `b87e7d5060850c6cb5168d5b2165cd1e79603babc9ec41d4d2f8eeb58e83d6d8`.
- Compiled the 27-page paper with no unresolved references, LaTeX errors, or
  overfull boxes introduced by this revision. Existing narrow-table underfull
  warnings remain unrelated to V20.
- Visually inspected PDF pages 13--15; the V20 table, result figure, caption,
  and transition to V21 are legible and non-overlapping.
- `pytest` reports `31 passed`; Ruff and Python compilation pass.
- No manuscript baseline or response-letter workflow exists in this research
  draft, so a reviewer redline was not generated.

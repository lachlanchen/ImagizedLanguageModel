# V48 Measured-Result Publication Revision Plan

Date: 2026-08-15

Status: approved internal evidence update; written before editing the active
TeX manuscript.

## Revision Stage

Internal major revision after completion of the preregistered V48 production
run and strict development audit. This is one bounded scientific evidence
update, not a reviewer response, general prose rewrite, or authorization to
design or report V49 as if it had run.

## Goal

Bring the active image-native manuscript from V47 through the completed V48
result. The revision must state both sides of the result:

1. dense visual-future supervision removes V47's terminal-position collapse
   and preserves a real ordered-history signal in a compact image-only model;
2. the deterministic inverse-DCT point writer produces conditional-average,
   speckled rasters, loses identity after visible rereading, and rapidly
   compounds errors in closed loop.

The resulting claim is a narrowed research direction, not a complete or usable
ILM claim.

## Frozen Baseline And Evidence

- Active manuscript: `publication/ilm-image-native/ilm-image-native.tex`
- Pre-revision Git baseline:
  `5b947bb788f8a6b5e9ab46632eb3b31efc14b852`
- Pre-revision TeX SHA-256:
  `05b459df91dedb2f16d6b786985ab70a17c45a11fc87d3ad9f076e7fd730c1da`
- Pre-revision PDF SHA-256:
  `c30dbf5e04c24440b364e72790b77dab9a5d1a75f1e7c7caf13ed64a8cba4c40`
- V48 checkpoint SHA-256:
  `d281f8c8403d07b2662bc6d091145287f218727b3c8df2f2ea87da04c70165f3`
- Training summary SHA-256:
  `fe658885bed9ebc10cb8ff4cf61a487c2b0334b6feabf3e2108059f101ed0c7b`
- Development report SHA-256:
  `2d3e5857ce7c50c8b3b85f5056f7c9f26b3ed3b6eb162c68c3ad7ca5a94742fa`
- Development summary SHA-256:
  `a1371bd7be804100e351e83e04b1df6d56dcd03fdb13cc89cdfc64d461f59d9d`
- Raster evidence SHA-256:
  `bce170e2e5342c0d7d61735e98bd14f8668c00583ac63f3c218bd3a8326aeeea`
- Audit erratum SHA-256:
  `1cc9bdd777da6079b0cf0dbdaf88645114c5c25c77c283bdf92300730aa99f20`

There is no named TeX baseline or active redline convention in this manuscript
directory. Git commit `5b947bb` is the immutable baseline for this revision
unit. A temporary redline may be generated from Git for verification but must
not replace the active manuscript or create a silent baseline.

## Supported Measured Claims

Only the following V48 claims may be added unless another tracked report is
explicitly cited:

- The deployed/student boundary receives continuous raster streams and emits
  continuous image fields or visible rasters; it receives no token IDs,
  Unicode IDs, strings, OCR, glyph lookup, visual codebook, candidate bank, or
  external language model.
- The from-scratch model has 16,278,401 trainable parameters and predicts four
  future `32 x 32` writing images densely from a 64-cell raster context.
- Production training uses 10,000 updates and 160,000 image segments, takes
  1,369.04 seconds, and peaks at 0.4921 GiB allocated CUDA memory. Peak
  train-or-audit allocation is 2.1490 GiB.
- The strict report passes all 23 integrity checks, reproduces V42 top-1
  `0.19970703125` in-process, and leaves the frozen partition closed.
- Immediate full-history top-1 is `0.1806640625`, above symbolic bigram
  `0.12255859375` and shuffled history `0.16162109375`, but below matched V42
  `0.19970703125`.
- Ordered target log probability gains `0.16133` nat over shuffled history.
- Horizon-one prediction beats its offset control; horizons two through four
  do not all clear the fixed control margin, although every horizon remains
  diverse.
- Terminal top-1 remains stable from visible length 63 (`0.17383`) to 64
  (`0.17432`), eliminating V47's localized terminal collapse.
- Counterfactual arm accuracy is `0.53223`, below the frozen `>0.55` gate.
- Direct visible-reread identity is `0.14893`, pixel F1 is `0.46614`, identity
  retention is `0.82431`, blank rate is zero, and the identity and retention
  gates fail.
- Four-step closed-loop mean identity is `0.04395`, but step identity falls
  from `0.14453` to `0.01563`, `0.00391`, and `0.01172` after generated pixels
  are fed back.
- Eleven of sixteen preregistered gates pass. The result remains
  non-qualifying.
- The evidence supports retaining an ordered causal visual reader while
  separating an invertible raster carrier from a foreground-sensitive
  identity geometry and replacing deterministic multimodal point output with
  a sampled conditional raster distribution trained to recover from visible
  errors. This is a next-hypothesis boundary, not a measured V49 result.

## Intended Scope

### Files Allowed To Change

- `references/v48_measured_result_publication_revision_plan.md` -- plan and
  execution record.
- `publication/ilm-image-native/generate_v48_result_figure.py` -- deterministic
  evidence-backed result renderer.
- `publication/ilm-image-native/figures/visual_future_block_language_v48_result.png`
  -- generated result figure.
- `publication/ilm-image-native/ilm-image-native.tex` -- bounded V48 update.
- `publication/ilm-image-native/ilm-image-native.pdf` -- rebuilt manuscript.
- Temporary untracked build/redline/render directories used for verification.

The existing tracked files under
`publication/ilm-image-native/evidence/v48/` are inspect-only inputs and must
remain byte-identical.

### Manuscript Locations Allowed To Change

1. Abstract: append one compact V48 paragraph after V47.
2. Hero caption: extend the measured-series qualification through V48.
3. Contributions: add one V48 item after V47.
4. Rename the bounded experiment subsection from `V42--V47` to `V42--V48`.
5. Results: append one V48 mathematical description, measured table, evidence
   figure, gate outcome, and mechanism diagnosis immediately after V47.
6. Evaluation synthesis: extend only the late V42--V47 evidence paragraph with
   V48.
7. Conclusion: add the measured V48 result and replace the obsolete next-step
   wording with the V48-supported reader/metric/distributional-writer boundary.
8. Product-target limitation: extend the experiment range from V42--V47 to
   V42--V48 where needed.

### Figure Requirements

The V48 result figure must be rendered deterministically from the four
hash-pinned publication evidence files. It must include:

- the 64-cell raster reader and four-future field structure;
- V42, bigram, shuffled-history, and V48 immediate top-1 values;
- the four horizon identities and controls;
- direct visible identity, visible F1, retention, and closed-loop decay;
- the actual target/proposal/visible/rollout raster sheet;
- the exact 11/16 non-qualifying decision and compute receipts; and
- a restrained diagnosis that point prediction averages multimodal raster
  futures.

It must not use decorative generated art as experimental evidence or imply
readable autonomous output.

## Explicitly Forbidden Changes

- Do not claim V48 is a usable ILM, understands language generally, answers
  etymology questions, generates historical forms, matches Qwen, is more
  efficient than token LMs, or reads like a human.
- Do not open the frozen partition, select another checkpoint, tune thresholds,
  rerun training, change gates, or round failed thresholds into passes.
- Do not change V34--V47 measured values or reinterpret their frozen outcomes.
- Do not describe V49 as implemented, trained, or validated.
- Do not add external citations, broad prior-art prose, README changes, or
  unrelated manuscript cleanup in this revision unit.
- Do not stage or modify `.auto-readme-work/`.

## Proposed Edits

| Step | Location | Edit | Rationale | Acceptance criterion |
| --- | --- | --- | --- | --- |
| 1 | V48 figure generator | Pin and validate all evidence hashes and frozen decision facts | Prevent attractive but unsupported result art | Generator rejects altered evidence and outputs one deterministic figure |
| 2 | Abstract and contribution list | Add bounded V48 result and claim limit | Keep the paper's headline current | Values and 11/16 decision match the strict report |
| 3 | V42--V48 result subsection | Add model equations, table, measured image, and diagnosis | Expose the tested mechanism and falsifying evidence | Reader, writer, controls, raster loop, and failures are all explicit |
| 4 | Evaluation and conclusion | Synthesize what V48 fixes and what it rejects | Turn the result into a precise next decision | No complete-ILM or efficiency overclaim; next mechanism remains prospective |
| 5 | PDF and trace audit | Build twice, inspect modified pages, and attempt redline | Verify source and presentation | No undefined references or new layout defects; locations are PDF-backed |

## Verification Plan

1. Assert all four publication evidence hashes before rendering the figure.
2. Run `python -m ruff check` and Python compilation on the figure generator.
3. Visually inspect the generated figure at original resolution.
4. Run `scripts/latex_build.sh publication/ilm-image-native/ilm-image-native.tex`
   and confirm the two-pass build succeeds.
5. Search the compiled PDF with `pdftotext -layout` for the V48 abstract,
   model, metrics, figure caption, diagnosis, and conclusion.
6. Rasterize and visually inspect every modified PDF page, including page
   breaks around the V48 figure and table.
7. Attempt a temporary redline against Git baseline `5b947bb`; record an
   unavailable or failed `latexdiff` honestly.
8. Rehash evidence files and verify they are unchanged.
9. Run `git diff --check` and inspect staged paths before committing.
10. Commit and push only this revision unit; leave `.auto-readme-work/`
    untouched and unstaged.

## Response-Letter Impact

None. There is no reviewer response letter in scope. The plan and Git baseline
provide the revision trace.

## Execution Record

Pending.

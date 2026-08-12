# V24 Result Publication Plan

Date: 2026-08-13

## Purpose

Publish the completed Visual Packet Reread Stream V24 evidence chain without
expanding its claim beyond the measured fixed packet grammar. This revision
closes the gap between the committed V24 implementation/evaluators and the
repository's README and manuscript, which currently stop at V23.

## Supported Claim

V24 is accepted as a bounded image-only, variable-length prompt and two-frame
answer stream. It locates visibly headed packet roles, performs the fixed
same/other relation, renders an unseen Chinese glyph, rereads that generated
frame, and renders its bound label. The single frozen run passed all
preregistered gates over 107 sealed identities.

V24 does not establish arbitrary sentence understanding, factual recall,
etymology explanation, page continuation, unrestricted output length, 3D or
movie generation, human-level reading, Qwen parity, or efficiency superiority.

## Allowed Changes

- Add `docs/visual-packet-reread-stream-v24-result.md` as the canonical result
  receipt.
- Add a reproducible result-figure generator under
  `publication/ilm-image-native/` and its measured PNG under `figures/`.
- Update the top-level `README.md` with a V24 status section and relabel V23 as
  the prior prompt test.
- Update `publication/ilm-image-native/ilm-image-native.tex` in the abstract,
  contribution list, experiment/results narrative, discussion, and conclusion
  to include V24.
- Update `publication/Makefile` only if needed to make the V24 figure
  reproducible through the existing build workflow.

## Evidence Sources

- Fixed protocol: `references/visual_packet_reread_stream_v24_protocol.md`
- Training receipts: `artifacts/visual_packet_stream_v24_evidence/`
- Fresh paired audit:
  `artifacts/visual_packet_stream_v24_paired_audit/paired_development_audit.json`
- Opaque review:
  `artifacts/visual_packet_stream_v24_opaque_review/review_result.json`
- One-shot frozen evaluation:
  `artifacts/visual_packet_stream_v24_frozen/frozen_evaluation.json`

Artifact metrics may be reported but ignored checkpoints, private review
artifacts, and generated caches must not be committed.

## Figure Design

The V24 figure will be generated from recorded artifacts and an actual opaque
review card. It will show:

1. the variable packet input and two-frame autoregressive output contract;
2. a real candidate output example;
3. fresh paired candidate/control causal metrics; and
4. the one-shot frozen performance and claim boundary.

The figure must not expose hidden transcriptions as model inputs or imply
open-ended language capability.

## Verification

1. Generate the figure and inspect the PNG at original resolution.
2. Run focused figure/script checks if added.
3. Build the manuscript through the repository Makefile.
4. Inspect the compiled PDF page containing the V24 figure and extract its text
   to confirm captions, tables, references, and limitations are present.
5. Run the full test suite.
6. Confirm `git diff --check` and that only intended files are staged.

## Out Of Scope

- Changing the sealed V24 training data, model, trainer, selected checkpoints,
  thresholds, audit, review, or frozen evaluator.
- Repeating the frozen evaluation.
- Beginning V25 training in the same revision unit.
- Updating old experiment receipts except where a forward pointer is required.
- Treating the agent opaque review as a human-subject evaluation.

## Redline Note

This repository uses a continuously evolving research manuscript rather than a
journal response package, and no manuscript baseline/redline convention is
currently present. Verification will therefore use the Git diff plus the newly
compiled PDF; no baseline file will be overwritten.

## Execution Record

Status: completed on 2026-08-13.

- Added the canonical V24 result receipt and a reproducible measured figure.
- Updated the README, abstract, contributions, experiment narrative, aggregate
  results, and conclusion with the same bounded claim.
- The manuscript builds successfully through `publication/Makefile` to a
  38-page PDF (`7,457,557` bytes).
- Inspected compiled pages 21--25 at 125 dpi. The V24 equations, paired-control
  table, frozen table, measured figure, caption, and claim boundary are readable
  and do not overlap or clip.
- The final LaTeX log has no unresolved-reference, missing-character, or
  overfull-box warning. Existing underfull bibliography/table warnings remain
  outside this revision's scope.
- `PYTHONPATH=. pytest -q` passes: `186 passed in 23.55s`.
- `git diff --check` passes.

The first plain `pytest -q` attempt failed during collection because this source
tree is not installed as a Python package. Rerunning with the repository's
required `PYTHONPATH=.` environment produced the passing result above; no code
change was needed.

### Deviation

The original plan allowed a `publication/Makefile` update if needed. None was
needed. The first PDF build fell back to XeLaTeX because newly inserted Chinese
header literals were unsupported by the default PDFLaTeX font. The prose now
uses role names and Unicode code-point references while the actual glyph images
remain in the measured figure; the final two-pass PDFLaTeX build succeeds.

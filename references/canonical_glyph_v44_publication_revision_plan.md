# Canonical Glyph V44 Publication Revision Plan

Status: approved internal evidence update, written before editing the active
TeX manuscript.

Date: 2026-08-14

## Problem

The README and active paper currently stop at V43. V44 has now completed its
preregistered production training, matched-base development audit, and
post-result scale/direction diagnosis. The public account must report that V44
removes V43's memorization gap but fails to improve generalizable binding and
damages the accepted natural-language calibration.

## Scientific Claim Change

Before: the predicted continuous visual plan is the dominant V43 bottleneck;
the next test is a frozen-base, nonrepeating residual binding path.

After: that exact V44 test is negative. The residual scores nearly identically
on consumed and unseen training pairs, so repeated-pool memorization is fixed,
but development arm accuracy is only `0.53418`, unseen-train accuracy is
`0.57422`, and matched natural top-1 falls from `0.19434` to `0.17432`.
Post-result interpolation shows no scale reaches gate; the learned displacement
is negatively aligned with the target displacement and moves anchors toward
the corpus-mean image field. The next representation test must remove common
visual modes before another reader or writer composition.

The revision may not claim V44 passes, that V42 is a complete ILM, that V43's
writer is reopened, or that a centered representation already works. The
frozen partition remains unopened.

## Allowed Files And Locations

- `README.md`: current evidence summary and next bounded decision.
- `publication/ilm-image-native/ilm-image-native.tex`: abstract tail,
  V42--V44 evidence section, synthesis, and conclusion.
- `publication/ilm-image-native/ilm-image-native.pdf`: rebuilt active paper.
- `publication/ilm-image-native/evidence/v44/**`: compact immutable production,
  development, and diagnostic receipts.
- `publication/ilm-image-native/generate_v44_result_figure.py`: deterministic
  hash-pinned figure generator.
- `publication/ilm-image-native/figures/canonical_glyph_binding_v44_result.png`:
  generated evidence figure.
- `references/canonical_glyph_binding_v44_result.md`: bounded result note.
- `scripts/diagnose_canonical_glyph_binding_v44.py`: post-result diagnostic.

## Figure Content

The V44 figure will show only measured evidence:

1. frozen V42 plus the 1.736M-parameter tangent residual;
2. one-pass 24,000-pair and 1,024-pair holdout boundaries;
3. matched natural retention failures;
4. development, consumed, and unseen pair results against fixed gates;
5. the residual-strength sweep; and
6. common-field drift and negative target-displacement alignment.

The figure must label V44 as rejected, the diagnostic as post-result, and the
writer/frozen partition as closed.

## Out Of Scope

- Changing V44 source, checkpoint, seeds, losses, thresholds, or reports.
- Opening the V43 writer or frozen partition.
- Implementing or claiming success for the next centered-field experiment.
- Rewriting older experiments except where their trajectory connects to V44.
- Formal redlines, because no stable manuscript baseline is designated.

## Verification

1. Copy only compact V44 receipts and pin every SHA-256 in the figure script.
2. Regenerate the figure deterministically and inspect it as an image.
3. Run Ruff and Python compilation on the diagnostic and figure scripts.
4. Run the focused V42--V44 tests.
5. Build the active paper twice to resolve references.
6. Inspect the abstract, V44 section, figure, and conclusion from rendered PDF
   pages.
7. Search extracted PDF text for the fixed metrics and rejected claim.
8. Run `git diff --check`, commit the revision atomically, and push.

## Redline Note

The repository has no designated baseline TeX for this continuously evolving
paper. The resulting Git commit is the traceable revision baseline.

## Execution Record

Completed on 2026-08-14:

- copied the compact training, development, and post-result diagnostic receipts;
- regenerated the V44 figure byte-for-byte from hash-pinned evidence;
- passed Ruff and Python compilation for both evidence scripts;
- passed all 18 focused V42--V44 tests, with one upstream `fontTools` warning;
- built the 85-page active paper through two LaTeX passes with resolved V44
  references;
- checked extracted abstract, evidence, and conclusion text for the fixed
  metrics and rejection language;
- visually inspected the abstract, V44 table, full-page result figure, and
  conclusion at rendered PDF resolution; and
- confirmed that the writer and frozen partition remain closed in every public
  claim.

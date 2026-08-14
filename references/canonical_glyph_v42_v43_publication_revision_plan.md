# Canonical Glyph V42/V43 Publication Revision Plan

Status: approved internal evidence update, written before editing the active TeX
manuscript.

Date: 2026-08-14

## Problem

The active README and image-native paper stop at V41. They therefore describe
V39.1 as the latest language result and V41 as the latest positive mechanism,
although the project has since completed the preregistered V42 and V43 runs.
The public account must distinguish the first bounded positive image-native
language evidence from the still-failing binding and raster-quality gates.

## Scientific Claim Change

Before: the project has a qualified image-conditioned glyph motor but no
positive natural-language result.

After: V42 provides bounded positive evidence that an image-only causal model
uses ordered Chinese raster history to predict a next glyph image better than
unigram, bigram, and shuffled-history controls. V43 preserves that bounded
signal and improves autonomous raster quality, but remains partial because
exact-suffix counterfactual arm accuracy is `0.54492 < 0.60` and generated
pixel F1 is `0.44507 < 0.55`.

The revision must not claim general question answering, historical-form
generation, Qwen parity, superiority to token LMs, a complete ILM, or a passed
V43 gate. The frozen partition remains unopened.

## Allowed Files And Locations

- `README.md`: hero qualification, current-evidence section, evidence
  trajectory, and relevant file links.
- `publication/ilm-image-native/ilm-image-native.tex`: abstract tail, hero
  caption, contribution list, one new V42/V43 subsection before Data,
  evaluation summary, and conclusion wording where the stale latest-result
  claim appears.
- `publication/ilm-image-native/evidence/v42/**`: small immutable summaries and
  the measured V42 target/generated sheet.
- `publication/ilm-image-native/evidence/v43/**`: small immutable summaries,
  measured V43 target/generated sheet, and post-result diagnostic receipt.
- `publication/ilm-image-native/generate_v43_result_figure.py`: deterministic
  evidence-backed figure generator.
- `publication/ilm-image-native/figures/canonical_glyph_flow_v43_result.png`:
  generated result figure.
- `references/canonical_glyph_flow_v43_result.md`: full bounded result and next
  decision.

## Figure Content

The new figure will use only measured artifacts. It will show:

1. the image-only `64 x 32 x 32` causal reader and bank-free spatial writer;
2. V42/V43 natural-language and control metrics;
3. counterfactual pair and generated-pixel gates with explicit failures;
4. real alternating target/generated glyph cells from V43; and
5. the post-result separation between a strong exact-plan writer ceiling and a
   weak predicted plan.

The figure must label development evidence, failed gates, and evaluator-only
oracle diagnostics. It may not visually imply a complete model.

## Out Of Scope

- Changing V42 or V43 protocols, gates, seeds, or measured reports.
- Opening or discussing results from the frozen partition.
- Editing older experiment numbers except where needed to connect the evidence
  trajectory.
- Retrospective algorithm changes to V43.
- A broad multilingual README rewrite; translation synchronization is a
  separate documentation unit.

## Verification

1. Hash every copied evidence artifact and pin those hashes in the figure
   generator.
2. Regenerate the figure from tracked evidence.
3. Run Ruff and Python compilation on the generator and V43 diagnostic.
4. Build `publication/ilm-image-native/ilm-image-native.pdf` with the project
   Makefile.
5. Inspect the generated figure and relevant manuscript pages as images.
6. Search extracted PDF text for V42/V43 metrics and qualification wording.
7. Run `git diff --check`, commit the bounded revision atomically, and push.

## Redline Note

This repository has no designated baseline TeX for the continuously evolving
ILM paper. A formal redline is therefore not generated. The Git commit is the
traceable baseline for this revision unit.

## Execution Record

Completed on 2026-08-14:

- copied eight compact V42/V43 receipts into the tracked publication evidence
  tree and pinned every SHA-256 in the deterministic figure generator;
- regenerated and visually inspected the measured result figure;
- rebuilt the active manuscript twice to resolve references, producing an
  82-page PDF with no build error;
- visually inspected the abstract, V42/V43 result section, evidence figure,
  conclusion, and adjoining page breaks;
- verified extracted PDF text contains the fixed metrics, failed gates, frozen
  split status, and bounded claim language;
- passed Ruff and Python compilation for the figure and diagnostic scripts;
- passed all 12 focused V42/V43 tests; and
- passed `git diff --check`.

The only test warning is an upstream `fontTools` deprecation notice. It does
not affect the measured result or generated artifacts.

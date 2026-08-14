# Noise-Limited Retinal Field V45 Publication Revision Plan

Status: approved internal evidence update, written before editing the active
TeX manuscript.

Date: 2026-08-14

## Problem

The README and active paper end with V44's representational diagnosis and name
a centered, variance-balanced raster field as the next bounded experiment. The
preregistered V45 production audit has now run from frozen code and passed all
13 gates. The public record must report the qualified representation without
turning a fixed geometry audit into an unsupported language-model claim.

## Scientific Claim Change

Before: V44 shows that raw normalized DCT geometry is dominated by common image
modes. A reversible, training-only, variance-balanced representation must pass
held-out geometry tests before another causal language run is justified.

After: V45 fits a fixed zero-parameter matrix-power field from 8,000
training-only canonical rasters. Direction plus log radius is exactly
invertible. Across 14,144 audited fit, held-font, and translated rasters, the
maximum FP64 DCT error is `4.2633e-14`, binary pixel accuracy and ink F1 are
both `1.0`, and no output is blank. The weighted common resultant falls from
`0.71900` to `0.01524`; effective rank rises from `145.08` to `185.38`. On the
pinned 1,024-pair V44 holdout, candidate-pair cosine falls from `0.56319` to
`0.06402`, fifth-percentile displacement norm rises from `0.36976` to
`0.77724`, effective displacement rank rises from `122.80` to `144.95`, and
stable rank rises from `60.82` to `67.79`. Held-font and one-pixel-shift
retrieval gates pass. V45 therefore qualifies the representation at `13/13`
gates in `66.51` seconds with `0.30212` GiB peak allocated VRAM.

The fixed V42-coordinate retrofit is report-only and degrades matched natural
top-1 from `0.19434` to `0.15039`. This is expected evidence against retrofitting
an old reader trained in another geometry. It does not negate the V45 geometry
qualification, and it requires the next causal language core to train from
scratch in V45 coordinates.

The revision may not claim that V45 itself learns language, improves V42
language accuracy, yields readable writing, establishes LLM parity, or proves
compute superiority. The V43 writer and frozen evaluation partition remain
closed.

## Allowed Files And Locations

- `README.md`: current-evidence heading, V45 result summary, evidence links,
  trajectory, and next bounded decision.
- `publication/ilm-image-native/ilm-image-native.tex`: abstract tail,
  contribution list, V42--V45 evidence section, synthesis, limitations, and
  conclusion.
- `publication/ilm-image-native/ilm-image-native.pdf`: rebuilt active paper.
- `publication/ilm-image-native/evidence/v45/report.json`: compact immutable
  production audit receipt copied from the official artifact.
- `publication/ilm-image-native/evidence/v45/checkpoint_receipt.json`: compact
  checkpoint, protocol, field-state, corpus, and holdout hashes without model
  tensors.
- `publication/ilm-image-native/generate_v45_result_figure.py`: deterministic,
  hash-pinned evidence renderer.
- `publication/ilm-image-native/figures/noise_limited_retinal_field_v45_result.png`:
  generated result figure.
- `publication/ilm-image-native/generate_figures.py` and
  `publication/ilm-image-native/figures/ilm_v_yan_readme_hero.png`: change only
  the embedded experiment-boundary footer from stale V38 wording to the current
  V45 representation/language boundary.
- `references/noise_limited_retinal_field_v45_result.md`: bounded decision and
  metric audit.
- This revision plan's execution record.

## Figure Content

The V45 figure will show only measured evidence:

1. the reversible raster-to-DCT-to-matrix-power-to-direction-plus-radius path;
2. the train-only statistic boundary and zero trainable parameters;
3. exact reconstruction over all seven audited banks;
4. common-resultant removal and covariance-rank gain;
5. pinned held-pair separation and displacement-rank gains;
6. held-font and one-pixel-shift continuity;
7. the report-only V42 retrofit decline; and
8. the authorized next step: a separately preregistered causal reader trained
   from scratch in V45 coordinates, while writer and frozen split stay closed.

The figure must label V45 as a qualified representation, not a qualified
language model. Its values must be read from hash-pinned evidence rather than
typed as independent constants.

## Out Of Scope

- Changing the V45 transform, corpus, fit bank, seeds, holdout, gates, or
  measured report.
- Retrofitting or fine-tuning V42, V43, or V44.
- Implementing the next causal language experiment in this revision unit.
- Opening the V43 writer or frozen evaluation partition.
- Updating translations before the English scientific account is verified.
- Formal redlines, because no stable manuscript baseline is designated.

## Verification

1. Copy the official V45 report and a tensor-free checkpoint receipt; pin their
   SHA-256 values in the figure generator.
2. Independently recompute all 13 gates from the tracked report.
3. Regenerate the figure deterministically and inspect it at full resolution.
4. Run Ruff and Python compilation on V45 source, evaluator, and figure code.
5. Run the focused V42, V44, and V45 tests.
6. Build the active paper through at least two LaTeX passes.
7. Search extracted PDF text for V45 qualification, `13/13`, the retrofit
   decline, and the writer/frozen boundary.
8. Visually inspect the abstract, V45 table/figure, synthesis, and conclusion
   from rendered PDF pages.
9. Run `git diff --check`, commit the revision atomically, and push.

## Redline Note

The repository has no designated baseline TeX for this continuously evolving
paper. The resulting Git commit is the traceable revision baseline.

## Execution Record

Completed on 2026-08-15:

- copied the official production report and created a tensor-free checkpoint
  receipt with matching checkpoint, field-state, corpus, protocol, partition,
  and held-pair hashes;
- independently recomputed the tracked report's gate map with the repository
  evaluator and matched all 13 passing gates exactly;
- generated the V45 figure twice with identical SHA-256
  `021cecb9256c9206a82a7cb4fa8aca6d950dc3769bae90085d97dd28d5ada542`;
- passed Ruff and Python compilation for the V45 model, evaluator, audit script,
  tests, and both touched figure generators;
- passed all 23 focused V42--V45 tests, with one upstream `fontTools`
  deprecation warning;
- rebuilt the 87-page active paper through two LaTeX passes with no undefined
  citation/reference, LaTeX error, fatal error, or overfull-box report;
- searched extracted PDF text for `13/13`, the `0.19434` to `0.15039` retrofit
  decline, the qualified-representation wording, and the next-reader boundary;
- visually inspected the abstract and concept hero on pages 2--3, V45 table on
  page 68, V45 result figure on page 69, and synthesis/conclusion on page 84;
- corrected only the concept hero's stale V38 footer and float placement after
  PDF inspection, leaving its illustrative answer explicitly unmeasured; and
- confirmed that the next causal reader requires separate preregistration and
  that the V43 writer and frozen partition remain closed.

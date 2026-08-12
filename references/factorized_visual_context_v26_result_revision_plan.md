# V26 Result Publication Revision Plan

Date: 2026-08-13

Status: completed internal result revision; plan written before editing the
active manuscript

## Problem

The fixed V26 training run and development audit are complete, but the tracked
README, research goal, and manuscript still stop at V25. The repository needs a
concise, reproducible account of the V26 negative result without weakening the
preregistered gates or presenting an internal-state change as language use.

## Scientific Claim

V26 implements the preregistered factorized visual-context model and completes
8,000 BF16 updates on one RTX 4090. It preserves the image-only student
boundary, achieves a perfect evaluator retina-bank oracle, and changes its
history residual across pixel-identical suffix pairs. It nevertheless fails to
bind those history changes to the correct next-glyph distribution:

- full-history top-1 is `0.000488`, below image unigram `0.014160` and symbolic
  bigram `0.135254`;
- the only non-invariant mechanism gate to pass is full-versus-last target log
  probability (`+0.170451` nat);
- full-versus-suffix-4 and full-versus-shuffled gains are only `+0.019729` and
  `+0.003692` nat;
- suffix-4 pair ranking and swapped-residual target accuracy are both exactly
  `0.500` despite mean history-residual difference `4.628258`; and
- the frozen split remains sealed and no writer is trained.

The supported interpretation is a clean falsification of this particular
factorization and objective. It is not evidence against image-native language
modeling in general, and it is not evidence of useful natural-language ability
or efficiency relative to a text model.

## Allowed Files And Locations

- Add `docs/factorized-visual-context-v26-result.md` as the complete receipt.
- Add `publication/ilm-image-native/generate_v26_result_figure.py` and its
  generated PNG under `publication/ilm-image-native/figures/`.
- Update the current-result section near the top of `README.md`, retaining V25
  as prior evidence.
- Update the milestone, status, and next-experiment passages in
  `docs/first-imagized-language-model-goal.md`.
- Update the abstract, contributions, V25-to-V26 methods/results transition,
  discussion, and conclusion in
  `publication/ilm-image-native/ilm-image-native.tex`.
- Rebuild the stable active artifact
  `publication/ilm-image-native/ilm-image-native.pdf`.

## Out Of Scope

- No changes to V26 code, checkpoints, audit JSON, protocol, thresholds, data,
  or partitions.
- No frozen evaluation and no pixel-writer training.
- No retrospective reclassification of V23 or V24.
- No claim of human-like reading, general question answering, arbitrary
  historical-form generation, Qwen parity, or language-model efficiency.
- No broad prose, bibliography, localization, or repository refactor.

## Figure Contract

The V26 figure must be generated directly from
`artifacts/factorized_visual_context_v26_evidence/development_audit.json`. Its
generator must reject smoke-only evidence, a changed architecture or protocol
hash, an opened frozen split, language/mechanism selection, an absent retina
oracle, or an unclean student boundary. The figure will show:

1. the appearance/history/particle architecture;
2. fixed natural-language controls;
3. the pixel-identical suffix-4 intervention;
4. internal-state difference versus chance target binding; and
5. the exact negative verdict and next localized hypothesis.

## Verification

1. Run the V26 figure generator and inspect the PNG visually.
2. Build the active paper with `latexmk` and require a clean exit.
3. Inspect the compiled pages containing the V26 table and figure.
4. Confirm the reported numbers against the fixed audit JSON.
5. Run the full Python test suite.
6. Confirm no ignored checkpoint or `.auto-readme-work/` content is staged.
7. Commit the bounded revision atomically and push `main`.

## Baseline And Redline Note

This repository keeps one stable active TeX file and does not contain a named
manuscript baseline or a redline build workflow. The Git parent of this
revision is the immutable baseline; `git diff` and the resulting commit provide
the revision trace. No separate redline artifact will be invented.

## Acceptance Criteria

- Every displayed quantitative claim is present in the fixed V26 audit.
- The distinction between changed hidden state and useful conditional behavior
  is explicit.
- The frozen split and writer remain unopened.
- The figure and PDF are readable and tracked.
- Tests and LaTeX build pass, then the atomic commit is pushed.

## Execution Record

- The evidence-derived figure was generated twice with identical SHA-256
  `593ca76cfe5af51306f67e151cf86ae8b3ee82636979b5b1c0c383d69e74b1cf`.
- The figure was inspected at its original `2400x1700` resolution; its branch
  diagram, labels, values, and verdict panels are readable and non-overlapping.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error
  ilm-image-native.tex` completed successfully and produced a 45-page PDF.
- Compiled PDF pages 28--30 were inspected for the V26 equations, tables,
  result prose, figure, and caption. Front-matter pages 1--3 were also checked
  after the abstract update. No overlap or clipping was observed.
- `PYTHONPATH=. pytest -q` completed with `229 passed` and one existing
  fontTools deprecation warning.
- `python -m py_compile
  publication/ilm-image-native/generate_v26_result_figure.py` passed.
- A direct `jq -e` audit confirmed the architecture, endpoint, window counts,
  reported natural and pair metrics, perfect retina oracle, failed selections,
  and sealed frozen state against the fixed JSON receipt.
- LaTeX auxiliary files were cleaned; the active PDF was retained.
- The tracked changes do not include `.auto-readme-work/`, model checkpoints,
  training logs, frozen images, cookies, or credentials.

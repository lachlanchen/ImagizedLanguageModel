# V47 measured-result manuscript revision plan

## Revision stage

Internal major revision after completion of the preregistered V47 development
experiment. This is a bounded evidence update, not a response to a reviewer and
not a general rewrite of the manuscript.

## Problem addressed

The active image-native manuscript ends at V46 and therefore does not record
the completed V47 test of the proposed codec-spherical mechanism. V47 is a
scientifically useful negative result: its frozen V34 visual field remains
qualified, but the single normalized glyph-vector causal objective loses both
natural continuation and counterfactual binding. Leaving the paper at V46
would overstate identity--radius factorization as the next supported path and
omit evidence that directly tests it.

## Frozen baseline and evidence

- Active manuscript: `publication/ilm-image-native/ilm-image-native.tex`
- Pre-revision Git baseline: `a6cc8d9e955c7e552b8f6adc58b6a958aeeddfea`
- Pre-revision TeX SHA-256:
  `dbd4aa4bd87e2c2b53632041dd44aba2690cf588f4dc86462ba63dee2830ab4b`
- Pre-revision PDF SHA-256:
  `e7f421537a194c39625b58e603ff03f8720388cfd2df7b65cd82e021a40d725e`
- V47 checkpoint SHA-256:
  `9bec353a278c56aa79f538a22fd0143f3d60b9366a7f782399941811d12663fc`
- Training summary SHA-256:
  `f474b2e5a572349cc8f26350906fd5ed2a431387ac1c0e80466d25739b7fa471`
- Development report SHA-256:
  `984d3dc380dc4c7b54f93bfce2bd8e4cb9dcb45524348adeacb441749ecea49d`
- Development summary SHA-256:
  `01b3a579774be93eb0260545ceb42bab792fe917269555c2f12877c8ce76237f`
- Raster triplet SHA-256:
  `0a3e51460c0abd4c2b43363c79fc2029db7a290004f8d2cc35cbfc8532e76c95`

There is no named TeX baseline or project redline convention in this manuscript
directory. Git commit `a6cc8d9` is the immutable baseline for this unit. A
temporary redline may be generated from Git for verification, but it will not
be treated as an active manuscript or silently replace the baseline.

## Supported measured claims

The revision may state only the following V47 claims unless another tracked
report is cited explicitly:

- Architecture: frozen 7.423M-parameter V34 EMA codec plus a 23.953M-trainable
  causal spherical model; 31.376M total parameters.
- Boundary: runtime and student training examples use image tensors only; no
  token IDs, Unicode IDs, OCR, strings, glyph lookup, visual codebook, candidate
  bank, or external language model is supplied to the student.
- Training: 10,000 updates, exactly 80,000 unique counterfactual rows consumed
  without replacement, 2,286.17 seconds, and 2.78614 GiB peak allocated CUDA
  memory.
- Frozen-field preflight passes: canonical reconstruction ink F1 0.99997,
  canonical decode--reread cosine 0.99996, held-sans top-1 0.97168, and
  held-serif top-1 0.90137.
- Fixed development language audit on 2,048 windows: full top-1 0.02539,
  symbolic-bigram top-1 0.12256, unigram top-1 0.01416, shuffled top-1 0.02539,
  and full target log probability -6.79318 versus -6.79904 shuffled.
- Counterfactual full-arm accuracy is 0.52930 on 512 pairs.
- Autonomous generation on 256 examples: identity top-1 0.02344, pixel F1
  0.37732, blank rate 0, and selected-proposal/visible-reread cosine 0.63081.
- Six of sixteen preregistered gates pass. The result is
  `non-qualifying-development-result`.
- Interpretation: the visual codec remains a valid retina/actuator, but one
  normalized glyph vector with anchor-plus-sample prediction is not a
  sufficient causal language state. This does not reject image-native language
  modeling or continuous visual states generally.

## Planned edits

### Allowed files

- `references/v47_measured_result_paper_revision.md`
- `publication/ilm-image-native/evidence/v47/training_summary.json`
- `publication/ilm-image-native/evidence/v47/development_report.json`
- `publication/ilm-image-native/evidence/v47/development_summary.json`
- `publication/ilm-image-native/evidence/v47/target_generated_reread_triplets.png`
- `publication/ilm-image-native/generate_v47_result_figure.py`
- `publication/ilm-image-native/figures/codec_spherical_glyph_language_v47_result.png`
- `publication/ilm-image-native/ilm-image-native.tex`
- `publication/ilm-image-native/ilm-image-native.pdf`
- a temporary, untracked redline/build directory used only for verification

### Allowed manuscript locations

1. Abstract: append one compact V47 paragraph after V46.
2. Hero caption: extend the measured-series references through V47.
3. Contributions: add one V47 item after V46.
4. Rename the bounded section from `V42--V46` to `V42--V47`.
5. Results: add a V47 mathematical description, measured table, evidence
   figure, and interpretation immediately after V46.
6. Discussion synthesis: extend the V42--V46 paragraph through V47.
7. Conclusion: add the measured V47 result and replace the obsolete proposed
   identity--radius intervention with the next inference supported by V47.
8. Product-target limitation: change `V42--V46` to `V42--V47`.

### Figure requirements

The result figure must be generated deterministically from the four pinned V47
evidence files. It must show the actual target/generated/reread raster sheet,
the 6/16 frozen decision, exact principal metrics, parameter/runtime receipts,
and a restrained conclusion. It must not depict illustrative model success or
use generated decorative art as experimental evidence.

## Explicitly out of scope

- No claim that V47 is a usable ILM, understands language generally, matches a
  local Qwen model, or improves efficiency over token LLMs.
- No sealed evaluation, post-hoc threshold change, checkpoint selection,
  retraining, or reinterpretation of failed gates.
- No change to V34--V46 measured values.
- No new V48 architecture, experiment, or speculative result in this revision.
- No general prose cleanup, bibliography expansion, README change, or unrelated
  publication edit.

## Verification and acceptance criteria

1. Evidence copies match the frozen source hashes above.
2. The figure generator rejects altered evidence hashes and validates the
   10,000-update checkpoint, 80,000 unique rows, clean boundary, and 6/16 gate
   partition.
3. Every manuscript number agrees with the copied development/training reports.
4. `scripts/latex_build.sh publication/ilm-image-native/ilm-image-native.tex`
   succeeds in two passes.
5. `pdftotext -layout` confirms the V47 abstract, result table, interpretation,
   conclusion, and figure caption are present in the compiled PDF.
6. The output figure is inspected at original resolution and the final PDF is
   rendered/inspected on pages containing the V47 material.
7. A temporary latexdiff against Git baseline `a6cc8d9` is attempted if the
   installed tool supports the manuscript; any failure is recorded rather than
   hidden.
8. `git diff --check`, focused source searches, and repository status pass;
   unrelated `.auto-readme-work/` remains untouched and unstaged.

## Response-letter impact

None. There is no reviewer response letter in scope. The plan itself is the
trace record for this internal measured-result revision.

## Deviations and build record

To be completed after execution.


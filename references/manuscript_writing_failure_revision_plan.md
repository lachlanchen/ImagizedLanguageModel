# Manuscript revision: locate the writing-interface failure

Stage: internal research update under the standing manuscript-maintenance
request. No journal review or submission is implied. The current goal is
useful autonomous image-language generation; this revision reports measured
progress and failures without claiming that goal is complete.

## Scope and evidence

Preserve the current 50-slide, 54-native-chart publication as a Git baseline
at `947058e7`. Update its abstract, add one bounded Results subsection and a
Supplementary Note, and append two editable evidence figures. Keep prior
study methods and figures intact. All changes must be staged away from active
files before building and visual review.

The new text may cite only completed retained results: the 512-question
diversity trial and its independent audit, fixed-state first-image likelihood,
matched normal/erased/text native-reader baseline, and the completed glyph
primitive. Sources are the corresponding `references/evidence/` directories
and result documents. Stage B language continuation is running and must not
be described as successful or completed. If its assessment completes before
the revision is built, update this plan first to define any additional scope.

Replace the abstract's current emphasis on 32/32 fitting with the broader
495/512 training fit, zero exact development answers, the lower reference
likelihood, and the reader baseline's 12/16 literal responses. Explain the
distinction between helper text, autonomous pixel answers, relative question
preferences, and semantic accuracy. Mention clear primitive glyph formation
as a rendering result only. Preserve the motivation of unencoded writing and
the independent image-input/image-output objective, along with external
pretraining and unachieved generalization/efficiency limits.

Figure S36 should combine source-bound training/development measurements,
first-image likelihood gaps and native-reader literal counts, each labelled
with its own metric and denominator. Figure S37 should show the explicit
continuous glyph loop and unrepaired primitive reconstructions, with clear
separation between completed primitive training and unvalidated language
generation. Use native editable shapes/charts and linked data workbooks;
quantitative figures must not be AI-generated. Preserve all prior ZIP parts
except presentation registration metadata, and verify old slide appearances.

## Allowed files

- New builder/text/figure modules in `publication/ilm-nature/` and staged
  artifacts under `artifacts/`.
- `frontmatter.tex`, one Results insertion in `article_content.tex`, one
  input in `supplementary_information.tex`, new bounded result/note inputs.
- Matching current PDFs, redlines, figure PDF/PNG/SVG, editable deck and its
  review PDF after visual review.
- Publication README's current-result summary and evidence/build receipts.
- Baseline manifest, this plan and verification notes under `references/`.

Out of scope: changes to completed experiment code/protocol/data, alteration
of measured outputs, physical-law or human-brain claims, historical-script
success, small-foundation or matched-quality efficiency claims, and wholesale
rewriting of unrelated historical results. No response letter is needed.

## Verification and delivery

Authenticate all evidence hashes, validate chart data against retained
measurements, and preserve the earlier publication bytes by baseline commit.
Render new native slides and inspect alignment, readable labels, metric scope
and unchanged model pixels. Compile article/review/supplement and matching
latexdiff copies only while the project's neural job is absent and its GPU
mutex is available. Respect workstation RAM/pressure limits and preserve
foreign jobs. Reject unresolved references, citations, LaTeX errors and new
overfull boxes. Inspect changed PDF pages and retain their actual locations
and hashes; never invent review completion.

After the staged package passes these checks, promote to stable filenames,
commit and push. Copy the verified PDFs and editable figures to the existing
authorized Nutstore ILM project folder, retaining the immediately preceding
package and verifying local hashes. Distinguish local copying from cloud
propagation. This plan authorizes no success claim before actual evidence.

Status: plan recorded; TeX files and current publication artifacts have not
yet been changed by this revision.

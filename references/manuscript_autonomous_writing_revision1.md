# Recognition and autonomous-writing manuscript revision

Stage: internal research manuscript revision, under the continuing ILM goal.
Status: plan prepared before TeX changes; pilot evaluation is still pending.

## Scope and reason

The active manuscript does not yet contain the completed recognition-loss
comparison, the complete reader diagnostic on generated pages, or the native
full-trajectory preflight. Add these and the final registered 32-update pilot
result as one bounded unit, after complete pilot assessment. Preserve the
distinction between visual reconstruction, copying and semantic answering.
Do not characterize successful code execution as a working language model.

Use baseline commit `5c458a2bc2f490769ca94a084293f5961e75b44d`. Its active deck
has 38 slides and 44 editable charts. Its existing figures, slide pixels,
masters and measured source images must remain unchanged. The stable active
TeX files are `frontmatter.tex`, `article_content.tex`, and
`supplementary_information.tex`; do not rename them to generic main/sm files.

## Allowed changes

- Add one short latest-result sentence to the abstract only after final pilot
  counts are verified. Preserve the general image-input/image-output aim,
  external-pretraining disclosure and unachieved language/efficiency claims.
- Add a compact results paragraph comparing the previous flow-only copying
  (6/32 seen, 10/32 held out), recognition loss (6/32, 9/32), and the pilot's
  actual counts. Avoid implying this is a matched-from-scratch comparison:
  the pilot continues the recognition checkpoint for 32 additional updates.
- Add Supplementary Notes 47 and 48 as separate included files, after the
  existing history-prefix note and before Reproduction. Note 47 covers the
  recognition comparison and complete autonomous reader diagnostic. Note 48
  gives the precise final-image objective, native-precision correction,
  full-trajectory execution proof, complete pilot result and limitations.
- Append native editable Figures S24 and S25 as slides 39 and 40. S24 shows
  recognition results and the assisted/free loss distinction; S25 shows the
  actual noise-to-image learning path, full cohort copying scores and the
  first two held-out cases (000 Chinese, 001 English), fixed before the new
  evaluation. Use raw PNGs and native charts/diagram objects; no generated
  illustration is experimental evidence.
- Add dedicated figure/build verification code, source receipts, matching
  PDF/PNG/SVG figure exports, refreshed manuscript/review/SI/redline PDFs,
  publication README and locally verified Nutstore package/latest aliases.

## Claims and exclusions

The reader diagnostic is not an independent quality evaluator. Keep its
14/32 and 19/32 transcripts distinct from CPU evaluation's 6/32 and 9/32.
Report that it can ignore repeated characters and duplicate lines. Describe
full-trajectory optimization as an application of DRaFT, not a newly invented
algorithm. Report complete runtime parameters, local trainable parameters,
allocated memory, time and inherited pretraining. Do not claim historical
script understanding, general language learning, parity with an LLM, small
from-scratch training, or quality gains unless directly established.

Do not rewrite earlier experiments, replace their figures, remove failed
outputs, alter the pilot's prospective progress/quality rules or select a
favourable checkpoint. Other pending representation/history experiments are
outside this revision unit; retain their source reports without inserting
unassessed claims. There is no reviewer response letter in this internal unit.

## Verification and build sequence

1. Confirm the existing owned training process is terminal. Complete all 64
   fresh native inference calls with no teacher/targets, then the unchanged
   separate evaluator and raw-image review. Authenticate actual tensors,
   input/seed bindings, all 32 training draws and every exported PNG.
2. Revalidate the full evidence before updating this plan with the final
   result wording. Freeze the baseline source and artifacts in private QA.
3. Inspect the existing source slide style and use the established native
   OOXML workflow if the local ArtifactTool runtime remains unavailable.
   Preserve all 38 earlier slide XML parts and rendered pixels, all 44 charts
   and prior figure assets; append only. Keep chart data and diagrams editable.
4. Follow the workstation policy: no figure/LaTeX heavy build while the model
   or another project-owned heavy job holds the build/GPU lock. Use the
   existing build slot when idle; do not rerun an old 36-slide generator.
5. Compile article, review, SI and matching redlines. Check overfull boxes,
   citations, labels, numbering and figure references from logs and extracted
   PDF text. Render all new figure pages and modified article/SI pages;
   inspect layout and source-image fidelity. Record actual page anchors.
6. Check PDF structure, baseline preservation, native editability, plot data
   and no missing raw images. Sync only the reviewed complete package to
   Nutstore, verify local hashes, and make no unverified cloud-sync claim.
7. Record any deviations and completed checks here, then commit and push the
   finished revision. Keep the full ILM goal active if language evidence fails.

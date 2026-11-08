# Recognition and autonomous-writing manuscript revision

Stage: internal research manuscript revision, under the continuing ILM goal.
Status: manuscript, review, supplement, two editable figures and matching
redlines built and visually checked on 8 November 2026. Local delivery and
commit verification are recorded below.

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
- Update the one outdated Discussion summary of free copying to say that
  reliable copying remains unproved; preserve its separation from the
  source-crop component. Add the two primary method citations to the shared
  bibliography. No other interpretation or earlier experiment is rewritten.
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

## Assessed result and permitted wording

The 32-update pilot finishes with all 64 native outputs, unchanged independent
CPU scoring and complete raw before/after galleries. OCR exact is 15/32 seen
and 13/32 held out, versus 6/32 and 9/32 at its start. All 32 IDs/noise seeds
reproduce the prospective streams; 22 distinct seen examples are sampled.
Training takes 1,135.452 seconds at 12.057 GiB; native assessment takes 518.400
seconds at 8.031 GiB. Both jobs are terminal and their owned runtimes exited.

Use the result wording: "Direct final-image continuation increases OCR-exact
copying from 6/32 to 15/32 on seen inputs and from 9/32 to 13/32 on held-out
inputs, but still fails reliable writing." Preserve the canonical-font scope,
OCR limitations, absent matched additional-training control, corrected input
precision and lack of semantic QA. Do not call this a validated causal repair,
usable language model or matched-quality efficiency gain. Full source evidence
is in `references/evidence/autonomous_writer_result/result.json`.

## Execution and verified locations

The result sentence appears in the article abstract on page 1. The new
Results paragraph begins “Frozen recognition supervision also fails reliable
copying” on page 12, right column, and on page 13 of the line-numbered review.
The Discussion now says free writers fail *reliable* copying. Note 47 is on
supplement page 53 and Figure S24 on page 54. Note 48 occupies pages 55–56,
with Figure S25 on page 57. Equations 24–31 give both objectives and the
full native sampling trajectory. The two new primary citations are 22–23.

The final article/review/SI have 16/29/61 pages; current-baseline redlines have
29/61 pages. All compile without overfull boxes or unresolved references or
citations and pass qpdf checks. Complete changed-page overviews and the new
scientific pages were rendered; the visual-review receipt distinguishes
overview inspection from full-size reading. New figures retain native charts,
embedded workbook values and original evidence PNGs. Native package and
geometry checks find zero defects, with 40 slides and 47 charts. All 38 prior
slide appearances, all 107 prior figure assets and the first 52 supplement
pages are unchanged. Actual PowerPoint-app inspection and Artifact Tool are
unavailable; Arial renders with Liberation Sans.

Dedicated current builders are `autonomous_writing_figures.py`,
`build_autonomous_writing_publication.py` and
`build_autonomous_writing_redlines.py`. The old producers remain unchanged.
The baseline manifest can be reconstructed from Git on a fresh checkout.
An initial build check expected a different caption prefix; it was corrected
to the actual `Fig. S24`/`Fig. S25` captions without changing the figures.
The early S25 layout was refined to separate the noise and backward-gradient
arrows and enlarge the two complete example pages. No model outputs changed.

The publication README and engineering goal now expose the current failure
boundary. `references/ilm_direction_assessment_20261108.md` explains why losses
and execution checks have not qualified a model; its proposed matched
continuation remains unrun and requires a separate protocol. No new model
optimization or repaired outputs were introduced in this revision unit.
There is no reviewer response letter for this internal revision.

Delivery target: `ILM_NatureStyle_20261108_AutonomousWriting` with the five
stable latest aliases in the local Nutstore ILM folder. Hash verification
establishes local copying only; cloud propagation is not asserted. The ILM
goal remains active and unachieved.

Out-of-scope discovery: the repository-profile checker fails for missing
`CITATION.cff`, funding metadata and incomplete translation parity. A separate
check on the unchanged HEAD versions confirms these gaps predate this unit.
No profile, funding or translation-completeness claim is made here. The
research artifact checks above pass; the profile check does not.

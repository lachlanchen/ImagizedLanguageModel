# Completed conditional-writing results: publication revision plan

Stage: internal research-draft revision, under the ongoing Nature-style
manuscript and editable-figure maintenance task. No external reviewer response
or journal submission is involved. This plan is created before TeX, PDF or
PowerPoint edits for this revision unit.

## Evidence and scientific purpose

The current paper reports completed writing-component experiments and a
complete-actor execution preflight, while conditional learning is prospective.
That description is now incomplete. V68 has completed 10,000 updates and an
independently verified 384-generation audit. It has zero exact Chinese answers
and fails three absolute answer-quality requirements, despite resolved question
dependence and correct stopping. A frozen 32-case diagnostic shows poor
prediction with correct history, including training examples, while supplied
writing codes reconstruct the images faithfully.

Use these source-bound records:

- `references/codec_language_v68_training_result.md` and its integrity receipt.
- `references/codec_language_v68_result.md` and the exact autonomous-verification
  receipt, SHA-256 `ca09f2fb88374406cc2ddfc8e44637776ddbea2fc5c18240ebce83b8df9c026c`.
- `references/codec_language_v68_diagnosis_result.md` and its raw-measurement
  receipt, SHA-256 `69446b336ee39016758ad9229bbb5d48788dfcf931a44303fa69e83275c2b461`.
- Original raw rasters and immutable reports in the two local V68 run folders.

The revision must distinguish reconstruction, assisted prediction and
autonomous generation. It must not call input dependence semantic correctness,
claim a successful language model, or identify an untested repair as a result.

## Preservation before editing

Freeze the current committed `publication/ilm-nature/` package in a new named
baseline, for example `baselines/v67-writing-completed/`, with a file-hash
manifest and source commit. Exclude nested baselines and build caches from
that snapshot; preserve the active TeX, style, bibliography, deck, PDF and
figure assets. Do not overwrite an existing baseline. Verify every copied
file against Git and retain the existing V63 baseline unchanged.

Keep the active filenames stable. The existing 22 editable slides, their
rendered content, and every existing figure asset must remain unchanged.
Extend the deck and add new standalone figures. Reuse the existing Lamp+LCD
style, native chart/workbook conventions and source-raster placement methods.

## Bounded manuscript changes

Allowed files are the active `frontmatter.tex`, `article_content.tex`,
`supplementary_information.tex`, package README, new data/figure helpers,
redline builder, validation/sync helpers and their evidence manifests.
Do not alter old experimental reports, data, protocols, source models or
historical numerical results.

1. **Abstract:** replace the prospective endpoint beginning “These reconstruction
   and execution results do not establish autonomous language” with the actual
   V68 failure and the limited exact-history diagnosis. Retain the general
   image-language aim and external-pretraining boundary; keep the abstract
   concise by shortening historical detail rather than adding promises.
2. **Results:** retain the existing preflight paragraph as execution evidence,
   then add the completed adaptation, original six language decisions, matched
   blank control, bootstrap interval and first fixed raw outputs. Introduce
   the assisted diagnostic and its known-history/known-length limitation.
3. **Discussion:** replace “predicted-code selection and free image feedback
   still require autonomous language tests” with the observed rejection.
   Explain that errors precede generated history and remain on training cases.
   Feature information, capacity, loss and optimization remain alternatives;
   no new optimizer, density or teacher-transfer method is yet demonstrated.
4. **Methods and supplement:** add the complete training budget and provenance,
   pre-generation layout correction, full-width/stop scoring, article bootstrap,
   diagnostic selection and exact cached-history procedure. Record all timing
   boundaries, measured memory, zero diagnostic updates, reused development,
   and unopened sealed data. Separate neural replay from offline physical checks.

## Figures and editable deck

Add main Figure 10 for autonomous outcomes and Supplementary Figure S9 for
the assisted diagnosis, with captions and references in the active text.
Append two slides to the 22-slide native editable deck; use native data charts
backed by the actual receipts. Use deterministic first cases, never best-case
selection. Preserve complete source strips or clearly identify any display-only
excerpt without changing scoring or hiding nonblank pixels. Distinguish supplied
codes, supplied history/length and prompt-only output directly in panel labels.
Any pipeline diagram should use editable shapes and match actual code.
Retain existing AI illustrations as historical/conceptual assets; do not use
synthetic writing as experimental output.

## Build, inspection and delivery

Use the PDF and Presentations skills for the new authoring unit, alongside
paper-revision and publisher-voice. Begin their required edit markers before
modifying those artifacts. Inspect the established build/sync scripts before
extending them. Rebuild the article, review, supplement, both matching redlines,
24-slide review deck and two new standalone figure PDFs. Render and visually
inspect all changed pages, both new slides/figures and affected redline pages.

Require clean TeX references and no unexplained overfull boxes; verify actual
counts and rounded values against receipts, exact source-raster placement,
unchanged older slides/figures and unchanged baseline hashes. Report any
native PowerPoint/ArtifactTool fallback honestly. Record build commands,
source/output hashes, visual scope and limitations in the verification receipt.

Sync the complete verified package and matching latest aliases into a fresh
Nutstore `Projects/ILM` version folder. Verify local byte equality; do not
claim remote cloud propagation without evidence. Commit and push the finished
revision unit. A language failure remains a failure after publication styling.

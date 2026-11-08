# Complete native QA learning result: manuscript revision plan

Stage: complete registered experiment and raw review available; publication
build waits for the separate active native flow training job to finish.
No heavy manuscript/figure build is admitted alongside that model stage.
This plan prepares one result unit; it does not qualify the independent ILM.

Lightweight preparation may run while the separate model job owns GPU0:
an isolated CPU builder may validate the completed QA export, append native
OOXML slides to immutable baseline bytes and write draft TeX fragments under
`artifacts/native_qa_publication_stage_v1/`. It must not load a neural model,
start Office/LaTeX, modify active manuscript/figure files or update Nutstore
aliases. This staging is not a rendered publication or a visual approval.
The final locked build and PDF review below remain required.

## Baseline and current evidence

Freeze publication commit `daef4bf`: 43 native slides, 51 native charts,
149 figure assets, 16-page article, 29-page review and 68-page supplement.
The active publication tree is unchanged from that commit when this plan is
recorded. Use immutable Git bytes for baseline manifests and redlines; retain
active filenames and reading style. `.auto-readme-work/` is outside scope.

The new experiment is registered in
`references/native_paired_qa_learning_protocol.md`. It starts from the fixed
ink-mass checkpoint, performs four permutations of 32 natural training
questions for 128 updates, and evaluates only the final model. Its complete
assessment has 320 new native calls: 256 initial/final QA and control calls,
plus 64 final copying calls. Initial copying is reused from its authenticated
complete report, not newly generated. Training-only reader supervision and
retained generator pretraining must remain explicit.

Third-pass training-page inspection shows faint, corrupted source fragments.
Those changing-weight/changing-noise training outputs are not a final or
held-out result. Their loss trajectory cannot populate any answer-accuracy
claim. The separate visual-semantic candidate and single-step preflight have
now completed execution checks; a separate 1,536-update learning run is active.
Those experiments remain outside this result revision.

The [complete result](native_paired_qa_learning_result.md) now retains all
384 original answer pages and 128 training pages. Both initial and final
models score 0/32 seen and 0/32 held-out source-accepted OCR-exact answers,
0/16 both-correct held-out passage pairs and zero in both erased-input cohorts.
Held-out copying falls from 19/32 to 16/32. All six prospective requirements
fail. Independent export authenticates the actual tensors, and bound raw
inspection covers all 384 original answers without transcript or score repair.
These complete observations can populate the scoped revision after model
execution releases the shared resource lock.

## Required evidence before execution

- Exact terminal coordinator state, final 128-update checkpoint, complete
  optimizer/RNG/schedule bindings and all 128 actual pre-update training pages.
- Complete source-bound assessment with 320 new answer pages and 64 reused
  initial copying pages, authenticated against actual tensors.
- Independent export recomputation of source-accepted and canonical OCR
  counts, both-correct passage pairs, erasure effects, copying retention and
  every prospective requirement. Do not relax thresholds after seeing results.
- Complete raw review of all initial/final normal, erased-input and copying
  pages. Preserve uncertain transcription, extra text, omissions, hallucinated
  strokes and disagreement with OCR. Raw review does not silently repair scores.
- Exact runtime costs and memory from the real training/inference reports,
  with no estimate substituted for an actual measurement.

The evaluator limitation is material: canonical reference OCR is 23/32 on
seen pages and 26/32 on held-out pages, while the registered seen requirement
is 24/32. This is a mismatch at the reference typography, not an upper bound
on a different generated style. Preserve the requirement and disclose the
limitation; do not use OCR alone to certify writing.

## Allowed writing changes

`publication/ilm-nature/frontmatter.tex`: adjust only the latest result and
qualification sentences after the full result is known. The current final
sentences say that copying/reconstruction have not established reliable
conditional image-language generation and that broader requirements remain
unachieved. The successor may state the observed initial/final held-out QA
counts and whether the registered developmental rule passes. Preserve failed
requirements and distinguish a narrow reused-development signal from the
full ILM objective. Do not increase the abstract into an experiment ledger.

`article_content.tex`: add one compact Results paragraph after the copying
and full-page preflight paragraphs. Report complete initial/final normal QA,
paired-question and erased-input behavior, copying retention and the declared
decision. Give the actual training cost; explain the implication for scaling
this unchanged route. Avoid attributing the copying improvement to semantic
learning. Existing earlier experiment descriptions remain stable.

Add Supplementary Note 52 in a new included TeX file, with complete method,
geometry, source/data/schedule bindings, inference-only information boundary,
loss terms, gradient/native equality checks, exact counts and costs, evaluator
calibration, all prospective requirements and complete evidence locations.
Document the early save-path failure as an execution correction before any
optimizer update, retaining the failed attempt and exact replay evidence.
Do not describe it as a quality-selected restart. Explain that the initial
model is assessed after training under the fixed prospective schedule.

No new Introduction, foundation-training claim, historical-script result,
text-LLM efficiency comparison or physics theorem enters this revision.
No reviewer response letter is involved.

## Editable figures and scope

Append native PowerPoint Figures S29–S30 as slides 44–45 using the existing
editable OOXML/chart/workbook approach. Preserve all prior slide parts,
appearances, chart workbooks and figure assets. Successor builders must not
overwrite historical producer scripts or baseline packages.

Figure S29: complete initial/final normal seen/held QA counts, both-correct
held-out pairs, held-out normal versus both erased-input counts, and copying
retention. Use explicit denominators (32 questions or 16 passage pairs),
native editable charts and exact numerical labels. Show the registered
qualification and reused-development/single-schedule limits. A separate
training-loss diagnostic, if needed, must identify changing weights/noise
and must not be labelled semantic accuracy.

Use an editable count table for the five identically zero QA measures and
a native chart with embedded workbook for copying. Zero-height bars obscure
the QA counts without adding a useful comparison. This layout preserves all
five QA denominators, both checkpoints and the complete copying result.

Figure S30: the fixed first held-out passage pair, its complete question
inputs, references and original normal/control answers. Choose by the
registered index, never by correctness or visual appeal. Keep every displayed
model answer complete and unrepaired. Make the layout readable; use source
question insets only when labelled and linked to their exact original
rectangles, with complete pages retained in the evidence. Include the
initial/final distinction and reveal extra writing. The complete 384-page
export remains available beyond the representative figure.

## Build, verification and delivery

Wait for all owned model jobs to finish, then reserve the shared
publication/model lock. Do not run heavy LibreOffice/LaTeX work alongside
an owned model stage or race a queued model for its admission slot.
Create successor figure, manuscript and redline builders. Rebuild article,
review and supplement, with matched redlines against the 43-slide baseline.
Keep completed figure and paper source filenames stable.

Check every native chart and embedded workbook against the complete exported
result. Authenticate evidence image pixels in the compiled PDFs. Verify old
slide parts/assets and the established style are preserved. Inspect actual
new slide PNGs, changed scientific PDF pages and redlines; resolve overlaps,
unreadable labels, missing references and LaTeX/PDF structural errors.
Record final page anchors and build status from the reviewed PDFs.

Copy only the reviewed package and established latest aliases to the local
Nutstore ILM project folder, with hash verification and prior alias backups.
Local copy verification does not establish cloud propagation. Commit and
push the completed result/publication unit. Until all conditions above are
met, leave the existing manuscript and its qualification unchanged.

## Isolated preparation completed

`publication/ilm-nature/prepare_native_qa_publication.py` has run on CPU with
CUDA hidden. It authenticates the complete public evidence, recomputes every
displayed QA/copy count, preserves all 43 baseline slides and 51 chart parts,
and stages a 45-slide deck with one additional native chart/workbook. Figure
S29 uses the zero-count QA table and copying chart. Figure S30 retains all
twelve normal/erased initial/final answer images of the first held-out pair,
both complete inputs and labelled source/reference display insets. Neither
model pages nor their contrast are edited.

The same preparation writes an abstract result sentence, compact Results
paragraph and draft Supplementary Note 52 under the ignored staging folder.
The [preparation record](evidence/native_qa_publication_preparation.json)
binds the builder, evidence, original images, chart data and staged bytes.
Native XML, displayed counts and embedded workbook values have been checked.
It has not started Office/LaTeX, rendered slides, visually verified a new PDF,
included these fragments in the active manuscript or updated Nutstore.
All active publication files remain unchanged. Do not promote the staged
deck or TeX until the locked build and visual review above are complete.

```bash
CUDA_VISIBLE_DEVICES='' python publication/ilm-nature/prepare_native_qa_publication.py \
  --output artifacts/native_qa_publication_stage_v1
```

Use the existing publication Python environment and a fresh output directory.
The model Python is separate and need not be changed for publication tools.

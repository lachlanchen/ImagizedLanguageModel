# Reward calibration failures: bounded manuscript revision

Stage: measured-results update under the existing manuscript-maintenance goal.
The preceding goal turn is a verified wait. The original white-target trainer
remains the sole model worker. This revision reports two completed CPU controls;
it makes no new model call, training intervention or language-quality claim.

## Scope and evidence

Use the completed `whole_answer_reward_calibration_20261116.json` and
`selective_visual_verifier_calibration_20261116.json` under `references/evidence/`,
their preregistrations and the 64 unchanged original final normal QA outputs.
Both controls retain 9 strict complete-correct, 9 readable-wrong and 46
uncredited/uncertain outcomes. Do not rerun the calibrations or change any
frozen reading, reference, image, threshold or acceptance decision.

The ordered image-edit reward separates known correct from readable wrong
answers, but its maximum uncredited reward exceeds its minimum known-correct
reward. The selective verifier accepts 7/9 known-correct complete answers,
0/9 readable wrong answers and one uncertain output. Its 9,581 physical
prototypes use training pixels only. Both prospective calibrations fail.
Uncertainty is not proof that an output is semantically wrong. No independent
human adjudication is present. No policy fit or student-language gain follows.

## Allowed revision

1. Preserve the active reading/review/redline PDFs, all TeX and style inputs,
   and every existing figure asset at commit `66317774` in a hash-bound private
   baseline. Keep the cumulative redline baseline `0f14fa3f` and all prior
   versioned local deliveries. Active manuscript filenames remain stable.
2. Add `reward_calibration_results.tex` after the saved-score Results and
   `reward_calibration_methods.tex` after its Methods through
   `article_content.tex`. State all cohort denominators, prospective failures,
   the training-only boundary, bounded-corruption assumption, complete-stream
   scoring, resource scope and absence of model updates. Report the later
   explicit-reward refusal guard as implementation, not learned behavior.
3. Update only `complete_continuation_frontmatter.tex` and the relevant
   reward/language discussion in `complete_continuation_discussion.tex`.
   Before: the summary ends with current physical-writer limitations.
   After: it also identifies the failed training-reward controls, while
   preserving the negative complete-language verdict, whole-model size and
   running white-target trial's unmeasured quality. Keep the summary concise,
   preferably at most 210 words. Do not alter earlier numerical results.
4. Add native Figure 29, one editable PowerPoint and a caption. Panel a uses
   all 64 records to compare minimum known-correct, maximum readable-wrong
   and maximum uncredited rewards for each candidate. Retain exact source
   extrema and group sizes in the data/notes. Panel b shows every selective
   acceptance count in a native table. Panel c retains exactly the three
   registered disagreements: all known-correct answers not fully accepted
   and every uncredited answer fully accepted. Embed the original answer PNGs
   without repair, reference prototypes or inferred transcriptions. Native
   labels state the frozen review and verifier outcome. Do not select only
   successes or imply that a geometric match settles an uncertain reading.
5. Keep earlier figure assets, captions and widths unchanged. Add the literal
   Figure 29 plate to `complete_continuation_figures.tex`. A source-bound
   wrapper may reuse the verified article builder and inspection machinery.
   Build privately and promote only after automated checks and actual PDF
   review. Preserve all prior embedded image contents.

No new architecture, data acquisition, reward tuning, student evaluation,
README/i18n redesign, title change or paper restructuring is in scope. No
reviewer response letter is required for this internal measured-results update.
Source revisions bound by the completed calibrations remain preserved in Git;
the policy helper's subsequent guard must not be mistaken for the historical
source used in those controls. None of the 115 live training sources may change.

## Execution and acceptance

One lightweight CPU publication owner under `publication_cpu.lock`, two threads,
at least 48 GiB available host RAM, low pressure and at most 2 GiB child-tree
RSS. Native figure export has a cumulative 180-second ceiling; article builds
plus automated inspection have 180 seconds; detailed page renders have a
separate 180 seconds. Debit failed attempts and preserve their logs. Reuse
existing shared dependencies. Acquire no GPU mutex and launch no heavy GUI,
duplicate model worker, office instance or unrelated build.

Verify original model identities and all 115 source hashes before and after
publication operations. Check every native chart cache, literal workbook,
table cell and embedded original PNG against the completed data. Inspect the
whole final figure and every PDF page in overview, then read all affected
summary, Results, Discussion, Methods, equation, plate and caption pages in
detailed renders. Record overview and detailed review separately. Check text
bounds, equation/figure references and footer clearance. Never infer an Office
application review from package validation or renderer export.

Commit and push the completed scientific revision. A separate local-delivery
commit may create a new Nutstore package, preserve the preceding package and
update only the hash-guarded owned latest-PDF aliases. Cloud propagation remains
unverified unless actually observed. Keep the full ILM goal active.

## Completion record

The revision is promoted in `publication/ilm-nature/evidence/reward_calibration_article_build.json`. Reading/review/redline PDFs contain 53, 93 and 95 pages. All 241 pages received actual layout overview and 38 affected pages detailed visual review. Native Figure 29 has one editable chart with a literal workbook, one table and three unchanged original PNGs. All 332 prior figure assets and 153 prior embedded image contents remain. Four figure-export attempts total 21.067 seconds; the first missing runtime-path failure and subsequent label-layout corrections remain preserved. Article build plus automated inspection takes 38.307 seconds, and detailed rendering 4.646 seconds separately. All 115 model sources and original controller/trainer/bootstrap identities stay unchanged. No neural call or optimizer update occurs. Office application export, independent human adjudication and cloud propagation remain unverified. Local delivery follows separately; the full ILM goal remains active.

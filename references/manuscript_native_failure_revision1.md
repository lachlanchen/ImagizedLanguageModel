# Completed native prediction failure: manuscript revision plan

Status: scoped, not yet executed. Baseline: git b4855be. This unit updates the
research manuscript with the completed native training, original fixed audit
and fixed-cohort prediction diagnosis. It does not report the newly running
adaptive-readout experiment as a language result.

## Scientific change

The current article describes retained native context as untrained execution
and says training remains an experiment. That account is now incomplete.
The declared 40,000-update training has finished and fails the original
Chinese quality gates. On 128 normal Chinese development questions it has
0 exact reference images, foreground F1 0.365044 and 0/64 both-correct pairs.
Question erasure gives F1 0.254036; the paired gain is 0.111009 with a 95%
article-bootstrap interval [0.085778, 0.140177]. All streams stop. English is
untrained transfer and gives 0/64 exact normal images and F1 0.058610.

The same 32 fixed training questions give free F1 0.462347, correct-history
F1 0.502406 and supplied-code F1 0.999953, with exact counts 0/32, 0/32 and
28/32. All 32 first blocks agree between free and correct-history generation.
Thus conditional prediction fails even under correct history; generated
history cannot be the sole explanation. This does not uniquely identify a
successful repair or establish semantic accuracy.

The trained actor has 7,583,221,875 complete parameters, including 46,899,713
trainable parameters. Training takes 45,426.51 active seconds and 18.2977 GiB
peak allocated memory. The earlier execution fixture's slightly different
parameter count remains a historical execution measurement. Preserve the
external-pretraining disclosure and all failed outcomes. The general ILM
objective remains unachieved.

## Permitted edits

- `publication/ilm-nature/frontmatter.tex`: replace the abstract's now-outdated
  execution-only account with the completed native failure and diagnostic;
  retain the image-language objective and avoid a successful-language claim.
- `article_content.tex`: update the paragraph introducing native continuation
  and add its completed result and fixed-cohort controls before Discussion.
  Clarify the conditional-code prediction failure without selecting a unique
  architectural cause.
- `supplementary_information.tex`: append Note 37 before Reproduction, with
  training/data/parameter provenance, the unchanged six gates, normal/erased
  results, reused development status, all32 diagnostic controls, verification
  scope and the original failed diagnostic launch. Preserve Note36 as the
  previous execution/visibility experiment.
- Add Figure S14 and a 29th slide to the editable figure deck. Panel a shows
  Chinese normal/erased foreground F1 on a 0–100% scale. Panel b shows all32
  free/correct-history/supplied-code F1 and exact counts. Panel c shows the
  first fixed question pair's four raw writing paths. Include no synthetic,
  repaired or selectively chosen successful answer images.
- Add repeatable figure/publication builders and source-bound receipts;
  update publication README entries. Rebuild matching article, review, SI,
  redlines, editable PPT and figure-review PDF. Copy the verified package and
  current aliases to the existing local Nutstore ILM destination.

Existing 28 native slides, 25 charts, individual figures, prior baselines,
experimental sources, targets, thresholds and source receipts are preserved.
Only deck membership/content-type parts may change while appending the new
slide. No new architecture, adaptive training outcome, semantic score, text
model parity, general historical-script understanding or efficiency advantage
is claimed by this revision. No response letter is involved.

## Verification

Authenticate `references/evidence/native_chinese_v80_result.json` and
`native_corpus_fit_v80_result.json` against their original training, audit and
CPU verification artifacts. Check native chart caches and embedded workbook
values. Every embedded sample PNG must match its unaltered source bytes.
Compare all28 prior native slide parts and fresh-rendered appearances; retain
the canonical figure PDF prefix pixel-for-pixel.

Use the existing incremental LaTeX and redline tooling. Reject overfull boxes,
undefined citations/references and structural PDF errors. Render and inspect
the updated abstract, result paragraph, SI note, new figure and redlines.
Use the existing headless conversion route without an idle desktop. Check
shared RAM/swap and project processes before rendering; do not overlap heavy
builds or a second GPU/model job. Verify local Nutstore copies and five LATEST
aliases after visual review; cloud propagation is not inferred from a local
copy. Record actual build/render/sync results here when the unit is executed.

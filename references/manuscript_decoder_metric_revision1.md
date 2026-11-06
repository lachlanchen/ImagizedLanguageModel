# Decoder-sensitive loss: completed-result manuscript revision

Status: executed and verified. Baseline: `af9571c`. This plan was written before
the TeX changes.

Add the completed, failed V82 comparison to the maintained article, review,
supplement and editable figures. Both arms produce 0/32 exact autonomous
images on seen training questions. Control/metric F1 is 85.983512/70.009368%;
correct-history F1 is 92.430062/75.116411%; supplied-code reconstruction is
28/32 exact and 99.995316% in both arms. All four original repair gates fail.
The result rejects scaling this repair and does not qualify language learning.

Permitted changes:

- Append one paragraph after the V81 result in `article_content.tex`, pointing
  to Supplementary Note 39 and Figure S16. Preserve the V81 result as history.
- Replace the abstract's short final readout sentence with a concise statement
  that readout and decoder-sensitive repairs both fail. Include the new matched
  scores without displacing the principal development result or adding a
  successful, priority or efficiency claim.
- Append Note 39 before Reproduction in `supplementary_information.tex`:
  specify the state-only decoder-Jacobian objective, its conditional-mean
  optimum and limited interpretation; matched warm weights/budgets; the owning
  unpadded-history execution; all three writing controls; failed gates; total
  versus trained parameters; actual cost and excluded interrupted work;
  independent verification scope; and the separate post-hoc position diagnosis.
  Add the precise reproduction entry. Preserve earlier scientific text.
- Append slide 31 / Figure S16 with editable native charts on fixed 0–100%
  axes for free and correct-history F1. Show reference, control and metric raw
  PNGs for the first four fixed questions, without outcome selection, crop,
  character repair or replacement. Preserve embedded PNG bytes and aspect
  ratios. Source-derived labels state 0/32 exact and supplied-code fidelity.
- Preserve all 30 prior native slides and their renders, every prior individual
  figure, the canonical 30-page figure-review prefix and the V68 redline baseline.
  Add repeatable current builders, source-bound receipts and publication README
  status. Rebuild article/review/SI, both redlines, figure/review PDFs and deck.
- Render and inspect the new figure and all changed PDF pages. Verify native
  chart caches/workbooks against full-precision JSON, raw image bytes, PDF text,
  layout/reference logs and qpdf. Record exact reviewed pages and limitations.
  Copy only verified current artifacts to the existing Nutstore ILM folder,
  preserve prior aliases with distinct backups, and hash-check the local copy.
  Cloud propagation remains unverified.
- Refresh the root README's stale manuscript notice and displayed current
  measured figure. Keep the status notices in all 11 languages consistent.

Layout amendment after rendering: the new review text leaves a single final
bibliography line on its own page. Permit standard single spacing within the
review copy's bibliography only, preserving its 12-point body and 1.5 body
spacing, all citation wording and the article/supplement style. The matching
redline inherits that review formatting. Rebuild and inspect the affected
reference pages rather than accepting the orphan page.

No response letter is involved. The experimental protocols, source code,
targets, checkpoint state, seeds, sampler and quality gates are out of scope.
The untested image-generator baseline is also outside this manuscript result
unit. Its weight transfer may proceed, but a second model or heavy build must
not overlap an active project GPU model. Record execution and deviations here
before committing the completed revision.

Execution: the article is 14 pages, the review 26 and the supplement 37;
V68-based redlines are 27/37 pages. Note 39 is on supplement page 33,
Figure S16 on page 34 and the reproduction entry on page 36. The deck and
canonical figure-review PDF have 31 slides/pages. Both native chart caches
and workbooks retain the full-precision values, categories and fixed 0–100%
scales. Twelve source PNG objects retain their exact bytes and aspect ratios.
All 30 earlier native slide parts and fresh rendered appearances, 70 earlier
individual figure assets and the first 32 supplement pages are preserved.

All seven PDFs pass qpdf and compile without overfull boxes or undefined
references/citations. Every changed manuscript/review/SI/redline page was
rendered and visually inspected in overviews; the new native figure was
inspected at readable size. The source-bound visual-review receipt records
the current PDF hashes and page scope. A compact main-text paragraph avoids
a sparse extra article page; review-only single-spaced references remove
the one-line bibliography spill. The marked copy retains its natural
27-page layout.

The workspace's requested artifact-operation marker helper is absent; its
attempted invocation fails with MODULE_NOT_FOUND and is retained privately.
The existing native OOXML/LibreOffice and LaTeX route was used, without a
fabricated marker or artifact-tool success claim. Inherited MuPDF structure
tree ancestry warnings remain during canonical prior-slide assembly; qpdf
and rendered appearances pass, without a tagged-accessibility claim.

The local Nutstore package `ILM_NatureStyle_20261107_DecoderMetric` and five
current PDF/deck aliases are copied and hash-verified. Previous aliases are
preserved as `before_decoder_metric` backups. Cloud propagation is unverified.
No model process or GUI stack is retained as publication evidence. The
independent image-generator baseline remains outside this manuscript unit.

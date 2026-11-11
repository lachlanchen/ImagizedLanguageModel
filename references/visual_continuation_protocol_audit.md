# Continuation trial: stop supervision and complete pilot auditing

The fixed complete-corpus fit remains unchanged. This CPU-only work checks
record boundaries and the interpretation of its registered evaluation rules.
It adds no optimizer update, model sample, benchmark case or revised threshold.

## Book boundary scope

The source builder prefers sentence/clause punctuation when cutting long
paragraphs and otherwise retains paragraph/record boundaries. Registered book
targets are complete **records**, not necessarily complete documents or
sentences. The current source also contains headings and lists of work titles.
A terminal punctuation heuristic cannot certify semantic completeness, and
its absence does not establish a bad example: headings and poetry may end
without sentence punctuation.

The boundary audit verifies every registered book ID, source-text hash, prefix
rule and complete target length against the unchanged source manifest. It
records literal sentence/clause marks, closing quotes and record lengths.
No target or stop label is changed. These measurements qualify how training
termination should be interpreted; actual natural stopping and coherent book
continuation still require the registered autonomous evaluation.

The [complete boundary receipt](evidence/visual_continuation_book_boundaries_20261112.json)
finds 2,935 direct sentence marks and 3,796 after optional closing quotes,
or 3,825 when semicolons are also admitted. All 4,762 prefix/target boundaries
match their registered source records. Lengths range from 50 to 641 glyphs;
65 records have length 640 and 39 have length 641. The source splitter's
inclusive punctuation search can retain a mark at position 641. Complete
targets remain preserved; the existing preflight already covers the resulting
513-glyph target maximum. Samples ending without sentence punctuation include
headings and lists of titles. These observations neither certify every example
as narrative prose nor identify a unique cause of any future generation failure.

## All four QA-erasure arms

The written rule limits **each QA-erasure arm** to at most two complete exact
answers. There are four 32-question arms: question-blank and passage-blank
inputs in both fixed-cell and proportional layouts. The existing bound
`review_visual_continuation.py` computes its final pilot expression from only
the two proportional erasure rows. Its proportional normal inputs are QA-only,
so its normal-row selection does not accidentally select a book aggregate.

A separate prospective protocol audit applies the literal erasure limit to all
four arms. The normal-minus-best-erased comparison remains proportional, as
registered for the proportional-answer pilot. All numeric minima, exact-case
retention, book criteria, scores and frozen readings remain unchanged. The
original summary is preserved, and a failed original pilot cannot be rescued
by the supplemental audit. Qualification requires the complete protocol audit
and subsequently untouched confirmation; neither report marks the full ILM
goal achieved.

This code is separate from the live registered pipeline. Its registration is
committed before new evaluation images or readings exist. The audit checks
complete generation/review coverage, original aggregates and all 732 reviewed
PNG hashes. It does not reinterpret glyphs, repair transcriptions or substitute
an independent language score. The regression cases are aggregate-rule tests,
not model-performance demonstrations.

```bash
# After the original complete opaque review and original scoring:
CUDA_VISIBLE_DEVICES='' python scripts/audit_visual_continuation_protocol.py pilot \
  --run artifacts/visual_continuation_full_logger_recovery_v2
```

# Natural visual reading questions: data preparation plan

Date: 2026-11-05. Prepare data before choosing or training a new model. The
goal remains independent image-input/image-output language, including future
book questions and historical forms. This dataset isolates whether a model
can read an actual paragraph and answer a natural Chinese question.

Use the author CMRC 2018 release at Git revision
`c0eb1b6ba219847457e6af3180da722bbeb656af`, SQuAD-style train and development
files only. Preserve its CC-BY-SA-4.0 license and author attribution. Do not
download test labels. Official source:
<https://github.com/ymcui/cmrc2018>.

Use the complete paragraph with input format `资料：{context}问题：{question}回答：`.
Keep prompts at most 480 visible cells and reference answers at most 16.
Remove whitespace for the existing Chinese cell renderer, retaining original
source text and remapping annotated offsets exactly. Reject unsupported
visible glyphs instead of silently dropping them. Never crop around a gold
answer or provide an answer position/length to the student.

Keep valid source-contained short reference answers, excluding an answer that
also appears verbatim in the question. Drop a record if it has no remaining
reference. Report every rejection reason and all counts. Remove training
records sharing an article title or normalized paragraph with retained
development inputs. Preserve official split roles and hashes.

Record same-paragraph development question pairs with disjoint valid answer
sets as a natural counterfactual evaluation set. Select at most one pair per
paragraph deterministically by record ID, without model scores. Store IDs and
references as evaluator metadata; future model calls receive rendered images
only. This is a natural reading task, not a synthetic fixed grammar or OCR-only
benchmark. It does not by itself establish free language or etymology ability.

Validate all kept annotated offsets, prompt/answer limits, source containment,
glyph coverage and cross-split separation. Store raw data and prepared text
only under ignored `data/`, with a portable summary for the research record.
No QA training or new acceptance claim is part of preparation.

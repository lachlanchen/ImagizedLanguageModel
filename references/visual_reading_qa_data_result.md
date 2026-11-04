# Prepared natural Chinese visual reading questions

Prepared 2026-11-05 from pinned CMRC 2018 source data. This is data preparation,
not a trained QA model or an evaluation result.

| Split | Questions | Disjoint articles/paragraphs | Prompt cells, median / maximum |
| --- | ---: | ---: | ---: |
| Train | 3,211 | 1,134 | 374 / 480 |
| Development | 1,321 | 421 | 378 / 480 |

All examples retain their full paragraph and natural question. No paragraph
is cropped using a gold answer. References are source-contained, at most 16
visible cells, with exact offset remapping after whitespace removal. No
eligible reference appears verbatim in the question. Unsupported visible
glyphs cause rejection rather than deletion. Article-title and paragraph-hash
overlap between splits is zero.

The preparation also identifies **393 same-paragraph question pairs** with
disjoint reference-answer sets, at most one pair per paragraph. These can test
whether changing the question changes the answer while the visible source
inventory stays fixed. The pairs do not enter training. Official test labels
were not downloaded.

The input-only 480-cell limit excludes 4,952 train and 1,600 development
questions. No eligible short reference remains for 1,911 and 262, respectively.
Unsupported input glyphs exclude 68 and 36. Invalid individual source offsets
are counted separately; a record may retain another valid reference. All
filters and original annotated text remain in the local preparation receipts.
These constraints define a short-paragraph subset, not the complete CMRC task.

Files are under `data/visual_qa/cmrc2018_short/`: train/development JSONL,
counterfactual-pair IDs, the original license and a SHA-pinned preparation
report. Raw author data are cached once in `data/raw/cmrc2018/`. Neither raw
nor prepared source text is committed. Future model calls must receive only
rendered image tensors; strings, annotated offsets and source IDs are host or
evaluator metadata.

Source revision: `c0eb1b6ba219847457e6af3180da722bbeb656af`.
License: CC-BY-SA-4.0. Cite Cui et al., *A Span-Extraction Dataset for Chinese
Machine Reading Comprehension*, EMNLP-IJCNLP 2019.
[Author repository](https://github.com/ymcui/cmrc2018),
[paper](https://aclanthology.org/D19-1600/).

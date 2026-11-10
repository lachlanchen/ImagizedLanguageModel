# Prospective Chinese reading beyond the reused CMRC questions

This is a research evaluation, not training data or evidence of a working
language model. Freeze the cohort before the genuine-copy qualification is
available. Complete that run's physical qualification and original forty-stream
autonomous assessment first. Do not select a model or alter training using this
cohort's results. Known-form copying, correct option letters and full image
answers remain separate requirements.

Use the mixed-genre development file from the authors' [C3 repository](https://github.com/nlpdata/c3),
pinned to `1f41682ebb684d63afe7c583c8f38e137fc8a826`. Its
[source notice](https://github.com/nlpdata/c3/blob/1f41682ebb684d63afe7c583c8f38e137fc8a826/license.txt)
limits use to non-commercial research. Preserve that notice with the raw source
and derivative input pages in ignored data/artifacts directories. Do not place
the corpus under the repository's code license. C3 is described by Sun, Yu, Yu
and Cardie, *Investigating Prior Knowledge for Challenging Chinese Machine
Reading Comprehension*, TACL 2020 ([paper](https://arxiv.org/abs/1904.09679v3)).

Acquire only the pinned development JSON, README and notice; authenticate each
by its pinned Git blob and record SHA-256. No model weights, train split or test
labels are downloaded. Take sixteen hash-ordered documents, with the first
question in source order. Reject only an exact normalized full-context overlap
with local CMRC train/development data or an input font/layout incompatibility.
Record every rejected document. Do not filter by correct answer, answer length,
canonical glyph coverage, model score or output quality. This is held out from
the present adaptation; exposure in the inherited body's pretraining is unknown.

Each document supplies four complete image inputs: original and cyclically
rotated options, each asking either for the option letter or the full correct
option phrase. The nonzero rotation is fixed from the document ID. The correct
letter must change and the correct phrase must remain identical. These paired
conditions test sensitivity to the options shown in the image, while the phrase
condition still requires writing a complete answer. A letter-only result cannot
establish general image-language generation. No passage-erasure claim is made.

Render all passage text, question and options on a 960x1024 grayscale page.
Use the existing authenticated Noto SC face, trying 24, 22 and 20 pixels in that
order until all four pages fit. The header and body occupy fixed separate areas;
no clipping, truncation or answer-aware crop is permitted. The rendering function
receives no answer identity. The answer is present only among the provided
options, as specified by the source task. Correct references live in a separate
evaluator file. Report unsupported reference forms without dropping a case.
Inspect all sixteen paired preview boards and representative full-size pages
before any model sees them.

Any later assessment must freeze its checkpoint and source hashes before the
first case, use the image-only loop without references or runtime glyph banks,
and retain every generated pixel and natural stop/cap decision. Proposed caps
are eight glyphs for a letter and128for a phrase; neither uses reference length.
Score complete raw outputs after output-first review. Report option-letter
correctness and two-view consistency separately from full-phrase correctness,
pixel metrics and stopping. Keep all failures. Inherited pretraining, repeated
development use after this first run and non-independent manual grading must
remain visible. No successful autonomous C3 result exists at preparation time.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_c3_visual_reading.py --stage acquire
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_c3_visual_reading.py --stage prepare
```

These commands perform source acquisition and CPU input preparation only. They
do not start another GPU job or change the genuine-copy run. Model assessment
requires its own executable, resource admission and reviewed inputs after the
current pipeline is terminal; this protocol does not claim it has run.

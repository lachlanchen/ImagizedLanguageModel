# Complete visual sequences for ILM training

The new corpus contains **13,785 training sequences**. It uses 4,762 source
passages from fourteen Chinese Wikisource works and 9,023 complete CMRC
question-answer pairs. All training sequences contain 7,255,223 glyph
positions, including repeated QA passages. The continuation and answer
targets contain 2,576,239 glyph positions. Counts refer to prepared data.
Model training on this corpus remains pending.

| Partition | Book passages | QA pairs | Total glyph positions | Target glyph positions |
| --- | ---: | ---: | ---: | ---: |
| Training | 4,762 | 9,023 | 7,255,223 | 2,576,239 |
| Development | 2,112 | 941 | 1,722,616 | 1,098,415 |

The book split holds out two whole works, 欒城集 and 西遊記. CMRC keeps
paragraphs together, preserves the existing held contexts and excludes the
official development contexts and titles. The verifier checked all 132
shards and found zero overlap in source groups, exact sequence text or
exact input pixels across the partitions. It inspected the offsets through
the full teacher and visual arrays. Two actual input previews were also
reviewed. See the [corpus evidence](../references/evidence/visual_sequence_corpus_20261111.json).

The prepared glyph stream uses 32×32 binary ink with a fixed Noto Sans CJK
face and blank spacing tiles. Its previews wrap the stream into rows for
inspection. The corpus records 300 font coverage rejections and 21 QA
records without valid full targets. The original source records remain
available. Real scanned pages, alternate hands and historical raster forms
require additional visual input and generation tests.

## Training boundary

Use `VisualSequenceCorpus` to open the packed visual inputs and target
boundaries. The default item contains pixels and their alignment offsets.
The separate `offline_teacher_alignment` call opens source unit IDs for
teacher prefill. Each source span ends on a complete UTF-8 boundary and
retains the range of original teacher units, including cases with more
than one unit per visual span. The teacher can thus provide a contextual
target at every span boundary from a single forward through a sequence.
Historical form preparation can use the separate grouped glyph corpus.

```python
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus

training = VisualSequenceCorpus("data/visual_sequence_corpus_v1", split="train")
sample = training[0]
ink = sample["ink"]  # [glyphs, 1, 32, 32], uint8
```

Fit the span feedback encoder and writer using full sequences and actual
teacher states. The current 4090 run still tests the earlier first-image
writer. The next fit uses all available continuation targets. Compare the
teacher's native sequence with matched visual feedback to measure state
drift, then evaluate actual complete generated images. Track compute per
answer and per passage along with correctness. See the
[model decision and math](../references/visual_chunk_coverage_result.md).

The teacher cache now has a verified schedule for 2,380,620 continuation,
answer and terminal states. Its source loader and resumable GPU stage are
available. See the [teacher context stage](visual-sequence-teacher-states.md).

The experiment should retain fixed training checkpoints and evaluate the
same reserved prompts at those checkpoints. The TeX and PowerPoint figures
can then pair actual complete-answer scores with the corresponding output
images and show physical writing loss separately. See the
[engineering goal](first-imagized-language-model-goal.md).

## Reproduction

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_visual_sequence_corpus.py \
  --output data/visual_sequence_corpus_reproduction
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/verify_visual_sequence_corpus.py \
  --data data/visual_sequence_corpus_reproduction
```

The preparation reuses the authenticated local teacher archive to read
its tokenizer metadata. Preparation and verification perform no neural
teacher forward. Source hashes, font hash, partition counts, rejected IDs
and shard hashes are recorded locally. Source attribution remains bound to
the Wikisource manifest and CMRC release.

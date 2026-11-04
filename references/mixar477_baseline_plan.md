# Pinned multilingual pixel baseline: exploratory check

Date: 2026-11-05, before running generation. Use the author-released 477M
MIXAR stage-2 checkpoint pinned in the research refresh. This is an external
pretrained model, not a local trained result. No local fine-tuning, weight
selection, temperature search or reranking is part of this check.

Use the author's bundled renderer, full FP32 weights, image-only model calls,
greedy zero-logit thresholding, 32 new two-patch blocks and the documented
blank-block stopping rule. Save every raw output for six fixed prompts:

- The capital of France is
- Water freezes at
- The word alphabet comes from
- 中国的首都是
- 汉字由
- 汉字“言”的本义是

Strings enter only the host renderer. The model receives continuous/binary
image tensors and spatial attention masks. Preserve model/source/font hashes,
framework versions, actual pixels, timing and allocated memory. Generation is
local with no OCR or external model call. The prompts are demonstrations, not
a statistical benchmark or evidence of general reasoning.

The renderer has an import-time automatic Conda installer when Cairo is
missing. Explicitly import and validate native bindings before importing it;
never permit that fallback. A stopped first import attempt reached metadata
solving but made no shared environment transaction. Use a project-owned
environment, reuse an existing Torch installation and native system bindings,
and keep any per-process library compatibility settings isolated.

After generation inspect all outputs, including blank or incorrect ones.
Retain external attribution even if the baseline is better than the local
model. Do not count external pretraining as single-4090 training or use an
attested-looking output as evidence of historical glyph provenance.

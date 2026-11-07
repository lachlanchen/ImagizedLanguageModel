# Whole-request canvas control with fixed learned weights

The state-alignment learner fails Chinese copying on full square inputs.
The earlier dense checkpoint gives 13/16 Chinese and 16/16 English copies,
but its copying inputs are 1024 by 256 strips, not 1024-square pages. Their
retained ink is byte-identical after restoring white bottom padding. Model,
training, layout and sampling differ across those experiments, so their
scores alone cannot identify a compression mechanism.

This no-optimization experiment fixes the completed dense checkpoint
`016429ec5454cdd681693d881df4c6b7d52fb43dcae7b9e15857d2d9cd32b483`
and evaluates three layouts on the complete 160-condition assessment:

1. Legacy: the exact 32 original copying strips, 48 Chinese question-pair
   conditions and 80 English question/passage-erasure conditions. Replay
   every retained original decision sequence, input hash and seed exactly.
   A mismatch stops the experiment before altered layouts are evaluated.
2. Square: restore only exactly white bottom padding to copying strips;
   other original whole square pages remain unchanged. Original source
   pixel positions and all ink must match the 1024-square page inventory.
3. Tight: from each complete square page, find the bounding box of every
   pixel unequal to RGB white, include a 16-pixel margin and round boundaries
   outward to multiples of 16. Crop only the white exterior. Restoring white
   pixels at its original coordinates must reconstruct the source exactly.

Tight preprocessing uses no OCR, font metric, transcript, answer, character
segmentation or semantic role. It keeps the whole request, passage and
question, including punctuation. No content is resized. Erased conditions
may have different extents; this is a geometry intervention, not a claim of
holding latent features or spatial positions fixed. VAE features, interpolated
positions, context lengths and sampled latent draws can change with layout.

All arms use the original per-case latent/readout seeds, same BF16 body,
FP32 codec, gated dense visual adapters, generic request, native mask and
96-decision top-one/EOS readout. There are 480 primary calls and no optimizer
updates, new model weights, teacher text or correct answer history. Each
condition is kept regardless of the result.

Report all copying exactness by language, Chinese/English normal and paired
QA correctness, erasure outcomes, real input sizes/context lengths and
invocation costs. Keep the original alternatives, normalization and EOS
requirements. The native 432 English source reference is still 11/32;
beating a failed full-page baseline is not sufficient to qualify a reader.
No outcome establishes independent continuous image answers, efficiency at
matched LLM quality or the full ILM goal.

Related visual-context compression work includes
[FastV](https://arxiv.org/abs/2403.06764) and
[VisionZip](https://arxiv.org/abs/2412.04467). This pixel-based canvas control
does not implement their learned-attention selection or claim a new pruning
algorithm. CMRC 2018/SQuAD 1.1 retain CC-BY-SA-4.0 attribution to Cui et al.
(2019) and Rajpurkar et al. (2016); Show-o2 source/model are NUS Show Lab,
Apache-2.0.

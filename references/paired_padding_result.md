# Faster visual-copy training still produces incorrect writing

The completed padding experiment gives **6/32 seen strings and 10/32
validation strings exactly recognized by offline OCR**. The clean reference
pages score 27/32 and 28/32 with the same recognizer. OCR is imperfect, but
the complete raw galleries independently show missing Chinese characters,
repeated English letters, extraneous marks and duplicated lines. The declared
requirement of 30/32 legible exact requested strings alone in each split fails.
This is partial copying, not a usable language model or evidence of question
answering. The two copying runs contain no natural-QA optimization.

All 64 cases remain in original order, reference above and actual output below:
[training 00–15](evidence/paired_padding_result/all_cases_00_15.png),
[training 16–31](evidence/paired_padding_result/all_cases_16_31.png),
[validation 00–15](evidence/paired_padding_result/all_cases_32_47.png), and
[validation 16–31](evidence/paired_padding_result/all_cases_48_63.png).
The [complete export](evidence/paired_padding_result/result.json) retains every
input, target and answer PNG, the eight unadapted controls, source reports and
the 512-update journal. All 72 generated PNGs match the retained postprocessed
decoder tensors; no OCR repair, font substitution or output selection is used.
This authenticates exported pixels, not an independent neural replay.

## Matched preprocessing experiment

Both arms start from the same externally pretrained undistilled
FLUX.2-klein-base-4B revision and train fresh rank-16 adapters on the same first
32 copying examples for 512 updates. Source code, target images, generic numeric
condition, hyperparameters, sampled training IDs and sigma draws match. The
first eight autonomous cases also use matched sampling seeds. The
[original square-canvas result](paired_image_writing_result.md) remains separate.

| Measurement | Square input | Strip input |
| --- | ---: | ---: |
| Input dimensions | 1024 × 1024 | 1024 × 256 |
| Native image-reference positions | 4,096 | 1,024 |
| 512-update invocation time (s) | 795.624 | 287.182 |
| Training peak allocated GPU memory (GiB) | 9.501 | 8.437 |
| Fixed eight-case screen, OCR exact | 2/8 | 3/8 |
| Complete 32/32 copy observation, OCR exact | Not performed | 6/32 train; 10/32 validation |

Removing exactly white bottom padding preserves every source ink pixel and its
coordinates. Independently restoring white padding reconstructs all 160
prepared copying inputs exactly. Targets remain byte-identical. However, the
VAE features change: all 64 target tensors are bit-exact across caches, while
the maximum input-feature difference in the retained extent is 1.6796875.
This is a preprocessing comparison, not a pure attention-mask ablation.

The 2.77-fold reduction in this training invocation's time does not establish
an advantage at useful language quality. It excludes source acquisition,
preparation, setup failures and separate controls. Each arm is a single run;
there is no independent-seed replication or matched text-model comparison.

## Complete observation and decision

The full strip cohort contains 64 logical outputs: the original eight samples
are retained, and only the remaining 56 are newly generated. Original seeds
are preserved; other seeds follow the separately specified
[full-cohort protocol](paired_padding_full_cohort_protocol.md). All calls use
50 native steps, guidance one and a 512 × 256 answer canvas. The fresh generator
receives only input images, the fixed generic condition and learned state;
reference answers and OCR run separately.

Failures are visible even among training examples. Training case 004 reduces a
twelve-character Chinese string to a few glyphs; case 011 duplicates a requested
English phrase on two lines. Validation case 000 drops its final character,
case 002 also omits the final character, and case 008 drops the fourth
character. These errors reject reliability
without deciding ambiguous OCR cases in the model's favour. The galleries
also contain correct copies. Their presence supports limited transfer of
copying and does not establish grammar, semantics or instruction following.

The deployed transformer has 3,899,903,488 parameters, including 24,358,912
locally trained adapter parameters; its VAE has 84,046,115. A single generic
condition is compiled offline through an externally pretrained Qwen3 encoder,
not claimed identical to the stock FLUX encoder. No question-specific text
embedding enters inference. There is no basis for claiming a small foundation
model trained from scratch or parity with an 8B language model.

The current checkpoint is not qualified for use or scale-up. The next learning
test must resolve exact content and order before extending answer length or
claiming language competence. Training-only recognition supervision and a
native unified understanding/generation backbone are hypotheses, not proven
repairs. Reproduction uses `scripts/export_paired_padding_result.py` and the
[paired-image execution guide](../docs/paired-image-learning.md).

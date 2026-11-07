# Contextual pixel spans: a grounded copying component of ILM

Prospective protocol, 7 November 2026. The current free image writers and
attention-only reader have failed. The prior V58 dynamic image-block reader
also fails paired-question qualification. This experiment does not rename
that method or replace the general language-generation objective with
extraction. It tests a specific component: a learned answer selector on the
already calibrated Qwen3-VL image/context body, followed by exact source-pixel
copying. Free answers absent from the prompt remain a required, separate part
of the full ILM goal.

## Representation and actual runtime

Use complete 32-pixel-high Pango writing strips, the existing white-column
folding operation, image-budget transform and authenticated numeric task
vectors of V77. Do not resize individual glyphs or attach a language
vocabulary. Reuse the existing Qwen3-VL-8B archive; its full externally
pretrained body is retained, not counted as a small trained-from-scratch model.
Before caching new inputs, reproduce four retained V77 normal-image suffix
memories exactly. No text projection, tokenizer or answer generation enters
this encoder. No source weights are downloaded or duplicated.

Anonymous units are maximal consecutive columns containing black ink in the
original strip. They need not be characters: radicals or letters may form
multiple units. Their boundaries depend only on the current pixels. Each
unit receives bilinearly sampled final contextual visual features at three
horizontal points and three vertical points of its original folded location.
Keep all final numeric task-frame states as the question-dependent memory.
The copy motor emits the original columns from a predicted first through last
unit, with a fixed two-pixel white horizontal frame. It generates an actual
answer PNG; it does not render a predicted Unicode string or use a stored
glyph inventory. Restriction: inputs are prepared binary writing strips,
not arbitrary scanned-page segmentation.

Train a new width-256 query-conditioned span head. Layer-normalize the
4,096-dimensional visual and task-frame features, project them to 256, and
score each unit from its visual projection, the mean projected task memory,
their elementwise product and four physical geometric features. A shared
two-layer nonlinear network supplies start/end scores. A softmax normalizes
over every visible contiguous span of 1-128 ink units, with a learned length
bias. Marginalize all source spans whose complete pixels equal the target
image; no answer location or candidate whitelist enters neural conditioning.
This is an observed-image distribution, not a density for unseen images.

The copying/generation distinction has prior art in See, Liu and Manning,
ACL 2017, https://aclanthology.org/P17-1099/. The project's V58 already applies
dynamic block marginalization to images. The present change is the contextual
source and physical-column representation. Neither generic copying nor
frozen-feature probing is claimed as a novel language paradigm.

## Fixed data and optimization

Authenticate the existing CMRC short train/development JSONL files. Pick
128 training paragraphs with two different-reference questions each, using
seed 2026110701 and sorted source IDs within a paragraph; use both questions
(256). Independently pick 32 development paragraphs/two questions (64), seed
2026110702. Require no article-title or paragraph-hash overlap between train
and development. This is repeatedly used public development, not sealed data.
No source-model outcome selects examples.

Render whole source prompts with the existing CompleteLineRenderer. Use
Pango's logical source positions and the supplied source annotation only to
prepare exact source-rendered reference images. Reject and count unsupported
font/layout/overlapping-boundary cases before paragraph selection. Preserve
all original annotation alternatives that can be represented. Training
receives reference pixels, never their offsets. Prepare 64 training random
copy strings (32 Chinese, 32 English) and 32 new copying strings (16 each),
seed 2026110703, lengths 5-12 Chinese marks or 2-4 English words. These copies
are distinct across partitions and use the same physical renderer.

Keep inputs and references separate. Feature caching and autonomous inference
must not open references. Evaluate 64 normal, 64 question-erased, 64
passage-erased QA requests and 32 normal copies. Erasure removes only the
declared source pixels with original layout retained. In addition evaluate
the first 32 training QA records to distinguish fitting from generalization.
All generated images and chosen boundaries remain unmodified.

Learn only the new head for exactly 3,200 updates, batch 8, with complete
shuffled passes of all 320 training images. AdamW: peak 3e-4, 100-update
linear warm-up, cosine decay to 3e-5, betas (.9,.95), decay .01, clipping 1.
Seed 2026110704. Use FP32 trainable parameters and contractions. Save the
final checkpoint; no development selection, early stopping or post-score
optimization. Check finite loss/gradients and save optimizer/RNG/cursor state.
Cache source features in BF16 without changing the source's FP32 contextual
computation. One GPU0 job at a time; never use GPU1 or leave a GUI resident.

## Decision and claim boundary

Report exact source-rendered answer images, multiset ink-column F1 as a
separate unordered diagnostic, both-correct question pairs, all erasure
outputs, fit outputs, invocation times, peak memory, full runtime parameters,
trainable parameters and inherited pretraining. Normalization for exactness
removes only all-white horizontal exterior; all 32 original rows remain
unchanged. No OCR or text comparison repairs outputs.

The component is qualified only if new copying is at least 30/32, natural
answers at least 48/64, both questions correct in at least 20/32 paragraphs,
and normal-minus-question-erased exact accuracy is at least .25. Every
criterion is fixed before calls. A pass qualifies this prepared-strip
extractive component only. It does not qualify free language generation,
raw-page understanding, arbitrary historical-script semantics, Qwen parity,
matched-quality efficiency or the complete ILM goal. Failure rejects this
head/representation at the specified budget without choosing a unique cause.

CMRC 2018 is Cui et al. (2019), CC-BY-SA-4.0. The source Qwen model retains
its original license and attribution. Synthetic copies have no private text.

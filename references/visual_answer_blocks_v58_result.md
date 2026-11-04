# Whole observed image answers: V58 result

The completed 19,597,363-parameter model selects and ends a contiguous sequence
of actual prompt glyph images. On the fixed 256 CMRC development questions,
25 answers are exact (9.766%), against 3 with question pixels blanked and 21
for matched additional glyph-answer training. Raster multiset F1 is 27.033%,
9.169% and 22.140%, respectively. F1 is not order-sensitive or the official
word-segmented CMRC metric; exact answer match enforces order and termination.

The block-minus-glyph F1 difference is 4.893 percentage points, source-article
bootstrap 95% interval [0.469, 9.047]. The exact-answer difference is only
1.563 points [-1.509, 4.652], unresolved. Extra training accounts for much of
the change from V57's 10/256; it would be incorrect to attribute that entire
increase to the block head. Normal-minus-blank F1 is 17.864 points
[13.438, 22.289]. Both questions in a shared paragraph are correct in only
4/393 pairs (1.018%). Six of seven registered gates pass; paired correctness
fails. The registered decision remains **reject_reading_model**.

Both arms start from identical V57 copy EMA weights and complete 4,000 updates
before development opens. The block arm takes 210.995 s, peak 0.967 GiB
allocated CUDA memory on physical GPU 0, an RTX 4090 D. Matched glyph training
takes 302.237 s and 1.056 GiB. These costs exclude earlier foundation training.
The audit takes 98.951 s and 0.142 GiB. All 256 fixed supplied-image motor
targets remain exact. Block outputs end through predicted boundaries rather
than a forced reference length or a maximum-length safety termination.

The probability law normalizes over all visible contiguous blocks of lengths
1–16 and marginalizes duplicate image sequences. Training uses target pixels,
not answer locations. The decoder receives no references, OCR, Unicode IDs or
persistent glyph inventory. Extractive reading does not establish free image
language generation, historical-form semantics or a generally useful assistant.
This is explicitly reused development, without an independent seed replication.

Artifacts are under `artifacts/visual_answer_blocks_v58_20261105/`: production
checkpoints in `block/` and `glyph/`, immutable audit and all raw predictions
in `development/`, native-image inference/export in `standalone/`.
The block checkpoint SHA-256 is
`4b2d79fb0b2dd403f6a1bc2581dcd28a847948d05ab47c632589159c7f8a66ca`.
The inference-only export is 78,772,107 bytes, SHA-256
`357c2fec17cf5977f956c8529189b2fde46ed727d4c02bb48355fd4db12a281b`.
Replaying its emitted native prompt PNG reproduces answer tensors, PNGs and
boundary events exactly. The raw answer `.pt` contains an `images` tensor.

[Registered protocol](visual_answer_blocks_v58_protocol.md),
[training](../scripts/train_visual_answer_blocks_v58.py),
[audit](../scripts/eval_visual_answer_blocks_v58.py),
[native-image inference](../scripts/infer_visual_answer_blocks.py).

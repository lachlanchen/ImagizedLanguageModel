# V59: frozen visual-language foundation probe

Frozen before feature extraction, readout implementation or training,
2026-11-05. V58 passes six of seven gates but answers both questions correctly
in only 4/393 shared-paragraph pairs. Train fit is much better than development
fit. Test whether stronger pretrained pixel-language features help evidence
selection before spending a larger budget fine-tuning a backbone.

This is a foundation/readout comparison, not a new general-ILM qualification
or a controlled separation of pretraining data from model capacity.

## Frozen encoders and inputs

Compare the completed V58 block reader with the released MIXAR-477M stage-2
pixel model. Pin MIXAR revision `c5d5bcfbad2f290296fd83810558f396c6a6333a`,
weight SHA-256 `be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8`.
Reuse the existing shared cache; do not duplicate weights. Its author reports
multilingual mC4 pretraining and about 62 hours on four H200s. That external
cost is not local one-4090 training. Preserve Apache-2.0 attribution. The
deployed model, if retained, consists of a frozen image encoder plus learned
image-block readout, with no external model call.

Use the exact V57/V58 CMRC JSONLs and canonical Noto CJK 26-pixel rendering
inside 32-pixel cells. This is deliberately identical across encoders but is
a font/layout shift from MIXAR's released 32-pixel renderer. A weak frozen
probe would not establish that the author model lacks language competence.
No new data, source cropping, reference length or source answer offsets enter
either encoder. Both consume prompt pixels only; all answers remain outside
feature extraction.

Append one blank cell for odd-length prompts, solely to form 64-pixel-wide
two-cell blocks. MIXAR consumes the resulting binary image strip in its native
white=+1/black=-1 convention. For a matched source unit, take the student's
state after each complete pair as well. Repeat each pair state over its two
source cells; discard the extra blank position from readout candidates. This
is fully observed prompt reading, not a future-glyph prediction claim.

Project MIXAR's 896-dimensional states to 384 through a fixed semi-orthogonal
matrix (CPU QR, seed 20265900). Student states are already 384. RMS-normalize
these semantic features. Append 128 fixed projected normalized-DCT image
coordinates (semi-orthogonal matrix, seed 20265901), RMS-normalized, and a
two-dimensional within-pair phase indicator. Both readouts thus receive 514
continuous features per visible image cell. These are not character IDs.
Record projector hashes. Cache float16 features; deployed readout must use
the same quantization. Extract normal train/development features and
question-blanked development features without using answer annotations.

## Matched learned readout

Each arm trains fresh identical heads: start/end query and key projections
514-to-384 plus 16 length biases. The law, source bounds, exact duplicate-block
marginalization and mode decoder match V58. Source images and reference image
sequences remain the only answer supervision. The frozen encoders and their
native free writers are not updated. The readout cannot invent an answer
absent from its prompt; this probe tests grounded reading only.

Each arm makes 4,000 updates, batch eight, uniform questions with replacement,
training seed 20265910 and data seed 20265911. AdamW betas (.9,.95), rate 3e-4,
weight decay .05, norm clip one, 200 warmup updates, cosine to .1 of peak,
EMA .999. Loss is reference block NLL divided by total reference cells plus
terminal events, as in V58. Use FP32 readout training, high matmul precision,
deterministic algorithms and saved optimizer/RNG/source hashes. Unit tests
must verify source packing/projection dimensions, equality marginalization,
normalization and gradients. Interrupted/resumed smoke weights and EMA must
match before production. Both final EMA arms finish before development scores
open; no best-checkpoint selection.

Run encoder preparation and trained arms sequentially on physical GPU 0 only.
Resource ceiling is 16 GiB allocated VRAM and four hours per phase. Cache
preparation cost, external pretraining, readout training and inference are
reported separately. Do not start another heavy project model job concurrently.

## Frozen evaluation

Use the same 256 primary IDs and 393 paired questions as V57/V58, explicitly
reused development. Evaluate normal and blank-question answers for both
encoders, exact stopped block match, exact-raster multiset F1, paired-question
both-correct rate, raw first examples and resource use. No OCR, repair,
reference-informed decoding or global glyph inventory. Bootstrap by source
article with 2,000 paired draws, seed 20266019.

The probe is useful only if MIXAR-minus-student F1 has a positive lower 95%
interval, MIXAR primary exact match >=15%, F1 >=35%, question-dependent F1 has
a positive lower interval and >=2% of question pairs have both answers exact.
All source-boundary, split and finite-resource checks must also pass. Report
the outcome even if negative. Passing these probe gates would identify a
candidate visual foundation, not establish high-quality general language,
from-scratch single-GPU pretraining, arbitrary historical-script semantics or
parity with Qwen 8B. A later native-image inference path and separately
registered learning experiment would still be required.

# Matched complete-writing development images

Specified 5 November 2026 before rendering the development sources. This is
data/interface preparation for broader independent image-language learning,
not a new training run or an evaluation of V63. Its frozen sources, canonical
CMRC image cache, sampling and gates remain unchanged.

The complete-line training cache already contains 3,193 Chinese CMRC examples
and 9,783 English Dolly instructions. It uses the fixed 24-pixel Noto Sans CJK
JP renderer, protected full ink bounds and ordinary whitespace, with at most
240 prompt blocks, 128 answer blocks and two white termination blocks.
Development images for a future comparison need that exact configuration.

Read only the prepared development partitions: 1,321 CMRC questions and 1,470
Dolly instructions. Pin their existing source hashes, the training cache
report hash, renderer provenance and code before preparation. Original
training/development partitions are not changed. Do not read the Dolly sealed
test source, call a teacher/actor, or use model scores for admission. The CMRC
development questions have been reused in earlier experiments; these are not
a fresh final language test. Dolly's exact-source split does not guarantee
semantic-near-duplicate or foundation-pretraining exclusion.

Use exactly the prior training admission rule: preserve the entire flat
prompt and the entire first recorded reference/response at fixed font size,
or retain the rejected ID and explicit reason. No source is cropped, shrunk,
rewritten or filtered by answer quality. Record category and width/height
counts. Retain first four source-index examples per source without quality
selection. Verify pack/unpack equality against every complete rendered image
and pin every packed file. The training cache and all original source files
must retain their hashes after preparation.

Add a separate bit-only loader that can read both the old complete-line
training cache and the new explicitly evaluation-only cache. Require an
explicit purpose and reject a development cache opened for training. Return
only physical pixels and spatial loss boundaries. The inference accessor
returns an owning copy of prompt pixels only, with no answer storage, source
text, IDs, category, Unicode table or font dependency. A target accessor is
for offline teacher-forced evaluation only. Validate packed bit count, type,
physical widths, capacity, hashes and white termination. Locking a later
experiment requires pinning the report hash as well as its preparation code.

Run CPU-only checks of real training and development rows, image round trips,
source/hash preservation, split refusal, tamper detection and prompt-storage
isolation. Do not run a second GPU/model-generation job. No language score,
production recipe, checkpoint choice or passing threshold is established by
this preparation. A future matched training protocol, real full-model memory
preflight, autonomous answers and unscored final evaluation remain necessary.

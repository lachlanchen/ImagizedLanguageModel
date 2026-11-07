# Retain full-resolution glyphs with native visual windows

This control follows the complete Chinese interface assessment: original and
resized images both give 0/32 exact answers, original text gives 16/32, and
432 codec reconstructions have 0.705221 character error. It is specified
before execution. No new optimizer, dataset, model weights or teacher service
is admitted by this diagnostic.

Keep complete original 1024-square image pixels, their FP32 native VAE
sampling and their matched seeds. The resulting image-patch grid is 64 by 64.
Retain its complete row-major order and all 4096 visual positions in the native
language body. Do not prune the contextual memory or resize glyphs.

Change only the semantic visual transformer before fusion. Divide its raw
64-square projected patch grid into overlapping 27-square windows with evenly
spaced origins [0, 18, 37] on each axis. Encode each window independently
using the source's own 729 position embeddings and unchanged semantic layers.
Average encoded features uniformly where windows overlap, restore the original
64-square order, and fuse with the unchanged full-image generative embeddings.
The native time embedding, role tokens, complete-context attention mask,
language body and cached readout remain unchanged. All patches must have
positive coverage. No OCR, word/character segmentation or answer crop enters
this computation.

At a 27-square grid, there is exactly one window. Before full-page calls,
require bit-exact prefix and mask equality with the existing source helper
on the first four fixed native-432 English normal inputs. Run both complete
96-decision native readouts and require exact ID replay against each other
and the retained source observation. Report these eight calls separately.
Failure stops the intervention before full-page assessment.

Then assess every original Chinese condition (48 calls) and every original
English condition (80 calls), with the original instructions, references,
condition order, latent/readout seeds, BF16 body and 96-decision cap. Keep
the already measured original/native-scale source outcomes as controls, not
replacement results. Retain all 32 full-resolution Chinese codec
reconstructions and assess them independently with the same one-thread
Tesseract chi_sim+eng/PSM6 transcript check. This tests reconstruction
readability, not an exact measure of latent information.

The intervention jointly changes semantic position handling and its attention
field; it does not isolate either mechanism. Averaged contextual features may
also lose information. A recovery would identify a useful visual interface
for this source, not a trained independent ILM. Failure would reject zero-shot
use of this window assembly. No image-answer generation claim follows.

Related primary work: Shi et al.,
[When Do We Not Need Larger Vision Models?](https://arxiv.org/abs/2403.13043v2)
and its [official S2 implementation](https://github.com/bfshi/scaling_on_scales)
motivate applying a frozen encoder within its native spatial scale. The
present overlap/semantic-feature control is not a replication of S2, and
that paper does not validate this particular Chinese reading interface.

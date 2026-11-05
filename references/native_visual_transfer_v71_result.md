# Reading through numeric image controls

The vocabulary-free imported contextual body answers four of eight checked
Chinese image questions exactly at the declared native sampling budget,
versus zero of eight with question ink erased. At original image scale,
one of eight answers is exact. The matched text helper answers three exactly.
These reused development cases establish bounded reading capability. They
do not qualify an image-output language model.

| Route | Exact / 8 | Character accuracy | Terminated / 8 |
| --- | ---: | ---: | ---: |
| Original image | 1 | 59.721% | 8 |
| Original image, question erased | 0 | 6.057% | 6 |
| Native-budget image | 4 | 69.354% | 8 |
| Native-budget image, question erased | 0 | 3.561% | 7 |
| Matched source text | 3 | 77.110% | 8 |

Numeric controls are compiled once from the authenticated source's chat/task
embeddings. The image interface contains these fixed vectors and the complete
imported vision/contextual body, with zero-output contextual adapters and
no vocabulary module. A temporary source text projection and tokenizer live
only in the separate diagnostic helper. There are no optimizer updates and
no generated image answers in this experiment.

The geometry uses factor-32 dimensions, a minimum of 1,048,576 pixels and a
maximum of 4,194,304, with declared Pillow RGB bilinear interpolation. This
reproduces the budget calculation; it does not prove equality to native C++
resized pixels. Upsampling adds no textual information. Both task controls
and observer scale differ from the failed earlier actor, so the comparison
does not isolate one causal repair.

All 32 original/transformed input images pass independent verification.
All 742 recorded helper projection decisions replay as exact argmax choices,
with no numerical near-tie disagreement. This verifies retained states and
symbolic projection, not an independent replay of the complete visual body.
Generation takes 232.533 seconds including 126.192 seconds of setup, with
18.983 GiB peak allocated GPU memory including the temporary helper head.
Independent CPU verification takes 38.511 seconds. GPU 0 is released afterward.

The [compact evidence record](evidence/native_visual_transfer_v71_result.json)
binds implementation hashes, raw generation and verification digests, and all
route summaries. Literal exact matching penalizes some valid aliases, such
as `Marilyn Manson` against a Chinese reference. Reference inclusion is not
counted as semantic correctness. The sample is too small and reused to support
general accuracy estimates. The next experiment must generate full physical
answer images without the diagnostic text projection.

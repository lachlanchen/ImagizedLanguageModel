# Autonomous reading and paired-question assessment

Specified during the 4,096-update visual-access run, before its final readout.
The bound learner, native prefix and source-loader files remain unchanged.
This assessment concerns the temporary native text head. It cannot qualify
image-answer generation or the complete ILM.

First retain all twelve matched before/after diagnostic responses. Four copies
belong to training; four copies and four questions are held out from this
optimizer. Report those groups separately. If the screen improves, retain a
separate complete observation of all 32 held-out copies and all 32 held-out
natural questions. Do not pool the overlapping diagnostic cases into that
64-case denominator. Validation articles are disjoint from training articles,
but the public data have already been used in exploratory project work and
are not a sealed benchmark.

The complete held-out input-only index and separate references were prepared
before the final readout. VAE seeds preserve the declared cache seeds;
readout seeds are 20268621+i in complete-cohort order. Every changing question
enters through its complete source image. Evaluation loads neither cached
training inputs nor correct answer histories. The language body and its output
vocabulary remain frozen diagnostic components.

For each of the sixteen held-out passage pairs, reconstruct the source page
with its original font and verify every RGB pixel and geometry field. Erase
from the beginning of the question paragraph through the bottom of the page.
This removes the question and the constant answer marker; it preserves every
context pixel. The boundary is determined by source paragraph layout, without
finding or cropping the answer. Require both erased pages of a pair to be
pixel-identical. Run two normal questions and one shared erased page per
passage: 32 normal calls and 16 erased calls. Do not count the shared erased
answer as two independent model calls.

Use latent seed 20269021+p and readout seed 20269121+p for passage p, with
the same seeds for its two questions and erased page. Native top-one decoding,
96-decision cap and the generic instruction stay unchanged. Each run loads
the complete 4,096-update checkpoint with its original implementation binding.
The intervention is an additional observation with new, predeclared seeds;
do not silently substitute its scores for the full validation observation.

Report both-correct passage pairs, ordinary exact answers, changed outputs
after erasure and correct normal answers unchanged after erasure. Output
sensitivity alone is insufficient: two wrong answers can differ. A shared
erased answer cannot correctly answer both distinct questions. Interpret the
erasure alongside ordinary correctness and retain every response.

Canonical exact match applies NFC, removes Chinese whitespace and collapses
English whitespace. It preserves case, punctuation, quotation marks and all
written characters. The stricter exact-stopped score additionally requires
native EOS. Alternative correct wording needs separate inspection against the
source passage; do not repair or silently rewrite generated strings. A supplied
character-multiset F1 ignores order and cannot establish language capability.

CPU evidence export requires all 4,096 contiguous finite optimizer records,
all 384 declared training IDs and no validation IDs. It binds raw observations,
full input/reference PNGs, the checkpoint hash and reports. Model weights and
runtime caches remain outside Git. The image-writing stage remains required.

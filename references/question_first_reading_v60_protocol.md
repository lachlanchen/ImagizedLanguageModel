# V60: read the question before the passage

Registered 2026-11-05 before new training or development outputs. V59 frozen
readout evaluation is running; its scores have not been inspected. V58's
locked block model answers 25/256 questions exactly and both questions in only
4/393 shared-paragraph pairs. Extra matched glyph training reaches 21/256,
so the V58 exact-match improvement attributable to a block head is unresolved.

## Hypothesis and fixed intervention

A causal reader currently assigns passage states before it encounters the
question. Its final query must recover evidence from question-independent
keys. Presenting the question first makes every later passage state a function
of that question: h_i = f(Q, X_1,...,X_i). Test this input-order intervention
without changing architecture, answer support, parameter count or data.

Initialize both arms from the same completed V58 block EMA checkpoint,
SHA-256 `4b2d79fb0b2dd403f6a1bc2581dcd28a847948d05ab47c632589159c7f8a66ca`.
The context-first control keeps `资料：{context}问题：{question}回答：`.
The question-first arm uses `问题：{question}资料：{context}回答：`.
The exact same source characters, paragraph, question and answer images occur
in each prompt, in different order. No answer crop, source offset, text ID,
answer length or reference-informed mask enters the reader or decoder.
Rendering, block law, duplicate-sequence marginalization, mode decoding and
16-cell maximum remain V58. Native continuation writer remains trained by the
natural/motor batches; an observed-block answer is still extractive, not
general free writing or demonstrated historical-script understanding.

Keep V58's 4,000 updates, batch four, three QA batches per natural/motor batch,
same uniform question sampling, optimizer/rates, warmup, cosine, clipping,
EMA and deterministic math-SDPA/BF16 regime. Use fresh matched training seed
20266010 and data seed 20266011. Preserve optimizer, all RNG states, exact
source/data/foundation hashes and production checkpoints. A four-update
interrupted/resumed GPU run must match uninterrupted weights and EMA exactly
before production. Both arms finish before their development scores open.
Run sequentially on physical GPU 0, at most 16 GiB allocated VRAM and four
hours per phase, without another heavy project job in parallel.

## Evaluation and decision

Reuse explicitly the fixed V57/V58 256 development IDs, all 393 question
pairs and the 256 supplied-image motor targets. Each arm receives its own
trained prompt order. Evaluate normal and question-blanked prompts, exact
stopped image sequences, exact-raster multiset F1, both-correct question pairs,
all-development conditional block NLL, raw first eight answers and resource
use. Blanking is done at that order's actual question positions and leaves
the passage and layout unchanged. Labels enter scoring only after decoding.
No OCR or glyph inventory; a block answer must exactly copy visible source
pixels and its predicted boundaries.

Use paired bootstrap over source articles, 2,000 draws, seed 20266119.
The order intervention qualifies as useful only if question-first minus
context-first F1 has positive lower 95% interval, question-first exact match
is at least 15%, F1 at least 35%, normal-minus-blank F1 has positive lower
interval, at least 2% of pairs have both answers exact, supplied-image motor
copy remains exact, and split/boundary/resource checks pass. These are bounded
reading gates, not high-quality general-language qualification. Publish a
negative result as well. No best-epoch or threshold selection on development.

If successful, the next goal is independent image-language continuation and
instruction following, with a new held-out test; this reused development set
cannot establish final generalization or equivalence to a large text model.

# Visual readers and glyph writers: what is still missing

Primary sources checked on 7 November 2026. The complete ILM remains
unqualified. A smaller visual encoder or a better font generator is useful
only if a learned system uses it to produce correct, readable answers from
unseen image prompts. Neither component establishes that behavior alone.

## The compact reader is already an attempted initialization

The existing `Team-PIXEL/pixel-m4` cache matches the exact revision and hashes
used by [the V32 implementation](../ilm/visual_lm/visual_semantic_raster_transducer.py):

| Item | Independently checked value |
| --- | --- |
| Revision | `56bfcbf71e98f613ee00f8efb7a607bf0058f1e6` |
| Weight bytes | 449,450,757 |
| Weight SHA256 | `6aa0642d46fe211727fefc5ac6e0bc28efa8511d1f0d9e5eee1987fa821600bc` |
| Configuration SHA256 | `90789708a8b064d848977d256ce4e3e20ba51f57d6803a59295bf401d514f83f` |

This check reads files without loading a neural model or downloading weights.
The private receipt is
`artifacts/contextual_pixel_adaptation_review/pixel_m4_reuse_review.json`.
It establishes reuse identity, not a new quality result.

V32 already combined this reader with a causal planner and continuous image
writer. The [subsequent V36 review](visual_semantic_plan_v36_research.md)
records the failed V32 generation and the weak PIXEL-M4 prompt-to-answer
retrieval probe. That probe does not reject every possible adaptation of this
encoder. It does rule out presenting the same cached initialization as a
newly discovered solution. The
[author repository](https://github.com/ilkerkesen/pixel-m4) describes a masked
autoencoder and an 86M encoder after discarding its reconstruction decoder.
Keeping a reconstruction decoder does not establish autonomous language use.

## Newer papers have different output contracts

| Primary source | Actual task or interface | Consequence for this project |
| --- | --- | --- |
| [Full Glyph Images Beat Token Embeddings, v1, Sections 3.1–3.4](https://arxiv.org/html/2607.03994v1) | The visual branch replaces input embeddings, but predicts character classes using cross-entropy. It also receives patch and global classification supervision. | Evidence about visual input learning is relevant. Its accuracy does not qualify a writer that emits new image answers. |
| [MPIXEL, v1, Sections 2–3 and Appendix A.3](https://arxiv.org/html/2610.29681v1) | ViT-MAE pretraining at 86M and 307M, followed by classification, parsing, NER and inference tasks. The authors report approximately 1,350 and 2,500 H100 GPU-hours of pretraining. | This is a possible source of visual-language representations, with external pretraining costs. It is not the cached PIXEL-M4 checkpoint or a demonstrated single-card free writer. The linked `ilkerkesen/mpixel` repository returned 404 during this review; code and weights from that release have not been verified locally. |
| [VecGlypher, v1, Sections 3.4–3.5](https://arxiv.org/html/2602.21461v1) | A supplied target character and style description or exemplars condition autoregressive SVG path-token generation. | Useful glyph geometry does not select the language content of an answer. Its symbolic SVG interface differs from the required image-only actor. No local adaptation or generation is claimed. |

The Full Glyph paper describes a global ViT over the complete input sequence
and next-character supervision at every position. The causal visibility of
that visual encoder would need implementation verification before reusing
the reported accuracy as a prefix-only language baseline. This is an
unresolved protocol question, not a demonstrated leakage finding. Its
vision-only auxiliary supervision also needs to be accounted for when
interpreting a comparison as changing only the input representation.

These are source reviews, not independent reproductions. The already reviewed
[PixelBench interface](reading_use_literature_20261105.md) likewise emits
palette indices through a text vocabulary; it does not fill the missing
image-language writer by itself.

## Implication for the current learning decision

The [running contextual comparison](contextual_pixel_adaptation_protocol.md)
continues unchanged. It tests answer selection with a physically exact source
copying motor. Its distribution has no support for answers absent from the
input. Even a successful comparison cannot qualify free composition.

The [completed writing controls](decoder_metric_v82_result.md) show that
supplied answer codes can reconstruct readable writing while autonomous
conditional prediction remains inaccurate. Correct history does not fully
repair the result. This justifies investigating the learned conditional path;
it does not establish one unique failure mechanism or admit an unchanged
longer run.

A subsequent generation experiment must retain actual prompt-only free
outputs and include answers absent from their inputs. Reconstruction,
classification, retrieval and source copying remain supporting measurements.
No new model job, changed training protocol, benchmark pass or validated
replacement architecture follows from this review.

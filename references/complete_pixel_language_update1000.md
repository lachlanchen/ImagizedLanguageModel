# Complete image answers improve on the 32 training questions

At the first registered 1,000-update assessment, **17/32 complete training
answers match their reference images pixel exactly**, versus 0/32 initially.
Foreground F1 rises from 0.1303 to **0.9435**. All 32 streams stop without a cap,
30 have a correct reference length, and four of the 16 passage pairs have both
distinct answers exact. These are seen-question fitting results. The separate
development prompts have not yet been generated or used for model selection.

The independent light audit verifies all 32 output PNGs against actual emitted
tensors, owning complete references, saved receipts and learned stop events.
Every physical score matches the producer. All 32 full output/reference pairs
were actually inspected on four plates with original pixels preserved. Many
answers are readable, including multi-block Chinese phrases and dates. Longer
damaged or truncated outputs remain, notably cases 12, 14, 28 and 31; several
other answers have small physical defects. No manual image repair or independent
semantic score enters the result.

The complete [00–07](evidence/complete_pixel_language_update1000/update1000_00_07.png),
[08–15](evidence/complete_pixel_language_update1000/update1000_08_15.png),
[16–23](evidence/complete_pixel_language_update1000/update1000_16_23.png) and
[24–31](evidence/complete_pixel_language_update1000/update1000_24_31.png)
plates retain every case in source order, including failures. The owning
[first prompt](evidence/complete_pixel_language_update1000/00/prompt.png) and
[second prompt](evidence/complete_pixel_language_update1000/01/prompt.png)
show different questions over the same training passage. Their answer images
are both exact. These examples are the first original source pair, rather
than a selection of the best-looking outputs.

Matched whole-cohort image NLL under true preceding images falls from 9.88395
to 0.006959 nats per physical piece. Autonomous scores instead use the model's
own emitted images as history, its learned stopping decisions and no supplied
answer length. At this intermediate assessment, generation reuses authenticated
full native contexts previously computed from the owning prompt PNGs. The final
assessment must invoke the actual PNG-input actor again and verify all native
prefixes remained unchanged. The lighter intermediate audit does not replay the
full inherited reader or re-encode every feedback record; the registered final
CPU audit requires the latter after the model process exits.

The same complete actor contains about 7.54B parameters, with 13.856M trained
parameters. It inherits external foundation pretraining, trained image-history
interfaces and the prior physical raster fit. This is not a lightweight
foundation model trained from scratch, nor does exact fitting establish new
language understanding. No tokenizer, language output vocabulary, OCR service
or font renderer is used in generation.

The [full audit](evidence/complete_pixel_language_update1000/independent_audit.json),
[producer report](evidence/complete_pixel_language_update1000/producer_report.json)
and [review manifest](evidence/complete_pixel_language_update1000/review_and_manifest.json)
bind the complete images and measurements. A local adapter-only checkpoint at
`artifacts/complete_pixel_language_v1/milestone_models/actor_update01000.pt`
preserves this model state without duplicating frozen foundation weights.
The registered writing-fit requirements, at least 24/32 exact and F1 at least
0.98, remain unmet. Training continues under the original schedule; its only
early-stop rule is 32/32 exact training answers at a scheduled milestone. Useful
unseen-prompt generation and the broader ILM goal remain unqualified.

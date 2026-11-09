# Complete image writing meets the bounded training-fit criteria

At 2,000 updates, **29/32 complete training answers are pixel-exact**, with
**0.996086 foreground F1**. Both distinct answers are exact in **14/16 passage
pairs**. All 32 outputs have a correct reference length, stop under the learned
decision and avoid the safety cap. This passes the prospectively registered
writing-fit criteria of at least 24/32 exact answers and F1 at least 0.98.
These are seen-question fitting results; development generation remains pending.

| Registered assessment | Complete training answers exact | Foreground F1 |
| --- | ---: | ---: |
| Initial state | 0/32 | 0.130254 |
| Update 1,000 | 17/32 | 0.943528 |
| Update 2,000 | 29/32 | 0.996086 |

The independent light inspection checks all raw image tensors, exact PNGs,
owning complete references, learned stop events and 107 image-history feedback
records. Each feedback record contains the actual emitted image and the
recorded feature hash matches the generated stream. All scores reproduce the
producer. Independent encoder recomputation remains part of the terminal full
CPU audit; no neural model is loaded by this milestone inspection.

Every output/reference pair was actually inspected on the complete
[00–07](evidence/complete_pixel_language_update2000/cases_00_07.png),
[08–15](evidence/complete_pixel_language_update2000/cases_08_15.png),
[16–23](evidence/complete_pixel_language_update2000/cases_16_23.png) and
[24–31](evidence/complete_pixel_language_update2000/cases_24_31.png) plates.
Cases 10, 11 and 17 remain non-exact. Case 10 contains a visibly distorted
internal glyph; the other two have small physical defects. Long phrases that
were damaged or truncated at update 1,000 now match, including cases 12, 14,
28 and 31. No generated pixel is repaired, replaced or rendered from a font.
An independent semantic score is not assigned from these physical measurements.

Matched whole-cohort image NLL under correct preceding image history falls
from 9.88395 to 0.006959 at update 1,000 and 0.001138 at update 2,000. The
autonomous writing scores instead use emitted images as history and supply
neither target images nor answer length. Intermediate generation uses the
owning full native prompt KV computed from actual PNGs. The final assessment
must invoke the PNG reader again and verify that source contexts stayed exact.

This milestone retains 8,000 example draws, or 250 exposures per question.
The exact update-two replay adds one actual optimizer call and four executed
examples, giving 2,001 recorded calls and 8,004 executed examples through this
milestone. Recorded active optimizer time is **1,919.738 seconds (32.0 minutes)**,
including those calls' sampling, batching, forward/backward, clipping, update
and synchronization. It excludes model loading, context capture, answer
generation, checkpoint saving, external pretraining and prior raster learning.
No matched-quality efficiency claim against a conventional LLM follows.

The same externally pretrained 7.54B actor has about 13.856M trained parameters.
This result does not establish a lightweight foundation learned from scratch,
new language understanding or historical-script competence. The previous
continuous-code writer and this direct physical density have different
initialization, decoder, precision and schedules; their results are not an
isolated causal comparison of losses.

The [inspection receipt](evidence/complete_pixel_language_update2000/independent_audit.json),
[producer report](evidence/complete_pixel_language_update2000/producer_report.json)
and [raw-image review](evidence/complete_pixel_language_update2000/review_and_manifest.json)
retain every case and relevant measurement. The adapter-only model is preserved
at `artifacts/complete_pixel_language_v1/milestone_models/actor_update02000.pt`,
with its [hash and source checkpoint](evidence/complete_pixel_language_update2000/retained_model.json).
Training continues under the unchanged rule: stop only at 32/32 exact training
answers at a registered milestone, otherwise finish at update 8,000. Held-out
Chinese questions, English instructions and erased controls remain necessary
to assess transfer. The general ILM objective remains incomplete.

# All complete training answers match at update 4,000

The native image-to-image actor produces **32/32 pixel-exact complete training
answers**, with foreground F1 **1.0**, at the registered 4,000-update assessment.
Both distinct answers match in **16/16 passage pairs**. Every output stops at
the correct reference length, and no output reaches the safety cap. This
triggers the registered training-fit stop rule. Fresh-PNG final training and
held-out generation are separate assessments and remain pending at this
milestone.

| Assessment | Complete training answers exact | Foreground F1 |
| --- | ---: | ---: |
| Initial state | 0/32 | 0.130254 |
| Update 1,000 | 17/32 | 0.943528 |
| Update 2,000 | 29/32 | 0.996086 |
| Update 4,000 | 32/32 | 1.000000 |

Independent scoring checks the original PNGs against their emitted tensors,
complete owning references, learned stop events and all 107 recorded
image-history feedback records. All scores reproduce the producer. The full
terminal audit must still independently re-encode the feedback images and
check source-prefill freshness, training exposure and held-out outputs.

All original output/reference pairs were actually viewed on the
[00–07](evidence/complete_pixel_language_update4000/cases_00_07.png),
[08–15](evidence/complete_pixel_language_update4000/cases_08_15.png),
[16–23](evidence/complete_pixel_language_update4000/cases_16_23.png) and
[24–31](evidence/complete_pixel_language_update4000/cases_24_31.png) plates.
No output pixels are repaired or replaced by a font renderer. The three
non-exact answers at update 2,000 now match their complete references.

Matched whole-cohort image NLL, measured with correct preceding image history,
falls from 9.883952 initially to 0.006959, 0.001138 and 8.659788e-9 at the three
registered milestones. Autonomous generation instead reads its own emitted
images and receives neither the reference images nor an answer length.
Intermediate assessments reuse authenticated native prompt KV from the
original PNGs. They do not independently invoke the PNG reader again.

The run retains 16,000 training-example draws, or 500 exposures per question.
The actual update-two replay adds one call and four executed examples:
4,001 optimizer calls and 16,004 executed examples through this milestone.
Measured active optimizer time is 3,789.292 seconds (63.2 minutes), excluding
initialization, context capture, generation, checkpoint saving and inherited
pretraining. This is not a matched-quality efficiency comparison with an LLM.

The [independent inspection](evidence/complete_pixel_language_update4000/independent_audit.json),
[producer report](evidence/complete_pixel_language_update4000/producer_report.json)
and [complete review manifest](evidence/complete_pixel_language_update4000/review_and_manifest.json)
retain every case. The adapter-only model is preserved locally as
`artifacts/complete_pixel_language_v1/milestone_models/actor_update04000.pt`;
its [receipt](evidence/complete_pixel_language_update4000/retained_model.json)
binds the original rolling checkpoint.

This establishes complete seen-question fitting, not general language
understanding. The actor retains a 7.54B externally pretrained foundation and
trains approximately 13.856M parameters. Useful unseen-question answers,
historical-script competence and a lightweight foundation learned from
glyphs remain unproven. The complete ILM objective remains active.

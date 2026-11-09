# Broader pixel-language training: the 4,000-update assessment

At 4,000 updates, the 512-question experiment improves physical writing but
does not demonstrate useful answers to unseen questions. Independent CPU
review reproduces all 112 complete-image scores and stopping traces. There
are three pixel-exact answers among the 64 assessed training questions and
none among the 24 normal development prompts. Qualitative inspection finds
no usable correct complete development answer: most strings contain broken
forms, and two date-like responses give the wrong years.

This is an interim result. The registered 16,000-update run continues with
unchanged architecture, data, objective and assessment schedule. It has not
completed its final assessment or independent feedback-encoder audit.

| Fixed assessment group | Cases | Pixel-exact complete answers | Foreground F1 | Learned stops | Cap failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original training controls | 32 | 1 | 0.500820 | 32 | 0 |
| Added training examples | 32 | 2 | 0.513467 | 32 | 0 |
| Chinese development, normal | 16 | 0 | 0.329256 | 16 | 0 |
| Chinese development, question erased | 16 | 0 | 0.264601 | 16 | 0 |
| English development, normal | 8 | 0 | 0.030339 | 8 | 0 |
| English development, prompt erased | 8 | 0 | 0.050661 | 8 | 0 |

Every autonomous response starts from its original prompt PNG, observes its
own emitted images and uses learned stopping with the fixed 128-block cap.
Neither the reference answer nor its length enters generation. None of the
32 assessed training pairs or eight Chinese development pairs has both
answers pixel-exact. Chinese normal-minus-erased F1 is 0.064654; English is
−0.020322. These overlap differences do not establish semantic correctness.

Whole-training correct-history image NLL falls from 9.749791 initially to
0.251025. This averages physical-piece NLL within blocks and questions; it is
not semantic accuracy or the unscaled joint likelihood of a full answer.
All 512 questions contribute. The retained training history contains 16,000
example draws: 416 questions have 32 exposures and 96 have 28. The earlier
[32-question experiment](complete_pixel_language_result.md) used 500 exposures
per question at its final fit. Data diversity, repetition and total compute
therefore cannot be treated as a controlled data-only comparison.
The broader training cohort itself remains incompletely fitted. This interim
observation does not isolate insufficient optimization, weak reusable glyph
learning or failure to select the answer from the question and passage.

All 24 normal development outputs were viewed in source order on
[output-only sheets](evidence/pixel_language_diversity_interim04000/output_only/)
before the current reference comparison. They contain no reference answers
and preserve every original output pixel. These cases were reviewed in
earlier experiments, so this is not a blinded or sealed evaluation. The
[source-first notes](evidence/pixel_language_diversity_interim04000/source_first_notes.json)
are retained separately from the
[later qualitative comparison](evidence/pixel_language_diversity_interim04000/manual_review.json).
There is no independent human grading or formal semantic-accuracy estimate.

Two date-like strings illustrate why readable writing alone is insufficient:

| Development index | Approximate source-first reading | Required by the pictured passage |
| --- | --- | --- |
| 12: year of the Treaty of Bern | 1829年 | 1874年 |
| 15: Jurassic Coast World Heritage listing | 2003年 | 2001年 |

The [original outputs](evidence/pixel_language_diversity_interim04000/raw_outputs/),
the two [inspected prompt PNGs](evidence/pixel_language_diversity_interim04000/inspected_inputs/)
and their [reference PNGs](evidence/pixel_language_diversity_interim04000/inspected_references/)
remain unrepaired. Those two examples were chosen after review because their
outputs were date-like; all other responses are retained as well. All 24
erased outputs and the first four examples from each training group were
also viewed. The other 56 training outputs were checked numerically and
retained, rather than claimed to have been visually read.

Both date-like strings occur for year questions; erasing those questions gives
broken, non-date-like strings instead. This is consistent with coarse output
format responding to the input. Two cases do not establish a general mechanism,
and the wrong years show that format alone does not solve factual selection.

The [CPU assessment audit](evidence/pixel_language_diversity_interim04000/audit.json)
authenticates every output tensor, PNG, score and stopping event, plus the
feedback-file hashes. It does not replay the feedback encoder or the full
imported reader. The
[owning-state receipt](evidence/pixel_language_diversity_interim04000/owning_state_receipt.json)
binds the assessed learned state to its original checkpoint. Its preservation
uses no GPU or inference and is fixed by the registered assessment time,
independently of quality. All 112 raw outputs, all 28 display sheets and their
[retention hashes](evidence/pixel_language_diversity_interim04000/retention_manifest.json)
are available for inspection.

The deployed actor still contains a 7.54B-parameter pretrained body with
13.856M locally trained parameters. This result establishes neither a small
foundation trained from scratch nor a matched-quality efficiency advantage
over an LLM. English is untrained transfer in this Chinese training stage;
historical forms and long-form writing are not evaluated here. The central
criterion remains correct, readable answers to new image prompts. If the
completed run still fails that criterion, the next diagnostic should compare
answer likelihood under the two different questions about each passage,
with identical correct image history, before another architecture change.
The subsequent [saved-prefix diagnosis](pixel_language_first_block_diagnosis.md)
localizes a physical error to the first image in every normal development
response. It specifies a first-block likelihood comparison to separate
conditional preference from local greedy decoding after the live run exits.

The review is reproducible on CPU after the owning assessment and fixed-state
receipt exist:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_pixel_language_assessment.py \
  --run artifacts/pixel_language_diversity_training_v2 --update 4000 \
  --output artifacts/pixel_language_diversity_review04000_recheck
```

The output directory must be new. Model training remains governed by the
[registered protocol](pixel_language_diversity_training_protocol.md).

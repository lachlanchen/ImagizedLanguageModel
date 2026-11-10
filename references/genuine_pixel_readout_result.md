# Direct pixel loss qualifies genuine-input glyph writing

Broader genuine visual contexts strongly improve glyph copying, and a pixel-only
readout objective passes the fixed physical qualification. The architecture,
parameter count, image-feedback encoder and decoder are unchanged. Complete
autonomous assessment is now complete: the writer retains the eight training
controls but still fails most development answers. This is a copying result,
not a useful general language model.

| Readout | Training-copy mean F1 | Held-wording mean F1 | Held cases reaching F1 0.95 /256 | Original-history mean F1 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Preserved language baseline |0.591677|0.606911|6|0.999996|Fail|
| Genuine contexts, anchor MSE + cosine + pixel loss |0.998227|0.951778|194|0.999993|Fail|
| Same genuine contexts, pixel loss only |0.998447|0.972856|229|0.999997|Pass|

The held-wording requirement was fixed at mean F1 at least 0.95 and at least 80%
of cases reaching 0.95. The anchor-loss candidate misses the latter requirement
(194/256, 75.78%); it is not promoted to autonomous assessment. The pixel-only
candidate reaches 229/256 (89.45%). Both retain a first-state non-stop fraction
of 1.0 and original-history quality. The training-copy mean requirement is 0.99;
the permitted original-history loss is 0.01 relative to the same baseline. No
threshold was changed after the result.

| Readout | Exact training-copy images /8172 | Exact held-wording images /256 | Exact original-history images /3855 |
| --- | ---: | ---: | ---: |
| Preserved baseline |0|0|3851|
| Genuine contexts, anchor losses |6683|6|3849|
| Genuine contexts, pixel only |7652|42|3852|

All three fixed 64-triplet boards for each fitted candidate were actually
inspected. The held sample still contains wrong punctuation and altered or
incomplete strokes. Pixel-only qualification is not perfect copying, unseen
glyph identity, a human grade of every form, or language accuracy. Every one
of the 24,566 baseline/candidate glyph images per comparison is retained in
lossless packed PNGs, with every packed pixel checked against its saved tensor:
[anchor-loss archive](evidence/genuine_glyph_readout_v1/archive.json) and
[pixel-only archive](evidence/direct_pixel_readout_g_only_v1/archive.json).
Each archive also retains all per-state metrics and the actually viewed boards.

![Held-wording target, baseline and pixel-only outputs](evidence/direct_pixel_readout_g_only_v1/copy_held_gold_baseline_candidate.png)

The data are 8,172 training images covering 4,086 visible canonical forms under two
copy instructions, plus 256 fixed held-wording cases. The empty form is excluded.
Each target is pasted as its exact 32×32 bitmap into the 512×256 input; no target
ID is supplied to the reader. All held identities are seen in training, and
the 3,855 original nonterminal histories were already seen by the original fit:
these routes test wording transfer and retention, respectively.

The frozen externally pretrained reader cached all 8,428 inputs in 9,651.28 seconds,
including 70.07 seconds of memory-pressure waiting, with peak allocation 17.723 GiB.
The measured first-batch/single-image state difference is 0.00032425, below the
declared 0.002 limit. No temporary vocabulary head or teacher service produces
these new states. The [cache summary](evidence/genuine_glyph_readout_v1/cache_summary.json)
records geometry and resources. This cache cost and inherited pretraining must
not be omitted from an efficiency account.

Both fits start from the preserved language checkpoint and use 18,000 batch-128
AdamW updates with the prescribed seed, learning rate, schedule and equal
copy/history sampling. Only G's 16,791,552 parameters train. Removing source-anchor
MSE and cosine losses leaves the existing balanced pixel BCE through frozen D.
The motor vector is no longer required to equal a source input embedding; the
actual emitted image still determines the next body embedding through frozen E.
The pixel-only fit takes 90.54 seconds with 0.695 GiB peak GPU allocation, compared
with 101.77 seconds and 0.759 GiB for the anchor-loss fit. These are small-readout
costs, not foundation-model training or a matched-quality text-LLM speed result.
Parent random-index traces were not retained; the prescribed sampler and seed,
not verified bitwise sample equality, are matched.

The [CPU reader-state audit](evidence/genuine_copy_state_audit_v1/report.json)
recovers 252/256 held identities by each of two fixed cosine rules against the
training views. It finds 8,428 distinct recorded state byte strings and no
conflicting exact-state group. This is evaluator-only geometry: no gallery
selects or repairs runtime output, and absence of collisions does not prove
semantics or learnability. Its complete measurements are retained, including
the less accurate cross-training-wording comparison. It makes no neural call.

The second declared arm, which would additionally train D's input block,
fails its shared-memory-pressure admission before training. Its failed-stage
record is preserved; it is not an unsuccessful trained algorithm and is not
needed to establish the loss-only physical result. The simpler passing arm
entered the original forty-stream autonomous assessment after actual image
review. All forty raw streams were then inspected on ten boards before reference
scoring. The [complete stream archive](evidence/direct_pixel_language_g_only_v1/archive.json)
retains every output PNG, viewed board and unchanged score.

| Complete-stream route | Cases | Exact canonical-reference images | Mean foreground F1 | Stops | Caps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training controls | 8 | 8 | 1.000000 | 8 | 0 |
| Chinese development, normal | 16 | 0 | 0.341013 | 5 | 11 |
| Chinese development, question erased | 16 | 0 | 0.073464 | 2 | 14 |

The prior baseline has one exact canonical development image and mean F1
0.354426, also with eleven caps. Thus better isolated copying does not improve
this complete-answer comparison. Several outputs remain readable, including
`288条`, `1874年` and `2001年`, but their pixels are not exact. The newly readable
`85分` is incomplete relative to `85分钟`; it is not counted as correct. Exact
canonical pixels, qualitative readability and semantic correctness are different
measurements. The cases have been seen in earlier experiments and have no
independent human grade. The assessment takes 450.26 seconds, including model
loading and resource guards. The C3 cohort remains excluded from fitting and
model selection; its frozen-checkpoint assessment is separate.

The runtime still inherits a 7.5B visual/contextual body and uses a 47.128M bridge.
No new small foundation, historical-form understanding, general English,
Unicode-independent unseen-form accuracy, priority claim or LLM-efficiency
advantage follows from this controlled copying result. The
[genuine-context protocol](genuine_glyph_readout_protocol.md) and
[direct-pixel protocol](direct_pixel_readout_protocol.md) preserve the recipes.

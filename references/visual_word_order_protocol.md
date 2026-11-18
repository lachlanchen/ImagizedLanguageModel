# Does a visual reader learn Mandarin word order?

A grammatical sentence and an unacceptable rearrangement can contain exactly the same images. Their relative score then tests whether context affects the reader's prediction of writing. The experiment follows the Mandarin minimal-pair method of Liu and colleagues. It uses the authors' main ZhoBLiMP release at revision `e8402762a563c13def19080a31e2cbb8d89e4ef3`. The complete benchmark covers more phenomena than the order-only sample here. [Paper](https://arxiv.org/abs/2411.06096) · [Original data](https://github.com/sjtu-compling/ZhoBLiMP/tree/e8402762a563c13def19080a31e2cbb8d89e4ef3)

Use the native eight-component reader at all five scheduled updates, including initialization. Compare the final model with initialization, chance, and a smoothed visual bigram estimated from exactly the visual training targets. The complete-response trial and its blind readings remain separate.

## Sample

Keep distinct pairs of at least two glyphs that share the first glyph and contain identical multisets of rendered cells. The first glyph serves as the shared unscored anchor. Use the original training rasterizer and its pinned font. Require the same multiset of source characters before rendering, and distinct physical sequences afterward. Traverse paradigms alphabetically. Within each, sort by SHA-256 of the paradigm name and pair ID joined by a slash. Keep at most 24 eligible pairs per paradigm. Exclude physical duplicates across selected pairs in either orientation and exact occurrences of either complete alternative anywhere within a training record, including its prompt. Record all selected pair IDs, coverage and exclusions. The substring screen only tests this particular fitted corpus.

## Scores

For cells with encoder outputs z₁,…,zₙ and visual histories hᵢ, sum the conditional log writing-code densities for z₂,…,zₙ and the Bernoulli log probabilities for continuing at 1,…,n−1 and ending at n. Evaluate both alternatives under supplied image histories. The continuous code density depends on the encoder's coordinates. Identical target multisets cancel any additive per-target score unrelated to order. The neural reader receives only the encoded pixels. Use deterministic CPU float32 neural operations and sum scores in float64. Record density and stopping contributions separately.

For the baseline, count transitions between exact physical bitmaps at the same predicted positions, including one end event per training record. For previous bitmap a, interpolate the conditional counts with a smoothed unigram using (count(a,b) + T(a) P(b))/(count(a) + T(a)), where T(a) is the number of observed successor types. A previously unseen context uses P(b). Apply add-one smoothing to the unigram over training-image types, the end event and one unknown type. A context-free unigram gives an exact tie on every eligible pair by construction.

Score the good-minus-bad margin as one, zero or half for a win, loss or tie within 1e−6. Report accuracy for each paradigm, macro accuracy and micro counts. Draw 10,000 paired bootstrap samples of whole paradigms with seed 20261118 for descriptive 95% intervals on final accuracy and its differences from initialization and the bigram. A lower endpoint above .5 for accuracy and above zero for improvement supports a bounded learned ordering preference. A lower endpoint above zero against the bigram additionally supports an advantage over this local transition control. Otherwise retain the measured limits. Templates and related paradigms restrict generalization.

No optimization, image generation, model selection, DRCD access or autonomous answer inspection occurs in this diagnostic. A positive result supports sentence discrimination and leaves question answering, free writing, historical transfer and matched-quality compute for their own tests. Use two CPU threads, no CUDA, 48 GiB available host RAM at entry, 24 GiB retained, 4 GiB peak process RSS and 20 minutes wall time. Preserve partial evidence on failure. Acquire the pinned archive separately in the ignored data directory.

```bash
mkdir -p data/evaluation_only/zhoblimp_e8402762
curl --fail --location https://raw.githubusercontent.com/sjtu-compling/ZhoBLiMP/e8402762a563c13def19080a31e2cbb8d89e4ef3/data/ZhoBLiMP.tar.gz -o data/evaluation_only/zhoblimp_e8402762/ZhoBLiMP.tar.gz
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. python scripts/evaluate_visual_word_order.py
```

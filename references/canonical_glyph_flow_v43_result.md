# Canonical Glyph Language V42 and Flow V43 Result

Date: 2026-08-14

Decision: **V42 establishes a bounded positive image-native language result;
V43 is rejected as a complete mechanism but qualifies a useful spatial writer
and sharpens the reader bottleneck.**

## Runtime Boundary

The deployed student receives ordered floating-point glyph rasters and emits a
glyph raster. It has no string, tokenizer, token or Unicode ID, character ID,
OCR result, vocabulary embedding or output, visual codebook, quantizer, glyph
lookup, external language model, or deployed candidate bank. Generated pixels
are thresholded, re-encoded by the same fixed image field, and become the only
feedback for another generation step.

Text is used only by host-side data preparation to find public-domain Chinese
windows and exact-suffix counterfactual pairs. The development bank identifies
an output after generation; it never supplies or selects a glyph during the
runtime path.

## V42: First Bounded Positive Language Evidence

V42 trains a `24,346,497`-parameter causal image model for 10,000 updates. It
completes in `1,374.70` seconds on one RTX 4090 D with `0.63194 GiB` peak
allocated CUDA memory. Its production checkpoint SHA-256 is:

`a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870`

On 2,048 fixed development windows:

| Route | Next-image top-1 | Target log probability |
|---|---:|---:|
| full ordered raster history | `0.19971` | `-5.25531` |
| shuffled earlier history | `0.18359` | `-5.42922` |
| final four raster cells | `0.15186` | `-5.60338` |
| final raster cell | `0.08057` | `-6.23706` |
| symbolic bigram evaluator | `0.12256` | `-5.05400` |
| image-unigram evaluator | `0.01416` | `-6.37330` |

The full visual model beats unigram and bigram top-1, shuffled-history top-1,
and shuffled-history target log probability by the preregistered margins. This
is evidence that ordered raster history carries useful next-glyph language
information. It is not evidence of broad language understanding.

V42 does not pass the complete conjunction. Exact-suffix counterfactual arm
accuracy is `0.53027`, and autonomous generated pixel F1 is `0.37308`; the
required values are greater than `0.60` and `0.55`.

## V43: Diverse Binding Attempt and Spatial Flow Writer

V43 initializes from the pinned V42 reader. Stage A uses 5,000 train-only
same-four-glyph-suffix pairs for 3,000 updates; Stage B freezes that reader and
trains a `5,693,697`-parameter conditional spatial rectified-flow writer for
5,000 updates over 128 raster targets per update.

| Stage | Time | Peak CUDA | Checkpoint SHA-256 |
|---|---:|---:|---|
| binding | `207.89 s` | `0.61394 GiB` | `1eceb6e63899b954172d8c38314c00ad05dbd007c1448d4fe99392c432e98794` |
| writer | `272.09 s` | `1.19858 GiB` | `f2f142f38a68e9add2f6a3db2d1128c59eebbd6351e4bbeb58cd7eaee6fe0539` |

The final model has `30,040,194` parameters. It preserves all four V42 language
control wins on a newly seeded 2,048-window development audit:

- full top-1 `0.17822` versus bigram `0.13721`, unigram `0.02295`, and shuffled
  history `0.14063`;
- full target log probability `-5.43490` versus shuffled `-5.71436`;
- generated identity top-1 `0.04297` versus unigram `0.02295`;
- zero blank generated fields and `1.34845` generated/target ink-density ratio.

Two preregistered gates fail:

| Gate | Measured | Required | Decision |
|---|---:|---:|---|
| exact-suffix arm accuracy | `0.54492` | `> 0.60` | fail |
| generated pixel F1 | `0.44507` | `> 0.55` | fail |

The frozen partition remains unopened. V43 may be described as a partial
reader/writer advance, not a qualified complete ILM.

## Post-Result Diagnosis

The diagnostic below was run only after the V43 decision and cannot change its
claim status.

### Stage A memorized its small pair pool

The corpus has approximately 215,340 eligible train suffix-four pairs. V43
used 5,000 and revisited them about 4.8 times. On matched 512-pair samples:

| Pair set | V42 arm accuracy | V43 arm accuracy |
|---|---:|---:|
| sampled V43 training pairs | `0.55078` | `0.99219` |
| unseen train pairs after the pool | `0.57422` | `0.58301` |
| development pairs | `0.53613` | `0.54688` |

The `+44.14`-point seen-pair gain but only `+0.88` and `+1.07` points on unseen
train and development pairs identifies memorization, not a generalized visual
history-binding rule.

### The writer works when the visual plan is correct

On 256 post-result development examples, the normal predicted plan gives
`0.46110` pixel F1. Choosing the best of four samples with target-side pixel F1
raises this only to `0.52006`, so sample selection is not the main failure.
Supplying the evaluator's exact target ink plan, solely as an oracle motor
diagnostic, raises ordinary anchor-selected output to `0.88240` pixel F1 and an
oracle-selected output to `0.92032`.

The direct V43 anchor inverse reaches `0.46315` F1, slightly below matched V42
at `0.47541`. Therefore V43's spatial flow is capable of readable rendering;
the autonomous continuous language plan is the dominant bottleneck.

## Next Decision

Do not enlarge or retrain the writer first. The next preregistered reader
experiment should:

1. freeze the accepted V42 base path to prevent language forgetting;
2. train a small residual long-history binding path rather than all reader
   weights;
3. consume at least one nonrepeating pass over 24,000 or more of the available
   train pairs;
4. align pairwise anchor differences with pairwise target-image differences,
   in addition to candidate assignment;
5. retain ordinary natural-window distillation to the V42 anchor; and
6. reuse the qualified V43 writer only after the counterfactual reader gate
   improves on development.

No historical-form, word-origin, or broad prompt-answer claim should be opened
until this bounded canonical Chinese dependency test passes.

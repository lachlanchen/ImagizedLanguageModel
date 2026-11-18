# Which part of a visual writer fails when glyphs appear in a new order?

The reader produced 801 exact four-character answers to 26,213 previously unseen Chinese idiom gaps. The same architecture trained with shuffled local views produced 805. In a separate experiment the writer received the **actual target glyphs** and copied 3,514 of 3,845 unfamiliar combinations correctly through the first character. Only 159 prefixes survived the second character. This gives a concrete place to intervene before spending another training run on passage understanding.

## Inferring the missing expression

All neural arms read passage images and decoded ink without consulting an answer inventory at inference. The matched fits saw 142,878 distinct training windows four times each. Supplying the four glyphs before and after a gap through a separate learned branch yielded 801 correct answers, versus 797 without the branch. Its paired advantage over shuffled inputs was −0.015 percentage points (95% source-group interval −0.137 to 0.108). Training-only local counts solved 1,504. The aligned reader wrote 405 answers on cases the count baseline missed. The first newly correct expression was 不以为然. On the first lost case it wrote the legible 相依为命 where the passage called for 不谋而合. In 256 cases that shared a local window and recorded answer with a training passage, the same model solved 39 when given the training passage itself, 37 with the held-out passage and 37 with only the local window.

## Drawing a known expression

| Information given to the frozen writer | Exact training expressions (3,845) | Rare expressions (360) | New combinations (3,845) |
| --- | ---: | ---: | ---: |
| Mean state from passages with that target | 98 | 0 | — |
| One state fitted per known target | 3,649 | 273 | — |
| Two states fitted per known target | 3,757 | 313 | — |
| Shared controller reading each supplied glyph | 2,759 | 110 | 0 |
| Shared controller trained with shuffled image assignments | 1 | 0 | 0 |

Rare expressions occurred one to five times in the training passages. Fitted states use the known answer. In 182 of 196 single-state failures the emitted path outranked the target under the writer's own probability model. Increasing beam width alone cannot make the target more probable than a higher-scoring path already found. The first state-optimization attempt exited before producing states because cuDNN required a training-mode GRU for input gradients. The successful run kept all weights frozen and enabled this mode only on the single-layer GRU with zero dropout.

The novel-combination test preserves the position-specific inventories of familiar glyphs and changes their ordering. These are synthetic strings used for a physical copying test. Directly feeding their true landmarks to the renderer reproduces all four source bitmaps on 3,833 cases. With the learned controller, exact prefixes at lengths one through four numbered **3,514, 159, 6 and 0**. On observed expressions they numbered **3,635, 2,999, 2,862 and 2,759**. Stopping alone cannot account for the gap. The writer emitted exactly four glyphs on 3,500 new combinations.

## Model implication

Let an ordered passage be images \(x_1,\ldots,x_n\). A literal ink sum \(S=\sum_t x_t\) assigns exactly the same observation to every permutation. No learner reading only \(S\) can distinguish two equally likely orders of the same images with accuracy above one half. Tiling images in an ordered grid is invertible when the reading order and cell boundaries are known. It preserves the information a learner needs to distinguish reversed sentences.

For target images \(y_t\), let \(A_t\) denote correct reproduction of the first \(t\) glyphs. The identity \(P(A_4)=P(A_1)\prod_{t=2}^4P(A_t\mid A_{t-1})\) requires no independence assumption. On new combinations, the observed probability of preserving the second glyph given a correct first glyph was 159/3,514 (4.5%), compared with 2,999/3,635 (82.5%) on training expressions. The measurements implicate history-dependent generation. They do not distinguish the contributions of the hidden state, emitted-pixel feedback and beam selection.

A subsequent frozen comparison changed how the writer used history. Both arms received exactly four output positions, committed a four-path beam at each boundary and produced identical first-position pixels. With drawing history carried forward, the writer reproduced 2,484 observed expressions and zero new strings. Resetting its drawing history from each supplied glyph image yielded 1,564 observed expressions and **1,450/3,845 new strings**. The reset arm had the same per-position counts on observed and permuted strings (3,599, 2,935, 3,027 and 2,581). This follows from the deterministic glyph-local computation and the matching glyph marginals. The two conditions scored 94 and 115 of the 360 rare expressions. Resetting passes the test for an effect of history and misses the 90% threshold for a usable motor. Fixed-length copying cannot validate autonomous stopping.

These results motivate a simpler factorization. Give the passage planner ordered visual history and train it to choose a compact physical description of each next glyph. Render that description with a glyph-local motor. In the existing system, 48 measured ink bits and the frozen motor already reproduce all targets in 3,833 of 3,845 new strings when the bits are supplied. The actual language test must infer those bits from the pictured passage alone. Score both complete strings and first-error position against the existing neural writer, shuffled passages and training-only local counts. A correct drawing of a contextually wrong idiom remains a language error. This test of familiar parts in unfamiliar combinations follows the experimental distinction developed by Lake and Baroni. [Lake and Baroni, ICML 2018](https://proceedings.mlr.press/v80/lake18a.html)

Related work already demonstrates rendered-text pretraining and autoregressive text-image generation. The contribution measured here is the contrast between correct local marks, compositional copying and independent contextual writing under a small GPU budget. [Rust et al., ICLR 2023](https://arxiv.org/abs/2207.06991) · [PIXAR](https://openreview.net/pdf/d6c690afdcd5b53644445c8640a54011c249dab7.pdf)

The two neighborhood fits together took 55.7 seconds on one RTX 4090 D with 295.4 MiB peak allocated device memory. The two shared-control fits and evaluation took 44.7 seconds. These incremental costs exclude earlier model training and data preparation. A claim of 1,000-fold savings requires a baseline at matched language quality.

[Fixed evaluation, physical images and receipts](evidence/ordered_neighborhood_20261118/) · [Shared-control protocol](shared_visual_writing_control_protocol.md) · [Prefix recount using the saved outputs](../scripts/report_visual_writing_prefixes.py)

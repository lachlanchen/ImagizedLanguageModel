# Canonical Glyph Binding V44 Result

Date: 2026-08-14

Decision: **rejected-or-partial** (`8/14` preregistered gates pass).

The frozen partition remained sealed and the V43 writer was not opened.

## Production Receipt

- checkpoint:
  `artifacts/canonical_glyph_binding_v44_20260814/checkpoint_final.pt`
- checkpoint SHA-256:
  `4b8e9d8e2c0e6cfa459c5a4f7bcf77e2e8921eb61f6eb34578fac086db93867d`
- frozen V42 checkpoint SHA-256:
  `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870`
- total parameters: `26,082,433`
- trainable residual parameters: `1,735,936`
- updates: `3,000`
- unique training pairs consumed once: `24,000`
- untouched training-partition pairs: `1,024`
- elapsed training time: `210.87 s`
- peak allocated training memory: `0.19848 GiB`
- V42 state exact after training: yes
- image-only runtime boundary: clean

The training-pair receipt is
`f542c084a51de95a2dd73d11954cc25ca1e2377788ddbe8c4739da2be2ff63d5`.
The nonoverlapping holdout receipt is
`e41637c5e3846e47d19ade6312205cd30c2c96c65730bce75ee8ab4a0745154c`.

## Fixed Development Result

The evaluator ran V42 and V44 on the same 2,048 natural windows and 512
development suffix pairs.

| Metric | Matched V42 | V44 | Fixed requirement |
|---|---:|---:|---:|
| natural full top-1 | `0.19434` | `0.17432` | V44 at least V42 - `0.015` |
| natural full log probability | `-5.23837` | `-5.77730` | V44 at least V42 - `0.10` |
| natural shuffled top-1 | `0.18604` | `0.15430` | full gain above `0.015` |
| natural bigram top-1 | `0.13672` | `0.13672` | full gain above `0.01` |
| development pair arm accuracy | `0.52441` | `0.53418` | above `0.60` |
| development shuffled-pair accuracy | `0.51367` | `0.51855` | full gain above `0.04` |
| consumed training-pair accuracy | report only | `0.57471` | diagnosis |
| unseen training-pair accuracy | report only | `0.57422` | above `0.60` |
| consumed-minus-unseen gap | report only | `0.00049` | below `0.10` |

V44 preserves the four absolute V42-style ordered-language controls. It fails
both matched-base retention gates, all three development binding gates, and the
unseen-pair accuracy gate. The near-zero consumed-versus-unseen gap shows that
the one-pass data design fixes V43's repeated-pool memorization problem, but it
does not create the missing binding rule.

## Post-Result Diagnosis

The preregistered decision is unchanged. A fixed interpolation audit evaluates
the frozen V42 anchor (`alpha=0`), V44 (`alpha=1`), and intermediate or
extrapolated residual strengths.

- Best development arm accuracy is only `0.53809` at `alpha=0.75`.
- No tested residual strength reaches `0.60`.
- The learned update-difference cosine with the true target-image difference is
  `-0.02289` on development and `-0.03395` on unseen training pairs.
- On natural windows, anchor cosine with the corpus-mean image field rises from
  `0.77712` at V42 to `0.88183` at V44.
- Raw target cosine rises from `0.60298` to `0.65413`, while exact top-1 falls
  from `0.19434` to `0.17432` and target log probability falls by `0.53893`
  nat.

This is not a residual-scale error. The learned update points away from the
signed target displacement and toward the shared image field dominated by
background and common stroke structure. Cosine-to-target alone can improve
under this common-mode collapse while discriminative next-image probability
gets worse.

## Consequence

Do not reopen the V43 writer and do not train another larger residual on the
same raw normalized DCT target. The next bounded experiment should change the
continuous visual target geometry while preserving an invertible image path:

1. estimate image-field mean and covariance from training rasters only;
2. separate common background/stroke modes from identity-bearing residual ink;
3. predict a centered, variance-preserving continuous image field for language;
4. keep the reversible full field available to the raster writer; and
5. require the new representation itself to improve held-out pair displacement
   and natural retrieval before another causal reader is trained.

This remains image-native: the proposed statistics and targets come from
rasters, not text, Unicode, token IDs, OCR, a visual codebook, or a deployed
candidate bank.

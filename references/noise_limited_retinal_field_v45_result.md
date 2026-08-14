# Noise-Limited Retinal Field V45 Result

Date: 2026-08-14

Decision: **qualified fixed retinal representation; 13/13 preregistered gates
pass.** V45 is not a language-model qualification. It authorizes a separately
preregistered causal language core trained from scratch in V45 coordinates.
The V43 writer and frozen evaluation partition remain closed.

## Immutable Receipts

- protocol SHA-256:
  `125702090b87534e1ab8bb8f3000765c65937608df056a0c5b771a09b1976029`
- corpus manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`
- V45 field checkpoint SHA-256:
  `0e5947d85a8baeff99d92996ee8434d3aceab39e64042c9b6ec1a142aa174534`
- V45 field-state SHA-256:
  `08b57734ac3ded0c1438cc4bf963d92357ce1f1d31ae49ee6548c56c19db019d`
- pinned V44 holdout-pair SHA-256:
  `e41637c5e3846e47d19ade6312205cd30c2c96c65730bce75ee8ab4a0745154c`
- tracked report SHA-256:
  `1d7c0867870b89a86b3da91913e0877cf9ef6b420a2eddcd0118f56c10357e7d`

The official report is
`publication/ilm-image-native/evidence/v45/report.json`. The tensor-free
checkpoint receipt is beside it. The reload audit reconstructs the field with
strict state loading and reproduces its field-state digest.

## Fixed Transform

For signed binary raster `x`, V45 computes the orthonormal DCT field `d`, fits
a weighted training-only mean `mu` and covariance eigensystem `(U, lambda)`,
and applies

```text
A = U diag((max(lambda, 0) + 0.50 mean(lambda))^-0.10) U^T
z = A(d - mu)
u = z / ||z||
q = log ||z||.
```

The deployed representation is the continuous pair `(u, q)`. It contains no
token ID, Unicode ID, character ID, OCR system, visual codebook, quantizer,
glyph lookup, finite candidate bank, or external language model. `A`, `mu`,
and the eigensystem are fixed buffers; trainable parameter count is zero. The
inverse

```text
d = mu + A^-1(exp(q) u)
```

is exact up to floating-point error.

## Exact Raster Path

The audit covers the 8,000 fit forms, two held fonts over 1,024 identities,
and four one-pixel translations over the same identities: 14,144 rasters in
seven banks.

| Metric | Measured |
|---|---:|
| maximum FP64 DCT reconstruction error | `4.2633e-14` |
| minimum binary pixel accuracy | `1.00000` |
| minimum ink F1 | `1.00000` |
| maximum blank rate | `0.00000` |
| all fields finite | yes |

## Training-Only Geometry

| Weighted fit-bank metric | Raw normalized DCT | V45 | Change |
|---|---:|---:|---:|
| common resultant length | `0.71900` | `0.01524` | `-97.88%` |
| covariance effective rank | `145.08` | `185.38` | `1.278x` |
| covariance stable rank | `72.95` | `97.12` | `1.331x` |

The result is not full whitening. A matrix power of `0.10` with a ridge equal
to half the mean variance removes the dominant common direction while retaining
font and translation continuity. Full ZCA is retained only as a descriptive
control.

## Pinned Pair Holdout

The 1,024 pairs are reconstructed from V44's seed and exactly match the
preregistered receipt. They share the same final four raster images but differ
in earlier history and target image.

| Held-pair metric | Raw normalized DCT | V45 | Gate |
|---|---:|---:|---:|
| candidate-pair cosine | `0.56319` | `0.06402` | reduction `0.49918` |
| displacement norm, fifth percentile | `0.36976` | `0.77724` | `2.102x` |
| displacement effective rank | `122.80` | `144.95` | `1.180x` |
| displacement stable rank | `60.82` | `67.79` | `1.115x` |

All four fixed displacement gates pass.

## Nuisance Continuity

| Retrieval audit | Raw normalized DCT | V45 |
|---|---:|---:|
| Noto Sans CJK Bold top-1 | `0.96582` | `0.96582` |
| Noto Serif CJK Medium top-1 | `0.82129` | `0.83984` |
| four one-pixel shifts, mean top-1 | `0.75781` | `0.76660` |

No held font loses more than the fixed `0.01` allowance. No shift loses more
than `0.02`, and both group means improve.

## Resource And Boundary Audit

- elapsed wall time: `66.5084` seconds
- peak allocated CUDA memory: `0.30211` GiB on one RTX 4090 D
- trainable parameters: `0`
- frozen evaluation images instantiated: no
- V43 writer opened: no
- checkpoint reload verified: yes

## Report-Only V42 Retrofit

Applying V45 coordinates after a frozen V42 reader reduces matched natural
top-1 from `0.19434` to `0.15039` and mean target log probability from
`-5.23837` to `-7.13590`. This was preregistered as a non-gating diagnostic:
V42 learned anchors in raw normalized-DCT geometry, so changing coordinates
after training is not a valid test of learning in V45 geometry.

The diagnostic rejects a calibration shortcut. It requires the next causal
reader to train from initialization with V45 targets and feedback.

## Bounded Conclusion

V45 resolves the specific V44 target-geometry failure: it supplies an exact,
continuous, image-only field in which common raster modes are suppressed,
discriminative target differences are larger and higher-rank, and ordinary
font/translation continuity is retained. This is a necessary representation
result, not sufficient language evidence.

The next authorized experiment must be preregistered before implementation,
train a causal raster reader from scratch in V45 direction-plus-radius space,
use the same public-domain Chinese stream and matched V42 controls, and beat
V42's natural-language and counterfactual-binding baselines before any writer
or frozen evaluation data is opened.

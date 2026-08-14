# Noise-Limited Retinal Field V45: Research Decision

Date: 2026-08-14

Decision: test a fixed, invertible, noise-limited partial-whitening transform
before training another language reader.

## Measured Problem

V42's target is the unit-normalized orthonormal DCT of a thresholded `32 x 32`
glyph raster. It provides the first bounded positive ordered-raster language
result in this repository. V44 shows why that target is not yet a good binding
space. Its learned residual raises target cosine but lowers exact retrieval and
target probability. The residual-update difference is negatively aligned with
the true target-image difference, while anchor cosine with the corpus-mean field
rises from `0.77712` to `0.88183`.

The algebra explains the failure. Every thresholded signed raster has the same
Euclidean norm under an orthonormal DCT, but a mostly white glyph cell also has a
large shared DC coefficient and correlated low-frequency stroke modes. Unit
normalization preserves those common modes. A cosine objective can therefore
move toward the average page cell and improve raw target cosine without moving
toward the identity-bearing target displacement.

This is a representation defect. More pair repetitions, a larger residual, or
another writer run cannot correct it reliably.

## Relevant Evidence

The intervention is conservative rather than a claim that biological vision is
fully understood.

- Atick and Redlich's efficient-coding account argues that early vision should
  reduce input redundancy, while explicitly limiting whitening when noise would
  be amplified: [What Does the Retina Know about Natural Scenes?](https://doi.org/10.1162/neco.1992.4.2.196).
- Simoncelli and Olshausen connect natural-image statistics, redundancy
  reduction, and neural representation, while noting that second-order
  decorrelation is not the whole visual problem:
  [Natural Image Statistics and Neural Representation](https://doi.org/10.1146/annurev.neuro.24.1.1193).
- ZCA whitening is the symmetric whitening transform that remains maximally
  similar to the original coordinates:
  [Optimal Whitening and Decorrelation](https://arxiv.org/abs/1512.00809).
- VICReg and Barlow Twins show why variance preservation and covariance control
  matter for learned visual representations:
  [VICReg](https://arxiv.org/abs/2105.04906) and
  [Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a.html).
- The degree of whitening affects generalization; full isotropy is not an
  automatic optimum:
  [Exploring the Gap between Collapsed and Whitened Features](https://proceedings.mlr.press/v162/he22c.html).

Together these results support removing measured redundancy while retaining a
noise floor and an exact image path. They do not justify full whitening or a
lossy semantic code.

## Options Rejected

### Larger V44 residual

Rejected. No tested interpolation of the trained V44 residual reaches the fixed
binding gate, and its update points away from target displacement.

### Full PCA or ZCA whitening

Rejected. A training-only probe raises effective rank strongly, but full
whitening reduces one-pixel-shift retrieval from `0.8984` to `0.1191` in one
direction. Low-variance raster modes contain rendering noise and fragile edge
details; amplifying them equally is not a retinally or statistically sound
choice.

### Learned autoencoder or visual codebook

Rejected for this test. It would mix representation learning, compression, and
language learning, making the V44 diagnosis harder to falsify. A codebook would
also reintroduce a discrete inventory at the exact boundary under study.

### Mean centering only

Retained as a control, not selected. It removes the common direction and is
robust, but does not improve effective rank. V45 asks whether a small amount of
variance balancing improves target displacement without sacrificing retinal
continuity.

## Selected Transform

For signed raster DCT field `d`, estimate a frequency-weighted mean `mu` and
covariance `Sigma` from 8,000 training-only canonical glyph rasters. Let

```text
Sigma = U diag(lambda) U^T
mean_variance = trace(Sigma) / 1024
A = U diag((max(lambda, 0) + 0.5 * mean_variance)^-0.10) U^T
z = A (d - mu)
u = z / ||z||
q = log ||z||
```

The language direction is `u`; the radial scalar `q` prevents unit
normalization from destroying information. The exact inverse is

```text
d = mu + A^-1 (exp(q) * u).
```

The transform has no learned parameters, finite visual vocabulary, OCR path, or
symbolic runtime. It is a continuous image statistic. The dense matrices can be
folded into fixed input/output projections later; the immediate audit keeps them
explicit for traceability.

The exponent `0.10` and ridge `0.50 * mean_variance` are frozen from a
training-only exploratory Pareto sweep. On those exploratory rasters, this point
changes common resultant length from `0.7190` to `0.0152`, effective rank from
`145.1` to `185.4`, and average four-direction one-pixel-shift top-1 from
`0.7576` to `0.7666`. It preserves one held development font exactly and
improves the other from `0.8213` to `0.8398` top-1.

On the untouched 1,024-pair V44 training-partition holdout, exploratory target
geometry changes as follows:

| Raster-only metric | Raw V42 field | Selected field |
|---|---:|---:|
| candidate-pair cosine | `0.56319` | `0.06402` |
| 5th-percentile target-delta norm | `0.36975` | `0.77724` |
| target-delta effective rank | `122.80` | `144.95` |
| target-delta stable rank | `60.83` | `67.82` |

These values were reconstructed with V44's pinned seed and verified against
holdout receipt
`e41637c5e3846e47d19ade6312205cd30c2c96c65730bce75ee8ab4a0745154c`.
They remain design observations, not preregistered capability evidence. The V45
protocol repeats the audit in repository code.

## Scientific Boundary

A passing V45 would qualify an invertible image representation for another
causal experiment. It would not establish language learning, next-glyph
prediction, readable generation, historical-form understanding, or superiority
to token models. The frozen language partition and V43 writer remain closed.

The next causal model may start only if every V45 representation gate passes.
That model must be trained in the new field from the beginning; retrofitting a
V42 anchor into new coordinates is a report-only diagnostic because V42 was
optimized under the old cosine geometry.

# Canonical Glyph Language V42: Frozen Development Protocol

Date frozen: 2026-08-14

## Question

Can one compact model learn held-out Chinese continuation from an ordered stream
of glyph rasters and emit the next glyph as a raster, without token IDs, Unicode
IDs, OCR, a character classifier, a visual codebook, or a candidate bank in the
deployed path?

V42 tests only this question. It deliberately uses one fixed font and one glyph
per visual time step. Historical forms, handwriting, layout, instruction tuning,
and the V41 style motor are excluded until this language gate passes.

## Prior Evidence And Design Change

V25--V31 showed weak order sensitivity but failed conditional target binding.
V34 established a useful learned glyph renderer, while V35 showed that its
reconstruction latent is not automatically a predictive language coordinate.
V39 failed answer planning. V41 established only that a target-supplied glyph
motor can improve a degraded source image; it did not predict glyph content.

The V42 change is therefore structural:

1. remove form variation from the first language proof;
2. replace a drifting learned target geometry with an exact, invertible visual
   field transform;
3. train next-field compatibility against images dynamically present in the
   batch, with positives defined only by exact raster equality; and
4. model the continuous next-field distribution with a sample-based energy
   score rather than treating the conditional mean as the generated glyph.

This combines the pixel-language premise of PIXEL/PIXAR, the continuous
autoregressive observation of MAR, and the strictly proper sample-based energy
objective used by CALM. It does not import their tokenizers, vocabularies, OCR
stages, or text-conditioned runtime components.

## Data

The fixed corpus is:

```text
data/visual_grammar/chinese_wikisource_public_domain.jsonl
SHA-256 76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03
```

It contains 7,017 records from 16 public-domain Chinese Wikisource works. V42
inherits the V25 record partition and never trains on development or frozen
records. Both original and offline simplified views may be rendered, but the
student receives only their resulting images.

Every visible character is rendered in:

```text
/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
font size 26, no augmentation, 32 x 32 grayscale ink raster
```

The training sample is a length-65 visual stream. The model receives cells
1--64 and predicts cells 2--65 at every causal position:

```text
context: B x 64 x 1 x 32 x 32 floating images
target:  B x 64 x 1 x 32 x 32 floating images
```

Strings and source metadata may exist in the host-side dataset object for
auditing, but the student collator must remove them.

## Continuous Visual Coordinate

Let an ink raster be `x in [0,1]^(32 x 32)`. V42 first thresholds the canonical
training raster at 0.5 and maps it to a signed image:

```text
s(x) = 2 * 1[x >= 0.5] - 1.
```

Let `D` be the orthonormal 32-point DCT-II matrix. The visual field is

```text
z(x) = vec(D s(x) D^T),       k(x) = z(x) / ||z(x)||_2.
```

For a binary 32 x 32 field, `||z(x)||_2 = 32` by Parseval's identity. The
inverse map is exact before thresholding:

```text
s_hat = D^T unvec(z_hat) D,   x_hat = 1[s_hat >= 0].
```

This is a fixed continuous image transform, not a learned dictionary. It has no
entry per character and applies unchanged to arbitrary input rasters.

## Causal Model

The fixed production configuration is:

```text
field width          1024 continuous DCT coefficients
model width           384
causal layers           8
attention heads          6
MLP ratio                3
maximum context          64 glyph images
noise width             128
energy samples            4
```

Each input field is linearly projected and processed by rotary causal
self-attention. At position `t`, the model emits a normalized visual anchor
`q_t` and a hidden state `h_t`.

For flattened predictions `q_i` and target image fields `k_j`, positives are
defined by exact equality of the corresponding binary target rasters. The
dynamic visual contrastive loss is

```text
L_nce = -mean_i log [sum_(j in P_i) exp(a q_i^T k_j)
                     / sum_j exp(a q_i^T k_j)],
```

where `a` is learned and bounded. The candidate matrix exists only inside a
training minibatch; it is not a vocabulary and is not saved as a deployed
lookup table.

The bank-free generator receives `h_t`, `q_t`, and continuous noise `epsilon`.
It emits samples `g_r(h_t, q_t, epsilon_r)` on the same unit visual sphere. For
`beta = 1`, V42 minimizes the empirical energy score

```text
L_energy = 2 E_r ||g_r - k||_2 - E_(r != s) ||g_r - g_s||_2.
```

The first term rewards target fidelity. The second prevents the sample family
from being trained as a conditional mean when multiple futures are plausible.
Auxiliary anchor cosine and decoded binary-field losses stabilize the bounded
single-GPU run.

## Inference Boundary

Deployed inference accepts only floating glyph images. It:

1. computes their fixed continuous DCT fields;
2. runs causal visual attention;
3. samples one or more next fields from noise;
4. selects a sample using only the predicted anchor, not an external bank;
5. applies inverse DCT and a zero-level threshold; and
6. rereads the generated raster for the next autoregressive step.

The deployed path must not receive strings, bytes, code points, token IDs,
character IDs, OCR output, a glyph lookup, a persistent candidate bank, or an
external language model.

## Fixed Development Audit

The evaluator may use labels and candidate images, but these never enter the
student call. The audit uses:

- 2,048 deterministic development windows;
- the 1,024 most frequent training Han raster identities;
- full 64-cell history;
- last-cell-only history;
- suffix-4 history;
- a suffix-4-preserving shuffle of cells 1--60;
- an all-blank history;
- image-unigram and symbolic-bigram controls built only from training text;
- 512 same-suffix, different-target counterfactual pairs when available; and
- bank-free generated-field identity and pixel quality on 256 windows.

Candidate ranking is evaluator-only. Autonomous output is produced before and
without candidate images; the bank is used afterward only to measure which
raster identity was drawn.

## Development Gates

The language mechanism passes only if all of these hold:

1. full-history top-1 exceeds image unigram by more than 0.03;
2. full-history top-1 exceeds symbolic bigram by more than 0.01;
3. full-history target log probability exceeds shuffled history by more than
   0.05 nat;
4. full-history top-1 exceeds shuffled history by more than 0.015;
5. same-suffix counterfactual assignment exceeds 0.60 arm accuracy;
6. bank-free generated identity top-1 exceeds image unigram;
7. generated binary pixel F1 exceeds 0.55;
8. generated blank rate is below 0.02;
9. the image-only boundary audit passes; and
10. peak allocated CUDA memory is below 18 GiB.

The frozen record split remains sealed in V42. A development pass authorizes a
separate frozen-confirmation protocol; it does not authorize an LLM-parity,
general-understanding, cross-font, historical-form, or efficiency claim.

## Stop Rules

- A smoke run validates shape, causality, determinism, checkpoint, and evaluator
  plumbing only.
- Hyperparameters may be calibrated before a production declaration, but every
  calibration run must be marked exploratory and cannot be reported as frozen
  evidence.
- Once a production run begins, thresholds, audit windows, and controls cannot
  change in response to its result.
- Failure means diagnose the failed stage. Do not add calligraphy, page layout,
  more parameters, or instruction data to hide a language failure.

## Claim Boundary

A passing V42 would support one narrow claim: a compact model learned useful
canonical-font Chinese next-glyph continuation from image history and generated
the next glyph through a continuous image field without a deployed symbolic
vocabulary. It would not yet establish semantics, question answering,
etymology, arbitrary historical forms, human-like reading, or superiority to a
token language model.

# Continuous Glyph Representation Codec V34: Result

Date completed: 2026-08-14

Decision: `continuous-codec-qualified`

## Claim Tested

V34 tested whether a compact, codebook-free convolutional codec can encode
arbitrary `32 x 32` writing rasters into normalized 768-dimensional continuous
vectors and directly reconstruct pixels for unseen Chinese fonts and held-out
historical-character families on one RTX 4090.

This was a visual-interface experiment. It did not test next-patch prediction,
prompt following, factual knowledge, language understanding, or autonomous
generation.

## Frozen Run

- protocol:
  `references/continuous_glyph_codec_v34_protocol.md`;
- protocol SHA-256:
  `c2370374f202714e217236f7634f464eb98bed6a0f8afe898b9658614df7ce51`;
- source commit: `d8a7b43`;
- updates: `6,000 / 6,000`;
- checkpoint:
  `artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt`;
- checkpoint SHA-256:
  `a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33`;
- raw model SHA-256:
  `2d3450053c1c274d3f6494ee5c0b95b202b9d4aff00fa0c34458ffefa5916efa`;
- checkpoint finite-state audit: passed; and
- sealed split opened only after the EMA development gate passed.

The model has `7,423,361` trainable parameters. Its public codec methods accept
pixels or continuous latent vectors only. The boundary receipt found no token,
vocabulary, Unicode, character-ID, OCR, retrieval, quantization, codebook, or
runtime-teacher path.

## Data Receipt

Rendered public-domain stream:

- manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- 7,017 records;
- 6,743 train, 135 development, and 139 sealed records before rendering
  variants; and
- train, development, and sealed used distinct Noto CJK font files.

Historical stream:

- SQLite SHA-256:
  `c563e8587d7dcacf73704c0fb7816f6d830db11122e0a3da62678b3a7119f738`;
- ordered SVG manifest SHA-256:
  `3c4064441563c88dffe0c36d42cce0c381bf8b401b764b87484edfb4aa7db99c`;
- 84,626 indexed SVG records;
- 75,111 train glyphs from 4,954 modern-character families;
- 4,789 development glyphs from 299 held-out families; and
- 4,726 sealed glyphs from 300 held-out families.

All forms associated with a modern character remained in one split. Training
used a fixed-seed permutation of the train records. Evaluation retained
canonical order. The local historical corpus and raster cache were not copied
into Git.

## Optimization Evidence

- optimizer: AdamW, betas `(0.9, 0.95)`, weight decay `0.01`;
- learning rate: 250-update warmup to `2e-4`, cosine decay to `2e-5`;
- batch per update: all active patches from eight rendered strips plus 128
  historical glyphs;
- latent augmentation: independently selected half of patches, Gaussian sigma
  sampled uniformly from `[0, 0.05]`;
- objective: boundary-weighted BCE plus `0.5` replicate-padded Sobel L1 plus
  `0.5` ink Dice;
- training patches: `4,519,796`;
- training time: `321.306` seconds;
- throughput: `14,066.96` patches/second; and
- peak allocated CUDA memory: `2,552,274,944` bytes (`2.38 GiB`).

The final logged batch had total loss `0.0007855`. Training loss is not used as
held-out evidence.

## Development Results

The preregistered selection route was EMA. It evaluated 4,398 rendered patches
from 56 held-out serif-font strips and 4,096 historical glyphs from held-out
modern-character families.

| Development metric | Raw | EMA | Gate | EMA pass |
| --- | ---: | ---: | ---: | :---: |
| Clean rendered ink F1 | 0.997863 | 0.997864 | >= 0.985 | yes |
| Clean rendered edge F1 | 0.997226 | 0.997270 | >= 0.980 | yes |
| Clean rendered exact patch | 0.726467 | 0.735789 | report | - |
| Clean rendered OCR retention | 0.999740 | 1.002567 | >= 0.950 | yes |
| Noisy rendered ink F1 | 0.997819 | 0.997836 | >= 0.970 | yes |
| Noisy rendered edge F1 | 0.997174 | 0.997244 | >= 0.950 | yes |
| Noisy rendered exact patch | 0.722829 | 0.732606 | report | - |
| Historical ink F1 | 0.998506 | 0.998311 | >= 0.960 | yes |
| Historical edge F1 | 0.998132 | 0.997934 | >= 0.940 | yes |
| Historical exact patch | 0.877441 | 0.869629 | report | - |
| Blank false-ink rate | 0.000000 | 0.000000 | < 0.005 | yes |
| Mean latent per-dimension std | 0.963901 | 0.963368 | >= 0.10 | yes |

EMA target OCR accuracy was `0.723911`; reconstructed OCR accuracy was
`0.725770`, producing retention slightly above one. The target-normalized
metric therefore indicates that reconstruction did not lose the OCR signal
available in the rendered targets; it does not imply perfect OCR. Paired target
versus reconstruction OCR agreement was `0.973179`.

Every mandatory development condition passed, including finite state, complete
updates, and the resource ceiling.

## Sealed Results

The sealed EMA audit evaluated 4,363 patches from 56 strips in the sealed font
and 4,096 glyphs from sealed historical-character families.

| Sealed metric | Value | Sealed/dev ratio | Ratio pass |
| --- | ---: | ---: | :---: |
| Clean rendered ink F1 | 0.997793 | 0.999929 | yes |
| Clean rendered edge F1 | 0.997291 | 1.000021 | yes |
| Clean rendered OCR retention | 1.006088 | 1.003512 | yes |
| Noisy rendered ink F1 | 0.997749 | 0.999913 | yes |
| Noisy rendered edge F1 | 0.997226 | 0.999981 | yes |
| Historical ink F1 | 0.998068 | 0.999756 | yes |
| Historical edge F1 | 0.997747 | 0.999813 | yes |

All required sealed metrics retained more than 97% of their development value.
The sealed exact-patch rates were `0.771946` for clean rendered patches,
`0.767591` under latent noise, and `0.879639` for historical glyphs. Blank
false ink remained zero.

## Visual Evidence

Deterministic galleries are stored under:

```text
artifacts/continuous_glyph_codec_v34_20260814/galleries/raw/
artifacts/continuous_glyph_codec_v34_20260814/galleries/ema/
```

The development and sealed rendered galleries place target, clean
reconstruction, and `sigma=0.03` reconstruction on adjacent rows. Historical
galleries place each target beside its reconstruction. Visual inspection agrees
with the numerical result: differences are sparse stroke pixels rather than
character substitutions or blank collapse.

## Decision and Limits

V34 qualifies the continuous visual codec as a candidate interface for a
separately preregistered causal model. It establishes that this 7.4M-parameter
component can preserve unseen modern and historical writing at low compute
without a discrete visual vocabulary.

It does **not** establish that a causal model can predict the next visual
representation, understand a prompt, generate coherent language, answer an
etymology question, or operate independently as an ILM. Those claims require
autonomous generation plus correct-prompt, shuffled-prompt, blank-prompt,
paraphrase, and counterfactual controls. No causal result should be inferred
from codec reconstruction.

## Reproduction

```bash
PYTHONPATH=. python scripts/train_continuous_glyph_codec_v34.py \
  --device cuda:0 \
  --out artifacts/continuous_glyph_codec_v34_20260814
```

Use `--resume` only with the exact source files and checkpoint receipt recorded
by the interrupted run.

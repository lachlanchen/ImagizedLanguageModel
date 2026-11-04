# Joint exact raster language: matched result

Date: 2026-11-05. Both arms finished before development opened. Overall
decision: **reject as a complete visual language mechanism**. Retain the joint
checkpoint as the stronger measured conditional density, with exact copying.

## Reproducible experiment

Run: `artifacts/joint_exact_raster_v55_20261105/`.
Both arms start from V54 EMA. The model and exact normalized pixel law are
unchanged. The controlled intervention enables likelihood gradients into the
reader; its dropout remains disabled in both arms. The unused contrastive
temperature stays fixed.

| Arm | Updated parameters | Extra updates | Seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Frozen reader | 2,434,592 | 8,000 | 218.011 | 0.40016 |
| Joint reader/density | 18,711,840 | 8,000 | 342.378 | 0.64775 |

Runtime size is 18,711,841 parameters for either arm. Costs exclude V42/V54
foundation training and data preparation. Both arms run sequentially on
physical GPU 0. The frozen reader remains bit-identical to initialization;
the joint reader changes. Final training generator states match, establishing
matched draws. Model, EMA and CPU/CUDA/training RNGs match between interrupted
and uninterrupted smoke runs. Gradient-isolation and causal-history tests pass.

Checkpoint hashes:

- Frozen: `0f993f484c7998629a1adf1dc94f60044c3f23f9ef95fd1459a4636c7f2b6c5a`.
- Joint: `ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394`.

## Held-out result

| Measurement | Frozen continuation | Joint learning |
| --- | ---: | ---: |
| Ordered target NLL, 512 windows | 7.57913 | 7.02581 |
| Earlier-shuffled NLL | 7.69784 | 7.21105 |
| Ordered NLL advantage over shuffle | 0.11871 | 0.18524 |
| Ordered first-sample target identity | 6/256 | 9/256 |
| Shuffled first-sample target identity | 6/256 | 8/256 |
| Ordered first-sample pixel F1 | 0.36432 | 0.36201 |
| Ordered first-sample exact target image | 5/256 | 9/256 |
| Greedy exact target image | 17/256 | 19/256 |
| Four-step identity, ordered / shuffled | 2.344% / 2.734% | 3.125% / 1.563% |
| Supplied-image greedy exact match | 256/256 | 256/256 |
| Supplied-image sample pixel F1 | 0.99993 | 0.99994 |

The paired joint NLL advantage over the matched frozen control is **0.55332
nat**, 95% source-record bootstrap CI **[0.44987, 0.66542]**. Earlier-history
advantages are 0.11871 [0.05165, 0.18587] frozen and 0.18524 [0.12798, 0.24422]
joint. Thus additional density training alone resolves an earlier-history
likelihood effect, and joint learning provides a further measured improvement.

The paired sampled-identity gain is 1.1719 percentage points, CI
[-0.3968, 2.8932]. It remains unresolved. The joint candidate passes 6/8 gates,
failing sampled quality and the ordered-generation advantage. The control
passes 5/8, additionally failing visible-feedback ordering. Both pass surface,
motor, exact-likelihood context effect and runtime checks. Do not revise these
gates after seeing the result.

All intervals use 2,000 grouped draws. NLL covers 169 source records; generation
covers 134. The post-generation bank covers every target and contains 9,932
distinct rasters. All primary samples precede bank construction. No sealed
records or independent training seed are evaluated. Paired controls do not
remove the limitation from repeated development use.

## Independent inference

`scripts/infer_joint_exact_raster_v55.py` exports a self-contained
75,226,167-byte checkpoint, SHA-256
`29e0b6cbcb3e56088ac4f8e7b92d0886d965048684c273f2986a394ce73dadb8`.
Its native-image greedy continuation reproduces the production checkpoint
pixel for pixel; its motor output matches the binarized input exactly.
Artifacts and receipts are under the run's `standalone/` directory. No
tokenizer, OCR, glyph lookup or external inference model chooses output pixels.
The export remains explicitly unqualified as a general language model.

## Interpretation

The same small visual model can improve a proper language-conditioned raster
likelihood through end-to-end learning on one GPU, while retaining a reliable
copy path. This is stronger than a frozen reader feeding a separately trained
writer. It is still weaker than prompt-dependent, coherent autonomous language.
Target identity compares one sample with one observed future; it does not by
itself measure all valid continuations. Raw readability, calibration, meaningful
longer continuations and counterfactual prompt behavior remain necessary.

The next evidence should separate canonical-form validity from choosing the
observed next form, and assess a pinned public multilingual pixel backbone as
an explicitly external baseline. Preserve the joint density and copying path;
do not replace this bounded improvement with a broad fluency or efficiency claim.

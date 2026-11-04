# Noisy Visual Energy Guidance V52: Measured Result

Date completed: 2026-11-04

Decision: reject the product-of-experts guidance mechanism

## Answer

V52 separates reading from writing in the strongest form tested so far: a
frozen image-only causal reader supplies visual history, a frozen unconditional
diffusion model supplies the glyph manifold, and a learned noise-aware critic
adds the history-dependent score. This separation preserves the reader and
produces coherent glyph-like images, but it does not generate the glyph implied
by the history.

The critic ranks completed target glyphs at `21.875%` top-1 in a 1,024-image
development bank. Generation from Gaussian noise is qualitatively different.
The first guided sample reaches `0/256` target identities, and critic selection
among four samples reaches only `1/256`. Shuffling the visual history gives the
same selected identity result. The four-step feedback result is also identical
for ordered and shuffled histories.

V52 is therefore a useful negative result: compatible reading and writing
components do not become a conditional visual language generator merely by
adding their scores.

## Frozen Construction

The registered mechanism is

\[
s_{\mathrm{cond}}(X_t,H,t)
=s_{\mathrm{write}}(X_t,t)
+\nabla_{X_t}R_\psi(H,X_t,t),
\]

where `H` is an ordered stream of visible glyph rasters, `s_write` is the
frozen V51 marginal glyph score, and `R` is the learned density-ratio critic.
The reverse process starts from Gaussian noise and uses 25 deterministic DDIM
levels. Four paired-noise samples are generated for each context; the critic's
clean-image energy selects one without a glyph bank.

The deployed path has no tokenizer, string, Unicode value, character ID, OCR
transcript, vocabulary embedding, codebook, quantizer, glyph lookup, candidate
bank, or external runtime model. The 1,024-image bank exists only in the
evaluator after output pixels have been produced.

| Component | Parameters | Training state in V52 |
| --- | ---: | --- |
| V42 causal raster reader | 16,277,249 | frozen |
| V51 unconditional glyph prior | 3,464,321 | frozen |
| V52 noisy visual energy critic | 3,525,441 | calibrated, then frozen |
| Combined runtime | 23,267,011 | below the 27M cap |

## Calibration Receipt

The critic completed all `4,000` registered updates in `537.104` seconds on
one RTX 4090. Peak allocated training memory was `0.377 GiB`. The final
checkpoint is

`artifacts/noisy_visual_energy_guidance_v52_20261104/checkpoint_final.pt`

with SHA-256

`635815395302568cb8440473dbd166cbee1828677165927d74b93f96912cc84a`.

The final logged high-noise batch remained finite, with `14.0625%` in-batch
top-1, `33.5938%` V42 teacher top-1, and a `+0.48016` ordered-minus-shuffled
target log-mass gap. These are optimization diagnostics, not generation
evidence.

## Locked Development Audit

The evaluator was committed before the registered audit. It used 512 fixed
development windows, a 1,024-image evaluator-only bank, 256 paired-noise
one-step generations, and 64 four-step feedback prefixes. The complete audit
took `16.157` seconds and peaked at `0.609 GiB` allocated CUDA memory.

### Completed-image compatibility

| Measurement | Ordered history | Shuffled history | Ordered gain |
| --- | ---: | ---: | ---: |
| critic identity top-1 | 0.218750 | 0.208984 | +0.009766 |
| target log probability | -5.046511 | -5.202268 | +0.155758 |
| V42 direct identity top-1 | 0.216797 | 0.201172 | +0.015625 |

The critic transfers the reader's candidate-ranking ability to its learned
image embedding. Its ordered top-1 advantage misses the strict `>0.01` gate by
`0.000234`, but this near miss is not the central failure.

### Autonomous image generation

| Measurement | Ordered | Shuffled | Zero guidance |
| --- | ---: | ---: | ---: |
| first-sample identity top-1 | 0.000000 | 0.000000 | 0.000000 |
| selected identity top-1 | 0.003906 | 0.003906 | 0.000000 |
| first-sample pixel F1 | 0.424361 | 0.423248 | 0.417379 |
| selected pixel F1 | 0.427190 | 0.426826 | 0.420587 |
| selected blank rate | 0.000000 | 0.000000 | 0.000000 |
| selected isolated-ink fraction | 0.010833 | 0.010346 | 0.010792 |
| selected/target ink ratio | 1.514245 | 1.512284 | 1.518063 |
| distinct selected evaluator identities | 204 | 212 | 215 |

The outputs are nonblank, diverse, and visibly glyph-like. Those properties do
not imply conditional language. Ordered history neither improves first-sample
identity over zero guidance nor improves selected identity over shuffled
history.

### Visible feedback

| Feedback measurement | Ordered | Shuffled |
| --- | ---: | ---: |
| step-1 identity top-1 | 0.015625 | 0.015625 |
| steps 2--4 identity top-1 | 0, 0, 0 | 0, 0, 0 |
| four-step mean identity top-1 | 0.003906 | 0.003906 |
| four-step mean pixel F1 | 0.371944 | 0.371970 |

The generated raster can be reread, but rereading unrelated glyph modes does
not create language dynamics.

## Gate Decision

| Frozen gate | Result | Decision |
| --- | ---: | --- |
| image-only runtime boundary | clean | pass |
| critic / combined parameter caps | 3.53M / 23.27M | pass |
| clean critic top-1 at least 0.15 | 0.218750 | pass |
| clean ordered advantage greater than 0.01 | 0.009766 | fail |
| ordered first-sample identity at least 0.05 | 0.000000 | fail |
| first-sample gain over zero guidance greater than 0.02 | 0.000000 | fail |
| selected identity at least 0.08 and ordered gain greater than 0.015 | 0.003906 / 0.000000 | fail |
| ordered four-step feedback advantage | 0.000000 | fail |
| blank, isolated-ink, and ink-ratio surface bounds | 0 / 0.010833 / 1.514245 | fail |
| at least 32 selected identities | 204 | pass |
| finite values, memory, and frozen hashes | clean | pass |
| evaluator bank absent from the model | clean | pass |

The fixed rule requires every gate. V52 is rejected.

The full report is

`artifacts/noisy_visual_energy_guidance_v52_20261104/development/development_report.json`

with SHA-256

`4d8e664a6582a74584658d7662932accddd559d5313f1162510e3e77257993a7`.

The deterministic five-column gallery has SHA-256

`2f279d77ea1eb0ae97ecc706a5adb6e70bf771d873c6a243653946d2591c1ddd`.

Its columns are target, ordered first sample, ordered selected sample, shuffled
selected sample, and zero-guidance selected sample. It shows a normal changing
target column beside four similar pseudo-glyph columns.

## Why The Mechanism Fails

The failure is not an absent writing manifold. V51 reconstructs real glyphs at
moderate corruption, and V52 outputs connected glyph-like modes from noise. It
is also not an inability to compare completed glyphs: the clean V52 critic
matches V42's `~22%` identity ranking.

The missing operation is **conditional mode formation**.

At diffusion level 99, the cumulative signal power is
`alpha_bar = 2.43e-7`; target signal has standard deviation only
`4.93e-4`. The correct history-dependent score therefore approaches zero at
the start distribution. The unconditional reverse trajectory begins choosing
a glyph mode before context is visually recoverable. When the conditional
energy becomes stronger at lower noise, its local pixel gradient can adjust
strokes but does not reliably cross from the already selected mode to the
history-implied mode. Four-sample critic selection cannot recover a target mode
that was almost never proposed.

This is the same distinction exposed by the audit:

```text
rank an already formed glyph:     works at a bounded level
form the right glyph from noise:   fails
```

Longer critic calibration, a guidance-strength sweep, or more samples would
not test a new mechanism and are not authorized as a response to this result.

## Replacement Direction

The next model should remove the unconditional-prior/critic decomposition. It
should learn the conditional visual future directly and should make a
multi-glyph block, rather than one isolated glyph, its smallest generated
language object.

The technically supported route is a compact causal **visual block diffusion**
model:

1. an ordered raster reader encodes the visible prompt or book line;
2. a conditional denoiser predicts a joint block of four to eight future glyph
   rasters, not independent per-pixel marginals;
3. every future cell receives its own noise level, following the causal
   sequence construction of [Diffusion Forcing](https://arxiv.org/abs/2407.01392);
4. continuous next-block distributions use diffusion loss rather than a
   categorical vocabulary, following [MAR](https://arxiv.org/abs/2406.11838);
5. clean target visual representations supervise intermediate denoiser states
   during training only, following the efficiency principle of
   [REPA](https://arxiv.org/abs/2410.06940); and
6. direct output rasters, ordered-versus-shuffled controls, and visible
   feedback remain the deciding evidence.

This changes the random variable and the learning problem. The condition
participates inside every denoising layer, while joint future cells can express
word- and phrase-level constraints before individual glyph modes settle. No
character vocabulary or deployed candidate bank is required.

The first experiment must remain bounded. It should test canonical Chinese
continuation and copy on one RTX 4090, compare one-glyph and four-glyph arms
under matched compute, and require correct visual-history advantage in direct
first samples. Instruction following, historical forms, book-page prediction,
and teacher-distilled knowledge remain later stages until this conditional
block mechanism passes.

## Claim Boundary

V52 does not establish a usable ILM, semantic understanding, instruction
following, etymology generation, historical-script control, human-like
reading, Qwen parity, or efficiency superiority over token language models. It
establishes a sharper negative fact: a compact image-only reader, a compact
glyph diffusion prior, and a learned noisy visual energy can each work in
isolation while their score sum still fails as a conditional language
generator.

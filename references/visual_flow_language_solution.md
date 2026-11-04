# Visual Flow Language: Minimal End-to-End Solution

Date: 2026-11-04

Status: selected implementation direction before end-to-end training

## Problem

The model must read an ordered stream of writing images and generate the next
writing image. Its runtime must not depend on token IDs, Unicode values, OCR,
a character vocabulary, a glyph lookup table, or an external language model.

The existing evidence isolates two useful components:

- V42 learned ordered Chinese continuation from 32-by-32 raster cells. Full
  visual history reached `0.19971` next-image top-1, compared with `0.18359`
  after shuffling earlier history and `0.12256` for the symbolic bigram
  control.
- V43 learned a compact spatial flow writer. Its pixel F1 rose to `0.88240`
  when supplied the correct target-derived visual plan.

Their frozen composition failed. The autonomous reader supplied an averaged
plan, and the separately trained writer could not turn that plan into a
coherent identity. Adding a more elaborate writer while keeping the reader
frozen would repeat the same separation error.

## Selected Model

Visual Flow Language trains the reader and writer together:

```text
ordered raster cells -> causal visual transformer -> continuous context state
                                                    |
Gaussian raster noise -> small spatial flow network -+-> next raster cell
```

For visible history $X_{\leq t}$, the causal reader produces

\[
h_t=R_\phi(X_{\leq t}).
\]

For the next raster $Y_t$ and Gaussian raster noise $E$, flow training samples
$u\sim U(0,1)$ and forms

\[
Z_u=(1-u)Y_t+uE,\qquad V^*=E-Y_t.
\]

The spatial writer predicts the velocity directly from the noisy raster, time,
and visual-language state:

\[
\widehat V=W_\theta(Z_u,u,h_t).
\]

Both $\phi$ and $\theta$ receive gradients from

\[
\mathcal L_{flow}=\mathbb E\|\widehat V-V^*\|^2_{ink}.
\]

This is the essential change. Noise supplies the choice variable for multiple
plausible next glyphs, while end-to-end training makes the causal state useful
for controlling that choice. The deterministic V42 ink plan is removed from
the writer path rather than used as an averaged target image.

A small visual contrastive term retains the measured V42 ordering signal:

\[
\mathcal L=\mathcal L_{flow}
+0.10\mathcal L_{visual\text{-}NCE}
+0.05\mathcal L_{anchor}.
\]

The old stochastic field generator is frozen and unused. The deployed model is
approximately 30 million parameters and fits comfortably on one RTX 4090.

## Runtime

Inference starts from one 32-by-32 Gaussian noise image, integrates the learned
flow toward a visible glyph, thresholds the terminal raster, and feeds those
pixels back through the same reader. Several samples may be drawn, but the
model's own visual anchor—not an evaluator bank—selects the emitted raster.

The runtime accepts and emits images only. Offline text remains permissible for
rendering training pages and evaluating known benchmark answers; it never
enters a model method.

## First Decision

The first run answers only whether joint training repairs the frozen
reader/writer mismatch. It compares against the measured V43 checkpoint on the
same development windows:

1. direct generated identity top-1;
2. generated pixel F1 and blank rate;
3. ordered-history versus shuffled-history generation;
4. condition-present versus condition-removed generation; and
5. finite four-cell visible feedback.

The route is retained only if generated identity and raster quality improve
together and generation measurably depends on ordered context. Training loss,
sharp-looking samples, or evaluator-selected best-of-many output is not enough.

## Scope

This first model uses canonical Chinese glyph cells because it isolates
language learning from typography. Historical forms, whole pages, prompt
answering, multilingual writing, and local-LLM-assisted data preparation are
later data expansions. They do not require a different runtime architecture:
each remains a visible raster stream read by the same causal model and written
by the same image flow.

The experiment does not yet establish general language understanding, useful
instruction following, etymology knowledge, historical-script generation, or
parity with token language models. It tests the smallest end-to-end mechanism
that could support those capabilities.

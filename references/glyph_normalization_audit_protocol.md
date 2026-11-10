# Retrospective normalization and first-policy audit

The current image readout and stop network begin with LayerNorm; the source
contextual body ends with RMSNorm. LayerNorm removes a per-vector mean, whereas
a linear source policy need not be invariant to that translation. This is a
testable architectural concern, not an established cause of the answer failures.

Use the same128 saved initial states and original lexical matrix as the
completed first-policy audit. Reproduce every prior CPU winner before testing
mean centering. Report source first-character matches before/after centering,
and how often the complete source unit changes. No image answer grade changes.

Also construct a controlled alias on the source normalization shell. For
nonzero final RMSNorm scale gamma, define weights w_i=gamma_i^-2,
m=sum(w_i*h_i)/sum(w_i), and h'=h-2m*1. Then

\[
 \sum_i(h'_i/\gamma_i)^2=\sum_i(h_i/\gamma_i)^2,
 \quad LN(h')=LN(h).
\]

The first identity follows by expanding the square and substituting m; the
second follows from translation invariance, including the same variance epsilon.
Check both numerically and compare source winners/first characters. Such pairs
are contrived states on the same RMS shell, not evidence that both are reached
by real image prompts or that every output error comes from normalization.
Retain zero-scale exceptions rather than divide by zero. Record actual weighted
mean ranges in the training first-state pool and the evaluated cohorts.

An eventual mean/std channel into the first affine layers could strictly extend
the existing networks with zero-initialized weights. Neither that architecture
nor a trained remedy is implemented or validated by this audit. Use hidden
CUDA, the common model/build mutex,48 GiB host entry and24 GiB live reserve.
The source head is offline-only; no model forward, optimizer update, new answer
image or training example is produced.

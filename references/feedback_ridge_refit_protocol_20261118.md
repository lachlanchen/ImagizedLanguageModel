# Direct output-layer fitting on the fixed reader features

The complete nonlinear calibration finishes both twenty-epoch arms and fails
its reader gate: weighted error is .578608694 affine and .542496895 nonlinear,
against the fixed .10 limit. All 33,941 rows appear in every epoch. The original
encoder/codec remain unchanged. Do not extend the fit, select an intermediate
head or claim language from this small alignment improvement.

This separate intervention changes the output-layer fitting procedure. Keep
the two prescribed final feature maps, corpus, source embeddings, alias targets,
loss masses and deployed arithmetic fixed. Fit the final linear coefficients
by one FP64 weighted ridge solve, on both the affine and nonlinear families.
No feature, source, threshold, epoch count or language loss is learned anew.
This is ordinary convex regression, not a novel physical mechanism. Ridge
regularization and numerical precision differ from the preceding AdamW fit;
the result cannot uniquely attribute that fit's shortfall to optimization.

## Probability-free native-coordinate objective

Let e_i be a BF16 source embedding, m_i its original corpus mass and mu/sigma
the existing fixed source mean/scale. Construct phi_i from a constant one,
the actual BF16 LayerNorm output, and (for the nonlinear head) its fixed BF16
GELU expansion. The last coefficients K are expressed in native coordinates:

\[
 L(K)=\sum_i c_i\|\phi_iK-(e_i-\mu)\|^2+\lambda\|K\|_F^2,
 \quad c_i=m_i/\max(\|e_i\|^2,10^{-8}),
\]
\[
 G=\sum_i c_i\phi_i^T\phi_i,\quad C=\sum_i c_i\phi_i^T(e_i-\mu),
 \quad\lambda=10^{-4}\operatorname{tr}(G)/d,\quad K=(G+\lambda I)^{-1}C.
\]

All coefficients, including the constant, receive the same fixed ridge.
Positive ridge handles the LayerNorm-induced column dependence; it is not tuned
on these outputs. Stream sufficient statistics in 1,024-row blocks and solve
by FP64 Cholesky. Require relative normal-equation residual <1e-7. Native
coefficient columns are divided by sigma to restore the original normalized
head weights/bias, keeping mu/sigma and all feature layers fixed.

The solve is optimal for this regularized, fixed-feature, exact-arithmetic
problem. Real inference retains FP32 stored weights and BF16 linear/GELU
operations; its actual error is separately measured after copying coefficients.
An ideal matrix loss is not a deployed error or a language result. Record both
and preserve rounding deterioration if present.

## Execution and fixed decision

Reuse the authenticated prepared rows and frozen-query cache from the terminal
calibration. Do not re-encode images, repeat its epochs or call a teacher body.
First reproduce the two final actual weighted errors within 1e-6. On each
family and each of the three whole training records, fit a temporary ridge
readout on every native position with equal per-position mass. Require finite
coefficients, the normal-equation check, actual BF16 native error below that
record's original head error, unchanged feature tensors and exact trial reset.
This is a solver execution/support preflight, not a language fit. Any failure
stops the single pipeline without retry or altered regularization.

Only after all six preflights pass, solve each family once on all 33,941 rows
with the original complete-corpus masses. Preserve original heads and save
separate final readouts. Evaluate actual weighted error, ideal matrix error,
coefficient norms, normal-equation residuals and all complete-record errors.
Keep the preceding reader-stage gate: nonlinear actual error <=.10, gain over
the refitted affine control >=.10, and actual error <=.10 on every complete
record. Do not select between the two families or tune lambda after seeing
results. A reader pass permits preparation of a separately registered live-
image conditional-writing evaluation only. It does not admit a language fit,
autonomous-language claim, historical-form claim or efficiency comparison.

All targets/IDs are training-only and no runtime embedding table, glyph bank,
nearest lookup, font renderer or lexical output head is introduced. Source
pretraining and the prior calibration remain part of the cost/provenance.
The previously frozen language outputs and readings never change.

## Resources and source binding

CPU tests compare the streamed solve with independently augmented least squares,
including dependent columns, verify native affine inversion and preserve inputs.
Bind those tests, implementation, fixed heads, cache, complete source preparation,
original registration and all accessed sources before execution; commit and push.
One owner uses the existing GPU0 mutex. Require 48 GiB host, 20 GiB free GPU0,
48 GiB disk and low PSI at entry; retain 24 GiB host, cap allocation at 8 GiB,
use two CPU threads, 1,800 monitored seconds and 300 seconds pressure waiting.
Use the existing 1,860-second external deadline plus thirty-second termination
grace. Preserve every failure and partial result. No automatic second worker,
retry, unchanged epoch extension or foreign-service change is permitted.

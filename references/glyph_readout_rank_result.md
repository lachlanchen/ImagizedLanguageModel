# The semantic output span has a measurable rank limit

The preserved G ends with a2048-to4096linear layer. Every predicted semantic
code therefore lies in its bias plus a subspace of dimension at most2048.
This limits exact regression to the full4,087-form anchor set, independently
of input-side nonlinear capacity.

For centered anchor covariance eigenvalues lambda_i, the minimum possible
uniform per-coordinate MSE for any rank-r affine output is

    sum of eigenvalues below the largest r /4096.

A CPU-only double-precision eigenspectrum gives **0.0601091 at rank2048**,
corresponding to a maximum **93.9876%** of anchor variance captured. Rank3072
reduces the lower bound to **0.00500010** (99.4999%variance). The best possible
projection into the *actual frozen trained output span and its bias* has
MSE **0.305180** across these forms. The current span is not the optimal
rank2048span for the broader inventory.

These are exact-regression geometry results, not pixel or language accuracy.
A nonlinear D may decode many useful images from a lower-dimensional subspace;
this calculation does not prove that2048hidden units cannot render all glyphs.
The best-affine bound also does not identify whether limited supervision,
bias, optimization or other choices produced the current trained span.

The analysis uses the same prepared semantic targets and authenticated
language checkpoint, four CPU threads,5.60s, no neural forward calls, optimizer
updates or image generation; CUDA remains uninitialized. It runs independently
of the sole GPU image-state cache. All original model files stay unchanged.
The [full covariance and span receipt](evidence/glyph_readout_rank_v1/report.json)
contains eigenvalues, resources and source/data/checkpoint hashes.

The next test already has a prospective
[genuine-context curriculum](genuine_glyph_readout_protocol.md). Its broader
real-state supervision can change G's learned span without changing its width.
That control should finish before claiming that widening is necessary. If its
regression/physical gates fail, a separately specified function-preserving wider
readout can test capacity using the same cached images and retained language
histories; no extra reader cache or source vocabulary is required.

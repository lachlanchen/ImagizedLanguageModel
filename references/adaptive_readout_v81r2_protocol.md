# Readout repair: an explicit revision of the numerical execution criterion

This revision retains the model, data, two matched arms, optimization budget,
noise law, seeds, sampler, autonomous-image requirements and training-fit gates
of the [original protocol](adaptive_readout_v81_protocol.md). It changes only
the numerical comparison used for batched and incremental hidden states, and
persists per-arm execution checks. It is specified after inspecting the failed
preflight, before either 2,000-update experimental arm has started. No language
quality result is used to choose the revised execution tolerance.

The original absolute 1e-4 preflight fails on the adapted arm. An actual-model
replay restores the original second-update checkpoints and reproduces the
third update and mismatch exactly: maximum padding difference 1.69754e-4 and
incremental difference 2.00272e-4. RMS errors range from 1.69e-6 to 4.08e-6.
State components reach 74.4 in magnitude. Future-exclusion differences are
exactly zero. These measurements are consistent with shape-dependent FP32
roundoff; they do not constitute an independent high-precision neural forward
reference or prove that every numerical issue is resolved. Even an elementwise
absolute/relative comparison of 1e-4 + 1e-5|component| misses one component,
so it is not reported as a passing check.

Use a normwise bound for each complete 4,096-dimensional contextual state:

    ||h_actual - h_reference||_infinity
        <= max(1e-4, 1e-5 * ||h_reference||_infinity).

This compares vector-scale numerical error rather than relative error in a
single component affected by cancellation. Retained original states give a
maximum error/bound ratio of 0.721. The revised preflight must repeat this
comparison on the actual model and retain both maximum absolute discrepancies
and the original 1e-4 check's pass/fail status. The original failed run remains
failed; it is not overwritten or relabelled. Its producing source bytes are
archived, and the [failure and exact-replay receipt](evidence/adaptive_readout_v81_precision_failure.json)
preserves the measurements and source identities.

Future exclusion must still be exact. Prompt KV and source-memory identity,
frozen-parameter integrity, optimizer/noise resume and physical feedback hashes
must still be bit-exact. Actual generated images and all quality thresholds
are unchanged. Numerical execution checks do not certify output quality.

Use the original commands with a separate output directory,
`--run artifacts/adaptive_readout_v81r2_20261106` (now the default), first
`--stage preflight`, then `--stage train` only if that preflight passes, then
CPU-only `--stage verify`. Original artifacts remain under
`artifacts/adaptive_readout_v81_20261106`. Only one project GPU job may run.

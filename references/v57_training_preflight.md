# V57 training preflight

2026-11-05, before production. The frozen V57 task and model law are unchanged.

Nine CPU tests pass across the raw-pixel decoder and copy model. The tests
include normalization of the copy/free/stop law, duplicate-image mass, exact
equality on random unencoded binary forms, future-image and source-position
masking, finite reader/head gradients, native copying and learned stopping.
Two further evaluator tests check sequence order, length, stopping and grouped
uncertainty. A raster F1 of one alone can credit a reordered answer; exact
sequence match therefore remains a separate primary measure.

The natural-stream smoke exposed an all-masked logsumexp gradient when a
matching image existed only at a future, masked source position. Copy mass now
uses finite visible attention entries, and the no-visible-match branch has a
finite backward path. A regression test covers that case. Production training
has not run on the faulty implementation.

Ordinary CUDA attention backward was not bit-reproducible across separate
smoke processes. Production therefore uses deterministic PyTorch algorithms,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the math SDPA backend. BF16 training and
high matmul precision remain as registered. A four-update GPU smoke, including
the natural/motor task, matches an interrupted two-plus-two run exactly for
every model and EMA tensor. Data, training, CPU and CUDA RNG states also match.
Peak allocated memory is 0.659 GiB in the copy smoke. Smoke outputs are not QA
capability evidence. The added source-closure checks affect provenance only.

The production arms remain fixed at 4,000 updates, batch four, sequentially on
physical GPU 0. Development performance stays closed until both arms finish.

# Offline check of identity in genuine copy states

The genuine-copy curriculum is still caching. Its sources, learning schedule,
qualification gates and automatic supervisor remain unchanged. This auxiliary
check is fixed before its readout fit and before the held-wording states exist.
It requires the completed, hash-authenticated cache and makes no model call.

For the 256 already fixed held identities, compare template 1 states with the
template 0 gallery, then compare held template 2 states with both training
templates. Report both raw cosine and cosine after subtracting the corresponding
training-gallery mean. Normalization is per vector. Never estimate the center
from held inputs. Merge the two training views by their maximum similarity per
form; rank all 4,086 visible forms. Report top-1, top-5, optimistic and pessimistic
mean identity ranks, and every case's target and winning cosine. No geometry is
selected after observing results. Template 1 is itself a training wording;
template 2 tests wording transfer on known identities.

Labels serve this offline evaluation only. No gallery, rank, winner, or corrected
form is passed into G or the image-only runtime. The audit produces no glyphs,
performs no optimizer update, and does not evaluate language accuracy. A high
score would show that this particular rule can recover identity from the reader
states. A low score would not exclude nonlinear information in those states.
Compare these observations with the subsequent physical readout qualification;
do not substitute retrieval for generated images or relax its gates.

Run after caching, without competing with a neural stage:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_genuine_copy_states.py \
  --run artifacts/genuine_glyph_readout_v1 \
  --output artifacts/genuine_copy_state_audit_v1
```

The calculation uses at most four CPU threads, query batches of 32 and a
120-second wall budget. It validates all cache shard hashes and record ordering.
No GPU, model loading, output-head inference, teacher or new dataset is required.
The present document specifies a diagnostic, not a completed result.

Before computing similarities, group all recorded FP32 state byte strings by
SHA-256, checking tensor equality on any matching hash. Record groups containing
different target forms. Authenticate the canonical training tensor and verify
that the target IDs in this cohort have distinct bitmap bytes. Loading these
CPU tensors is not a neural forward. A deterministic readout gives one output
for one such state. With distinct canonical targets, an exact-copy
upper bound on this fixed pool is

\[
\frac{1}{N}\sum_{\text{identical-state group }g}\max_c n_{g,c}.
\]

This optimistic bound says nothing about foreground F1 or generalization. No
conflicts would not prove semantic information, practical distinguishability,
learnability or language understanding. This is an additional CPU recording
check, fixed before the held-wording states or new readout results exist; it
changes no training, model, similarity geometry or qualification threshold.

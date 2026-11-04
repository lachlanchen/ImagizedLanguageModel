# V63 real preflight: completed implementation evidence

5 November 2026. All registered implementation checks pass. No V63
development score or language qualification exists at this checkpoint.

Training-only pair preparation admits 3,793 pairs across 939 contexts and
3,016 distinct questions. All retained reference strings and physical answer
sequences are disjoint within pairs. Two candidate pairs share a reference
string and are excluded. Manifest SHA-256:
`861f7c487223d65b5649cb4c33fb1e3de24a7be4663eea5676de4a07b00f2dbb`.
The pair-cache report SHA-256 is
`55d7e641fe4a51547d7819359cf5a233dbe894d73a7b7ee5feae21a51a17aec6`.

Both actors contain 489,998,208 parameters. One full optimizer/EMA update
on the longest admitted joined stream (250 native image blocks), retaining
all four graphs, peaks at 9.197 GiB allocated memory. Four-update checks
peak at 9.235 GiB for each arm. These measurements are feasibility checks,
not capability-normalized efficiency or training throughput comparisons.

For control and aligned arms independently, four accumulated updates match
two plus resumed two exactly in model, EMA, optimizer, Python/NumPy/Torch
CPU/CUDA RNG and pair-sampling RNG. Full pair journals also match exactly.
All parameter gradients are finite. An assignment-only backward gives
nonzero gradients through the physical input projection, first and last
contextual layers, image projection and mixture gate. Correct/swapped physical
stream assembly is checked against source prefix/suffix tensors.

Changing future answer pixels has zero effect on earlier predictions.
Strict-FP32 cached/prefill maximum logit difference is 0.00006485, below the
registered 0.0002 tolerance. The preflight uses training images only and
makes no external-model call. Its owned processes exit and release GPU 0
before production starts.

The immutable detailed receipt is
`artifacts/question_assignment_v63_20261105/preflight/report.json`. The
[protocol](question_assignment_v63_protocol.md) fixes the matched 2,000-update
production schedule and autonomous-generation gates. A successful preflight
does not establish useful image-language generation.

# Physical-head FP64 arithmetic repairs the captured numerical discrepancy

The full physical-head control passes all eight registered checks on the exact
captured states. Cached/full teacher-logit error falls from 1.24931e-4 to
1.74971e-13, under the unchanged 1e-4 limit. Maximum errors to the independent
FP64 CPU equations are 2.13385e-13 and 2.02505e-13. Both autonomous local block
trajectories match each other and the original FP32 pixel outputs exactly.
All original GPU traces replay bit exactly, target causality is exact, and
every parameter has a finite gradient after complete-image NLL backward.

All 2,666,944 parameter values, names, shapes and FP32 storage remain unchanged.
The small head casts parameters and activations for FP64 arithmetic; gradients
return to FP32 stored parameters. No vocabulary, glyph table, logit rounding
or output repair is introduced. The inherited reader is neither loaded nor
modified by this component control. This qualifies a numerical implementation
on these states, not language learning or general image quality.

In this execution, twenty timed two-block forward calls after five warmups
average 2.903 ms for the original head and 4.701 ms for the candidate. Peak
allocation is 0.179 GiB; the complete control takes 9.002 seconds. These are
small-head costs and exclude source pretraining, the native reader and full
answer generation. There is no efficiency claim against a conventional LLM.

The [normalization-only control](evidence/accurate_pixel_density_result/execution/report.json)
remains rejected: it passes the original cache gate but its full-path maximum
reference error increases from 7.28034e-5 to 7.45783e-5. Its measured forward
times belong to that separate execution and are not pooled with the full-head
comparison. Neither failed control is relabelled as passed.

The [protocol](fp64_pixel_density_protocol.md),
[source-bound result](evidence/fp64_pixel_density_result/execution/report.json)
and preparation retain every criterion and measurement. Original baseline and
candidate tensors remain in `artifacts/fp64_pixel_density_v1/execution/`.
No optimizer step or complete autonomous language answer is executed. The
next required check is actual native image-to-image training, optimizer replay,
generated-image feedback and learned stopping under this exact implementation.

# Auxiliary interference was not established on the prescribed records

The no-update diagnostic completes all three whole training records on both exact
final actors. The registered interference criterion is not met. Every measured
combined AdamW direction has a negative inner product with the conditional-flow
gradient, as does its flow-only counterfactual. These are first-order directions
on new batches, not measured post-update losses or semantic improvements.

The [public evidence](evidence/visual_context_gradient_result_20261117.json)
retains all group geometry, six complete losses, gradient-presence counts, global
clipping, exact actor/projection preservation checks and source hashes. No
optimizer update or answer generation occurs. The prior 792-stream failed
language result and all frozen readings remain unchanged.

| Final actor | Complete training record | Flow/auxiliary gradient cosine | g_F·d_combined | g_F·d_flow-only |
|---|---|---:|---:|---:|
| Learned context | 513-glyph book | .00943 | −.05140 | −.06618 |
| Learned context | 846-glyph QA | .00339 | −.50221 | −.65855 |
| Learned context | 1,042-glyph QA | .00315 | −.06512 | −.18238 |
| Frozen context | 513-glyph book | −.05839 | −.05564 | −.08108 |
| Frozen context | 846-glyph QA | .05880 | −.23621 | −.35880 |
| Frozen context | 1,042-glyph QA | −.00458 | −.10171 | −.23423 |

The prospective rule requires the treatment book and at least one QA record to
have a nonnegative combined inner product while the flow-only inner product is
below −1e-12. None qualifies. Some magnitudes are smaller under the combined
objective, particularly on the longest QA. This includes effects of global
clipping, auxiliary signals and the retained moments; it is not sufficient
evidence to select new loss weights. Negative cosine in two control cases does
not make the actual combined direction an ascent direction.

The flow-only calculation starts with the same retained optimizer history as the
combined calculation. It is not a comparison with a model trained exclusively
on flow. The three selected records do not establish a corpus-wide causal
explanation. No actual parameter update is applied, so no actual post-update
loss decrease is claimed.

## Execution and preservation

The first registered host-gradient execution fails at its 300-second pressure-
wait ceiling before completing a record measurement. Its monitored duration is
413.354 seconds; the outer supervisor records 424.910 seconds. It contributes
no interference finding, and its final tensor-preservation checks were not
reached. Peak VRAM was not saved. All failure evidence remains preserved.

The [separate recovery](visual_context_device_gradient_protocol_20261117.md)
changes only the saved detached FP32 flow gradient from host to device storage.
An exact source comparison verifies that the measurement function otherwise
remains unchanged, and all 56 original bindings are preserved. Records, actors,
moments, clipping, mathematical helper, RNG replay, resource guards and decision
criterion remain fixed. The successful monitored scope is **183.602 seconds**,
including **10.010 seconds** of pressure waiting, with **16.687 GiB** peak allocated
VRAM and **55.298 GiB** minimum available host RAM. Its supervisor scope is
195.196 seconds. Nested scopes must not be added as independent costs.

All six scalar losses repeat exactly across each record's three RNG-matched
passes. Exact model and training-only projection tensor hashes match before and
after both actors. The worker and supervisor exit successfully, releasing GPU0;
no foreign service is changed. The lower pressure in this execution does not
isolate the placement change as the sole cause because workstation state also
changed between attempts.

## Next decision

Do not select a loss-weight repair from this diagnostic. The strong opposing-
direction explanation is unestablished on the prescribed records. Conditional
writing remains malformed despite nonzero gradients and locally descending
first-order flow directions. The subsequent
[saved-output analysis](visual_context_saved_dependence_result_20261117.md)
finds distinct complete responses alongside recurrent damaged first tiles. It
separates matched sampling shapes from unequal-span comparisons, without new
neural calls, OCR or reading repair. Together these findings redirect the next
intervention toward useful conditional image selection. No new full fit is
admitted, and usable visual language remains unproved.

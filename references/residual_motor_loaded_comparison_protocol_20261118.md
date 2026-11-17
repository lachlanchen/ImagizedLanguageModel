# Serialized-plan boundary correction before the physical motor fit

The [physical-head execution](evidence/residual_motor_loading_failure_20261118.json)
passes resource admission but stops while reading the cached plans, before
constructing a model or making an update. All stored plans are uint8 bytes.
The shared generic unpacker compares with scalar 256; that scalar is outside
the tensor's dtype range. Its earlier unit checks used integer tensors.
Preserve this failed execution and its zero-update status.

The bounded correction widens serialized plan bytes to int64 before invoking
the unchanged shared validator. Exact pack/unpack recovery verifies that no byte
or bit value changes. The [CPU qualification](evidence/residual_motor_loader_result_20261118.json)
covers every one of 68,948 actual 256-bit plans under its prospective registration,
with no neural call, optimizer update, source-mask interpretation or language
measurement. A separate unit check covers all 256 byte values at every plan
position. Do not change the old shared module or repair source images.

The new committed worker differs from the original registered training body
only at this explicit input boundary and the already selected physical-motor
resource context. Copy the previous head registration's model, source pixels,
frozen codec, initialization, objective, sampling, update schedule, generation,
probability checks, fixed quality and resource limits. Preserve the original
40-GiB physical entry / 24-GiB runtime reserve and all GPU/RSS/time ceilings.
Use a new registration and output directory; no failed directory is reused.

Run once after committing and pushing the bound sources. Qualify actual complete-
source gradients, update/reset and frozen-codec identity before retaining a fit,
then execute all 6,000 updates and every prescribed actual image and probability
check. No automatic retry or chosen checkpoint. Independently recount all actual
outputs before language admission. Every original physical and complete-language
criterion remains unchanged; the earlier attempts measured no model quality.

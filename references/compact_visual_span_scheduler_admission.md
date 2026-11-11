# Compact sequence scheduler admission correction

The teacher and fit workers already use `full_reader_gpu`, which acquires the
project GPU mutex, requires 48 GiB available host RAM and low memory pressure,
audits owned runtimes, and requires 20 GiB free device memory before loading.
It records high global swap occupancy. Its runtime monitor retains the 24 GiB
host reserve and 22 GiB peak allocated GPU limit. Other project services are
left unchanged.

The first serial controller repeated the RAM and pressure checks but also
required global swap occupancy at or below 75%. This extra condition did not
match the teacher and fit admission it claimed to mirror. During extraction,
one observed host sample had 72.985 GiB available RAM, 81.864% swap occupancy,
and 0.01% memory-stall averages. The controller would withhold the next fit
even though its existing worker admission accepted that sample. An owned
process inventory found the active controller, teacher and agent runtime;
there was no obsolete project model or GUI process to clean up.

The corrected controller uses the worker's RAM and pressure checks for teacher
and fit stages. The child still repeats its exclusive admission and owned
process audit. Standalone inference retains its original 24 GiB and 75% swap
gate. This change does not alter a model, optimizer, target, batching budget,
checkpoint schedule, inference setting or evaluation cohort.

The corrected default run directory is
`artifacts/compact_visual_span_pipeline_v2`. Preserve the earlier controller's
request and terminal log. The successor must wait for that controller to exit.
A separate ownership-checked handoff waits for the active teacher to terminate
and verifies that no live child remains before stopping the old controller.
The active teacher is not restarted for this controller change. The successor
authenticates retained cache shards and resumes only missing work.

CPU checks cover the exact RAM and pressure boundaries, rejection below the
reserve or above either pressure limit, high-swap teacher/fit admission and
continued high-swap inference rejection. The actual resource sample passes
the corrected fit check and fails the original controller check. These are
scheduler checks; compact GPU fitting and language quality remain unmeasured.

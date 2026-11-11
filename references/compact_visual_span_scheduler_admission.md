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
process audit. This change does not alter a model, optimizer, target, batching
budget, checkpoint schedule, generation setting or evaluation cohort.

The corrected default run directory is
`artifacts/compact_visual_span_pipeline_v3`. Preserve the earlier controllers'
request and terminal log. The successor must wait for that controller to exit.
A separate ownership-checked handoff waits for the current worker to terminate
and verifies that no live child remains before stopping the old controller.
The first handoff preserved uninterrupted teacher extraction and the second
preserves the active fit. The successor authenticates retained cache shards
and checkpoints and resumes only missing work.

CPU checks cover the exact RAM and pressure boundaries and rejection below
the reserve or above either pressure limit. The actual resource sample passes
the corrected fit check and fails the original controller check. The fit then
started and saved real optimizer updates; its
[early checkpoint evidence](evidence/compact_visual_span_gpu_start_20261111.json)
reports execution separately from unmeasured language quality.

## Standalone generation admission

The second controller initially retained standalone generation's original
global-swap gate. Current generation uses the existing `diagnostic_gpu` path
after auditing project-owned runtimes. At swap occupancy above 75%, it
requires at least 64 GiB available host RAM and the same low pressure limits.
At ordinary swap occupancy, it retains the original 24 GiB admission. Both
routes require the exclusive project GPU mutex and 20 GiB free device memory.
The report records the actual admission. No swap setting, kernel policy or
foreign service is changed. Model arithmetic, flow steps, seed, budgets and
the standalone input boundary remain unchanged.

Checks cover both inference admission branches, their RAM and pressure
boundaries, and actual occupied-mutex rejection while the registered fit owns
GPU0. The rejected call did not initialize CUDA or allocate a second model.
Trained image generation and cold replay remain pending; those executions
must verify the final runtime and generated pixels.

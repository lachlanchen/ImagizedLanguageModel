# Resume the interrupted metric arm without changing the experiment

The original V82r2 process terminates externally before the metric arm's first
250-update checkpoint. Its execution session reports exit 143; the process
becomes a zombie and then disappears, and GPU0 memory returns to its idle
level. The exact external source of the termination is not established. The
last complete metric journal row is update 180; no metric checkpoint or
metric final answer exists. Do not infer the exact number of additional
uncheckpointed updates from the last journal row.

The control arm is complete at 2,000 updates. A CPU-only observation verifies
all 32 original references, raw autonomous PNGs, bound output files, final
checkpoint hash and independently recomputed scores. Control autonomous F1
is 85.983512%, with 0/32 pixel-exact answers; correct-history F1 is 92.430062%,
and supplied-code F1 is 99.995316%. All answers stop. These seen-case results
do not pass absolute autonomous fit. The matched metric comparison remains
unfinished; no relative treatment effect can yet be reported.

Before resuming, preserve the original training log, last status/runtime
handoff and unfinished metric journal under an interruption directory with
hashes. Retain all completed control files and all 32 metric initial-output
headers and payloads. Archive the unfinished metric journal before the new
invocation creates its production journal, so discarded partial optimization
is not mixed into the retained final trajectory.

Use the unchanged original trainer, all pinned source bytes and matching
preflight. Its existing resume logic skips the completed control report.
Because the metric arm has no checkpoint, it restores the original declared
V81 warm weights and fresh declared Adam, case sampler and noise generator.
It must receive the same 2,000 retained optimizer updates and 8,000 draws.
Retained initial generations must keep their original bindings. Discarded
interrupted computation is additional cost, not an extra part of the final
checkpoint's learning trajectory. Compare repeated logged updates with the
archived rows to check actual deterministic re-execution where available.

Launch the one replacement GPU0 process in a dedicated user-systemd service,
with CPU verification sequenced after successful training. Confirm the former
process is terminal and the project GPU lock is free before launch. Do not
start another model job or use another project's GPU1. Keep the service unit,
actual process identity and logs in the private runtime handoff. The service
exists to survive the interactive command lifetime; it changes no model,
optimizer, loss, data, seeds, sampler, output rule or original quality gate.

Preserve and report any replay or restart failure. The original full CPU
verifier and result exporter still require both completed arms. No successful
language, lightweight-model or efficiency claim follows from this recovery.

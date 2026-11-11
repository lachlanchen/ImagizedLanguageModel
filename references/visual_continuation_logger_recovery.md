# Continuation fit: event-logging failure and recovery

The first complete-corpus execution stopped after its first optimizer update,
before recording or saving that update. The event constructor supplied `kind`
both as the event category and through the returned training record, where it
identified a book or QA source. Python rejected the duplicate keyword. This is
an execution failure, with no complete-language result.

The original registration, initial checkpoint, failed worker receipt and serial
controller journal are preserved. The controller measured 139.558657 seconds
for this child, including startup and shutdown. The unsaved update is discarded;
it must not be counted as completed scheduled exposure. Its changed weights were
not archived. The saved checkpoint remains at step zero.

The correction separates `event_kind` from the source `kind`, including during
the real recovery replay. A regression check serializes both event categories.
A new execution uses fresh initial weights and the identical source pixels,
targets, schedule, loss, initialization, optimizer and fixed evaluation. It has
its own committed registration and output directory; the original execution is
not overwritten. Earlier child time counts against the cumulative 48-hour fit
budget. Data preparation is reused, with source changes explicitly recorded.

The [recovery registration](evidence/visual_continuation_full_logger_recovery_registration_20261112.json)
binds the correction and prior failure evidence. No threshold, target, cohort,
snapshot or stopping decision changes. Actual save/replay equality and complete
language performance still require their prescribed measurements.

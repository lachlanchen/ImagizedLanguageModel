# Preserve completed prefixes across runtime interruptions

The original full-cache process stopped unexpectedly after 810 complete cases
(including the four preflight cases). Its final status still described active
caching; both the cache owner and coordinator had exited. There was no logged
Python exception. No main optimizer update had run. The cause of the process
termination is not established by the receipt.

Recovery authenticates the prepared input hashes and every complete saved
tensor's case ID, labels, shapes, finite FP32 values and positional coordinates.
It retains all original files, status snapshots and logs. Partial writes remain
present but are not accepted as complete records. The CPU audit adopts 810
cases and leaves 2,366 cases to compute. A resumed calculation must first replay
an original case within the same 0.002 tolerance. It then writes only missing
cases and publishes a complete ordered manifest for the unchanged training code.

The first recovery attempt fails its inherited read-only diagnostic admission
before any neural forward: global swap occupancy had crossed 75%, activating
that helper's 64-GiB host reserve. Its failed zero-work receipt is preserved.
The recovery-specific admission now requires 48 GiB available host RAM, the
existing GPU mutex, low-pressure entry check and 20 GiB free device memory.
The live 24-GiB host floor and 22-GiB GPU allocation cap remain in force. The
measured full-reader preflight already runs with minimum available host RAM
49.10 GiB and peak GPU allocation 17.724 GiB. No foreign process is stopped,
and the parent resource-admission functions are unchanged. This is an explicit
execution-resource amendment, not a changed learning rate, dataset or metric gate.

The resumed cache and subsequent serial fit/qualification run inside one
project-owned detached tmux session, `ilm-glyph-tail-v2`. Status and evidence
live in files rather than depending on a tool terminal's lifetime. The session
exits when its command sequence ends. The coordinator preserves any nonzero
child exit for inspection and stops after qualification for actual image review.

Cross-attempt timings remain explicit. The interrupted attempt's exact exit
time and total pressure waiting were not retained, so a complete wall-time or
wait-time total must not be invented from the resumed-stage receipt alone.
Recovery changes no optimizer schedule or neural data calculation. It is not
evidence that the resulting model has learned usable language.

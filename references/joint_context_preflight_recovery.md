# Preserve the interrupted joint-context preflight

The first real-source preflight exits with code 143, SIGTERM, while its status
still says `admitting` with no case position. No QA gradient comparison, book
optimizer update or complete model result is recorded. The original 50 model/
test/protocol sources, registration, status and console are preserved unchanged
in the private recovery archive. The signal sender is unknown; no OOM event is
recorded in the process cgroup. This is not a failed language measurement.

Kernel logs also report a 595.99.02 userspace client against the loaded 595.84
NVIDIA kernel module. This does not establish the SIGTERM cause. The machine
already has a private, matching userspace installation and a per-process
`with-current-nvidia` wrapper. Reuse those verified files without installing
packages, changing system library links, restarting services, resetting GPUs or
interrupting the other GPU's workloads.

The [recovery record](evidence/joint_context_preflight_sigterm_recovery_20261113.json)
binds that wrapper, both matching libraries, the existing signal tracer and the
small recovery launcher. Launch exactly one explicit recovery in a durable
tmux session. `strace` records termination signals and their sender metadata;
it does not supply image data or change model parameters. A second signal must
be investigated from that trace, not treated as permission for a retry loop.

The original 50 source hashes, four complete cases, gradient tolerances,
initialization and resource ceilings remain unchanged. The registration gains
only a pointer to the source-bound execution-recovery record. Preserve the
original failed verdict and all earlier image readings.

The first observation after terminal state, boot time and process start ticks
give a conservative 155-second wall-time debit. The recovery shifts only the
instrumented start time, leaving at most 1,645 seconds of the original 1,800-second
allowance. No pressure wait was recorded: the bound monitor writes its waiting
status before every wait and does not restore `admitting` afterward. The original
allocated-memory peak is unavailable and must not be invented. The recovered
attempt's peak remains a separate measurement. Report the original upper-bound
cost, recovered attempt and cumulative upper bound distinctly.

The recovered report's `total_process_instrumented_seconds` includes the prior
debit; the recovery completion receipt separately records this attempt's wall
time. Passing the recovered computational checks would permit preparing a new
complete fit, not claiming language understanding. Preserve any subsequent
failure and all costs before another decision.

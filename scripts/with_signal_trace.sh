#!/usr/bin/env bash
# Record termination signals without tracing arguments, data or system calls.
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
    printf '%s\n' 'Usage: with_signal_trace.sh NEW_LOG -- COMMAND [ARG ...]' >&2
    exit 64
fi
trace_log=$1
shift 2
if ! command -v strace >/dev/null 2>&1; then
    printf '%s\n' 'strace is required.' >&2
    exit 127
fi
mkdir -p -- "$(dirname -- "$trace_log")"
if ! (umask 077; set -o noclobber; : > "$trace_log"); then
    printf '%s\n' 'Refusing an existing or unwritable signal log.' >&2
    exit 73
fi

# Request seccomp filtering to avoid unrelated syscall stops. Signals are
# still observed and delivered. If the tracer dies, its owned tracees die too.
exec strace -f -ttt -qq --interruptible=2 --kill-on-exit --seccomp-bpf \
    -e trace=none -e signal=SIGTERM,SIGINT -o "$trace_log" -- "$@"

# Trace an interrupted runtime

An exit code of 143 alone does not identify who stopped a process. Use the
installed `strace` to record signals delivered to a command and its children:

```bash
bash scripts/with_signal_trace.sh artifacts/runtime/assessment-signals.log -- \
  python scripts/compare_contextual_pixel_learning.py \
  --run artifacts/contextual_pixel_adaptation_v1 --resume
```

Use the experiment's verified Python environment, GPU0 settings and compatible
NVIDIA client libraries. Where the workstation's `with-current-nvidia` helper
is installed, put it immediately before the Python command. It selects the
shared userspace matching the loaded kernel driver without changing the
system installation. A completion supervisor must inherit that environment
as well, so its standalone inference children use compatible libraries.

The wrapper requires a new log path, records timestamps, traced PIDs and
SIGTERM/SIGINT metadata, and leaves the command's output on its usual streams.
The signal's `si_pid` and `si_uid` identify its sender when supplied by the
kernel. Syscall arguments, file contents and environment variables are not
traced. Keep runtime logs under ignored `artifacts/`.

Target signal handlers still run. If the tracer itself terminates, the kernel
kills its owned tracees so they cannot remain as unattended model processes;
this does not invoke their graceful cleanup handlers. A killed tracer cannot
guarantee a final log record. Do not infer a sender from an incomplete log.
The wrapper does not restart jobs or suppress termination.

After an interruption, verify actual process state, checkpoint hashes and
retained outputs before resuming. The comparison's existing `--resume` path
skips training arms whose completed reports and checkpoint hashes agree, and
continues an incomplete indexed assessment. Preserve interrupted invocation
logs and account for their time separately from the resumed portion. Signal
tracing does not change an incomplete comparison into a quality result.

The process-level checks use real signals and no model or GPU:

```bash
CUDA_VISIBLE_DEVICES='' python tests/test_runtime_signal_trace.py
```

They check sender metadata, the target's default termination, a custom cleanup
handler and exit status, and cleanup when the tracer stops. The diagnostic
depends on Linux ptrace/seccomp support and was verified with `strace` 6.8;
run these checks before using it on a long job.

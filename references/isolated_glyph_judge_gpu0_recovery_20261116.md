# GPU placement recovery for the fixed isolated-cell judge

The first registered transcription action stopped after one completed request
in 12.559 seconds. GPU0 stayed at 16 MiB while GPU1 increased from 8,719 MiB to
18,540 MiB, triggering the registered foreign-device guard. The response was
saved before the abort. There is no complete transcription freeze or calibrated
judge result. This is an **execution failure**, not a measured reader failure or
student language result. Preserve the first registration, sources, response,
owner receipt and failure report unchanged.

Read-only inspection then established that both existing local Ollama servers
are pinned to physical GPU1. Checking free GPU0 memory did not verify the
shared server's placement. The failed client's model expired naturally; the
shared daemon remains running. No foreign process was stopped.

## One changed execution condition

The recovery uses the same teacher manifest, runtime version, prompt, input
transform, ordered cell cohort, request options, parsing, scoring and rejection
rules in the [original protocol](isolated_glyph_judge_protocol_20261116.md).
Only serving placement changes: start a temporary loopback Ollama server on
port 11438, using the existing verified executable and weight store, with
`CUDA_VISIBLE_DEVICES` set to the physical GPU0 UUID. Do not copy weights or
alter either shared service. Bind the wrapper, original executor and helper,
old failure and response hashes, new runtime identity and exact prepared cell
inventory before inference. The new registration is distinct from the failed
one and must be committed and pushed before chat calls.

The temporary server is owned by one supervisor under the project GPU mutex.
Verify its actual environment, binary digest, metadata readiness and empty
resident-model list. Use one parallel request, one resident model and a 2,048
default context limit, retaining the original per-request options. Source
metadata requests during registration load no model. Stop the exact owned
server process group on completion or error, verify its port is closed, and
record device occupancy afterward. Leave foreign services and GPU1 workloads
untouched. The original host, disk, wall, request, client-RSS and GPU ceilings
remain in force; the server's resident host memory is not included in client
RSS, so report the host reserve and distinguish this scope.

All 526 nonblank physical masks are submitted in the original hash order,
including the one previously requested cell. Its first reading has not been
inspected, scored or used to change the prompt. This one duplicated request is
an explicit, prospectively registered placement recovery, not response selection
or a resumed automatic retry. Report both attempts and their cost. Freeze the
complete new cache before opening answer strings or old labels for scoring.
Any further execution failure stops this recovery; do not launch another
worker. Any failed calibration condition rejects the judge without fitting.

Before any recovery inference, metadata-only checks exposed two admission
issues. A recently closed TCP listener can retain TIME_WAIT sockets; the empty
port probe now allows address reuse while still refusing an active listener.
Ollama also returns identical model-default parameter fields in different line
orders. The recovery fingerprint sorts these fields while preserving their
values. The observed fields are presence penalty 1.5, temperature 1, top-k 20
and top-p .95; the fixed request continues to override temperature to zero.
Neither change inspects or selects a glyph response. Preserve the uncommitted
registration draft privately and finalize these bindings before the first
recovery chat call. A CPU regression distinguishes harmless parameter ordering
from a changed parameter value.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/calibrate_isolated_glyph_judge_gpu0.py register
# Commit and push the distinct recovery registration, then:
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/calibrate_isolated_glyph_judge_gpu0.py transcribe
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/calibrate_isolated_glyph_judge_gpu0.py score
```

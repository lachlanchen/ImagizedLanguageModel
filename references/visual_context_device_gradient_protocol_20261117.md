# Device placement recovery for the no-update gradient diagnostic

The first [registered diagnostic](visual_context_gradient_protocol_20261117.md)
ends at its unchanged pressure-wait ceiling during the first treatment book's
combined pass. Its monitored duration is 413.354 seconds, with zero completed
record-level measurements and zero optimizer updates. The supervisor records
exit code one. Its failure, status, log and exit identity are preserved in the
[failure evidence](evidence/visual_context_gradient_failure_20261117.json).
The run does not establish whether auxiliary objectives interfere with learning.
The planned final model/projection preservation checks were not reached.
Peak VRAM and the full pressure sample history were not saved by that failed
implementation; no retrospective values are inferred.

The copied flow gradient can occupy up to 5.463 GiB of host memory, using the
treatment's trainable parameter count as an upper bound. This additional host
allocation may contribute to pressure, but the original failure does not
identify the source of global pressure on the shared workstation. No foreign
process is stopped and no system cache or swap setting is changed.

The single execution change is to keep the detached FP32 flow-gradient copy on
its original GPU. The original measurement function is copied explicitly, and a
CPU source comparison requires exact source equality after replacing only
`parameter.grad.detach().cpu()` by `parameter.grad.detach().clone()` at the saved
flow-gradient assignment. Both copies retain the same FP32 values. All other
forward/backward computation, RNG resets, scalar losses, groups, clipping,
retained-moment AdamW displacement, stopping rules and prospective interpretation
remain identical. The original run, loader and resource guard are reused.

The [source comparison](evidence/visual_context_device_gradient_equivalence_20261117.json)
makes zero model calls. It does not establish that the changed memory placement
fits the device budget or avoids global pressure. The added device copy must
fit the same 22-GiB allocation ceiling; exceeding it fails the recovery rather
than relaxing the limit. The earlier four arithmetic checks remain valid because
the mathematical helper is unchanged.

Bind all 56 original inputs unchanged, plus the original registration, failed-run
evidence, new source, source comparison and this protocol. Commit and push the
new binding before execution. Use the same complete 513-, 846- and 1,042-glyph
records and both exact final actors. Do not crop records or increase the
300-second pressure-wait allowance, 1,200-second monitored deadline or external
1,260-second timeout with 30 seconds of termination grace. Preserve the original
48-GiB host/20-GiB free GPU admission, 24-GiB host reserve, low-pressure guard,
48-GiB disk reserve, one GPU0 owner and protection of foreign services.

One new serial owner may run only after the original owner is verified exited.
It makes no optimizer updates and generates no answers. The same source-group
interference criterion admits at most preparation of an objective-change
preflight, never a full fit or a language claim. Failure ends this recovery with
partial evidence retained; there is no automatic retry.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/diagnose_visual_context_device_gradients.py register
# Commit and push first; verify original owner exit and fresh resource admission.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=30s 1260s \
  python scripts/diagnose_visual_context_device_gradients.py run
```

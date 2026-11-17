# Execution continuation after a pre-model resource rejection

The treatment completes all 13,785 records, 8,172 updates and 108 shards. Its
CPU audit verifies the complete batch order, all final optimizer counters and
agreement of the final deployment with the full saved state. The original
serial controller then exits at the first initialization-checkpoint generation
stage because the unchanged host-memory-pressure entry guard rejects admission.
The failed directory contains only its request and terminal status: model loading
and generation were not reached. No language result follows from this event.

The fit reports 60,127.268 seconds, including 1,421.354 seconds of pressure
waiting, and 11.647 GiB peak allocated VRAM. Minimum sampled available host RAM
is 38.464 GiB. Device-wide usage observed elsewhere is a different quantity.
The audited final checkpoint SHA-256 is
`ebcbb2ac37773d5bc2525e36dd374e0aa122b95fb6da6465bd7de8b0b1385a04`.
These are training-execution measurements, not language or efficiency evidence.

## One bounded execution change

Register one continuation with
[`resume_visual_context_comparison.py`](../scripts/resume_visual_context_comparison.py).
Its only intervention is a finite admission queue before each remaining GPU
stage, including the cold replay's GPU child. Require two samples ten seconds
apart with at least 48 GiB available host RAM, 20 GiB free GPU0 memory, 48 GiB
free disk, memory-pressure `some avg10` at most .5 and `full avg10` at most .1,
and an available project GPU mutex. The total additional admission queue is
limited to 1,800 seconds and recorded separately. Original child entry and
runtime guards remain intact; a race that fails a child guard ends this recovery.
The queue changes scheduling, not numerical execution or acceptance criteria.

Preserve the original terminal controller, logs, exit receipt, failed request
and status. Bind those files, the completed treatment report/audit/ledger,
initial state receipt, full latest state, all five deployment checkpoints,
new execution code and this protocol. Revalidate every one of the original 599
bound inputs and sources and package versions. Commit and push this recovery
binding before launching its sole controller. Hold the original comparison's
owner lock; retain the original per-child GPU mutex and two CPU threads.
No shared weights are copied and no foreign process, swap or kernel setting changes.

## Prescribed remaining work

Skip the completed treatment fit and CPU audit. Generate all five prescribed
treatment snapshots and the 24 final long-book cases, then perform the original
traced cold replay. Use a new sibling directory only for the failed initial
generation (`generated_step_0000000_admission_recovery1`); preserve its original
two-file directory unchanged. Keep all later generation paths, allowing the
unchanged cold verifier and blind reviewer to consume the original layout.

Run the original complete frozen-context fit once, its CPU audit, all original
generation cohorts and its cold replay. Check identical initial full-model and
objective hashes across arms, then prepare the unchanged 792-stream opaque review.
The generation manifest records the recovery binding and both attempts. The
failed attempt's recorded monitor duration of 5.249 seconds counts against the
first snapshot's 7,200-second
budget; the recovery passes only the remaining time to that child. Record child
wall time and additional admission queue separately. This accounting does not
replace the original fit, preparation or inherited training costs.
The original failed process's outer wall time was not recorded separately;
its monitor duration must not be presented as that complete process cost.

No model, optimizer, loss, teacher cache, source record, seed, flow step, input,
cap, checkpoint, output-reading rule or quality threshold changes. There is no
new fit for the treatment, extra epoch, checkpoint selection or reward tuning.
No model output or reference answer is inspected to choose this recovery.

## Outcomes and checks

Any error ends this sole recovery and preserves every partial file. No automatic
retry or further recovery is admitted here. A successful execution still requires
all 792 visual readings frozen before new reference scoring and both traced cold
checks. The original complete-answer, pair, readability, dependence and book
criteria remain. Proportional presentation, untouched confirmation, historical
forms, etymology and matched-quality resource savings remain separate requirements.

CPU checks verify all 17 remaining stages, omission of only the completed
treatment fit/audit, all 792 streams, unchanged resource boundaries, cumulative
first-snapshot time and rejection of any failed directory containing outputs.
They test execution safety, not model quality.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python -m unittest tests.test_visual_context_admission_recovery -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/resume_visual_context_comparison.py register
# Commit and push the exact new binding before the one continuation launch.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/resume_visual_context_comparison.py run
```

# Recovery of the registered run after a resource-guard exit

The 512-question experiment stopped after update 12,128 when its host
RAM/swap guard rejected a native-context pool refill. The original process
exited. Its last durable checkpoint was update 12,000; the 128 subsequent
updates had not reached the next scheduled checkpoint.

After host memory recovered, the existing runner was explicitly resumed from
that checkpoint. GPU0 was free and the host passed the original admission
thresholds. No other project's process was stopped. The model, prepared images,
objective, sampling schedule, learning rates and 16,000-update target remain
unchanged. All 27 registered source files retain their original hashes.

The existing resume path restores learned parameters, Adam state, sampler and
CPU/CUDA random state. It preserves the interrupted history tail before
replaying from update 12,001. The cumulative active-process limit remains
43,200 seconds across both processes: 30,268.744605 seconds had elapsed at the
exit, leaving 12,931.255395 seconds when resuming. The budget was not reset.

The [replay report](evidence/pixel_language_diversity_resource_recovery/replay_report.json)
checks the actual resumed execution against the automatically preserved tail:

- All 128 history records for updates 12,001–12,128 match exactly, including
  sampled indices, losses, gradient summaries and learning rates.
- The original and replayed record bytes have the same SHA-256, and the first
  12,000 retained records are unchanged.
- The optimizer ledger retains the 128 executions discarded by recovery.
  It contains 12,256 actual calls through replayed update 12,128; these are
  distinct from 12,128 retained updates.

The [128 records](evidence/pixel_language_diversity_resource_recovery/replayed_history12001_12128.jsonl)
and [retention hashes](evidence/pixel_language_diversity_resource_recovery/retention_manifest.json)
are retained. Verification reads existing records and makes no neural-model
calls. No original model/Adam snapshot exists at update 12,128, so equality of
the logged measurements is not an independent proof that every tensor in the
replayed state is identical to the interrupted state.

Training has passed the interruption point and continues under the
[registered protocol](pixel_language_diversity_training_protocol.md).
The final assessment and feedback-encoder audit are still pending. Recovery
does not change the [8,000-update language-quality conclusion](pixel_language_diversity_interim08000_result.md):
useful unseen-answer generation remains unestablished. The final compute
accounting must include discarded optimizer calls and both processes' active
time.

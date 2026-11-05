# Incremental writing-state computation

The current free-writing loop calls conditions(memory, full_history) before
every emitted block and final stopping decision. This recomputes prompt and
history projections through all eight causal layers. Introduce a separate
inference-only cache of each layer's attention keys and values. Compute the
prompt/start prefix once, then process only the newly re-encoded physical
image block. It receives continuous image codes, never character identifiers.

Preserve every registered V75 training, inference and evaluation source byte.
The new cache is not adopted by that experiment or its queued copy probe.
It changes no trained parameters or mathematical conditional state in real
arithmetic, but actual backend rounding may differ. GPU answer pixels,
stopping and end-to-end latency require separate replay before adoption.

First verify hidden states on CPU against the existing full-history method,
using the completed 64,520,193-parameter V74 checkpoint. Use the first CMRC
cache record and the first admitted Dolly record with 128 answer blocks,
both from the authenticated full training cache. Compare every prefix from
zero through the complete actual image-code history, including the final
stopping condition. This is assisted numerical equivalence, not autonomous
generation or a language result. No neural reader, codec, sampler or GPU
is loaded or run in this check.

FP32 state agreement uses rtol=0.00001 and atol=0.00002; report maximum/RMS
error and all stopping-threshold decisions. Also report CPU BF16 arithmetic
when supported, without treating it as GPU replay. Check exact cache growth,
the registered 128-block capacity and refusal after parameter mutation.
Record CPU-only timings as informational measurements that exclude the
reader, denoiser, image codec and GPU execution. Projection-row and attention-
score counts describe work avoided, not measured end-to-end speedup.

Source and protocol are committed before executing the comparison. Preserve
the original checkpoint and use a fresh report path. The general ILM objective
and original held-out requirements remain unchanged.

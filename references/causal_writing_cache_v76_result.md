# Incremental writing states agree on the CPU check

An inference-only attention cache agrees with the existing full-history
computation across 132 prefix states from two real image-code histories.
Maximum FP32 absolute error is 0.00000191; all learned stopping-threshold
decisions agree. The 128-block history uses 181 projected positions per
layer instead of repeatedly processing 15,093 positions.

| Actual history | Compared states | Maximum absolute error | Full history, CPU | Cached, CPU |
| --- | ---: | ---: | ---: | ---: |
| CMRC, 2 blocks | 3 | 0.00000143 | 0.063 s | 0.035 s |
| Dolly, 128 blocks | 129 | 0.00000191 | 5.568 s | 0.916 s |

These are single four-thread CPU observations on a shared host. They cover
only conditional-state computation and exclude the image reader, denoising
head, codec and GPU execution. They do not establish end-to-end inference
speed or a speed advantage over a language model. Final K/V tensor storage
for the long FP32 history is 5,931,008 bytes.

The check loads the completed 64.52M-parameter V74 writer checkpoint and
uses authenticated prompt memory and reference image-code histories. It
checks every prefix, including the final stopping query, exact cache growth,
the 128-block limit and refusal after parameter mutation. It does not generate
an answer image or assess language understanding. CPU BF16 is explicitly
skipped because this host lacks the selected native BF16 support.

The cache is separate from all registered V75 sources. Current training,
the original autonomous audit and the queued copy diagnostic continue through
the existing computation. Floating-point arithmetic on GPU can differ;
actual output pixels, stopping and timing must be replayed before adopting
the cache in a qualified image-generation path.

The [protocol](causal_writing_cache_v76_protocol.md),
[verification receipt](evidence/causal_writing_cache_v76_cpu.json) and
scripts/verify_causal_writing_cache.py bind the exact sources, weights,
histories and numerical observations. The general ILM objective remains open.
